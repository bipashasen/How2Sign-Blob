import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE, VQVAE_B2F
from scheduler import CycleScheduler
import distributed as dist

from utils import *

datasets = {
    'hand2gesture': 1,
    'blob2full': 2,
    'hand2gesture4pixelsnail': 3,
    'facetranslationmultipleframes': 4
}

dataset = 4

sample_size = 25

def test(loader, model, device):
    model.eval()
    sample_size = 8
    latent_loss_weight = 0.25
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    criterion = nn.MSELoss()
    losses, recon_losses, latent_losses = [], [], []

    for i, data in enumerate(tqdm(loader)):
        img, S, gt = process_data(data, device)

        sample, gt = img[:sample_size], gt[:sample_size]

        with torch.no_grad():
            out, latent_loss = model(sample)

            recon_loss = criterion(out, gt)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            latent_losses.append(latent_loss.item())

            saveas = f"{sample_folder}/{str(i + 1).zfill(5)}.png"

            if dist.is_primary():
                save_image(torch.cat([sample, out], 0), saveas, sample_size)

                loader.set_description(
                (
                    f"mse: {recon_losses[-1]:.5f}; "
                    f"latent: {latent_losses[-1]:.3f}; "
                    f"loss: {losses[-1]:.5f}"
                ))

    if dist.is_primary():
        print(f'Mean MSE: {recon_losses.mean()}, Mean Latent: {latent_losses.mean()}, Mean Loss: {losses.mean()}')

def onlysample_validation(img):
    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    save_image(torch.cat([sample, out], 0), 
        f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
        sample_size)

def blob2full_validation(model, img):
    face, rhand, lhand = img

    face, rhand, lhand = face[:sample_size], rhand[:sample_size], lhand[:sample_size]
    sample = face, rhand, lhand

    gt = gt[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)
    
    save_image(torch.cat([face, rhand, lhand, out, gt], 0), 
        f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
        sample_size)

def base_validation(model, val_loader, device, epoch, i):
    for vi, batch in enumerate(tqdm(val_loader)):
        try:
            (vimg, vlabel) = batch
        except:
            vimg = batch
        vimg = vimg.to(device)

        sample = vimg[:sample_size]

        with torch.no_grad():
            out, _ = model(sample)

        def get_proper_shape(x):
            shape = x.shape
            return x.view(shape[0], -1, 3, shape[2], shape[3]).view(-1, 3, shape[2], shape[3])

        if sample.shape[1] != 3:
            sample = get_proper_shape(sample)
            out = get_proper_shape(out)

        if vi%(len(val_loader)//10) == 0:
            save_image(torch.cat([sample[:3*3], out[:3*3]], 0), 
                f"{sample_folder}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_{str(vi).zfill(5)}.png",
                sample_size)

def validation(model, val_loader, device, epoch, i):
    
    base_validation(model, val_loader, device, epoch, i)

def train(epoch, loader, val_loader, model, optimizer, scheduler, device, save_at):
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 8

    mse_sum = 0
    mse_n = 0

    for i, data in enumerate(loader):
        model.zero_grad()

        img, S, gt = process_data(data, device)

        out, latent_loss = model(img)
        
        recon_loss = criterion(out, gt)
        latent_loss = latent_loss.mean()
        
        loss = recon_loss + latent_loss_weight * latent_loss
        
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * S
        part_mse_n = S

        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

        if i % save_at == 0:
            model.eval()

            validation(model, val_loader, device, epoch, i)

            model.train()

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_loader, val_loader, model = get_loaders_and_models(args, transform, device, dataset, args.batch_size, test=args.test)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        model.module.load_state_dict(state_dict)

    # if test:
    #     test(loader, model, device)

    #     return

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = None
    
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(train_loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, train_loader, val_loader, model, optimizer, scheduler, device, args.save_at)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--save_at", type=int, default=4096)
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()

    print(args, flush=True)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
