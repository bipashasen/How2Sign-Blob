import argparse
import sys
import os
import os.path as osp
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist

from utils import *
from config import dataset, latent_loss_weight, sample_size

criterion = nn.MSELoss()
BASE_PATH = '/ssd_scratch/cvit/aditya1'

def run_step(model, data, device, run='train'):
    # img, S, ground_truth = process_data(data, device)
    img, ground_truth = data
    img, ground_truth = img.to(device), ground_truth.to(device)
    S = img.shape[0]
    out, latent_loss = model(img)
    
    # print(f'Output shape : {out.shape}, ground truth : {ground_truth.shape}')
    recon_loss = criterion(out, ground_truth)
    latent_loss = latent_loss.mean()
    
    if run == 'train':
        return recon_loss, latent_loss, S
    else:
        return img, out
        
def test(loader, model, device, save_dir):
    model.eval()
    
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    criterion = nn.MSELoss()

    losses, recon_losses, latent_losses = [], [], []

    for i, data in enumerate(tqdm(loader)):
        # img, S, ground_truth = process_data(data, device)
        img, ground_truth = data
        img, ground_truth = img.to(device), ground_truth.to(device)

        sample, ground_truth = img[:sample_size], ground_truth[:sample_size]

        with torch.no_grad():
            out, latent_loss = model(sample)

            recon_loss = criterion(out, ground_truth)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            latent_losses.append(latent_loss.item())

            saveas = f"{save_dir}/{i+1}.png"

            if dist.is_primary():
                save_image(torch.cat([sample, out], 0), saveas)

                loader.set_description(
                (
                    f"mse: {recon_losses[-1]:.5f}; "
                    f"latent: {latent_losses[-1]:.3f}; "
                    f"loss: {losses[-1]:.5f}"
                ))

    if dist.is_primary():
        print(f'Mean MSE: {recon_losses.mean()}, Mean Latent: {latent_losses.mean()}, Mean Loss: {losses.mean()}')

def onlysample_save(img, epoch, i):
    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    save_image(
        torch.cat([sample, out], 0), 
        f"sample/{epoch + 1}_{i}.png",)

def blob2full_validation(model, img):
    face, rhand, lhand = img

    face = face[:sample_size] 
    rhand = rhand[:sample_size]
    lhand = lhand[:sample_size]
    sample = face, rhand, lhand

    gt = gt[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)
    
    save_image(torch.cat([face, rhand, lhand, out, gt], 0), 
        f"sample/{epoch + 1}_{i}.png")

def base_validation(model, val_loader, device, epoch, i, save_dir):
    def get_proper_shape(x):
        shape = x.shape
        return x.view(shape[0], -1, 3, shape[2], shape[3]).view(-1, 3, shape[2], shape[3])

    for val_i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            sample, out = run_step(model, data, device, 'val')

        if sample.shape[1] != 3:
            sample = get_proper_shape(sample[:sample_size])
            out = get_proper_shape(out[:sample_size])

        if val_i % (len(val_loader)//10) == 0: # save 10 results
            save_image(
                torch.cat([sample[:3*3], out[:3*3]], 0), 
                f"{save_dir}/{epoch + 1}_{i}_{val_i}.png")

def validation(model, val_loader, device, epoch, i, save_dir):
    
    base_validation(model, val_loader, device, epoch, i, save_dir)

def train(model, loader, val_loader, optimizer, scheduler, device, epoch, validate_at, save_dir):
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    mse_sum = 0
    mse_n = 0

    for i, data in enumerate(loader):
        model.zero_grad()

        recon_loss, latent_loss, S = run_step(model, data, device)

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

        if i % validate_at == 0:
            model.eval()

            validation(model, val_loader, device, epoch, i, save_dir)

            model.train()

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    default_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    loader, val_loader, model = get_loaders_and_models(
        args, dataset, default_transform, device, test=args.test)

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

    if args.test:
        test(loader, model, device, args.save_dir)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        scheduler = None
        
        if args.sched == "cycle":
            scheduler = CycleScheduler(
                optimizer,
                args.lr,
                n_iter=len(loader) * args.epoch,
                momentum=None,
                warmup_proportion=0.05,
            )

        for i in range(args.epoch):
            train(model, loader, val_loader, optimizer, scheduler, device, i, args.validate_at, args.save_dir)

            if dist.is_primary():
                torch.save(model.state_dict(), f"{args.ckpt_dir}/vqvae_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )

    port = random.randint(51000, 52000)

    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--validate_at", type=int, default=4096)
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)

    random_dir = str(random.randint(10000, 99999)).zfill(5)

    parser.add_argument("--save_dir", type=str, default=osp.join(BASE_PATH, 'samples_' + random_dir))
    parser.add_argument("--ckpt_dir", type=str, default=osp.join(BASE_PATH, 'checkpoint' + random_dir))

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f'Saving ckpt and samples to : {args.ckpt_dir}, {args.save_dir}')

    args.n_gpu = torch.cuda.device_count()

    print(args, flush=True)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
