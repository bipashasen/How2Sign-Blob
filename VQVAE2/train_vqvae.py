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

import custom_datasets as cds

visual_folder = 'outputs_Full'
sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/VQVAE2'

# os.makedirs(visual_folder, exist_ok=True)
# os.makedirs('sample', exist_ok=True)
os.makedirs(sample_folder, exist_ok=True)

blob2full = False

def save_image(data, saveas, sample_size):
    utils.save_image(
        data,
        saveas,
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )

def get_b2f_loaders_and_model(args, transform, device):
    # dataset = datasets.ImageFolder(args.path, transform=transform)
    dataset = cds.Blob2Full('train', transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE_B2F().to(device)

    return loader, None, model

def get_loaders_and_model(args, transform, device, test=False):
    batch_size = 32

    model = VQVAE().to(device)

    if test:
        dataset = cds.HandGesturesDataset('test', transform=transform)
        sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)

        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=2
        ), model

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    dataset = cds.HandGesturesDataset('train', transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    val_dataset = cds.HandGesturesDataset('val', transform=transform)
    val_sampler = dist.data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size // args.n_gpu, sampler=val_sampler, num_workers=2
    )

    return loader, val_loader, model

def process_data(data, device):
    if blob2full:
        face, rhand, lhand, gt = [x.to(device) for x in data]

        img = face, rhand, lhand

        S = face.shape[0]
    else:
        img, label = data

        img = img.to(device)

        S = img.shape[0]
        gt = img

    return img, S, gt

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

            run_for = 10

            if val_loader:
                for vi in tqdm(range(run_for)):
                    (vimg, vlabel) = next(iter(val_loader))
                    vimg = vimg.to(device)

                    sample = vimg[:sample_size]

                    with torch.no_grad():
                        out, _ = model(sample)

                    save_image(torch.cat([sample, out], 0), 
                        f"{sample_folder}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_{str(vi).zfill(5)}.png",
                        sample_size)
            elif blob2full:
                face, rhand, lhand = img

                face, rhand, lhand = face[:sample_size], rhand[:sample_size], lhand[:sample_size]
                sample = face, rhand, lhand

                gt = gt[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)
                
                save_image(torch.cat([face, rhand, lhand, out, gt], 0), 
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    sample_size)
            else:
                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                save_image(torch.cat([sample, out], 0), 
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    sample_size)

            comm = dist.all_gather(comm)
            model.train()

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if args.test:
        loader, model = get_loaders_and_model(args, transform, device, test=True)

    else:
        loader, val_loader, model = get_b2f_loaders_and_model(args, transform, device)\
            if blob2full else get_loaders_and_model(args, transform, device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt))

    if test:
        test(loader, model, device)

        return

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
        train(i, loader, val_loader, model, optimizer, scheduler, device, args.save_at)

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

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--save_at", type=int, default=100)
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)
    # parser.add_argument("path", type=str)

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()

    print(args, flush=True)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
