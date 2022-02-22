import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler

from utils import *
from config import dataset, latent_loss_weight, sample_size

criterion = nn.MSELoss()
        
def test(loader, model, device):
    model.eval()
    
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    criterion = nn.MSELoss()

    losses, recon_losses, latent_losses = [], [], []

    for i, data in enumerate(tqdm(loader)):
        img, S, ground_truth = process_data(data, device)

        sample, ground_truth = img[:sample_size], ground_truth[:sample_size]

        with torch.no_grad():
            out, latent_loss = model(sample)

            recon_loss = criterion(out, ground_truth)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            latent_losses.append(latent_loss.item())

            saveas = f"{sample_folder}/{i+1}.png"

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

def train(model, loader, val_loader, optimizer, scheduler, device, epoch, validate_at):
    def process_data(data):
        return [x.to(device) for x in data]

    for i, data in enumerate(loader):
        model.zero_grad()

        '''
        pertubed: B x T x 3 x H x W
        everything else: B x T x 9 x H x W
        '''
        perturbed, source_aligned, target_original, aligned_mask, target_mask = process_data(data)

        denoised_perturbed = model(perturbed)

        assert source_aligned.shape == perturbed.shape

def main(args):
    device = "cuda"

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

    model = nn.parallel.DataParallel(model)

    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        model.module.load_state_dict(state_dict)

    if args.test:
        test(loader, model, device)
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
            train(model, loader, val_loader, optimizer, scheduler, device, i, args.validate_at)

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
    parser.add_argument("--validate_at", type=int, default=4096)
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()

    print(args, flush=True)

    main(args)

