import argparse
import sys
import os

import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler

from utils import *
from config import dataset, latent_loss_weight, sample_size

from datasets.face_translation_videos import FaceTransformsVideos
from models.video_vqvae import VideoVQVAE

criterion = nn.MSELoss()

n = 3
latent_loss_weight = 0.25
device = "cuda"

validation_at = 2096

sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/VQVAE2-FaceVideo'

def save_image(data, saveas):
    utils.save_image(
        data,
        saveas,
        nrow=data.shape[0]//2,
        normalize=True,
        range=(-1, 1),
    )

def get_loaders_and_models():
    train_dataset = FaceTransformsVideos('train', n)
    val_dataset = FaceTransformsVideos('val', n)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=2)

    model = VideoVQVAE(n).to(device)

    return train_loader, val_loader, model
        
def onlysample_save(img, epoch, i):
    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    save_image(
        torch.cat([sample, out], 0), 
        f"sample/{epoch + 1}_{i}.png",)

def run_step(model, data, run_type='train'):
    def process_data(data):
        return [x.to(device) for x in data]

    perturbed, source_aligned, target_original, aligned_mask, target_mask = process_data(data)

    denoised_perturbed, latent_loss = model(perturbed)

    assert source_aligned.shape == denoised_perturbed.shape

    source_loss = criterion(aligned_mask * denoised_perturbed, source_aligned)

    aligned_mask = torch.bitwise_not(aligned_mask.bool()).int()
    
    target_loss = criterion(aligned_mask * denoised_perturbed, source_aligned * target_original)

    recon_loss = source_loss + target_loss

    if run_type == 'train':
        return recon_loss, latent_loss
    else:
        return perturbed, denoised_perturbed

def validation(model, val_loader, device, epoch, i):
    for val_i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            sample, out = run_step(model, data, 'val')

        sample, out = sample[0], out[0]

        H, W = sample.shape[-2], sample.shape[-1]

        pre_out = out[:, -3:]
        post_out = out[-1, 3:].unsqueeze(0).view(-1, 3, H, W)
        out = torch.cat((pre_out, post_out), 0)

        if val_i % (len(val_loader)//10) == 0: # save 10 results
            save_image(
                torch.cat([sample, out], 0), 
                f"{sample_folder}/{epoch + 1}_{i}_{val_i}.png")

def train(model, loader, val_loader, optimizer, scheduler, epoch, validate_at):
    loader = tqdm(loader, file=sys.stdout)

    for i, data in enumerate(loader):
        model.zero_grad()

        '''
        pertubed: B x T x 3 x H x W
        everything else: B x T x 9 x H x W
        '''
        recon_loss, latent_loss = run_step(model, data)

        loss = recon_loss + latent_loss_weight * latent_loss

        loss.backward()
        optimizer.step()

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; "
                f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            )
        )

        if i % validate_at == 0:
            model.eval()

            validation(model, val_loader, device, epoch, i)

            model.train()

def main(args):
    default_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    loader, val_loader, model = get_loaders_and_models()

    model = nn.parallel.DataParallel(model)

    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        model.module.load_state_dict(state_dict)

    if args.test:
        # test(loader, model)
        pass
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
            train(model, loader, val_loader, optimizer, scheduler, i, args.validate_at)

            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
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

