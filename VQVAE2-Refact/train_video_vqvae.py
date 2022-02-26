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

from datasets.face_translation_videos3 import FaceTransformsVideos
from models.video_vqvae import VideoVQVAE

from loss import VQLPIPSWithDiscriminator

criterion = nn.L1Loss()

latent_loss_weight = 0.25
device = "cuda"

global_step = 0

adversarial_disc_start = 64

sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/{}'

def save_image(data, saveas, dp):
    utils.save_image(
        data,
        saveas,
        nrow=data.shape[0]//dp,
        normalize=True,
        range=(-1, 1),
    )

def get_loaders_and_models(args):
    train_dataset = FaceTransformsVideos('train', args.n, args.max_frame_len)
    val_dataset = FaceTransformsVideos('val', args.n, args.max_frame_len)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=2)

    model = VideoVQVAE(args.n).to(device)

    adversarial_model = VQLPIPSWithDiscriminator(disc_start=adversarial_disc_start).to(device)

    return train_loader, val_loader, model, adversarial_model
        
def onlysample_save(img, epoch, i):
    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    save_image(
        torch.cat([sample, out], 0), 
        f"sample/{epoch + 1}_{i}.png",)

def get_model_last_layer(model):
    return model.vqvae.dec.blocks[-1].weight

def get_adversarial_turn():
    return global_step % 2 # 0: generator, 1: discriminator

def get_adversarial_loss(model, adversarial_model, recon_loss, target, denoised_perturbed):
    last_layer = get_model_last_layer(model)

    shape = target.shape

    target = target.view(-1, 3, shape[-2], shape[-1])
    denoised_perturbed = denoised_perturbed.view(-1, 3, shape[-2], shape[-1])

    loss = adversarial_model(
        recon_loss,
        target, 
        denoised_perturbed,
        get_adversarial_turn(), 
        global_step, 
        last_layer=last_layer)

    target = target.view(shape)
    denoised_perturbed = denoised_perturbed.view(shape)

    return loss

def run_step(model, adversarial_model, data, target_by_source_mask=False, no_enlargement=True, run_type='train'):
    def process_data(data):
        return [x.to(device) for x in data]

    perturbed, source_aligned, target_original, aligned_mask, aligned_mask_no_enlargement, target_mask = process_data(data)

    # latent loss
    denoised_perturbed, latent_loss = model(perturbed)

    if not run_type == 'train':
        return perturbed, source_aligned, target_original, denoised_perturbed

    latent_loss = latent_loss.mean()

    assert source_aligned.shape == denoised_perturbed.shape

    # to reduce supervision on the edges? not sure if this will work.
    # aligned_mask = torch.randint(0, 2, aligned_mask.shape) * aligned_mask
    
    if no_enlargement:
        aligned_mask = aligned_mask_no_enlargement

    # reconstruction loss
    source_loss = criterion(aligned_mask * denoised_perturbed, aligned_mask * source_aligned)

    if target_by_source_mask:
        target_mask = torch.bitwise_not(aligned_mask.bool()).int()

    target_loss = criterion(target_mask * denoised_perturbed, target_mask * target_original)

    recon_loss = source_loss + target_loss

    # adversarial loss
    adversarial_loss = get_adversarial_loss(model, adversarial_model, recon_loss, target_original, denoised_perturbed)

    return recon_loss, latent_loss, adversarial_loss

def validation(model, adversarial_model, val_loader, device, epoch, i):
    def transform_video_dimension(x, H, W):
        pre_x = x[0, :-3].view(-1, 3, H, W)
        post_x = x[:, -3:].unsqueeze(0).view(-1, 3, H, W)
        return torch.cat((pre_x, post_x), 0)

    for val_i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            sample, source, target, out = run_step(model, adversarial_model, data, run_type='val')

        sample, source, target, out = sample[0], source[0], target[0], out[0]

        H, W = sample.shape[-2], sample.shape[-1]
        source = transform_video_dimension(source, H, W)
        target = transform_video_dimension(target, H, W)
        out = transform_video_dimension(out, H, W)

        if True: #val_i % (len(val_loader)//10) == 0: # save 10 results
            save_image(
                torch.cat([sample, source, target, out], 0), 
                f"{sample_folder}/{epoch + 1}_{i}_{global_step}_{val_i}.png", dp=4)

def train(
    model, 
    adversarial_model, 
    loader, 
    val_loader, 
    optimizer, 
    optimizer_adversarial, 
    scheduler, 
    epoch, 
    validate_at):
    global global_step

    loader = tqdm(loader, file=sys.stdout)

    for i, data in enumerate(loader):
        model.zero_grad()
        adversarial_model.zero_grad()

        '''
        pertubed: B x T x 3 x H x W
        everything else: B x T x 9 x H x W
        '''
        recon_weight = 1.0 if global_step < adversarial_disc_start else 0.25

        recon_loss, latent_loss, adversarial_loss = run_step(model, adversarial_model, data)

        if get_adversarial_turn() == 0: # generator
            loss = (recon_loss * recon_weight) + (latent_loss_weight * latent_loss) + adversarial_loss

            loss.backward()
            optimizer.step()

        else:
            adversarial_loss.backward()
            optimizer_adversarial.step()

        global_step += 1

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; "
                f"recon: {recon_loss.item():.3f} "
                f"adversarial: {adversarial_loss.item():.3f} "
                f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            )
        )

        if global_step % validate_at == 1:
            model.eval()

            validation(model, adversarial_model, val_loader, device, epoch, 0)

            model.train()

def main(args):
    start = 0
    global global_step
    loader, val_loader, model, adversarial_model = get_loaders_and_models(args)

    # model = nn.parallel.DataParallel(model)

    if args.ckpt:
        if ',' in args.ckpt:
            ckpts = args.ckpt.split(',')
            model.load_state_dict(torch.load(ckpts[0]))
            adversarial_model.load_state_dict(torch.load(ckpts[1]))
            start = int(ckpts[0].split('_')[-1].split('.')[0]) + 1
            global_step = (start-1)*13
            print(f'Loading from checkpoints {ckpts} from epoch {start} and {global_step} global_step')
        else:
            state_dict = torch.load(args.ckpt)
            state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
            model.vqvae.load_state_dict(state_dict)

    if args.test:
        # test(loader, model)
        pass
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        optimizer_adversarial = optim.Adam(adversarial_model.parameters(), lr=args.lr, betas=(0.5, 0.9))
        
        scheduler = None
        
        if args.sched == "cycle":
            scheduler = CycleScheduler(
                optimizer,
                args.lr,
                n_iter=len(loader) * args.epoch,
                momentum=None,
                warmup_proportion=0.05,
            )

        for i in range(args.epoch)[start:]:
            train(
                model, 
                adversarial_model, 
                loader, 
                val_loader, 
                optimizer, 
                optimizer_adversarial, 
                scheduler, 
                i, 
                args.validate_at)

            checkpoint_dir = f"checkpoint_{args.suffix}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(model.state_dict(), f"{checkpoint_dir}/vqvae_{str(i + 1).zfill(3)}.pt")
            torch.save(adversarial_model.state_dict(), f"{checkpoint_dir}/adversarial_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max_frame_len", type=int, default=48)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--validate_at", type=int, default=128)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)
    parser.add_argument("--save_folder", type=str, default="VQVAE2-FaceVideo")

    args = parser.parse_args()

    sample_folder = sample_folder.format(args.save_folder)
    os.makedirs(sample_folder, exist_ok=True)

    args.n_gpu = torch.cuda.device_count()

    args.batch_size = args.batch_size * args.n_gpu

    print(args, flush=True)

    main(args)

