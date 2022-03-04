import argparse
import sys
import os

import random
import cv2
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils, io

from tqdm import tqdm

from scheduler import CycleScheduler

from utils import *
from config import dataset, latent_loss_weight, sample_size

from datasets.face_translation_videos3 import FaceTransformsVideos
from models.video_vqvae import VideoVQVAE

from loss import VQLPIPSWithDiscriminator, SiameseNetworkFaceSimilarity

from config import *

criterion = nn.L1Loss()

latent_loss_weight = 0.25
device = "cuda"

global_step = 0

checkpoint_dir = f'checkpoint_{checkpoint_suffix}'

sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/{}'

sample_folder = sample_folder.format(save_folder)
os.makedirs(sample_folder, exist_ok=True)

pydir = os.path.join(checkpoint_dir, 'py')
os.makedirs(pydir, exist_ok=True)
os.system(f'cp train_video_vqvae.py {pydir}') # keep forgetting the state
os.system(f'cp config.py {pydir}') # keep forgetting the state

def save_image(data, saveas, dp):
    os.makedirs(sample_folder, exist_ok=True)

    utils.save_image(
        data,
        saveas,
        nrow=data.shape[0]//dp,
        normalize=True,
        range=(-1, 1),
    )

def save_frames_as_video(frames, video_path, fps=30):
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames: 
        video.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)) 
      
    cv2.destroyAllWindows() 
    video.release()
            
    # print(f'Video {video_path} written successfully')
        
def onlysample_save(img, epoch, i):
    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    save_image(
        torch.cat([sample, out], 0), 
        f"sample/{epoch + 1}_{i}.png",)

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

    disc_model = VQLPIPSWithDiscriminator(disc_start=adversarial_disc_start).to(device)

    facesim_model = SiameseNetworkFaceSimilarity().to(device)
    # facesim_model.load_state_dict(torch.load('facesim/ep_99.pt'))

    facesim_model.eval()

    models = [model, disc_model, facesim_model]

    return train_loader, val_loader, models

def get_model_last_layer(model):
    return model.vqvae.dec.blocks[-1].weight

def get_adversarial_turn():
    return global_step % 2 # 0: generator, 1: discriminator

def absolute_loss(x, xrec):
    return torch.abs(x - xrec)

def calc_disc_loss(models, recon_loss, target, denoised_perturbed, new_shape):
    last_layer = get_model_last_layer(models[0])

    return models[1](
        recon_loss.view(new_shape),
        target.view(new_shape), 
        denoised_perturbed.view(new_shape),
        get_adversarial_turn(), 
        global_step, 
        last_layer=last_layer,
        perceptual_loss=perceptual_loss)

def run_step(models, data, run_type='train'):
    def process_data(data):
        return [x.to(device) for x in data]

    # expand input
    data = process_data(data)
    perturbed = data[0] # convex hull of source pasted on target. 
    # types of issue - 1) scaling, 2) alignment, 3) lighting, 4) pose, 5) blur.
    source_aligned = data[1] # face of the source aligned.
    source_image_aligned = data[2] # full source image aligned
    target_original = data[3] # original target
    aligned_mask = data[4] # transformed background mask of transformed source.
    # this mask has enlarged convex hull of the source image (that is pasted on target)
    aligned_mask_no_enlargement = data[5] # the true mask of the non-enlarged convex hull
    target_mask = data[6] # face mask of the target image (full face gone) to get background.

    # latent loss
    denoised_perturbed, latent_loss = models[0](perturbed)

    if not run_type == 'train':
        return perturbed,\
            source_aligned,\
            source_image_aligned,\
            target_original,\
            denoised_perturbed

    shape = perturbed.shape
    new_shape = [-1, 3, shape[-2], shape[-1]]

    assert source_aligned.shape == denoised_perturbed.shape

    # Loss no. 1 - To get the codebook right.
    latent_loss = latent_loss.mean()
    
    # Loss no. 2 - To get the source face right.
    if loss_with_no_enlargement: 
        # if true, then we use the smallest convex 
        # hull of the source to calculate the loss.
        # default and recommended is True.
        aligned_mask = aligned_mask_no_enlargement

    source_loss = absolute_loss(
        aligned_mask * denoised_perturbed,
        aligned_mask * source_aligned)

    if target_by_source_mask:
        # if true, then the target-source is used for background loss.
        # default and recommended is False
        target_mask = torch.bitwise_not(aligned_mask.bool()).int()

    target_loss = absolute_loss(
        target_mask * denoised_perturbed, 
        target_mask * target_original)

    adaptive_weight = min( # eventually, we want to decay the source face loss.
        adaptive_source_rec_weight_start/(global_step+1e-8), 1.0)

    # source_recon_weight = 1.0 if global_step < adversarial_disc_start else adaptive_weight
    source_recon_weight = adaptive_weight

    recon_loss = (source_loss * source_recon_weight) + target_loss

    # Loss no. 3 - Adversarial loss, to see if the generated face looks real or not?
    # Loss no. 4 - LPIPS perceptual loss, to see the perceptual similarity of the 
    #              denoised output to the target (lighting mostly).
    adversarial_loss, perceptual_loss, recon_loss = calc_disc_loss(
        models, 
        recon_loss, 
        target_original, 
        denoised_perturbed, 
        new_shape)

    # Loss no. 5 - Once the reconstruction loss is decayed for source, how do we 
    #              make sure that the original face is preserved? 
    # face_similarity_loss = models[2](
    #     denoised_perturbed.view(new_shape).mean(1), # grayscale
    #     source_image_aligned.view(new_shape).mean(1)) # grayscale
    face_similarity_loss = torch.tensor([0.])

    # Aggregate and send.
    losses = {
        'latent_loss': latent_loss,
        'recon_loss': recon_loss, 
        'adversarial_loss': adversarial_loss,
        'perceptual_loss': perceptual_loss,
        'facesim_loss': face_similarity_loss,
    }

    return losses, source_recon_weight

def validation(models, val_loader, device, epoch):
    def transform_video_dimension(x, H, W):
        pre_x = x[0, :-3].view(-1, 3, H, W)
        post_x = x[:, -3:].unsqueeze(0).view(-1, 3, H, W)
        return torch.cat((pre_x, post_x), 0)

        # pre_x = x[:, :3].unsqueeze(0).view(-1, 3, H, W)
        # post_x = x[-1, 3:].view(-1, 3, H, W)
        # return torch.cat((pre_x, post_x), 0)

    for val_i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            sample, source, source_original, target, out = run_step(models, data, run_type='val')

        sample, source, source_original, target, out = sample[0], source[0], source_original[0], target[0], out[0]

        H, W = sample.shape[-2], sample.shape[-1]
        source = transform_video_dimension(source, H, W)
        source_original = transform_video_dimension(source_original, H, W)
        target = transform_video_dimension(target, H, W)
        out = transform_video_dimension(out, H, W)

        # save_st = [sample, source, source_original, target, out]
        save_st = {
            'sample': sample,
            'source': source,
            'source_original': source_original,
            'target': target,
            'out': out
        }

        if True: #val_i % (len(val_loader)//10) == 0: # save 10 results
            def denormalize(x):
                return (x.clamp(min=-1.0, max=1.0) + 1)/2

            for name in save_st:
                saveas = f"{sample_folder}/{epoch + 1}_{global_step}_{val_i}_{name}.mp4"
                # save_frames_as_video(save_st[name].detach().cpu(), saveas)
                frames = save_st[name].detach().cpu()
                frames = [denormalize(x).permute(1, 2, 0).numpy() for x in frames]

                save_frames_as_video(frames, saveas, fps=25)
def train(
    models, 
    loader, 
    val_loader, 
    optims,
    scheduler, 
    epoch, 
    validate_at):
    global global_step

    loader = tqdm(loader, file=sys.stdout)

    for i, data in enumerate(loader):
        for model in models:
            model.zero_grad()

        '''
        pertubed: B x T x 3 x H x W
        everything else: B x T x 9 x H x W
        '''
        losses, source_recon_weight = run_step(models, data)

        if get_adversarial_turn() == 0: # generator
            loss = (losses['latent_loss'] * latent_loss_weight) 
            loss += (losses['recon_loss'] * recon_weight) 
            loss += losses['perceptual_loss']
            loss += losses['adversarial_loss'] 
            # loss += losses['facesim_loss']

            loss.backward()
            optims[0].step()

        else:
            losses['perceptual_loss'].backward()
            optims[1].step()

        global_step += 1

        loader.set_description(
            (
                f"epoch: {epoch + 1}; "
                f"latent: {losses['latent_loss'].item():.3f}; "
                f"recon: {losses['recon_loss'].item():.3f} "
                f"percep: {losses['perceptual_loss'].item():.3f} "
                f"adversarial: {losses['adversarial_loss'].item():.3f} "
                # f"facesim: {losses['facesim_loss'].item()} "
                f"lr: {optims[0].param_groups[0]['lr']:.5f} "
                f"s_recon_w: {source_recon_weight:.3f}"
            )
        )

        if global_step % validate_at == 1:
            for model in models:
                model.eval()

            validation(models, val_loader, device, epoch)

            models[0].train()
            models[1].train()

def main(args):
    start = 0
    global global_step
    loader, val_loader, models = get_loaders_and_models(args)

    # model = nn.parallel.DataParallel(model)

    if args.ckpt:
        if ',' in args.ckpt:
            ckpts = args.ckpt.split(',')
            models[0].load_state_dict(torch.load(ckpts[0]))
            models[1].load_state_dict(torch.load(ckpts[1]))
            start = int(ckpts[0].split('_')[-1].split('.')[0]) + 1
            global_step = (start-1)*len(loader)
            print(f'Loading from checkpoints {ckpts} from epoch {start} and {global_step} global_step')
        else:
            state_dict = torch.load(args.ckpt)
            state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
            models[0].vqvae.load_state_dict(state_dict)

    if args.test:
        validation(models, val_loader, device, 0)
    else:
        optimizer = optim.Adam(models[0].parameters(), lr=args.lr)

        optimizer_adversarial = optim.Adam(models[1].parameters(), lr=args.lr, betas=(0.5, 0.9))
        
        optims = [optimizer, optimizer_adversarial]

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
                models,
                loader, 
                val_loader, 
                optims,
                scheduler, 
                i, 
                args.validate_at)

            os.makedirs(checkpoint_dir, exist_ok=True)

            if i % 100 == 0:
                torch.save(models[0].state_dict(), f"{checkpoint_dir}/vqvae_{str(i + 1).zfill(3)}.pt")
                torch.save(models[1].state_dict(), f"{checkpoint_dir}/adversarial_{str(i + 1).zfill(3)}.pt")

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
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()

    args.batch_size = args.batch_size * args.n_gpu

    if args.test:
        args.max_frame_len *= 20

    print(args, flush=True)

    main(args)

