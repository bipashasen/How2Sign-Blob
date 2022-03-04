'''
VQVAE-Videos in an non-autoregressive manner. 
'''
import argparse
import sys
import os

import random
import cv2
import numpy as np

import kornia

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils, io

from tqdm import tqdm

from models.combined_vqvae import VQVAE
from models.pre_combined_vqvae import TemporalAlignment

from TemporalAlignment.dataset import TemporalAlignmentDataset

from loss import VQLPIPSWithDiscriminator

# torch.autograd.set_detect_anomaly(True)

device = "cuda"

global_step = 0

run_combined_network = True

checkpoint_dir = 'checkpoint_{}'

criterion = nn.MSELoss()

latent_loss_weight = 0.25

sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/{}'


# preceptual_model = VQLPIPSWithDiscriminator(disc_start=2000000000000).to(device)

def save_frames_as_video(frames, video_path, fps=25):
    os.makedirs(sample_folder, exist_ok=True)
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames: 
        video.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)) 
      
    cv2.destroyAllWindows() 
    video.release()

def absolute_loss(x, xrec):
    return torch.abs(x - xrec)

def run_step(model, data, run_type='train'):
    def process_data(data):
        return [x.to(device).squeeze(0) for x in data]

    source_hulls, _, background, source_images = process_data(data)

    prediction, latent_loss = model(source_hulls, background)

    if not run_type == 'train':
        return source_hulls, prediction, background, source_images
    
    # recon_loss = criterion(prediction, source_images)
    recon_loss = absolute_loss(source_images, prediction)
    
    latent_loss = latent_loss.mean()

    # perceptual_loss, recon_loss = preceptual_model(recon_loss, source_images, prediction)
    perceptual_loss = torch.tensor([0.]).to(device)
    recon_loss = recon_loss.mean()

    return recon_loss, latent_loss, perceptual_loss

def validation(model, val_loader, device, epoch, mode='train'):
    for i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            source_hulls, prediction, background, source_images = run_step(model, data, run_type='val')

        saves = {
            'source': source_hulls,
            'background': background,
            'prediction': prediction,
            'source_images': source_images
        }

        if i % (len(val_loader) // 10) == 0 or mode != 'train':
            def denormalize(x):
                return (x.clamp(min=-1.0, max=1.0) + 1)/2

            for name in saves:
                saveas = f"{sample_folder}/{epoch + 1}_{global_step}_{i}_{name}.mp4"
                frames = saves[name].detach().cpu()
                frames = [denormalize(x).permute(1, 2, 0).numpy() for x in frames]

                save_frames_as_video(frames, saveas, fps=25)

def train(model, loader, val_loader, optimizer, scheduler, epoch, validate_at):
    global global_step

    loader = tqdm(loader, file=sys.stdout)

    for i, data in enumerate(loader):
        model.zero_grad()

        recon_loss, latent_loss, perceptual_loss = run_step(model, data)

        loss = recon_loss + (latent_loss_weight * latent_loss) + perceptual_loss

        loss.backward()
        optimizer.step()

        global_step += 1

        loader.set_description(
            ( 
                f"epoch: {epoch+1} "
                f"global_step: {global_step} "
                f"rec_loss: {recon_loss.item():.3f} "
                f"latent_loss: {latent_loss.item():.3f} "
                f"perceptual_loss: {perceptual_loss.item():.3f}"
            )
        )

        if global_step % validate_at == 1:
            model.eval()

            validation(model, val_loader, device, epoch)

            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(model.state_dict(), f"{checkpoint_dir}/temporal_{epoch+1}_{str(i + 1).zfill(3)}.pt")

            model.train()

def get_loaders_and_models(args):
    train_dataset = TemporalAlignmentDataset('train', args.max_frame_len)
    val_dataset = TemporalAlignmentDataset('val', args.max_frame_len)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=2)

    model = VQVAE().to(device)
    # model = TemporalAlignment().to(device)

    return train_loader, val_loader, model

def main(args):
    start = 0
    
    global global_step
    
    loader, val_loader, model = get_loaders_and_models(args)

    model = nn.parallel.DataParallel(model)

    def load_missing(model, pretrained_dict):
        model_dict = model.module.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)), flush=True)
        print('miss matched params:', missed_params, flush=True)
        model_dict.update(pretrained_dict)
        model.module.load_state_dict(model_dict) 

    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        try:
            model.module.load_state_dict(state_dict)
        except:
            load_missing(model, state_dict)
        start = int(args.ckpt.split('_')[-1].split('.')[0]) + 1
        global_step = (start-1)*len(loader)

    if args.test:
        validation(model, val_loader, device, 0, 'val')
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

        for i in range(args.epoch)[start:]:
            train(model, loader, val_loader, optimizer, scheduler, i, args.validate_at)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frame_len", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--validate_at", type=int, default=128)
    parser.add_argument("--checkpoint_suffix", type=str, default='')
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()

    args.batch_size = 1 # video-level

    args.max_frame_len = args.max_frame_len * args.n_gpu

    checkpoint_dir = checkpoint_dir.format(args.checkpoint_suffix)

    sample_folder = sample_folder.format(args.checkpoint_suffix)

    pydir = os.path.join(checkpoint_dir, 'py')

    os.makedirs(pydir, exist_ok=True)
    os.system(f'cp -r *.py {pydir}') # keep forgetting the state

    # if args.test:
    #     args.max_frame_len *= 20

    print(args, flush=True)

    main(args)

