import argparse
import sys
import os
import random
import os.path as osp

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist

from utils import *
from config import DATASET, LATENT_LOSS_WEIGHT, DISC_LOSS_WEIGHT, SAMPLE_SIZE_FOR_VISUALIZATION

criterion = nn.MSELoss()

bce_loss = nn.BCEWithLogitsLoss()

global_step = 0

sample_size = SAMPLE_SIZE_FOR_VISUALIZATION

dataset = DATASET

CONST_FRAMES_TO_CHECK = 16

BASE = '/ssd_scratch/cvit/aditya1/video_vqvae2_results'
# sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/{}'

# checkpoint_dir = 'checkpoint_{}'

def run_step(model, data, device, run='train'):
    img, S, ground_truth = process_data(data, device, dataset)

    out, latent_loss = model(img)

    out = out[:, :3]
    
    recon_loss = criterion(out, ground_truth)
    latent_loss = latent_loss.mean()
    
    if run == 'train':
        return recon_loss, latent_loss, S, out, ground_truth
    else:
        return ground_truth, img, out

def run_step_custom(model, data, device, run='train'):
    img, S, ground_truth = process_data(data, device, dataset)

    out, latent_loss = model(img)

    out = out[:, :3] # first 3 channels of the prediction

    return out, ground_truth

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

def jitter_validation(model, val_loader, device, epoch, i, run_type, sample_folder):
    for i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            source_images, input, prediction = run_step(model, data, device, run='val')
            
        source_hulls = input[:, :3]
        background = input[:, 3:]

        saves = {
            'source': source_hulls,
            'background': background,
            'prediction': prediction,
            'source_images': source_images
        }

        if i % (len(val_loader) // 10) == 0 or run_type != 'train':
            def denormalize(x):
                return (x.clamp(min=-1.0, max=1.0) + 1)/2

            for name in saves:
                saveas = f"{sample_folder}/{epoch + 1}_{global_step}_{i}_{name}.mp4"
                frames = saves[name].detach().cpu()
                frames = [denormalize(x).permute(1, 2, 0).numpy() for x in frames]

                # os.makedirs(sample_folder, exist_ok=True)
                save_frames_as_video(frames, saveas, fps=25)

def base_validation(model, val_loader, device, epoch, i, run_type, sample_folder):
    def get_proper_shape(x):
        shape = x.shape
        return x.view(shape[0], -1, 3, shape[2], shape[3]).view(-1, 3, shape[2], shape[3])

    for val_i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            sample, _, out = run_step(model, data, device, 'val')

        if sample.shape[1] != 3:
            sample = get_proper_shape(sample[:sample_size])
            out = get_proper_shape(out[:sample_size])

        if val_i % (len(val_loader)//10) == 0: # save 10 results
            save_image(
                torch.cat([sample[:3*3], out[:3*3]], 0), 
                f"{sample_folder}/{epoch + 1}_{i}_{val_i}.png")

def validation(model, val_loader, device, epoch, i, sample_folder, run_type='train'):
    if dataset == 6:
        jitter_validation(model, val_loader, device, epoch, i, run_type, sample_folder)
    else:
        base_validation(model, val_loader, device, epoch, i, run_type, sample_folder)

def train(model, disc, loader, val_loader, optimizer, disc_optimizer, scheduler, device, epoch, validate_at, checkpoint_dir, sample_folder):
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    mse_sum = 0
    mse_n = 0
    disc_sum = 0
    disc_n = 0

    global global_step

    for i, data in enumerate(loader):

        # disc step
        if global_step%2:
            model.zero_grad()
            disc.zero_grad()

            # generator step 
            prediction, ground_truth = run_step_custom(model, data, device)

            # generate the disc predictions 
            # print(f'Ground truth shape : {ground_truth.shape}, prediction shape : {prediction.shape}')

            # test the flow for any random sequence of frames
            random_index = random.randint(0, ground_truth.shape[0] - CONST_FRAMES_TO_CHECK - 1)

            ground_truth = ground_truth[random_index : random_index + CONST_FRAMES_TO_CHECK].unsqueeze(0).permute(0, 2, 1, 3, 4)
            prediction = prediction[random_index : random_index + CONST_FRAMES_TO_CHECK].unsqueeze(0).permute(0, 2, 1, 3, 4)

            disc_real_pred = disc(ground_truth)
            disc_fake_pred = disc(prediction.detach()) 

            disc_real_loss = bce_loss(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_fake_loss = bce_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            # backprop the disc loss
            disc_loss.backward()
            disc_optimizer.step()

            # print(f'Disc real prediction : {disc_real_pred.shape}, disc fake : {disc_fake_pred.shape}')
            part_disc_sum = disc_loss.item() * S
            part_disc_n = S

            comm = {"disc_sum": part_disc_sum, "disc_n": part_disc_n}
            comm = dist.all_gather(comm)

            for part in comm:
                disc_sum += part["disc_sum"]
                disc_n += part["disc_n"]

            if dist.is_primary():
                lr = optimizer.param_groups[0]["lr"]

                loader.set_description(
                    (
                        f"epoch: {epoch + 1}; disc step; disc loss: {disc_loss.item():.5f}; "
                        f"avg disc loss: {disc_sum / disc_n:.5f}; "
                        f"lr: {lr:.5f}"
                    )
                )
            
        # gen step
        else:
            model.zero_grad()
            disc.zero_grad()

            # train the generator 
            recon_loss, latent_loss, S, prediction, ground_truth = \
                        run_step(model, data, device)

            # generate the discriminator predictions on the generator output
            random_index = random.randint(0, ground_truth.shape[0] - CONST_FRAMES_TO_CHECK - 1)

            prediction = prediction[random_index : random_index + CONST_FRAMES_TO_CHECK].unsqueeze(0).permute(0, 2, 1, 3, 4)

            disc_fake_pred = disc(prediction)

            disc_fake_loss = bce_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))

            gen_loss = recon_loss + LATENT_LOSS_WEIGHT * latent_loss + DISC_LOSS_WEIGHT * disc_fake_loss

            gen_loss.backward()

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
                        f"epoch: {epoch + 1}; gen step; mse: {recon_loss.item():.5f}; "
                        f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                        f"lr: {lr:.5f}"
                    )
                )

        global_step += 1

        if i % validate_at == 0:
            model.eval()

            validation(model, val_loader, device, epoch, i, sample_folder)

            if dist.is_primary():
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), f"{checkpoint_dir}/vqvae_{epoch+1}_{str(i + 1).zfill(4)}.pt")

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

    loader, val_loader, model, disc = get_loaders_and_models(
        args, dataset, default_transform, device, test=args.test)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

        disc = nn.parallel.DistributedDataParallel(
            disc,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        try:
            model.module.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict)

    if args.test:
        # test(loader, model, device)
        validation(model, val_loader, device, 0, 0, args.sample_folder, 'val')
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        disc_optimizer = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)
        
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
            train(model, disc, loader, val_loader, optimizer, disc_optimizer, scheduler, device, i, args.validate_at, args.checkpoint_dir, args.sample_folder)

def get_random_name(cipher_length=5):
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join([chars[random.randint(0, len(chars)-1)] for i in range(cipher_length)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    # port = (
    #     2 ** 15
    #     + 2 ** 14
    #     + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    # )

    port = random.randint(51000, 52000)

    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--checkpoint_suffix", type=str, default='')
    parser.add_argument("--validate_at", type=int, default=512)
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)
    parser.add_argument("--gray", action='store_true', required=False)
    parser.add_argument("--colorjit", type=str, default='', help='const or random or empty')
    parser.add_argument("--crossid", action='store_true', required=False)
    parser.add_argument("--sample_folder", type=str, default='samples')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoint')

    args = parser.parse_args()

    # args.n_gpu = torch.cuda.device_count()
    current_run = get_random_name()

    # sample_folder = sample_folder.format(args.checkpoint_suffix)
    args.sample_folder = osp.join(BASE, args.sample_folder + '_' + current_run)
    os.makedirs(args.sample_folder, exist_ok=True)

    args.checkpoint_dir = osp.join(BASE, args.checkpoint_dir + '_' + current_run)
    # os.makedirs(args.checkpoint_dir, exist_ok=True)

    # checkpoint_dir = checkpoint_dir.format(args.checkpoint_suffix)

    print(args, flush=True)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
