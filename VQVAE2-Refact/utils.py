import argparse
import sys
import os

import cv2

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

# from models.vqvae import VQVAE, VQVAE_B2F
# from models.vqvae_conv3d import VQVAE
# from models.vqvae_wobbles import VQVAE
from scheduler import CycleScheduler
import distributed as dist

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def save_frames_as_video(frames, video_path, fps=30):
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames: 
        video.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)) 
      
    cv2.destroyAllWindows() 
    video.release()

def save_image(data, saveas, video=False):
    utils.save_image(
        data,
        saveas,
        nrow=data.shape[0]//2,
        normalize=True,
        range=(-1, 1),
    )

def process_data(data, device, dataset):
    if dataset >= 6:
        source, target, background, source_images, source_images_original = data
        
        img = torch.cat([source, background], axis=2).squeeze(0).to(device)
        source = source.squeeze(0)
        ground_truth = source_images.squeeze(0).to(device)

        S = source.shape[0]

        return img, S, ground_truth, source_images_original.squeeze(0).to(device)

    elif dataset == 2: 
        assert len(data) == 4
        face, rhand, lhand, ground_truth = [x.to(device) for x in data]
        img = face, rhand, lhand

        S = face.shape[0]

    else:
        if len(data) == 2: # case hand2gestures
            img, label = data
        else:
            img = data

        img = img.to(device)
        S = img.shape[0]
    
        ground_truth = img

    return img, S, ground_truth

def get_facetranslation_concatenated_on_different_channels(args, device):
    from models.vqvae import VQVAE, VQVAE_B2F
    from TemporalAlignment.dataset import TemporalAlignmentDataset

    model = VQVAE(in_channel=3*2).to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 96, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 130, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=True,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model

# takes a video with face perturbations as input 
# pretrained vqvae2 model fixes the alignment issues
def get_facetranslation_pretrained_wobbles(args, device):
    from TemporalAlignment.dataset import TemporalAlignmentDataset 
    from models.vqvae_wobbles import VQVAE

    # ckpt is the pretrained alignment checkpoint
    model = VQVAE(in_channel=3, ckpt='/ssd_scratch/cvit/aditya1/ckpts/varying_colorjitter.pt').to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 96, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 130, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=True,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model    

def get_facetranslation_video_discriminator(args, device):
    from TemporalAlignment.dataset import TemporalAlignmentDataset 
    from TemporalAlignment.models.video_discriminator import VideoDiscriminator

    model = VQVAE(in_channel=3, ckpt='/ssd_scratch/cvit/aditya1/ckpts/const_color_jitter.pt').to(device)
    # model = VQVAE(in_channel=3*2).to(device)

    disc = VideoDiscriminator(n_channels=3).to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 70, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 90, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=True,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model, disc  

# conv3d is applied before quantization
def get_facetranslation_latent_conv(args, device):
    from TemporalAlignment.dataset import TemporalAlignmentDataset 
    from models.vqvae_conv3d_latent import VQVAE

    model = VQVAE(in_channel=3*2).to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 90, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 126, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=True,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model  

# conv3d is appled before quanization and lpips loss is initialized
def get_facetranslation_latent_conv_perceptual(args, device):
    from TemporalAlignment.dataset import TemporalAlignmentDataset 
    from models.vqvae_conv3d_latent import VQVAE
    from loss import VQLPIPS

    model = VQVAE(in_channel=3*2).to(device)
    vqlpips = VQLPIPS().to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 30, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 50, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray,
        custom_validation_required=args.custom_validation)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=True,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model, vqlpips

# conv3d is applied after quantization
def get_facetranslation_quantization_conv(args, device):
    from TemporalAlignment.dataset import TemporalAlignmentDataset 
    from models.vqvae_conv3d import VQVAE

    model = VQVAE(in_channel=3*2).to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 90, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 126, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=True,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model  

# conv3d is applied after quantization and perceptual loss is added
def get_facetranslation_quantization_conv_perceptual(args, device):
    from TemporalAlignment.dataset import TemporalAlignmentDataset 
    from models.vqvae_conv3d import VQVAE
    from loss import VQLPIPS

    model = VQVAE(in_channel=3*2).to(device)
    vqlpips = VQLPIPS().to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 30, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 50, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=2)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=True,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model, vqlpips

def get_facetranslation_multipleframes_loaders_and_model(args, device):
    from datasets.face_translation_multiple_frames import FacialTransformsMultipleFramesDataset

    model = VQVAE(in_channel=3*3).to(device)

    train_dataset = FacialTransformsMultipleFramesDataset('train', 3)
    val_dataset = FacialTransformsMultipleFramesDataset('val', 3)

    sampler = dist.data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    sampler = dist.data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    return train_loader, val_loader, model

def get_facetranslation_loaders_and_model(args, device):
    from datasets.face_translation import FacialTransforms

    model = VQVAE().to(device)

    train_dataset = FacialTransforms('train')
    val_dataset = FacialTransforms('val')

    sampler = dist.data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    sampler = dist.data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    return train_loader, val_loader, model



def get_b2f_loaders_and_model(args, transform, device):
    from datasets.blob2full import Blob2Full

    dataset = Blob2Full('train', transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE_B2F().to(device)

    return loader, None, model

def get_handgesture_loaders_and_model(args, transform, device, test=False):
    from datasets.handgestures import HandGesturesDataset

    model = VQVAE().to(device)

    if test:
        dataset = cds.HandGesturesDataset('test', transform=transform)
        sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)

        return DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2
        ), None, model

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

def get_loaders_and_models(args, dataset, default_transform, device, test=False):
    if dataset == 1:
        return get_handgesture_loaders_and_model(args, default_transform, device)
    elif dataset == 2:
        return get_b2f_loaders_and_model(args, default_transform, device)
    elif dataset == 3:
        pass 
    elif dataset == 4:
        return get_facetranslation_multipleframes_loaders_and_model(args, device) 
    elif dataset == 5:
        return get_facetranslation_loaders_and_model(args, device)
    elif dataset == 6:
        return get_facetranslation_concatenated_on_different_channels(args, device)
    elif dataset == 7:
        return get_facetranslation_pretrained_wobbles(args, device)
    elif dataset == 8:
        return get_facetranslation_video_discriminator(args, device)
    elif dataset == 9:
        # conv3d is added to the output of the encoded representation before quantization
        return get_facetranslation_latent_conv(args, device)
    elif dataset == 10:
        # conv3d is added to the output of the quantization layer 
        return get_facetranslation_quantization_conv(args, device)
    elif dataset == 11:
        # conv3d is applied before quantization and lpips perceptual loss is added
        return get_facetranslation_latent_conv_perceptual(args, device)
    elif dataset == 12:
        # conv3d is applied after quantization and lpips perceptual loss is added
        return get_facetranslation_quantization_conv_perceptual(args, device)
    