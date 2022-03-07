import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from models.vqvae import VQVAE, VQVAE_B2F
from scheduler import CycleScheduler
import distributed as dist

from config import sample_folder

os.makedirs(sample_folder, exist_ok=True)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def save_image(data, saveas):
    utils.save_image(
        data,
        saveas,
        nrow=data.shape[0]//2,
        normalize=True,
        range=(-1, 1),
    )

def process_data(data, device):
    if len(data) == 4: # case blob2full
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

def get_facetranslation_loaders_and_model(args, device):
    from datasets.face_translation_multiple_frames import FacialTransformsMultipleFramesDataset

    model = VQVAE(in_channel=3*3).to(device)

    train_dataset = FacialTransformsMultipleFramesDataset(mode='train')
    val_dataset = FacialTransformsMultipleFramesDataset(mode='val')

    sampler = dist.data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    sampler = dist.data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    return train_loader, val_loader, model

def get_facetranslation_loaders_and_model_single_frame(args, device):
    from datasets.face_translation_single_frame_perturbed import FacialTransformsSingleFrameDatasetPerturbed

    model = VQVAE(in_channel=3).to(device)

    # defines the training mode and the number of channels 
    train_dataset = FacialTransformsSingleFrameDatasetPerturbed('train', 1)
    val_dataset = FacialTransformsSingleFrameDatasetPerturbed('val', 1)

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
        return get_facetranslation_loaders_and_model(args, device) 
    elif dataset == 5:
        return get_facetranslation_loaders_and_model_single_frame(args, device)

