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

sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/VQVAE2-FaceMultiFrames'

os.makedirs(sample_folder, exist_ok=True)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def save_image(data, saveas, sample_size):
    utils.save_image(
        data,
        saveas,
        nrow=data.shape[0]//2,
        normalize=True,
        range=(-1, 1),
    )

def process_data(data, device):
    # if dataset == 2:
    #     face, rhand, lhand, gt = [x.to(device) for x in data]

    #     img = face, rhand, lhand

    #     S = face.shape[0]
    # else:
    try:
        img, label = data
    except:
        img = data

    img = img.to(device)

    S = img.shape[0]
    gt = img

    return img, S, gt

def get_facetranslation_loaders_and_model(args, batch_size, device):
    from datasets.face_translation_multiple_frames import FacialTransformsMultipleFramesDataset

    model = VQVAE(in_channel=3*3).to(device)

    train_dataset = FacialTransformsMultipleFramesDataset('train', 3)
    val_dataset = FacialTransformsMultipleFramesDataset('val', 3)

    sampler = dist.data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    sampler = dist.data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size // args.n_gpu, sampler=sampler, num_workers=2
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

def get_handgesture_loaders_and_model(args, transform, device, batch_size = 32, test=False):
    from datasets.handgestures import HandGesturesDataset

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

def get_loaders_and_models(args, transform, device, dataset, batch_size, test=False):
    if dataset == 4:
        return get_facetranslation_loaders_and_model(args, batch_size, device) 