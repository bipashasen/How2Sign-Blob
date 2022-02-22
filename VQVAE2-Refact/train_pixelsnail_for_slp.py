import argparse
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import custom_datasets as cds
from custom_datasets import CodeRowVideos

from torch.nn import functional as F

try:
    from apex import amp

except ImportError:
    amp = None

from scheduler import CycleScheduler

from transformers import SLPModel

save_model_every = 5000

# 1 FOR EOS TOKEN (although, need to predict n through the n/w)
# There might not be any use of EOS token as we are not predicting EOS
# But let us keep it for now.
# 1 FOR PADDING_TOKEN
ntoken = 16212 + 2 

# Indexing from 0
PADDING_INDEX = ntoken-1
EOS_TOKEN = ntoken-2

def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss() # this thing will match between
    # generated encoding and the ground truth encoding
    # so padding shouldn't matter here at all. 
    # It will never encounter this index anyway

    if args.logdir is None:
        checkpoint_dir = 'slp_trans' if (args.sched is None) else 'slp_trans_sched'
    else:
        checkpoint_dir = args.logdir

    for i, (top, bottom, label, mask) in enumerate(loader):
        model.zero_grad()

        top, label, mask = top.to(device), label.to(device), mask.to(device)

        top = top.squeeze(0) # T x (32 x 32)
        bottom = bottom.squeeze(0) # T x (64 x 64)

        if args.hier == 'top':
            target = top
            out = model(label, top, mask)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out = model(label, bottom, mask, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )

        os.makedirs(checkpoint_dir, exist_ok=True)

        if i % save_model_every == 0:
            torch.save(
                {'model': model.state_dict(), 'args': args},
                f'{checkpoint_dir}/pixelsnail_{args.hier}_{str(epoch+1)}_{str(i + 1).zfill(3)}.pt',
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    args.gpu = torch.cuda.device_count()

    print(args)

    device = 'cuda'

    dataset = cds.HandGesturesDatasetForPixelSnail(args.path, slen=args.gpu*args.batch, PADDING_INDEX=PADDING_INDEX)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    model = SLPModel(args, ntoken, PADDING_INDEX)

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, scheduler, device)
