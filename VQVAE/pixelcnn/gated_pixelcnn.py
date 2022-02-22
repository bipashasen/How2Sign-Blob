import math

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import time
import os 
import sys
import random

"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
current_dir = sys.path.append(os.getcwd())
pixelcnn_dir = sys.path.append(os.getcwd()+ '/pixelcnn')
# models_dir = sys.path.append(os.getcwd()+ '../models')

from transformers import SLPModel
import utils

from models.vqvae import VQVAE

from tqdm import tqdm

"""
Hyperparameters
"""
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true")

parser.add_argument("--dataset",  type=str, default='LATENT_BLOCK',
    help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ckpt", type=str, required=False)
parser.add_argument("--img_dim", type=int, default=25)
parser.add_argument("--input_dim", type=int, default=1,
    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=512,
    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--vqvae_ckpt", type=str)
parser.add_argument("--pix_ckpt", type=str)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.batch_size = args.batch_size * torch.cuda.device_count()

"""
data loaders
"""
training_data, _, train_loader, test_loader, _ = utils.load_data_and_data_loaders('LATENT_BLOCK', 'image', args.batch_size, '')

ntoken = len(training_data.label_mappng)
EOS_INDEX, PADDING_INDEX = 0, 0

model = SLPModel(ntoken, args.n_embeddings, args.img_dim, args.n_layers, args.batch_size).to(device)

vqvae = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, 
              args.embedding_dim, args.beta, device)

generate_every = 200

if args.ckpt:
    model.load_state_dict(torch.load(args.ckpt))
elif args.pix_ckpt:
    model.pixel_cnn.module.load_state_dict(torch.load(args.pix_ckpt))

criterion = nn.CrossEntropyLoss().cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

"""
train, test, and log
"""

def input_code(code):
    # code_ones = torch.ones_like(code[0]).unsqueeze(0)

    # pad = args.batch_size - len(code) - 1

    # code_first = code_ones * EOS_INDEX
    # code_padding = (code_ones * PADDING_INDEX).repeat(pad, 1, 1)

    # return torch.cat((code_first, code, code_padding), axis=0)

    # index = random.randint(0, len(code)-1)

    # return code[index, :, :].unsqueeze(0).repeat(args.batch_size, 1, 1)

    return torch.ones_like(code[0]).unsqueeze(0).repeat(args.batch_size, 1, 1)

def run(loader):
    x, label, mask = next(iter(loader))
        
    x = x.squeeze(0)
    in_x = input_code(x).cuda()
    label = label.cuda()
    mask = mask.cuda()

    # Train PixelCNN with images
    logits = model(in_x, label, mask)
    logits = logits.permute(0, 2, 3, 1).contiguous()

    x_pad = (torch.ones_like(x[0]) * PADDING_INDEX).unsqueeze(0)
    x_pad = x_pad.repeat(logits.shape[0] - x.shape[0], 1, 1)
    out_x = torch.cat((x, x_pad), axis=0).cuda()

    return criterion(
        logits.view(-1, args.n_embeddings),
        out_x.view(-1)
    ), in_x, out_x, label, mask, logits

def train():
    train_loss = []

    tq = tqdm(range(8888))
    for i in tq:
        model.train()
        start_time = time.time()

        loss, in_x, out_x, label, mask, logits = run(train_loader)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if i % generate_every == 0:
            generate_samples(epoch, i, in_x, out_x, label, mask, logits.detach())

        tq.set_description((f'avg. loss: {np.array(train_loss).mean():.5f}'))

def test(epoch):
    iters = 1
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for i in tqdm(range(iters)):
            loss, in_x, out_x, label, mask, logits = run(test_loader)
            
            val_loss.append(loss.item())

        generate_samples(epoch, 'test', in_x, out_x, label, mask, logits)

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)

vqvae.load_state_dict(torch.load(args.vqvae_ckpt)['model'])

vqvae = vqvae.to(device)

vqvae.eval()

def get_min_encodings(min_encoding_indices, model, device, B):
    min_encodings = torch.zeros(
        min_encoding_indices.shape[0], model.vector_quantization.n_e).to(device)

    min_encodings.scatter_(1, min_encoding_indices, 1)

    # get quantized latent vectors
    z_q = torch.matmul(min_encodings, model.vector_quantization.embedding.weight)

    z_q = z_q.view(B, args.img_dim, args.img_dim, args.embedding_dim)

    # reshape back to match original input shape
    return z_q.permute(0, 3, 1, 2).contiguous()

def generate_samples(epoch, step, in_x, out_x, label, mask, logits):
    model.eval()

    save_root = '/home2/bipasha31/python_scripts/CurrentWork/SLP/temp'
    B = in_x.shape[0]

    def decode(x):
        with torch.no_grad():
            x = x.view(-1, 1)
            z_q = get_min_encodings(x, vqvae, device, B)

            return vqvae.decoder(z_q)

    out_x = decode(out_x)

    def save_img(decoded, save_type):
        data = torch.cat([out_x, decoded], axis=0)
        save_image(
            data,
            f'{save_root}/epoch_{epoch}_step_{step}_{save_type}.jpg',
            nrow=B,
            normalize=True,
            range=(-1, 1),
        )
    
    with torch.no_grad():
        # generated from multinomial distribution
        x_tilde = model.generate(in_x, label, mask)
        logits = torch.argmax(nn.Softmax(dim=-1)(logits), dim=-1)

        save_img(decode(logits), 'logits')
        save_img(decode(x_tilde), 'multinomial')

BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, args.epochs):
    print("\nEpoch {}:".format(epoch))
    cur_loss = test(epoch)
    train()

    if args.save or cur_loss <= BEST_LOSS:
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), f'results/{args.dataset}_{epoch}_pixelcnn.pt')
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
