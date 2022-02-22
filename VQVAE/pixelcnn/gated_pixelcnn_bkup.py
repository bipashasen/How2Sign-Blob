import math

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import time
import os 
import sys

"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
current_dir = sys.path.append(os.getcwd())
pixelcnn_dir = sys.path.append(os.getcwd()+ '/pixelcnn')
# models_dir = sys.path.append(os.getcwd()+ '../models')

from pixelcnn.models import GatedPixelCNN
import utils

from models.vqvae import VQVAE

from tqdm import tqdm

"""
Hyperparameters
"""
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
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

args = parser.parse_args()

args.batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
data loaders
"""
if args.dataset == 'LATENT_BLOCK':
    _, _, train_loader, test_loader, _ = utils.load_data_and_data_loaders('LATENT_BLOCK', args.batch_size)
else:
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=True, download=True,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_Size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=False,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

model = GatedPixelCNN(args.n_embeddings, args.img_dim**2, args.n_layers).to(device)

vqvae = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, 
              args.embedding_dim, args.beta, device)

vqvae.load_state_dict(torch.load(args.vqvae_ckpt)['model'])

vqvae = vqvae.to(device)

vqvae.eval()

if args.ckpt:
    model.load_state_dict(torch.load(args.ckpt))

# model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def get_min_encodings(min_encoding_indices, model, device, B):
    min_encodings = torch.zeros(
        min_encoding_indices.shape[0], model.vector_quantization.n_e).to(device)

    min_encodings.scatter_(1, min_encoding_indices, 1)

    # get quantized latent vectors
    z_q = torch.matmul(min_encodings, model.vector_quantization.embedding.weight)

    z_q = z_q.view(B, args.img_dim, args.img_dim, args.embedding_dim)

    # reshape back to match original input shape
    return z_q.permute(0, 3, 1, 2).contiguous()

"""
train, test, and log
"""

def train():
    train_loss = []

    tq = tqdm(range(8888))
    for i in tq:
        x, label = next(iter(train_loader))

        start_time = time.time()
        x = x.squeeze(0).unsqueeze(1)
        # print(f'x.shape: {x.shape}')
        
        if args.dataset == 'LATENT_BLOCK':
            x = (x[:, 0]).cuda()
        else:
            x = (x[:, 0] * (K-1)).long().cuda()
        label = label.cuda().view(-1)
        # print(f'label: {label.shape}')
       
        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        # print(f'logits: {logits.shape}')

        loss = criterion(
            logits.view(-1, args.n_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        tq.set_description((f'avg. loss: {np.array(train_loss).mean():.5f}'))

        # if (batch_idx + 1) % args.log_interval == 0:
        #     print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
        #         batch_idx * len(x), len(train_loader.dataset),
        #         args.log_interval * batch_idx / len(train_loader),
        #         np.asarray(train_loss)[-args.log_interval:].mean(0),
        #         time.time() - start_time
        #     ))


def test():
    iters = 200
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for i in tqdm(range(iters)):
            (x, label) = next(iter(test_loader))
            x = x.squeeze(0).unsqueeze(1)

            if args.dataset == 'LATENT_BLOCK':
                x = (x[:, 0]).cuda()
            else:
                x = (x[:, 0] * (args.n_embeddings-1)).long().cuda()

            label = label.cuda().view(-1)

            logits = model(x, label)
            
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )
            
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples(epoch):
    model.eval()
    B = 16
    n = int(math.sqrt(B))

    with torch.no_grad():
        label = torch.arange(1).expand(n, n).contiguous().view(-1)
        label = label.long().cuda()

        x_tilde = model.generate(label, shape=(args.img_dim,args.img_dim), batch_size=B)
        
        # print(x_tilde[0])

        saveas = 'prior_samples'
        os.makedirs(saveas, exist_ok=True)

        decoded = []

        with torch.no_grad():
            x_tilde = x_tilde.view(-1, 1)

            z_q = get_min_encodings(x_tilde, vqvae, device, B)

            decoded = vqvae.decoder(z_q)
            
        np.save(f'{saveas}/{epoch}', decoded.detach().cpu().numpy())

        save_image(
            decoded,
            f'/home2/bipasha31/python_scripts/CurrentWork/SLP/temp/dummy_{epoch}.jpg',
            nrow=B,
            normalize=True,
            range=(-1, 1),
        )

BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, args.epochs):
    # generate_samples(epoch)

    print("\nEpoch {}:".format(epoch))
    cur_loss = test()
    train()

    if args.save or cur_loss <= BEST_LOSS:
        # BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), f'results/{args.dataset}_{epoch}_pixelcnn.pt')
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
