import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from tqdm import tqdm

import sys
import distributed as dist
import utils
from models.vqvae import VQVAE, VQVAE_Blob2Full
from models.discriminator import discriminator

device = 'cuda:0'

def main(args):
    """
    Load data and define batch data loaders
    """
    items = utils.load_data_and_data_loaders(args.dataset, args.batch_size)
        
    training_loader, validation_loader = items[2], items[3]

    x_train_var = items[4]
    
    """
    Set up VQ-VAE model with components defined in ./models/ folder
    """
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                  args.n_residual_layers, args.n_embeddings, 
                  args.embedding_dim, args.beta, device)

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])

    model = model.to(device)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist.get_local_rank()],
        output_device=dist.get_local_rank(),
    )
    
    """
    Set up optimizer and training loop
    """
    model.eval()

    with torch.no_grad():
        print('Extracting codes for train set.')
        extract_code(training_loader, model)
        print('Extracting codes for val set.')
        extract_code(validation_loader, model)

root = '/scratch/bipasha31/vqvae_codes'
os.makedirs(root, exist_ok=True)

def extract_code(loader, model):
    for i, data in enumerate(tqdm(loader)):
        x, path = data
        x = x.to(device)

        name = path.split('/')[-1].split('.')[0]

        min_encoding_indices = model(x)

        saveas = f'{root}/{name}'
        np.savez_compressed(saveas, data=min_encoding_indices)

if __name__ == "__main__":
    # train_vqgan()
    # train_blob2full()

    parser = argparse.ArgumentParser()

    """
    Hyperparameters
    """
    timestamp = utils.readable_timestamp()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--dataset",  type=str, default='HandGestures')

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )+1

    print(args, f'port: {port}')

    dist.launch(main, args.n_gpu, 1, 0, f"tcp://127.0.0.1:{port}", args=(args,))
