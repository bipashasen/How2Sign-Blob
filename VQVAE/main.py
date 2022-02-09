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

visual_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/VQVAE'

os.makedirs(visual_folder, exist_ok=True)

verbose = False
save_idx_global = 0
save_at = 100
did = 0

models = {
    'gan': 0,
    'vae': 1
}

model_to_train = models['vae']

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'd_loss': []
}

device = 'cuda:0'

def main(args):
    """
    Set up VQ-VAE model with components defined in ./models/ folder
    """
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                  args.n_residual_layers, args.n_embeddings, 
                  args.embedding_dim, args.beta, device)

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])

    model = model.to(device)

    if args.test:
        loader = utils.load_data_and_data_loaders(args.dataset, args.batch_size, test=True)

        test(loader, model)

        return

    """
    Load data and define batch data loaders
    """
    items = utils.load_data_and_data_loaders(args.dataset, args.batch_size)
    training_loader, validation_loader = items[2], items[3]

    x_train_var = items[4]
    
    """
    Set up optimizer and training loop
    """
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    model.train()

    if model_to_train == models['gan']:
        train_vqgan(args, training_loader, validation_loader, x_train_var, model, optimizer)

    else:
        train(args, training_loader, validation_loader, x_train_var, model, optimizer)

def test(loader, model):
    for i, data in enumerate(tqdm(loader)):
        x, _ = data

        x = x.to(device)

        with torch.no_grad():
            _ = model(x, save_idx=f'{i}', visual_folder=visual_folder)

def train(args, training_loader, validation_loader, x_train_var, model, optimizer):
    global save_idx_global

    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        save_idx = None

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % save_at == 0:
            save_idx = save_idx_global
            save_idx_global += 1

            model.eval()

            with torch.no_grad():
                for vi in tqdm(range(10)):
                    (x, _) = next(iter(validation_loader))
                    x = x.to(device)
                    _, _, _ = model(x, verbose=verbose, save_idx=f'{save_idx}_{vi}', visual_folder=visual_folder)

            model.train()

        if i % args.log_interval == 0 and dist.is_primary():
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, optimizer, results, hyperparameters, args.filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))

def train_vqgan(args, training_loader, validation_loader, x_train_var, model, optimizer):
    global save_idx_global

    c_mse = nn.MSELoss()

    disc = discriminator().to(device)

    optim_D = optim.Adam(disc.parameters(), lr=args.learning_rate, amsgrad=True)

    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()
        optim_D.zero_grad()

        save_idx = None

        if i % save_at == 0 and i > 0:
            save_idx = save_idx_global
            save_idx_global += 1

        embedding_loss, x_hat, perplexity = \
            model(x, verbose=verbose, save_idx=save_idx, visual_folder=visual_folder)
            
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        '''
        adding the perceptual loss here - patch loss of real and fake
        '''
        B = args.batch_size
        D = 16 * 16

        ones = torch.ones((B, D), dtype=torch.float32, device=device)
        zeros = torch.zeros((B, D), dtype=torch.float32, device=device)

        if i % 2 == 0:
            fake = disc(x_hat).view(B, D)

            loss += c_mse(fake, ones)

        else:
            fake = disc(x_hat.clone().detach()).view(B, D)
            real = disc(x).view(B, D)

            d_loss = c_mse(real, ones) + c_mse(fake, zeros)

            results["d_loss"].append(d_loss.cpu().detach().numpy())

            d_loss.backward()
            optim_D.step()

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, optimizer, results, hyperparameters, args.filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Discriminator Loss', np.mean(results['d_loss'][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]), flush=True)

if __name__ == "__main__":
    # train_vqgan()
    # train_blob2full()

    parser = argparse.ArgumentParser()

    """
    Hyperparameters
    """
    timestamp = utils.readable_timestamp()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_updates", type=int, default=50000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--log_interval", type=int, default=3)
    parser.add_argument("--save_at", type=int, default=100)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--dataset",  type=str, default='HandGestures')

    parser.add_argument("--test",  action='store_true')

    # whether or not to save model
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--filename",  type=str, default=timestamp)

    args = parser.parse_args()

    args.save = True

    if args.save and dist.is_primary():
        print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

    args.n_gpu = torch.cuda.device_count()

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )+1

    print(f'port: {port}')

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, f"tcp://127.0.0.1:{port}", args=(args,))
