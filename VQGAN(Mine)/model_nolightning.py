import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision

from torchvision import utils
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset

from tqdm import tqdm

from dataset import HandGesturesDataset

device = 'cuda:0'
save_image_at = 20

epochs = 250

validation_at = 500

save_at = 100
save_dir = 'checkpoints'

os.makedirs(save_dir, exist_ok=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_dataloaders(batch_size):
    training_dataset = HandGesturesDataset('train')

    training_loader = DataLoader(training_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    val_dataset = HandGesturesDataset('val')

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    return training_loader, val_loader

def get_input(x):
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.to(memory_format=torch.contiguous_format)
    return x.float().to(device)

def save_image(x, xrec, idx, turn):
    saveas = f'/home2/bipasha31/python_scripts/CurrentWork/samples/VQGAN2/{idx}_{turn}.jpg'
    utils.save_image(
        torch.cat((x, xrec), axis=0),
        saveas,
        nrow=x.shape[0],
        normalize=True,
        range=(-1, 1),
    )

def get_last_layer(model):
    return model.module.decoder.conv_out.weight

def configure_optimizers(model, lr):
    opt_ae = torch.optim.Adam(list(model.module.encoder.parameters())+
                              list(model.module.decoder.parameters())+
                              list(model.module.quantize.parameters())+
                              list(model.module.quant_conv.parameters())+
                              list(model.module.post_quant_conv.parameters()),
                              lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(model.module.loss.discriminator.parameters(),
                                lr=lr, betas=(0.5, 0.9))
    return [opt_ae, opt_disc]

def validation(model, loader, t_updates, global_step):
    t_pbar = tqdm(range(t_updates))

    aelosses, disclosses = [], []
    rec_losses = []

    for i in t_pbar:
        batch = next(iter(loader))

        x = get_input(batch)

        xrec, qloss = model(x)

        if i % (t_updates//10) == 0:
            save_image(x, xrec, i, 'val')

        aeloss, log_dict_ae = model.module.loss(qloss, x, xrec, 0, global_step,
                                            last_layer=get_last_layer(model), split="val")

        discloss, log_dict_disc = model.module.loss(qloss, x, xrec, 1, global_step,
                                            last_layer=get_last_layer(model), split="val")
        
        rec_loss = log_dict_ae["val/rec_loss"]

        aelosses.append(aeloss)
        rec_losses.appen(rec_loss)
        disclosses.append(discloss)
        
        t_pbar.set_description("val/rec_loss", rec_loss)
        t_pbar.set_description("val/aeloss", aeloss)
        t_pbar.set_description("val/discloss", discloss)

    print(f'AELoss: {aelosses.mean():.3f}, DiscLoss: {disclosses.mean():.3f}, RecLoss: {rec_losses.mean():.3f}')

def train(model, training_loader, val_loader, learning_rate):
    opt_ae, opt_disc = configure_optimizers(model, learning_rate)

    n_updates, t_updates = 0.2*len(training_loader), 0.2*len(val_loader)
    n_updates, t_updates = int(n_updates), int(t_updates)  

    aelosses, disclosses = [], []

    for epoch in range(epochs):
        pbar = tqdm(range(n_updates))

        for i in pbar:
            batch = next(iter(training_loader))

            optimizer_idx = i%2

            model.train()

            opt_ae.zero_grad()
            opt_disc.zero_grad()

            x = get_input(batch)
            xrec, qloss = model(x)

            if i % save_image_at == 0:
                save_image(x, xrec, i, 'train')

            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = model.module.loss(qloss, x, xrec, optimizer_idx, i,
                                                last_layer=get_last_layer(model), split="train")

                pbar.set_description("train/aeloss", aeloss)

                aelosses.append(aeloss)

                aeloss.backward()
                opt_ae.step()

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = model.module.loss(qloss, x, xrec, optimizer_idx, i,
                                                last_layer=get_last_layer(model), split="train")
                pbar.set_description("train/discloss", discloss)

                disclosses.append(discloss)
            
                discloss.backward()
                opt_disc.step()

            if i % validation_at == 0:
                with torch.no_grad():
                    validation(model, val_loader, t_updates, i)

            if i % save_at == 0:
                torch.save({
                    'model': model.module.state_dict(),
                    'opt_ae': opt_ae.state_dict(),
                    'opt_disc': opt_disc.state_dict() }, f'{save_dir}/step_{i}.pt')

                pbar.set_description('saved: ', i)

        print(f'AELoss: {aelosses.mean():.3f}, DiscLoss: {disclosses.mean():.3f}')

if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="if testing")
    parser.add_argument(
        "-b",
        "--base",
        default="configs/dummy_config.yaml",
        help="paths to base configs. Loaded from left-to-right."
    )
    parser.add_argument(
        "-batch",
        type=int,
        default=4,
        help="batch size for training"
    )
    parser.add_argument(
        "-accu_grad",
        type=int,
        default=1,
        help="accumulate_grad_batches (0/1)"
    )
    
    opt, unknown = parser.parse_known_args()
    config = OmegaConf.load(opt.base) 

    ngpu = torch.cuda.device_count()

    learning_rate = opt.accu_grad * ngpu * opt.batch * config.model.base_learning_rate
    print(f'Setting learning rate to: {learning_rate} = {opt.accu_grad} (accu_grad) * {ngpu} (ngpu) * {opt.batch} (batch_size) * {config.model.base_learning_rate} (base lr)')

    model = instantiate_from_config(config.model)

    model = nn.DataParallel(model).to(device)

    training_loader, val_loader = get_dataloaders(opt.batch)

    train(model, training_loader, val_loader, learning_rate)
    