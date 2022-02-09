import os
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder

from torchvision import utils

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

from tqdm import tqdm

'''
For debugging
'''
import matplotlib.pyplot as plt

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, device_id, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device_id)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False, save_idx=None, visual_folder=None, return_codes_only=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, min_encoding_indices = self.vector_quantization(
            z_e)

        if return_codes_only:
            return min_encoding_indices
            
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        if save_idx is not None:
            def save_img(img, save_idx):
                # img = (img.detach().cpu() + 0.5).numpy()
                # img = np.transpose(img, (1,2,0))
                # fig = plt.imshow(img, interpolation='nearest')
                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                # plt.savefig('{}/{}_{}_{}.jpg'.\
                #     format(visual_folder, save_idx, i, dtype), bbox_inches='tight')

                saveas = f'{visual_folder}/{save_idx}.jpg'
                utils.save_image(
                    img,
                    saveas,
                    nrow=5,
                    normalize=True,
                    range=(-1, 1),
                )

            img = torch.cat((x[:5], x_hat[:5]), axis=0)
            save_img(img, save_idx)

        return embedding_loss, x_hat, perplexity

class VQVAE_Blob2Full(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, device_id, save_img_embedding_map=False):
        super(VQVAE_Blob2Full, self).__init__()
        '''
        Right Hand
        '''
        # encode image into continuous latent space
        self.encoder_rhand = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv_rhand = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization_rhand = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device_id)
        # decode the discrete latent representation

        '''
        Left Hand
        '''
        # encode image into continuous latent space
        self.encoder_lhand = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv_lhand = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization_lhand = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device_id)
        # decode the discrete latent representation

        '''
        Face
        '''
        # encode image into continuous latent space
        self.encoder_face = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv_face = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization_face = VectorQuantizer(
            n_embeddings, embedding_dim, beta, device_id)
        # decode the discrete latent representation

        '''
        Decoder
        '''
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False, save_idx=None, visual_folder=None):
        face, rhand, lhand, gt = x

        '''
        Right Hand
        '''
        z_e_rhand = self.encoder_rhand(rhand)

        z_e_rhand = self.pre_quantization_conv_rhand(z_e_rhand)
        embedding_loss_rhand, z_q_rhand, perplexity_rhand, _, _ = self.vector_quantization_rhand(
            z_e_rhand)

        '''
        Left Hand
        '''
        z_e_lhand = self.encoder_lhand(lhand)

        z_e_lhand = self.pre_quantization_conv_lhand(z_e_lhand)
        embedding_loss_lhand, z_q_lhand, perplexity_lhand, _, _ = self.vector_quantization_lhand(
            z_e_lhand)

        '''
        Face
        '''
        z_e_face = self.encoder_face(face)

        z_e_face = self.pre_quantization_conv_face(z_e_face)
        embedding_loss_face, z_q_face, perplexity_face, _, _ = self.vector_quantization_face(
            z_e_face)

        '''
        Combine
        '''
        z_q = z_q_rhand + z_q_lhand + z_q_face
        embedding_loss = embedding_loss_rhand + embedding_loss_lhand + embedding_loss_face
        perplexity = perplexity_rhand + perplexity_lhand + perplexity_face

        '''
        Decoder
        '''
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        if save_idx is not None:
            def save_img(img, save_idx, i, dtype):
                img = (img.detach().cpu() + 0.5).numpy()
                img = np.transpose(img, (1,2,0))
                fig = plt.imshow(img, interpolation='nearest')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig('{}/{}_{}_{}.jpg'.\
                    format(visual_folder, save_idx, i, dtype), bbox_inches='tight')

            for i in tqdm(range(min(face.shape[0], 8))):
                save_img(face[i], save_idx, i, '2_face')
                save_img(rhand[i], save_idx, i, '1_rhand')
                save_img(lhand[i], save_idx, i, '3_lhand')
                save_img(gt[i], save_idx, i, '5_gt')
                save_img(x_hat[i], save_idx, i, '4_reconstructed')

            # HACK
            os.system('cp /ssd_scratch/cvit/bipasha31/SLP/VQVAE/outputs_Blobs2Full/* /home2/bipasha31/python_scripts/CurrentWork/outputs')

        return embedding_loss, x_hat, perplexity
