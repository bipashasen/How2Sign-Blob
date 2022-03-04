import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
import numpy as np
import distributed as dist_fn

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Encode(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        n_res_block,
        n_res_channel,
        embed_dim,
        n_embed,
        decay,
        ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)

    def forward(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.encode_hull = Encode(in_channel,
            channel,
            n_res_block,
            n_res_channel,
            embed_dim,
            n_embed,
            decay)

        self.encode_background = Encode(in_channel,
            channel,
            n_res_block,
            n_res_channel,
            embed_dim,
            n_embed,
            decay)

        self.vq_encoding = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=(1, 0, 0)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=(1, 0, 0)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(128, 128, kernel_size=3, padding=(1, 0, 0)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=5, padding=(2, 0, 0)))

        self.gru = nn.GRU(256*4*2, 512, 3, bidirectional=True)

        self.predict_layer1 = nn.Linear(512*2, 128)
        self.predict_layer2 = nn.Linear(128, 6)

        self.combine = nn.Conv2d(64*2, 64, kernel_size=1)
        self.combine2 = nn.Conv2d(64, 64, kernel_size=1)

        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def stn(self, quant_t_hull, quant_b_hull, tenc):
        B, C, T, H, W = quant_t_hull.shape
        D, N, C_ = tenc.shape

        quant_t_hull = quant_t_hull.permute(0, 2, 1, 3, 4).view(-1, C, H, W)

        tenc = tenc.view(D*N, C_)

        theta = nn.ReLU()(self.predict_layer1(tenc))

        theta = nn.ReLU()(self.predict_layer2(theta)).view(-1, 2, 3)

        grid_t_hull = F.affine_grid(theta, quant_t_hull.size())
        quant_t_hull = F.grid_sample(quant_t_hull, grid_t_hull) # B x C x H x W

        grid_b_hull = F.affine_grid(theta, quant_b_hull.size())
        quant_b_hull = F.grid_sample(quant_b_hull, grid_b_hull) # B x C x H x W

        return quant_t_hull, quant_b_hull

    def merge_embeddings(self, hull, background):
        quant_t_hull, quant_b_hull, _, _, _ = hull
        quant_t_back, quant_b_back, _, _, _ = background

        T = quant_t_hull.shape[0]

        quant_t_hull = quant_t_hull.unsqueeze(0).permute(0, 2, 1, 3, 4)
        quant_t_back = quant_t_back.unsqueeze(0).permute(0, 2, 1, 3, 4)

        hull_enc = self.vq_encoding(quant_t_hull) 
        back_enc = self.vq_encoding(quant_t_back)

        hull_enc = hull_enc.permute(2, 0, 1, 3, 4).contiguous().view(T, 1, -1)
        back_enc = back_enc.permute(2, 0, 1, 3, 4).contiguous().view(T, 1, -1)

        encoding = torch.cat([hull_enc, back_enc], axis=-1)

        tenc, _ = self.gru(encoding)

        quant_t_hull, quant_b_hull = self.stn(quant_t_hull, quant_b_hull, tenc)

        quant_t_back = quant_t_back.permute(0, 2, 1, 3, 4).squeeze(0)

        quant_t = torch.cat([quant_t_hull, quant_t_back], axis=1)
        quant_b = torch.cat([quant_b_hull, quant_b_back], axis=1) 

        quant_t = self.combine2(nn.ReLU()(self.combine(quant_t)))
        quant_b = self.combine2(nn.ReLU()(self.combine(quant_b)))

        return quant_t, quant_b

    def forward(self, hull, background):
        hull, background = self.encode_hull(hull), self.encode_background(background)
        
        diff = hull[2] + background[2]

        joint_quant = self.merge_embeddings(hull, background)

        dec = self.decode(joint_quant[0], joint_quant[1])

        return dec, diff

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec