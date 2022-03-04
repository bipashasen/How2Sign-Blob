import numpy as np
import torch
import torch.nn as nn
import math

from .vqvae import VQVAE

from torch.nn import functional as F

class Conv3d(nn.Module):
    """Extended nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv3d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm3d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            try:
                out += x
            except:
                print(out.size())
                print(x.size())
        return self.act(out)

class TemporalAlignment(nn.Module):
	def __init__(self):
		super(TemporalAlignment, self).__init__()

		channels = [3, 32, 64, 128, 256, 512]

		convs = []
		n = len(channels)-1

		for i in range(n):
			k = (3, 3, 3) if i < n-1 else (3, 5, 5)
			mp = (1, 2, 2) if i < n-1 else (1, 3, 3)

			convs.extend(self._make_conv_layer(channels[i], channels[i+1], k, mp, temporal_padding=1))

		self.encode = nn.Sequential(*convs)

		self.gru = nn.GRU(channels[-1]*2, 512, 3, bidirectional=True)

		self.predict_layer1 = nn.Linear(channels[-1]*2, channels[-1]//2)

		self.predict_layer2 = nn.Linear(channels[-1]//2, 6)

		# Initialize the weights/bias with identity transformation
		self.predict_layer2.weight.data.zero_()
		self.predict_layer2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

		# self.post_conv = nn.Sequential(*self._make_conv_layer(3, 3, 3, 1, all_padding=1))
		self.vqvae = VQVAE()

	def _make_conv_layer(self, in_c, out_c, kernel, max_pool, temporal_padding=0, all_padding=0):
		padding = (temporal_padding, 0, 0) if temporal_padding > 0 else all_padding
		
		return [
			nn.Conv3d(in_c, out_c, kernel_size=kernel, padding=padding),
			nn.LeakyReLU(inplace=True),
			nn.Conv3d(out_c, out_c, kernel_size=kernel, padding=padding),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool3d(max_pool)
		]

	def stn(self, x, combined_encodings):
		B, T, C, H, W = x.shape
		D, N, C_ = combined_encodings.shape

		x = x.view(-1, C, H, W)

		combined_encodings = combined_encodings.view(D*N, C_)

		theta = nn.ReLU()(self.predict_layer1(combined_encodings))

		theta = nn.ReLU()(self.predict_layer2(theta)).view(-1, 2, 3)

		grid = F.affine_grid(theta, x.size())
		x = F.grid_sample(x, grid) # B x C x H x W

		return x.view(B, T, C, H, W)

	def forward(self, source, target):
		def change_dim2(encodings):
			B, C, T, _, _ = encodings.shape
			return encodings.permute(2, 0, 1, 3, 4).view(T, B, C)

		# source: B x T x C x H x W
		# 1 x 32 x 3 x 256 x 256 when T = 32. 

		source, target = source.unsqueeze(0), target.unsqueeze(0)

		source_permuted = source.permute(0, 2, 1, 3, 4)
		target_permuted = target.permute(0, 2, 1, 3, 4)
		
		source_encodings = change_dim2(self.encode(source_permuted))
		target_encodings = change_dim2(self.encode(target_permuted))

		concat_encodings = torch.cat([
			source_encodings, target_encodings], axis=-1)

		combined_encodings, _ = self.gru(concat_encodings)
		
		source_new = self.stn(source, combined_encodings)

		added = source_new + target

		return self.vqvae(added.squeeze(0))

