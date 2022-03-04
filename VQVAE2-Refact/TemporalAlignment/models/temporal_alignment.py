import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as F

class TemporalAlignment(nn.Module):
	def __init__(self):
		super(TemporalAlignment, self).__init__()

		channels = [3, 32, 64, 128, 256, 512]

		convs = []
		n = len(channels)-1

		for i in range(n):
			k = (3, 3, 3) if i < n-1 else (3, 5, 5)
			mp = (1, 2, 2) if i < n-1 else (1, 3, 3)

			convs.extend(self._make_conv_layer(channels[i], channels[i+1], k, mp, 1))

		self.encode = nn.Sequential(*convs)

		self.predict = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Linear(channels[-1], channels[-1]//4),
			nn.ReLU(inplace=True),
			nn.Linear(channels[-1]//4, 3*2)) # for now no scale/shear.

		# Initialize the weights/bias with identity transformation
		self.predict[3].weight.data.zero_()
		self.predict[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

	def _make_conv_layer(self, in_c, out_c, kernel, max_pool, temporal_padding=0):
		padding = (temporal_padding, 0, 0)

		return [
			nn.Conv3d(in_c, out_c, kernel_size=kernel, padding=padding),
			nn.LeakyReLU(inplace=True),
			nn.Conv3d(out_c, out_c, kernel_size=kernel, padding=padding),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool3d(max_pool),
		]

	def stn(self, x, theta):
		theta = theta.view(-1, 2, 3)

		grid = F.affine_grid(theta, x.size())
		x = F.grid_sample(x, grid) # B x C x H x W

		return x

	def forward(self, x):
		x_permuted = x.permute(0, 2, 1, 3, 4) # B x C x T x H x W
		encodings = self.encode(x_permuted).permute(2, 1, 0, 3, 4)
		encodings = encodings.view(encodings.shape[0], -1)
		theta = self.predict(encodings)

		x_new = self.stn(x.squeeze(0), theta)
		return x_new.unsqueeze(0)

