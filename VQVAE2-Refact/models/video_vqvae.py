import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as F

from .vqvae import VQVAE

class VideoVQVAE(nn.Module):
	def __init__(self, n):
		super(VideoVQVAE, self).__init__()

		self.n = n
		self.vqvae = VQVAE(in_channel=3*n)

	def forward(self, x):
		# B x T x 3 x H X W
		B, T, C, H, W = x.shape

		# B x T x 9 x H x W
		out = torch.zeros_like(x, device=x.device).repeat(1, 1, self.n, 1, 1)

		out[:, self.n-2] = x[:, :self.n].view(B, C*self.n, H, W)

		latent_losses = []

		for i in range(T)[self.n-1:]:
			sub_X = x[:, i-self.n+1:i+1].clone() # B x 3 x 3 x H x W
			sub_X = sub_X.view(B, C*self.n, H, W) # B x 9 x H x W
			sub_X[:, :3*(self.n-1)] = out[:, i-1, 3:].squeeze(1).detach().clone()
			
			out[:, i], latent_loss = self.vqvae(sub_X) # B x 9 x H x W, 1

			latent_losses.append(latent_loss)

		# B x T-self.n+1 x 9 x H x W
		return out[:, self.n-1:], torch.vstack(latent_losses).mean()