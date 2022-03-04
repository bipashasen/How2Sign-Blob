import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as F

class TemporalAlignment(nn.Module):
	def __init__(self):
		super(TemporalAlignment, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv3d(3, 3, kernel_size=(5,5,5), padding=(2,2,2)),
			nn.LeakyReLU(inplace=True)
			nn.Conv3d(3, 3, kernel_size=(5,5,5), padding=(2,2,2)),
			nn.LeakyReLU(inplace=True)
			nn.Conv3d(3, 3, kernel_size=(5,5,5), padding=(2,2,2)))

	def forward(self, x):
		return nn.Tanh()(x)

