import glob
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import random

import json

import torch

class FacialTransforms(Dataset):
	def __init__(self, mode):
		self.mode = mode

		self.H, self.W = 256, 256

		self.transformElements = [
			transforms.ToPILImage(),
			transforms.Pad((0, 0)),
			transforms.Resize((self.H, self.W)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]

		self.base = '/scratch/bipasha31/processed_vlog_dataset_copy/*/*/*/*/*.jpg'

		datapoints = glob.glob(self.base)
		random.shuffle(datapoints)

		train_split = int(0.95*len(datapoints))

		self.data = datapoints[:train_split] if mode == 'train' else datapoints[train_split:]

		print(f'Loaded {self.__len__()} datapoints for {mode} mode.')
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		raw_image = io.imread(self.data[idx])

		h, w, _ = raw_image.shape

		if h > w:
			padw, padh = (h-w)//2, 0
		else:
			padw, padh = 0, (w-h)//2

		self.transformElements[1] = transforms.Pad((padw, padh))

		return transforms.Compose(self.transformElements)(raw_image)
