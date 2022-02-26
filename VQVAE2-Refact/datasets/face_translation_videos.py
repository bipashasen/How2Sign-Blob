import glob
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

import random

import json

import torch

class FaceTransformsVideos(Dataset):
	def __init__(self, mode, n):
		self.mode = mode

		self.H, self.W = 256, 256
		self.n = n

		self.max_len = 36

		self.transformElements = [
			transforms.ToPILImage(),
			transforms.Pad((0, 0)),
			transforms.Resize((self.H, self.W)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]

		self.base = '/ssd_scratch/cvit/bipasha31/processed_image_frames/*/*'

		datapoints = sorted(glob.glob(self.base))

		train_split = int(0.9*len(datapoints))

		self.folders = datapoints[:train_split] if mode == 'train' else datapoints[train_split:]

		self.create_dataset()
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		raw_images = [io.imread(p) for p in self.data[idx]]

		h, w, _ = raw_images[0].shape

		if h > w:
			padw, padh = (h-w)//2, 0
		else:
			padw, padh = 0, (w-h)//2

		self.transformElements[1] = transforms.Pad((padw, padh))

		c_transforms = transforms.Compose(self.transformElements)

		new_images = [c_transforms(x).unsqueeze(0) for x in raw_images]

		out_images = [torch.vstack(new_images[i:i+self.n])
			.view(-1, self.H, self.W).unsqueeze(0)
				for i in range(len(new_images) - self.n + 1)]

		new_images = torch.vstack(new_images)

		out_images = torch.vstack(out_images)

		mask = np.zeros_like(out_images)

		return new_images, out_images, out_images, mask, mask

	def create_dataset(self):
		self.data = []

		def get_key(x):
			return int(x.split('/')[-1].split('.')[0])

		for folder in self.folders: 
			files = sorted(glob.glob(f'{folder}/*.jpg'), key=lambda x: get_key(x))

			if len(files) < self.n:
				continue

			self.data.append(files[:self.max_len])
