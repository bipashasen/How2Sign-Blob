import glob
import random
import json

from skimage import io
from sklearn.model_selection import train_test_split
import numpy as np

from torch.utils.data import Dataset
import torch
from torchvision import transforms

class FacialTransformsMultipleFramesDataset(Dataset):
	def __init__(self, mode, n):
		self.mode = mode

		self.H, self.W = 256, 256
		self.n = n

		self.transformElements = [
			transforms.ToPILImage(),
			transforms.Pad((0, 0)),
			transforms.Resize((self.H, self.W)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]

		# self.base = '/ssd_scratch/cvit/bipasha31/processed_image_frames/*/*'
		# self.base = '/scratch/bipasha31/processed_vlog_dataset_copy/*/*/*/*'
		self.base = '/ssd_scratch/cvit/aditya1/processed_vlog_dataset_copy/*/*/*/*'

		datapoints = glob.glob(self.base)
		random.shuffle(datapoints)

		train, test = train_test_split(datapoints, train_size=0.95)
		self.folders = train if mode == 'train' else test

		self.create_dataset()
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		raw_images = [io.imread(p) for p in self.data[idx]]

		new_images = []

		for raw_image in raw_images:
			h, w, _ = raw_image.shape

			if h > w:
				padw, padh = (h-w)//2, 0
			else:
				padw, padh = 0, (w-h)//2

			self.transformElements[1] = transforms.Pad((padw, padh))

			new_images.append(transforms.Compose(self.transformElements)(raw_image))

		return torch.vstack(new_images).view(self.n*3, self.H, self.W)

	def create_dataset(self):
		self.data = []

		def get_key(x):
			return int(x.split('/')[-1].split('.')[0])

		for folder in self.folders: 
			files = sorted(glob.glob(f'{folder}/*.jpg'), key=lambda x: get_key(x))

			if len(files) < self.n:
				continue

			files = [files[i:i+self.n] for i in range(len(files) - self.n)]

			self.data.extend(files)
