import glob
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import random

import json

import torch

class FacialTransformsDataset(Dataset):
	def __init__(self, mode):
		self.mode = mode

		self.transformElements = [
			transforms.ToPILImage(),
			transforms.Pad((0, 0)),
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]

		self.base = '/ssd_scratch/cvit/bipasha31/processed_image_frames/*/*/*.jpg'

		self.data = glob.glob(self.base)
		
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