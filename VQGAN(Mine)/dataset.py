import glob
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import random

import json
import cv2

import torch

class HandGesturesDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.base = '/scratch/bipasha31/How2Sign-Blobs/{}-set-rhands-images/*.jpg'.format(mode)

        self.data = glob.glob(self.base)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transforms(io.imread(self.data[idx]))