import cv2
import glob
import json
import random
import lmdb
import pickle

import torch
import re
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from collections import namedtuple

base = '/ssd_scratch/cvit/bipasha31/How2Sign-Blobs/{}-set-rhands-images/' 

if not os.path.exists(base.rsplit('/', 2)[0]):
    base = base.replace('/ssd_scratch/cvit', '/scratch')

root = base.rsplit('/', 2)[0]

class HandGesturesDataset(Dataset):
    def __init__(self, mode, transform, code=False):
        self.mode = mode
        self.code = code

        self.transform = transform

        self.base = '{}/*.jpg'.format(base.format(mode))

        # n = 300000 if mode == 'train' else 2000

        self.data = glob.glob(self.base)

        print(f'{len(self.data)} images loaded from {self.base.split("/")[3]} for {mode}')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        def read_image_from_path(path):
            # return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            return cv2.imread(path)

        # print(cv2.imread(self.data[idx]).shape)
        path = self.data[idx]

        obj = [self.transform(read_image_from_path(path)), []]

        if self.code:
            obj.append(path.split('/')[-1].split('.')[0])
        return obj