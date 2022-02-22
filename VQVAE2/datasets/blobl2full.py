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

class Blob2Full(Dataset):
    def __init__(self, mode, transform):
        self.mode = mode

        self.transforms = transform

        base = '/ssd_scratch/cvit/bipasha31/How2Sign-Detection/{}.txt'

        with open(base.format(mode)) as r:
            # folder, face_path, rhand_path, lhand_path, full_path
            data = json.load(r) 

        lhand_path_idx = 3

        lhand_files = [glob.glob('{}/*.jpg'.format(x[lhand_path_idx])) for x in data]
        lhand_files = [item for sublist in lhand_files for item in sublist]

        total_files = len(lhand_files)

        self.data = self.remove_with_errors(lhand_files)

        self.train_path, self.raw_path = 'train-set-images', 'raw_videos_images' 

        print(f'{len(self.data)} datapoints loaded and removed {total_files-len(self.data)} datapoints.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.get_dataitem(idx)

        return data if data is not None else \
             self.__getitem__(random.randint(0, len(self.data)))

    def get_dataitem(self, idx):
        def read_image_from_path(path):
            return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        try:    
            cropped_path = self.data[idx]

            lhand = read_image_from_path(cropped_path)
            rhand = read_image_from_path(cropped_path.replace('_left_', '_right_'))
            face = read_image_from_path(cropped_path.replace('_left_', '_face_'))

            face, rhand, lhand =\
                self.transforms(face), self.transforms(rhand), self.transforms(lhand)

            full_path = cropped_path\
                .replace(self.train_path, self.raw_path)\
                .replace('_left_', '')

            out = self.transforms(read_image_from_path(full_path))

            return [face, rhand, lhand, out]
        except:
            with open('errors.txt', 'a') as w:
                w.write('{}\n'.format(cropped_path))

            return None

    def remove_with_errors(self, files):
        errors = open('errors.txt').read().splitlines()

        return [x for x in files if not x in errors] 