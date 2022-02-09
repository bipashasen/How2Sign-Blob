import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import random
import json

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
            # return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            return cv2.imread(path)

        # try:    
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
        # except:
        #     with open('errors.txt', 'a') as w:
        #         w.write('{}\n'.format(cropped_path))

        #     return None

    def remove_with_errors(self, files):
        errors = open('errors.txt').read().splitlines()

        return [x for x in files if not x in errors]

class HandGesturesDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

        base = '/scratch/bipasha31/How2Sign-Blobs/{}-set-rhands-images/'

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.base = '{}/*.jpg'.format(base.format(mode))

        self.data = glob.glob(self.base)
        # self.make_dataset()

        print(f'{len(self.data)} datapoints loaded for {mode}.')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        def read_image_from_path(path):
            # return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            return cv2.imread(path)

        return self.transforms(read_image_from_path(self.data[idx])), self.data[idx]
        # return self.data[idx]

    def make_dataset(self):
        files = glob.glob(self.base)

        self.data = []
        print_at = len(files)/500
        for i, file in enumerate(files):
            self.data.append(self.transforms(cv2.imread(file)))

            if i % print_at == 0:
                print(f'Loaded {i+1}/{len(files)} datapoints')
