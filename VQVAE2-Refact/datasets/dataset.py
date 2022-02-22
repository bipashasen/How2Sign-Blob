import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb
import numpy as np
import glob

import random

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename

class NPZDataset(Dataset):
    def __init__(self, path, batch_size):
        self.data = glob.glob(f'{path}/*')

        self.batch_size = batch_size

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        np_path = self.data[index]

        with open(np_path) as r:
            data = np.load(np_path, allow_pickle=True)['data'].item()

        top, bottom, filename = data['top'], data['bottom'], data['filename']

        B = top.shape[0]

        if B > self.batch_size:
            index = random.randint(0, B-self.batch_size+1)
            e_index = index+self.batch_size
            top = top[index:e_index]
            bottom = bottom[index:e_index]

        return torch.from_numpy(top), torch.from_numpy(bottom), filename

class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename
