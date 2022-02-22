import cv2
import numpy as np
from torch.utils.data import Dataset
import glob
import json
import re
import random

class BlockDataset(Dataset):
    """
    Creates block dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')
        data = np.array([cv2.resize(x[0][0][:, :, :3], dsize=(
            32, 32), interpolation=cv2.INTER_CUBIC) for x in data])

        n = data.shape[0]
        cutoff = n//10
        self.data = data[:-cutoff] if train else data[-cutoff:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)

class LatentBlockDatasetImages(Dataset):
    """
    Loads latent blobk dataset for images
    """

    def __init__(self, file_path, train=True, transform=None):
        data = glob.glob(f'{file_path}/*.npz')

        train_split = int(0.85*len(data))

        self.data = data[:train_split] if train else data[train_split:]
        
        self.transform = transform

        with open('index_mapping.json') as r:
            self.label_mappng = json.load(r)

        print(f'loaded {len(data)} datapoints with {train_split} training and {len(data)-train_split} validation datapoints')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        item = np.load(path, allow_pickle=True)['data'].item()
        imgs = item['encodings'].cpu()

        mid = len(imgs) // 2
        img = imgs[mid]

        gloss = item['gloss'][0][0]
        gloss_index = self.label_mappng[gloss]

        return self.transform(img), gloss_index

class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset 
    """

    def __init__(self, file_path, train=True, batch_size=96, transform=None, random=False):
        data = glob.glob(f'{file_path}/*.npz')

        train_split = int(0.85*len(data))

        self.random = random

        self.batch_size = batch_size-1 # 1 for 0th index
        
        self.data = data[:train_split] if train else data[train_split:]
        self.transform = transform

        with open('index_mapping.txt') as r:
            self.label_mappng = json.load(r)

        print(f'loaded {len(data)} datapoints with {train_split} training and {len(data)-train_split} validation datapoints')

    def __getitem__(self, index):
        path = self.data[index]
        item = np.load(path, allow_pickle=True)['data'].item()
        imgs = item['encodings'].cpu()

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]
        
        B = imgs.shape[0]

        if B > self.batch_size:
            if self.random:
                index = random.randint(0, B-self.batch_size+1)
                imgs = imgs[index:index+self.batch_size]

            else:
                mid = len(imgs) // 2
                s = mid-(self.batch_size//2)
                imgs = imgs[s:s+self.batch_size]

        sentence = item['labels'][0][0]

        sentence, src_key_padding_mask = self.pad_sentence(sentence)

        return imgs, sentence, src_key_padding_mask

    def __len__(self):
        return len(self.data)

    def process_sentence(self, content):
        def isempty(x):
            return x.isspace() or x == ''

        content = content.replace('-', ' ')
        content = re.sub('[^a-zA-Z\d\s]', '', content)
        
        return [y.lower() for y in content.split(' ') if not isempty(y)]

    def pad_sentence(self, sentence):
        sentence = [self.label_mappng[x] for x in self.process_sentence(sentence)]

        src_key_padding_mask = [False]*len(sentence) 

        return np.array(sentence), np.array(src_key_padding_mask)

