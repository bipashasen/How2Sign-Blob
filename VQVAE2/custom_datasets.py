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

CodeRowVideos = namedtuple('CodeRowVideos', ['code', 'labels', 'indices', 'filename'])
# code - {top: [], bottom: []}
# labels - {'SENTENCE': str, 'START': int, 'END': int}
# indices - list of indices of frames extracted from the video
# filename - name of the videofile

base = '/ssd_scratch/cvit/bipasha31/How2Sign-Blobs/{}-set-rhands-images/' 

if not os.path.exists(base.rsplit('/', 2)[0]):
    base = base.replace('/ssd_scratch/cvit', '/scratch')

root = base.rsplit('/', 2)[0]

class HandGesturesDatasetForPixelSnail(Dataset):
    def __init__(self, path, slen, PADDING_INDEX):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.PADDING_INDEX = PADDING_INDEX

        self.fps = 30
        self.slen = slen*8

        with open(os.path.join(root, 'Labels/txts', 'index_mapping.txt')) as r: 
            self.label_mappng = json.load(r)

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

        skip_by = 3

        # T 
        codes = row.code[::skip_by]

        codes = codes[:self.slen]

        top, bottom = [x['top'] for x in codes], [x['bottom'] for x in codes]

        sentence, _, _ = row.labels

        slen = min(self.slen, len(codes))

        sentence, src_key_padding_mask = self.pad_sentence(sentence, slen)

        data = [top, bottom, sentence, src_key_padding_mask]
 
        return [torch.from_numpy(np.array(x)) for x in data]

    def process_sentence(self, content):
        def isempty(x):
            return x.isspace() or x == ''

        content = content.replace('-', ' ')
        content = re.sub('[^a-zA-Z\d\s]', '', content)
        
        return [y.lower() for y in content.split(' ') if not isempty(y)]

    def pad_sentence(self, sentence, slen):
        sentence = [self.label_mappng[x] for x in self.process_sentence(sentence)]

        masked, unmasked = True, False

        # want to learn from padding mask too, don't ignore it. 
        # Because the model is not autogressive that's why.
        src_key_padding_mask = [unmasked]*slen

        if len(sentence) >= slen:

            return sentence[:slen], src_key_padding_mask

        else:
            senlen, pad = len(sentence), slen-len(sentence)

            # src_key_padding_mask = ([unmasked]*senlen) + ([masked]*pad)

            return (sentence + [self.PADDING_INDEX]*(pad)), src_key_padding_mask

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