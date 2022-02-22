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