import sys
import random
from glob import glob
import os
import json
import os.path as osp

from skimage import io
from skimage.transform import resize
from scipy.ndimage import laplace
import numpy as np
import cv2
from skimage import transform as tf

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

from datasets.face_translation_videos3_utils import *

landmark_name = '_landmarks.npz'
bad_mp4s = 'bad_mp4s.json'
valid_folders = '/scratch/bipasha31/processed_vlog_dataset_copy/' + 'valid_folders.json'

def get_good_videos(video_dir):
    def get_video_and_segment(x):
        return '/'.join(x.split('/')[-4:])

    video_segments = glob(video_dir + '/*/*/*.mp4')

    if os.path.exists(bad_mp4s):
        with open(bad_mp4s) as r:
            bad_video_segments = json.load(r)

    else:
        bad_video_segments = []

    with open(valid_folders) as r:
        valid_video_segments = json.load(r)
        valid_video_segments = [x for x in valid_video_segments]

    video_segments = [
        x for x in video_segments 
            if not get_video_and_segment(x) in bad_video_segments
                and get_video_and_segment(x) in valid_video_segments]

    video_segments = [
        x for x in video_segments
            if len(glob(f'{x.split(".")[0]}/*.jpg')) > 3]
    
    return video_segments

def get_video_frames_perturbed(video_dir, batch_size):
    video_segments = get_good_videos(video_dir)
    
    def get_samples():
        video_id = random.randint(0, len(video_segments)-1)
        video_path = video_segments[video_id].rsplit('.', 1)[0]
 
        video_dir = glob(video_path)[0]
 
        frames = glob(video_dir + '/[0-9][0-9][0-9][0-9][0-9].jpg')
 
        max_index = max(1, len(frames) - batch_size - 1)
        index = random.randint(0, max_index)
 
        return video_dir, index, len(frames)
 
    def sample_frames(index, min_frame_length, video_dir):
        frames_sampled = list()
 
        for i in range(index, index + min_frame_length):
            current_frame = osp.join(video_dir, str(i).zfill(5) + '.jpg')
            frames_sampled.append(current_frame)
 
        return frames_sampled
 
    # generate a random source index to index from 
    source_video_dir, source_index, source_frames_len = get_samples()
    target_video_dir, target_index, target_frames_len = get_samples()
 
    min_frame_length = min(
        source_frames_len-source_index, 
        target_frames_len-target_index, 
        batch_size)
 
    # after generating the source and target indices, we need to sample the frames 
    source_frames_sampled = sample_frames(source_index, min_frame_length, source_video_dir)
    target_frames_sampled = sample_frames(target_index, min_frame_length, target_video_dir)
 
    assert len(source_frames_sampled) == len(target_frames_sampled)
 
    source_faces = list()
    source_background_masks = list()
    source_transformed_images = list()
    source_background_masks_no_enlargement = list()
    target_images = list()
    target_face_masks = list()
    perturbed_images = list()
 
    for i in range(len(source_frames_sampled)):
        # construct the paths from the source and target frame paths 
        source_image_path = source_frames_sampled[i]
        target_image_path = target_frames_sampled[i]
 
        source_landmark_npz = osp.join(source_image_path.rsplit('.', 1)[0]) + landmark_name
        target_landmark_npz = osp.join(target_image_path.rsplit('.', 1)[0]) + landmark_name
        
        # read the data and apply framewise transformation
        source_face_transformed, source_background_mask_transformed, source_image_transformed, source_background_mask_no_enlargement_transformed, \
        target_image, target_face_mask, perturbed_image = \
            generate_warped_image(source_landmark_npz, target_landmark_npz, 
                source_image_path, target_image_path)
 
        source_faces.append(source_face_transformed)
        source_background_masks.append(source_background_mask_transformed)
        source_transformed_images.append(source_image_transformed)
        source_background_masks_no_enlargement.append(source_background_mask_no_enlargement_transformed)
        target_images.append(target_image)
        target_face_masks.append(target_face_mask)
        perturbed_images.append(perturbed_image)
 
    return source_faces, source_background_masks, source_transformed_images, source_background_masks_no_enlargement, target_images, target_face_masks, perturbed_images

class FaceTransformsVideos(Dataset):
    def __init__(self, mode, n, max_frame_len):
        self.mode = mode

        # base = '/ssd_scratch/cvit/bipasha31/processed_video_talkingheads/*'
        base = '/scratch/bipasha31/processed_vlog_dataset_copy/*'

        self.H, self.W = 256, 256
        self.n = n

        self.max_len = max_frame_len

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        datapoints = sorted(glob(base))
        datapoints = [x for x in datapoints if os.path.isdir(x)]

        # train_split = int(0.9*len(datapoints))

        # self.videos = datapoints[:train_split]\
        #     if mode == 'train' else datapoints[train_split:]

        self.videos = datapoints
        
        print(f'Loaded {len(self.videos)} videos of {mode} split')
 
    def __len__(self):
        return len(self.videos)
 
    def __getitem__(self, index):
        video_dir = self.videos[index]
        
        source_faces, source_background_masks, source_transformed_images, source_background_masks_no_enlargement, target_images, \
            target_face_masks, perturbed_images = \
                get_video_frames_perturbed(video_dir, self.max_len)\

        # loading images ... 
        # perturbed images, source faces, target images
        perturbed_images = self.load_images(perturbed_images, transform_dim=False)
        source_faces = self.load_images(source_faces)
        source_transformed_images = self.load_images(source_transformed_images)
        target_images = self.load_images(target_images)
        
        source_background_masks = self.transform_dimensions(source_background_masks, vstack=True)
        source_background_masks_no_enlargement = self.transform_dimensions(source_background_masks_no_enlargement, vstack=True)
        target_face_masks = self.transform_dimensions(target_face_masks, vstack=True)

        return perturbed_images, source_faces, source_transformed_images, target_images, source_background_masks/255, source_background_masks_no_enlargement/255, target_face_masks/255
 
    def load_images(self, images, transform_dim=True):
        images = [self.transform(p).unsqueeze(0) for p in images]

        if transform_dim:
            images = self.transform_dimensions(images)

        return torch.vstack(images)

    # changing dimension of the images ... 
    def transform_dimensions(self, images, vstack=False):
        if vstack:
            images = [torch.tensor(x) for x in images]

        images = [torch.vstack(images[i:i+self.n])
            .view(-1, self.H, self.W).unsqueeze(0)
                for i in range(len(images) - self.n + 1)]

        if vstack:
            images = torch.vstack(images)

        return images