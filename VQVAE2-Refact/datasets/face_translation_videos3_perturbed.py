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
import datasets.perturbations as perturbations

landmark_name = '_landmarks.npz'
bad_mp4s = 'bad_mp4s.json'
# valid_folders = '/scratch/bipasha31/processed_vlog_dataset_copy/' + 'valid_folders.json'

def get_good_videos(video_dir):
    def get_video_and_segment(x):
        return '/'.join(x.rsplit('/', 2)[-2:])

    video_segments = glob(video_dir + '/*.mp4')
    with open('bad_mp4s.json') as r:
        bad_video_segments = json.load(r)

    with open('valid_folders.json') as r:
        valid_video_segments = json.load(r)
        valid_video_segments = [x+'.mp4' for x in valid_video_segments]

    video_segments = [
        x for x in video_segments 
            if not get_video_and_segment(x) in bad_video_segments
                and get_video_and_segment(x) in valid_video_segments]
    
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
    
    min_frame_length = min(
        source_frames_len-source_index,
        batch_size)
 
    # after generating the source and target indices, we need to sample the frames 
    source_frames_sampled = sample_frames(source_index, min_frame_length, source_video_dir)

    source_masks = list()
    source_images = list()
    source_faces_perturbed = list() # perturbations performed on the source faces
 
    for i in range(len(source_frames_sampled)):
        # construct the paths from the source and target frame paths 
        source_image_path = source_frames_sampled[i]
        source_landmark_npz_path = osp.join(source_image_path.rsplit('.', 1)[0]) + '_landmarks_compressed.npz'
        
        mask, source_image, source_face_perturbed = \
            perturbed_single_image(source_image_path, source_landmark_npz_path)

        source_masks.append(mask)
        source_images.append(source_image)
        source_faces_perturbed.append(source_face_perturbed)
 
    return source_masks, source_imags, source_faces_perturbed

def resize_frame(frame, resize_dim=256):
	h, w, _ = frame.shape

	if h > w:
		padw, padh = (h-w)//2, 0
	else:
		padw, padh = 0, (w-h)//2

	padded = cv2.copyMakeBorder(frame, padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=0)
	padded = cv2.resize(padded, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

	return padded

def apply_mask(mask, image):
	return ((mask / 255.) * image).astype(np.uint8)

def perturbed_single_frame(image_path, landmark_path):
    # returns mask, source_image, source_face_perturbed

    raw_image = io.imread(image_path)

    resized_image = resize_frame(raw_image)
    resized_image_copy = resized_image.copy()
    
    landmark = np.load(landmark_path, allow_pickle=True)['landmark']
    convex_mask = generate_convex_hull(resized_image, landmark[17:])
    face_segmented = apply_mask(convex_mask, resized_image)
    face_segmented_perturbed = perturbations.perturb_image_composite(face_segmented, landmark)

    mask = face_segmented_perturbed[..., 0] != 0
    resized_image[mask] = 0 # background image

    return mask, raw_image_copy, face_segmented_perturbed

def readPoints(nparray) :
	points = []
	
	for row in nparray:
		x, y = row[0], row[1]
		points.append((int(x), int(y)))
	
	return points

def generate_convex_hull(img, points):
	# points = np.load(landmark_path, allow_pickle=True)['mask'].astype(np.uint8)
	points = readPoints(points)

	hull = []
	hullIndex = cv2.convexHull(np.array(points), returnPoints = False)

	for i in range(0, len(hullIndex)):
		hull.append(points[int(hullIndex[i])])

	sizeImg = img.shape   
	rect = (0, 0, sizeImg[1], sizeImg[0])

	hull8U = []
	for i in range(0, len(hull)):
		hull8U.append((hull[i][0], hull[i][1]))

	mask = np.zeros(img.shape, dtype = img.dtype)  

	cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

	# convex_face = ((mask/255.) * img).astype(np.uint8)

	return mask

class FaceTransformsVideos(Dataset):
    def __init__(self, mode, n, max_frame_len, color_jitter_required=True):
        self.mode = mode

        base = '/ssd_scratch/cvit/bipasha31/processed_video_talkingheads/*'

        self.colorJitterRequired = color_jitter_required

        self.H, self.W = 256, 256
        self.n = n

        self.max_len = max_frame_len

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.colorJitterTransformation = [
			transforms.ToPILImage(),
			transforms.ColorJitter(brightness=(1.0,1.5), contrast=(1), saturation=(1.0,1.5)),# hue=(-0.1,0.1)),
			transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]

        with open('valid_videos.txt') as r:
            valid_videos = r.read().splitlines()

        datapoints = sorted(glob(base))
        datapoints = [x for x in datapoints if os.path.basename(x) in valid_videos]

        train_split = int(0.9*len(datapoints))

        # self.videos = datapoints[:train_split]\
        #     if mode == 'train' else datapoints[train_split:]

        self.videos = datapoints
        
        print(f'Loaded {len(self.videos)} videos of {mode} split')
 
    def __len__(self):
        return len(self.videos)
 
    def __getitem__(self, index):
        video_dir = self.videos[index]
        
        masks, source_images, source_face_perturbeds = \
                get_video_frames_perturbed(video_dir, self.max_len)

        # loading images ... 
        # masks, source_images, source_face_perturbations
        source_images = self.load_images_transformed(source_imags)

        if self.colorJitterRequired:
            source_face_perturbeds = [transforms.Compose(self.colorJitterTransformation)(p) for p in source_face_perturbeds]
        else:
            source_face_perturbeds = self.load_images_transformed(source_face_perturbeds)

        source_backgrounds = [apply_mask(masks[i], source_images[i]) for i in range(len(source_images))]
        source_backgrounds = self.load_images_transformed(source_backgrounds)
        
        return source_images, source_face_perturbeds, source_backgrounds

    def load_images_transformed(self, images):
        return [self.transform(p).unsqueeze(0) for p in images]
