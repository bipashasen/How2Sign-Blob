# A dataloader that takes in two random sequences of frames as input 
# Learns the alignment between the pairwise sequence of frames 
# Applies the transformation/alignment between the frames 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

import random
from glob import glob
import os
import json
import os.path as osp

from skimage import io
from skimage.transform import resize

import numpy as np
import cv2

def resize_frame(frame, resize_dim=256):
    h, w, _ = frame.shape

    if h > w:
        padw, padh = (h-w)//2, 0
    else:
        padw, padh = 0, (w-h)//2

    padded = cv2.copyMakeBorder(frame, padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=0)
    padded = cv2.resize(padded, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

    return padded

# Method to combine the face segmented with the face segmentation mask
def combine_images(face_mask, face_image, generate_mask=True):
    image_masked = face_mask.copy()
    if generate_mask:
        mask = face_image[..., 0] != 0
        image_masked[mask] = 0
    
    combined_image = image_masked + face_image
    
    return combined_image

# computes the rotation of the face using the angle of the line connecting the eye centroids 
def compute_rotation(shape):
    # landmark coordinates corresponding to the eyes 
    lStart, lEnd = 36, 41
    rStart, rEnd = 42, 47

    # landmarks for the left and right eyes 
    leftEyePoints = shape[lStart:lEnd]
    rightEyePoints = shape[rStart:rEnd]

    # compute the center of mass for each of the eyes 
    leftEyeCenter = leftEyePoints.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePoints.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) 
    
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    
    dist = np.sqrt((dX ** 2) + (dY ** 2)) # this indicates the distance between the two eyes 
    
    return angle, eyesCenter, dist

# code to generate the alignment between the source and the target image 
def generate_aligned_image(source_landmark_npz, target_landmark_npz, 
                            source_image_path, target_image_path,
                            target_face_mask_path, source_background_mask_path):
    
    source_image = resize_frame(io.imread(source_image_path))
    target_image = resize_frame(io.imread(target_image_path))
    
    source_landmarks = np.load(source_landmark_npz)['mask']
    source_rotation, source_center, source_distance = compute_rotation(source_landmarks)

    target_landmarks = np.load(target_landmark_npz)['mask']
    target_rotation, target_center, target_distance = compute_rotation(target_landmarks)

    # rotation of the source conditioned on the source orientation 
    target_conditioned_source_rotation = source_rotation - target_rotation
    
    # calculate the scaling that needs to be applied on the source image 
    scaling = target_distance / source_distance

    # apply the rotation on the source image
    height, width = 256, 256
    rotate_matrix = cv2.getRotationMatrix2D(center=source_center, angle=target_conditioned_source_rotation, scale=scaling)

    # calculate the translation component of the matrix M 
    rotate_matrix[0, 2] += (target_center[0] - source_center[0])
    rotate_matrix[1, 2] += (target_center[1] - source_center[1])

    target_face_mask = resize(io.imread(target_face_mask_path), (height, width), anti_aliasing=False)

    source_background_mask = resize(io.imread(source_background_mask_path), (height, width), anti_aliasing=False)
    
    source_transformed = cv2.warpAffine(source_image, rotate_matrix, (width, height), flags=cv2.INTER_CUBIC)

    source_background_mask_transformed = cv2.warpAffine(source_background_mask, rotate_matrix, (width, height), flags=cv2.INTER_CUBIC)
    target_background = np.multiply(target_image / 255.0, target_face_mask)
    # target_background = target_face_mask.astype(np.uint8) * target_image/255.
    # source_face_transformed = np.multiply(source_transformed / 255.0, source_background_mask_transformed)
    source_face_transformed = source_background_mask_transformed.astype(np.uint8) * source_transformed

    # generate the perturbed image 
    combined_image = combine_images(source_face_transformed / 255.0, target_background)

    return source_face_transformed, source_background_mask_transformed, target_image, target_face_mask, (combined_image*255).astype(np.uint8)

def get_video_frames_perturbed(video_dir, batch_size):
    video_segments = glob(video_dir + '/*.mp4')
    with open('bad_mp4s.json') as r:
        bad_video_segments = json.load(r)

    video_segments = [x for x in video_segments if not '/'.join(x.rsplit('/', 2)[-2:]) in bad_video_segments]
    
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
    target_images = list()
    target_face_masks = list()
    perturbed_images = list()

    for i in range(len(source_frames_sampled)):
        # construct the paths from the source and target frame paths 
        source_image_path = source_frames_sampled[i]
        target_image_path = target_frames_sampled[i]
        SOURCE_BASE_PATH = osp.join(source_image_path.rsplit('.', 1)[0])
        TARGET_BASE_PATH = osp.join(target_image_path.rsplit('.', 1)[0])

        source_landmark_npz = SOURCE_BASE_PATH + '_landmarks_compressed.npz'
        target_landmark_npz = TARGET_BASE_PATH + '_landmarks_compressed.npz'
        source_background_mask_path = SOURCE_BASE_PATH + '_face_mask_compressed.npz'
        target_face_mask_path = TARGET_BASE_PATH + '_background_mask_compressed.npz'

        if not osp.exists(source_image_path)\
            or not osp.exists(target_image_path)\
            or not osp.exists(source_landmark_npz)\
            or not osp.exists(target_landmark_npz)\
            or not osp.exists(source_background_mask_path)\
            or not osp.exists(target_face_mask_path):

            continue

        # read the data and apply framewise transformation
        source_face_transformed, source_background_mask_transformed, \
        target_image, target_face_mask, perturbed_image = \
                generate_aligned_image(source_landmark_npz, target_landmark_npz, 
                source_image_path, target_image_path,
                target_face_mask_path, source_background_mask_path)

        source_faces.append(source_face_transformed)
        source_background_masks.append(source_background_mask_transformed)
        target_images.append(target_image)
        target_face_masks.append(target_face_mask)
        perturbed_images.append(perturbed_image)

    return source_faces, source_background_masks, target_images, target_face_masks, perturbed_images

def save_frames(frames, video_path, fsp=30):
    height, width, layers = frames[0].shape
    print(f'video path {video_path}, {height}, {width}, {layers}')
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in frames:   
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) 
      
    cv2.destroyAllWindows() 
    video.release()
            
    print(f'Video {video_path} written successfully')

class FaceTransformsVideos(Dataset):
    def __init__(self, mode, n, max_frame_len):
        self.mode = mode

        self.H, self.W = 256, 256
        self.n = n

        self.max_len = max_frame_len

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        base = '/ssd_scratch/cvit/bipasha31/processed_video_talkingheads/*'

        datapoints = sorted(glob(base))
        datapoints = [x for x in datapoints if os.path.isdir(x)]

        train_split = int(0.9*len(datapoints))

        self.videos = datapoints[:train_split]\
            if mode == 'train' else datapoints[train_split:]
        
        print(f'Loaded {len(self.videos)} videos of {mode} split')
 
    def __len__(self):
        return len(self.videos)
 
    def __getitem__(self, index):
        video_dir = self.videos[index]
        # convert lists to tensor
        source_faces, source_background_masks, target_images, \
            target_face_masks, perturbed_images = \
                get_video_frames_perturbed(video_dir, self.max_len)\

        # loading images ... 
        # perturbed images, source faces, target images
        perturbed_images = self.load_images(perturbed_images, transform_dim=False)
        source_faces = self.load_images(source_faces)
        target_images = self.load_images(target_images)
        
        source_background_masks = self.transform_dimensions(source_background_masks, vstack=True)
        target_face_masks = self.transform_dimensions(target_face_masks, vstack=True)

        return perturbed_images, source_faces, target_images, source_background_masks, target_face_masks
 
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