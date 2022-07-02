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

import TemporalAlignment.perturbations as perturbations
# from datasets.face_translation_videos3_utils_bb import *
from datasets.face_translation_videos3_utils import *

requires_bb = False
extract_lip_region = False

if extract_lip_region:
    start_idx = 49
    end_idx = 61
else:
    start_idx = 0
    end_idx = 67

# # for convex hull
# start_idx = 0
# end_idx = -1

# for lip region
# start_idx = 49
# end_idx = 61

def perturbed_single_image(image_path, landmark_path):
    # returns mask, source_image, source_face_perturbed

    raw_image = io.imread(image_path)

    resized_image = resize_frame(raw_image)
    resized_image_copy = resized_image.copy()
    
    landmark = np.load(landmark_path, allow_pickle=True)['landmark']
    if requires_bb:
        convex_mask = generate_convex_hull_bb(resized_image, landmark[start_idx:end_idx])
    else:
        convex_mask = generate_convex_hull(resized_image, landmark[start_idx:end_idx])

    face_segmented = apply_mask(convex_mask, resized_image)
    face_segmented_perturbed, gt_transformations = perturbations.perturb_image_composite(face_segmented, landmark)

    face_background = apply_mask(np.invert(convex_mask), resized_image)

    mask = face_segmented_perturbed[..., 0] != 0
    resized_image[mask] = 0 # background image

    return mask, face_segmented, resized_image_copy, face_segmented_perturbed, gt_transformations, face_background

def get_video_frames_perturbed(video_dir, batch_size):
    def get_key(x):
        return int(x.split('/')[-1].split('_')[0])

    landmark_paths = sorted(glob(f'{video_dir}/*_landmarks.npz'), key=lambda x: get_key(x))

    # print(f'number of landmarks: {len(landmark_paths)}')
    # batch_size could be more than the number of jpgs itself,  
    # the second value could be negative in such a case.
    index = random.randint(0, max(5, len(landmark_paths) - batch_size - 1))
 
    # after generating the source and target indices, we need to sample the frames 
    source_landmarks_sampled = landmark_paths[index:index+batch_size]

    source_masks = list()
    source_images = list()
    face_segmenteds = list()
    source_backgrounds = list()
    gt_transformations = list()
    source_faces_perturbed = list() # perturbations performed on the source faces
 
    for i in range(len(source_landmarks_sampled)):
        # construct the paths from the source and target frame paths 
        source_landmark_path = source_landmarks_sampled[i]

        # check what image extension to use 
        file_without_extension = osp.join(source_landmark_path.rsplit('_', 1)[0])
        if os.path.exists(file_without_extension + '.jpg'):
            source_image_path = file_without_extension + '.jpg'
        else:
            source_image_path = file_without_extension + '.png'

        mask, face_segmented, source_image, source_face_perturbed, gt_transformation, source_background = \
            perturbed_single_image(source_image_path, source_landmark_path)

        face_segmenteds.append(face_segmented)
        source_backgrounds.append(source_background)
        source_masks.append(mask)
        source_images.append(source_image)
        source_faces_perturbed.append(source_face_perturbed)
        gt_transformations.append(gt_transformation)
 
    return source_masks, face_segmenteds, source_images, source_faces_perturbed, gt_transformations, source_backgrounds

def get_source_target_video_frames_perturbed(video_dir_source, video_dir_target, batch_size, keep_same_index=False):
    landmark_paths_source = sorted(glob(f'{video_dir_source}/*_landmarks.npz'))
    landmark_paths_target = sorted(glob(f'{video_dir_target}/*_landmarks.npz'))

    print(f'Source landmarks : {len(landmark_paths_source)}, target landmarks : {len(landmark_paths_target)}')

    # print(f'Landmark length {len(landmark_paths_source)}, target : {len(landmark_paths_target)}')

    # print(f'number of landmarks: {len(landmark_paths)}')
    # batch_size could be more than the number of jpgs itself,  
    # the second value could be negative in such a case.
    index_source = random.randint(0, max(5, len(landmark_paths_source) - batch_size - 1))
    index_target = random.randint(0, max(5, len(landmark_paths_target) - batch_size - 1))

    if keep_same_index:
        # TODO target and source index is kept the same
        index_target = index_source

    # print(f'Indices are : {index_source}, {index_target}')
 
    # after generating the source and target indices, we need to sample the frames 
    source_landmarks_sampled = landmark_paths_source[index_source:index_source+batch_size]
    target_landmarks_sampled = landmark_paths_target[index_target:index_target+batch_size]

    if len(source_landmarks_sampled) != len(target_landmarks_sampled):
        min_len = min(len(source_landmarks_sampled), len(target_landmarks_sampled))

        source_landmarks_sampled = source_landmarks_sampled[:min_len]
        target_landmarks_sampled = target_landmarks_sampled[:min_len]

    source_images = list()
    source_face_perturbeds = list()
    target_images = list()
    target_backgrounds = list()
 
    for i in range(len(source_landmarks_sampled)):
        # construct the paths from the source and target frame paths 
        source_landmark_npz = source_landmarks_sampled[i]
        target_landmark_npz = target_landmarks_sampled[i]

        extension = '.jpg'

        source_image_path = osp.join(source_landmark_npz.rsplit('_', 1)[0]) + extension
        target_image_path = osp.join(target_landmark_npz.rsplit('_', 1)[0]) + extension

        if not osp.exists(source_image_path):
            extension = '.png'
            source_image_path = source_image_path.replace('.jpg', extension)

        if not osp.exists(target_image_path):
            extension = '.png'
            target_image_path = target_image_path.replace('.jpg', extension)


        # print(f'Source image : {source_image_path}, target image : {target_image_path}')
        
        # read the data and apply framewise transformation
        output = generate_warped_image(source_landmark_npz, target_landmark_npz, 
                source_image_path, target_image_path)

        source_face_perturbed = output[0]
        target_image = output[4]
        target_without_face_features = output[7]
        source_image = output[8]
 
        source_face_perturbeds.append(source_face_perturbed)
        target_images.append(target_image)
        target_backgrounds.append(target_without_face_features)
        source_images.append(source_image)
 
    return source_face_perturbeds, target_images, target_backgrounds, source_images

def get_validation_datapoints(validation_datapoints_dir):
    # specify default path is no path is provided
    if validation_datapoints_dir is None:
        # validation_datapoints_dir = '/ssd_scratch/cvit/aditya1/acm_rebuttal/validation'
        # validation_datapoints_dir = '/ssd_scratch/cvit/aditya1/rebuttal_scores_validation/validation_scores_datadir'
        validation_datapoints_dir = '/ssd_scratch/cvit/aditya1/data_video_vqvae/metrics_baseline'

    print(f'Validation dir is : {validation_datapoints_dir}')

    # video_segments = glob(base + '/*_source')
    # video_segments = glob(validation_datapoints_dir + '/*/*.mp4') + glob(validation_datapoints_dir + '/*_source') + glob(validation_datapoints_dir + '/*_target')
    # video_segments = glob(validation_datapoints_dir + '/*') + glob(validation_datapoints_dir + '/*/*.mp4')

    # video_segments = glob(validation_datapoints_dir + '/*_source')
    # video_segments = glob(validation_datapoints_dir + '/*_source/[0-9][0-9][0-9]')
    # video_segments = glob(validation_datapoints_dir + '/*/[0-9][0-9][0-9][0-9][0-9]')

    video_segments = glob(validation_datapoints_dir + '/source_gt/*.mp4')

    # works for both directories and mp4 files
    def is_good_video(dir):
        return len(glob(f'{dir.split(".")[0]}/*_landmarks.npz')) > 10

    return [x.replace('.mp4', '') for x in video_segments if is_good_video(x)]

    # return [x for x in video_segments if is_good_video(x)]

def get_datapoints():
    base = '/ssd_scratch/cvit/aditya1/data_video_vqvae/processed_vlog_dataset_copy'
    valid_videos_json_path = os.path.join(base, 'valid_folders.json')
    # base = '/scratch/bipasha31/processed_vlog_dataset_copy'
    # valid_videos_json_path = base + '/valid_folders.json'

    def get_name(x):
        return '/'.join(x.split('/')[-4:])

    # with open(valid_videos_json_path) as r:
    #     valid_videos = json.load(r)

    video_segments = glob(base + '/*/*/*/*.mp4')

    # def is_good_video(dir):
    #     name = get_name(dir)
    #     return name in valid_videos\
    #         and len(glob(f'{dir.split(".")[0]}/*_landmarks.npz')) > 3

    def is_good_video(dir):
        return True

    return [x.replace('.mp4', '') for x in video_segments if is_good_video(x)]

class TemporalAlignmentDataset(Dataset):
    def __init__(
        self, 
        mode, 
        max_frame_len,
        case='jitter', 
        color_jitter_type='',
        cross_identity_required=False,
        grayscale_required=False,
        custom_validation_required=False,
        validation_datapoints=None):

        self.mode = mode

        self.colorJitterType = color_jitter_type
        self.custom_validation_required = custom_validation_required

        if cross_identity_required:
            self.colorJitterType = ''

        self.H, self.W = 256, 256

        self.max_len = max_frame_len

        self.case = case

        self.cross_identity_required = cross_identity_required

        # normal transform elements

        transformElements = [transforms.ToPILImage()]

        if grayscale_required:
            transformElements.append(transforms.Grayscale())

            normalize_mean, normalize_std = .5, .5
        else:
            normalize_mean, normalize_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        transformElements.append(transforms.ToTensor())

        if case == 'jitter':
            transformElements.append(transforms.Normalize(normalize_mean, normalize_std))

        self.transform = transforms.Compose(transformElements)

        self.colorJitterTransformationElements = [
			transforms.ToPILImage(),
			transforms.ColorJitter(brightness=(1.0,1.5), contrast=(1), saturation=(1.0,1.5)),
			transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]

        print(f'Running mode is : {mode}')

        if mode == 'train':
            self.videos = get_datapoints()
            print(f'Number of training datapoints loaded : {len(self.videos)}')
        else:
            self.videos = get_validation_datapoints(validation_datapoints)
            print(f'Number of validation videos loaded : {len(self.videos)}')
            print(f'The validation videos are : {self.videos}')
        
        # print(f'Loaded {len(self.videos)} videos of {mode} split')
 
    def __len__(self):
        return len(self.videos)
 
    def __getitem__(self, index):
        try:
            if self.case == 'jitter':
                if self.cross_identity_required:
                    if self.custom_validation_required:
                        return self.get_item_jitter_network_custom_validation(index)
                    else:   
                        return self.get_item_jitter_network_cross_identity(index)
                else:
                    return self.get_item_jitter_network(index)
            else:
                return self.get_item_alignment_network(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

        '''
        # print(f'Case is {self.case}')
        if self.case == 'jitter':
            # print(f'Saw case jitter')
            if self.cross_identity_required:
                print(f'Cross id required')
                if self.custom_validation_required:
                    print('Custom validation required')
                    return self.get_item_jitter_network_custom_validation(index)
                else:
                    return self.get_item_jitter_network_cross_identity(index)
            else:
                # print(f'Require jitter network')
                return self.get_item_jitter_network(index)
        else:
            return self.get_item_alignment_network(index)
        '''
        
    def get_item_jitter_network_custom_validation_old(self, index):
        # print(f'hello world')
        source_video_dir = self.videos[index]
        target_video_dir = self.videos[random.randint(0, self.__len__()-1)]

        if self.custom_validation_required:
            # print(f'Self videos path : {self.videos}')
            full_dir = '/ssd_scratch/cvit/aditya1/supplementary/testing_data_50_trial1'
            target_videos = glob(full_dir + '/*_target/[0-9][0-9][0-9]')

            target_video_dir = target_videos[random.randint(0, len(target_videos)-1)]

        else:
            try:
                target_type = source_video_dir.rsplit('_', 1)[1]
            except:
                target_type = None

        # print(f'Source video dir : {source_video_dir}, target type : {target_type}')
        keep_same_index = False

        # print(target_type)
        # if target_type == 'source':
        #     # get the target dir
        #     target_video_dir = source_video_dir.replace(target_type, 'target')
        #     keep_same_index = True
        # # elif target_type == 'target':
        # elif target_type == 'target':
        #     # get the source dir 
        #     target_video_dir = source_video_dir.replace(target_type, 'source')
        #     keep_same_index = True

        source_face_perturbeds, target_images, target_backgrounds, source_images =\
            get_source_target_video_frames_perturbed(
                source_video_dir, target_video_dir, self.max_len, keep_same_index=keep_same_index)

        # print(len(source_face_perturbeds))

        source_face_perturbeds = self.load_images_transformed(source_face_perturbeds)
        target_images = self.load_images_transformed(target_images)
        target_backgrounds = self.load_images_transformed(target_backgrounds)
        source_images = self.load_images_transformed(source_images)

        return source_face_perturbeds, [], target_backgrounds, target_images, source_images


    # custom method defined for computing the scores for the acm rebuttal 
    def get_item_jitter_network_custom_validation(self, index):
        # print(f'hello world')

        source_video_dir = self.videos[index]
        target_video_dir = self.videos[index].replace('source', 'target')

        print(f'Current source : {source_video_dir}, current target : {target_video_dir}')

        # if self.custom_validation_required:
        #     # print(f'Self videos path : {self.videos}')
        #     full_dir = '/ssd_scratch/cvit/aditya1/supplementary/testing_data_50_trial1'
        #     target_videos = glob(full_dir + '/*_target/[0-9][0-9][0-9]')

        #     target_video_dir = target_videos[random.randint(0, len(target_videos)-1)]

        # else:
        #     try:
        #         target_type = source_video_dir.rsplit('_', 1)[1]
        #     except:
        #         target_type = None

        # print(f'Source video dir : {source_video_dir}, target type : {target_type}')
        keep_same_index = False

        # print(target_type)
        # if target_type == 'source':
        #     # get the target dir
        #     target_video_dir = source_video_dir.replace(target_type, 'target')
        #     keep_same_index = True
        # # elif target_type == 'target':
        # elif target_type == 'target':
        #     # get the source dir 
        #     target_video_dir = source_video_dir.replace(target_type, 'source')
        #     keep_same_index = True

        source_face_perturbeds, target_images, target_backgrounds, source_images =\
            get_source_target_video_frames_perturbed(
                source_video_dir, target_video_dir, self.max_len, keep_same_index=keep_same_index)

        # print(len(source_face_perturbeds))

        source_face_perturbeds = self.load_images_transformed(source_face_perturbeds)
        target_images = self.load_images_transformed(target_images)
        target_backgrounds = self.load_images_transformed(target_backgrounds)
        source_images = self.load_images_transformed(source_images)

        return source_face_perturbeds, [], target_backgrounds, target_images, source_images


    def get_item_jitter_network_cross_identity(self, index):
        source_video_dir = self.videos[index]
        target_video_dir = self.videos[random.randint(0, self.__len__()-1)]

        source_face_perturbeds, target_images, target_backgrounds, source_images =\
            get_source_target_video_frames_perturbed(
                source_video_dir, target_video_dir, self.max_len)

        source_face_perturbeds = self.load_images_transformed(source_face_perturbeds)
        target_images = self.load_images_transformed(target_images)
        target_backgrounds = self.load_images_transformed(target_backgrounds)
        source_images = self.load_images_transformed(source_images)

        return source_face_perturbeds, [], target_backgrounds, target_images, source_images

    def get_item_jitter_network(self, index):
        video_dir = self.videos[index]

        _, target, source_images, source, _, background =\
            get_video_frames_perturbed(video_dir, self.max_len)

        if len(source) == 0\
            or len(target) == 0\
            or len(background) == 0\
            or len(source_images) == 0:
            print(f'Empty Directory, check again - {video_dir}')
            print(f'Source: {len(source)}, Target: {len(target)}, Background: {len(background)}, SourceImages: {len(source_images)}')
            assert False

        source = self.get_source_image_hull(source)
        target = self.load_images_transformed(target)
        background = self.load_images_transformed(background)
        source_images = self.load_images_transformed(source_images)

        return source, target, background, source_images, target

    def get_item_alignment_network(self, index):
        video_dir = self.videos[index]
        
        masks, _, source_images, source_face_perturbeds, gt_transformations, source_backgrounds = \
            get_video_frames_perturbed(video_dir, self.max_len)

        # we need to predict the inverse of the applied transformation
        gt_transformations = -1 * torch.tensor([
            [x['rotate_image'], x['translate_horizontal'], x['translate_vertical']]
            for x in gt_transformations
        ])

        source_backgrounds = self.load_images_transformed(source_backgrounds)

        source_images = self.load_images_transformed(source_images)

        source_face_perturbeds = self.get_source_image_hull(source_face_perturbeds)

        input = torch.cat([ source_face_perturbeds, source_backgrounds], axis=1)
        
        return input, source_images, gt_transformations

    def get_source_image_hull(self, source_face_perturbeds):
        if self.colorJitterType != '':
            elements = self.colorJitterTransformationElements

            if self.colorJitterType == 'const':
                brightness = random.uniform(1.0,1.5)
                saturation = random.uniform(1.0,1.5)

                elements[1] = transforms.ColorJitter(
                    brightness=(brightness, brightness), contrast=(1, 1), saturation=(saturation, saturation))

            color_jitter_transform = transforms.Compose(elements)

            return self.load_images_transformed(
                source_face_perturbeds, jitter=color_jitter_transform)
        else:
            return self.load_images_transformed(source_face_perturbeds)

    def load_images_transformed(self, images, jitter=None):
        transform_function = jitter\
            if jitter is not None else self.transform

        # print(f'The len of images are : {len(images)}')

        return torch.vstack([transform_function(p).unsqueeze(0) for p in images])

