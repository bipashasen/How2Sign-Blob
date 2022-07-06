### Disclaimer: Not a very efficient implementation, code written in a single night
from glob import glob
import os.path as osp
import os
import math
import gc
import argparse

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import mediapipe as mp
import cv2

import face_alignment
import torch


FRAME_LIMIT = 8000
FRAME_THRESHOLD = 50

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def read_frames(video_file):
    video_stream = cv2.VideoCapture(video_file)

    ret, frame = video_stream.read()

    frames = list()
    while ret:
        frames.append(frame)
        ret, frame = video_stream.read()

    return frames

def save_frames_as_video(video_path, frames, fps=25):
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames: 
        # video.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        video.write(frame)
      
    cv2.destroyAllWindows() 
    video.release()

def split_videos(videos, WRITE_DIR):
    total_frames = 0
    for video in tqdm(videos):
        video_stream = cv2.VideoCapture(video)

        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_count = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps

        print(osp.basename(video), fps, frame_count, duration)

        # divide the video into frames of 8000 each 
        ret, frame = video_stream.read()
        frames = list()
        
        ### processing a single video
        global_index = 0

        while ret:
            while ret and len(frames) < FRAME_LIMIT:
                frames.append(frame)
                ret, frame = video_stream.read()

            # break from the loop 
            # write the frames to the disk only if there are atleast 50 frames to write
            if len(frames) > FRAME_THRESHOLD:
                write_dir = osp.join(WRITE_DIR, osp.basename(video).split('.')[0])
                os.makedirs(write_dir, exist_ok=True)
                filename = os.path.join(write_dir, f'{str(global_index).zfill(3)}.mp4')
                print(f'Saving file {filename} with {len(frames)} number of frames')
                total_frames += len(frames)
                save_frames_as_video(filename, frames)
                global_index += 1
                frames = list()

        global_index = 0

    print(f'Total frames generated : {total_frames}, total frames in original video : {frame_count}')


# function to return the bb coordinates of the cropped face
def crop_face_coordinates(image, x_px, y_px, width_px, height_px):
    # using different thresholds/bounds for the upper and lower faces 
    image_height, image_width, _ = image.shape
    lower_face_buffer, upper_face_buffer = 0.25, 0.65
    min_x, min_y, max_x, max_y = x_px, y_px, x_px + width_px, y_px + height_px

    x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
    x_right = min(image_width, int(max_x + (max_x - min_x) * lower_face_buffer))
    y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
    y_down = min(image_height, int(max_y + (max_y - min_y) * lower_face_buffer))

    # compute the size of the image and adjust the width accordingly
    size = max(x_right - x_left, y_down - y_top)
    sw = int((x_left + x_right)/2 - size // 2)

    # draw the bounding box using the updated coordinates 
    # image_buffer1 = image.copy()
    # cv2.rectangle(image_buffer1, (sw, y_top), (sw+size, y_down), (0, 0, 255), 2)
    # return the left-top and right-down corners of the bounding box
    
    return sw, y_top, sw+size, y_down

# function to return the bounding box coordinates given the image 
def bb_coordinates(image):
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    result = results.detections
    image_height, image_width, _ = image.shape
    
    if result is None:
        return -1, -1, -1, -1
    
    bb_values = result[0].location_data.relative_bounding_box

    normalized_x, normalized_y, normalized_width, normalized_height = \
                bb_values.xmin, bb_values.ymin, bb_values.width, bb_values.height

    # the bounding box coordinates are given as normalized values, unnormalize them by multiplying by height and width
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    width_px = min(math.floor(normalized_width * image_width), image_width - 1)
    height_px = min(math.floor(normalized_height * image_height), image_height - 1)
    
    return x_px, y_px, width_px, height_px

def display_image(image, requires_colorization=True):
    plt.figure()
    if requires_colorization:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image)

# function to draw the bounding box given the coordinates
def draw_bb(image, x_px, y_px, width_px, height_px):
    image_copy = image.copy()
    cv2.rectangle(image_copy, (x_px, y_px), (x_px + width_px, y_px + height_px), (0, 0, 255), 2)
    display_image(image_copy)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def crop_get_video(frames, current_indexes, bounding_box, VIDEO_DIR, video_order_id, fps=30):
    left, top, right, down = bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']
    width, height = right - left, down - top
    video_path = osp.join(VIDEO_DIR, str(video_order_id).zfill(5) + '.mp4')
    
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for index in current_indexes:
        # crop out the frames according to the dimension of the bounding box 
        current_frame = frames[index]
        cropped = current_frame[top : down, left : right]
        
        video.write(cropped) 

    print(f'Writing to file : {video_order_id}')
    cv2.destroyAllWindows()


def process_frames(frames, VIDEO_DIR, video_order_id):
    # declaring global scopre for the video_order_id variable
    # global video_order_id
    
    current_frames = list()
    mean_bounding_box = dict()
    iou_threshold = 0.7
    frame_count = 0
    frame_writing_threshold = 30 # minimum number of frames to write
    bb_prev_mean = dict()

    for index, frame in tqdm(enumerate(frames)):
        image_height, image_width, _ = frame.shape
        x_px, y_px, width_px, height_px = bb_coordinates(frame)

        if x_px == -1:
            if len(current_frames) > frame_writing_threshold:
                crop_get_video(frames, current_frames, mean_bounding_box, VIDEO_DIR, video_order_id)
                video_order_id += 1

            # reset 
            current_frames = list()
            frame_count = 0
            mean_bounding_box = dict()
            bb_prev_mean = dict()

        else:
            left, top, right, bottom = crop_face_coordinates(frame, x_px, y_px, width_px, height_px)
            current_bounding_box = {'x1' : left, 'x2' : right, 'y1' : top, 'y2' : bottom}

            if len(mean_bounding_box) == 0:

                mean_bounding_box = current_bounding_box.copy()
                bb_prev_mean = current_bounding_box.copy()

                frame_count += 1
                current_frames.append(index)

            else:
                # UPDATE - compute the iou between the current bounding box and the mean bounding box
                iou = get_iou(bb_prev_mean, current_bounding_box)

                if iou < iou_threshold:
                    mean_left, mean_right, mean_top, mean_down = mean_bounding_box['x1'], mean_bounding_box['x2'], mean_bounding_box['y1'], mean_bounding_box['y2']

                    if len(current_frames) > frame_writing_threshold:
                        crop_get_video(frames, current_frames, mean_bounding_box, VIDEO_DIR, video_order_id)
                        video_order_id += 1

                    current_frames = list()
                    frame_count = 0
                    mean_bounding_box = dict()

                # Add the current bounding box to the list of bounding boxes and compute the mean
                else:
                    mean_bounding_box['x1'] = min(mean_bounding_box['x1'], current_bounding_box['x1'])
                    mean_bounding_box['y1'] = min(mean_bounding_box['y1'], current_bounding_box['y1'])
                    mean_bounding_box['x2'] = max(mean_bounding_box['x2'], current_bounding_box['x2'])
                    mean_bounding_box['y2'] = max(mean_bounding_box['y2'], current_bounding_box['y2'])

                    # update the coordinates of the mean bounding box 
                    for item in bb_prev_mean.keys():
                        bb_prev_mean[item] = int((bb_prev_mean[item] * frame_count + current_bounding_box[item])/(frame_count + 1))

                    frame_count += 1
                    current_frames.append(index)
                    
    # save the remaining frames as a video if there are remaining and satisfy the threshold 
    if len(current_frames) > frame_writing_threshold:
        crop_get_video(frames, current_frames, mean_bounding_box, VIDEO_DIR, video_order_id)
        video_order_id += 1

# write the main code for processing the videos
def process_videos(video_files, processed_videos_dir):
    for video_file in tqdm(video_files):
        video_stream = cv2.VideoCapture(video_file)
        print(f'Processing video file : {video_file}')
        print(f'Total number of frames in the current video : {video_stream.get(cv2.CAP_PROP_FRAME_COUNT)}')

        video_order_id = 0
        frames = list() 

        video_file_name = osp.basename(video_file).split('.')[0]
        VIDEO_DIR = osp.join(processed_videos_dir, video_file_name)
        os.makedirs(VIDEO_DIR, exist_ok=True)

        ret, frame = video_stream.read()

        # since the number of video files were already restricted to 8000 frames, no customization is required
        while ret:
            frames.append(frame)
            ret, frame = video_stream.read()

        # after reading all the frames, process the frames
        process_frames(frames, VIDEO_DIR, video_order_id)
        print(f'Done processing file : {video_file}, saving results to : {VIDEO_DIR}')
        
        # def frames
        frames = None
        gc.collect()
        video_stream.release()


def drawPolyline(image, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end+1):
        point = [landmarks[i][0], landmarks[i][1]]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (0, 255, 255), 2, 16)

# Draw lines around landmarks corresponding to different facial regions
def drawPolylines(image, landmarks):
    drawPolyline(image, landmarks, 0, 16)           # Jaw line
    drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(image, landmarks, 27, 30)          # Nose bridge
    drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(image, landmarks, 36, 41, True)    # Left eye
    drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

# Detect landmarks for the given batch
def generate_batch_landmarks(batches, fa):
    gpu_id = 0
    batch_landmarks = list()

    for current_batch in batches:
        current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2).to('cuda:{}'.format(gpu_id))
        # current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2)
        landmarks = fa.get_landmarks_from_batch(current_batch)
        batch_landmarks.extend(landmarks)

    return batch_landmarks

def resize_frames(frames, resize_dim=256):
    resized = list()
    for frame in frames:
        resized.append(resize_frame(frame))

    return resized

def resize_frame(frame, resize_dim=256):
    h, w, _ = frame.shape

    if h > w:
        padw, padh = (h-w)//2, 0
    else:
        padw, padh = 0, (w-h)//2

    padded = cv2.copyMakeBorder(frame, padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=0)
    padded = cv2.resize(padded, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

    return padded

def read_frames(filepath):
    frames = list()

    video_stream = cv2.VideoCapture(filepath)
    ret, frame = video_stream.read()

    while ret:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ret, frame = video_stream.read()

    return frames

def save_frames(frames, save_dir):
    for i, frame in enumerate(frames):
        filepath = osp.join(save_dir, str(i).zfill(5) + '.jpg')
        cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def save_landmarks(landmarks, save_dir):
    for i, landmark in enumerate(landmarks):
        filepath = osp.join(save_dir, str(i).zfill(5) + '_landmarks.npz')
        if len(landmark) > 0:
            np.savez_compressed(filepath, landmark=landmark)

# This code is used for detecting face along with generating landmarks for the face image 
def generate_image_landmarks(video_path, fa, good_filepath, bad_filepath):
    batch_size = 32

    print(f'Processing video file : {video_path}', flush=True)

    video_stream = cv2.VideoCapture(video_path)

    threshold_limit = 1000
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > threshold_limit:
        print(f'Execution failed for video : {video_path}, continuing')
        with open(bad_filepath, 'a') as f:
            f.write(video_path + '\n')
        return

    image_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video : {video_path}, Frames read : {total_frames}, image height and width: {image_height}, {image_width}', flush=True)

    folder_name = osp.dirname(video_path) # gets the dir path 

    # If debug is enabled true, then image frames go inside the specific XXXX folder 
    save_folder_path = os.path.join(folder_name, os.path.basename(video_path).split('.')[0]) # SpeakerVideos/AnfisaNava/video_id/XXXX

    frames = read_frames(video_path)

    # resize the frames to the required dimension
    resized = resize_frames(frames, resize_dim=256)
    frames = None

    batches = [resized[i:i+batch_size] for i in range(0, len(resized), batch_size)]
    processed = False
    while not processed:
        try:
            if batch_size == 0:
                # Write this as a bad file
                with open(bad_filepath, 'a') as f:
                    f.write(video_path + '\n')
                continue
            landmarks = generate_batch_landmarks(batches, fa)
            processed = True
        except Exception as e: # Exception arising out of CUDA memory unavailable
            print(e)
            batch_size = batch_size // 2
            print(f'Cuda memory unavailable, reducing batch size to : {batch_size}', flush=True)
            batches = [resized[i:i+batch_size] for i in range(0, len(resized), batch_size)]
            continue

    landmark_threshold = 68 # Ignore frames where landmarks detected is not equal to landmark_threshold
    frames_ignored = 0
    frame_ignore_threshold = 10 # reject video if more than 10% of frames are bad 

    for landmark in landmarks:
        # check if the landmarks were generated for the current frame
        if len(landmark) == 0:
            frames_ignored += 1

    # check if the video file needs to be processed 
    if total_frames == 0 or ((frames_ignored/total_frames)*100 > frame_ignore_threshold):
        print(f'Bad video {video_path}, ignoring!, frames ignored : {frames_ignored}, total frames : {total_frames}', flush=True)
        with open(bad_filepath, 'a') as f:
            f.write(video_path + '\n')
        return

    # create folder only if required 
    os.makedirs(save_folder_path, exist_ok=True)

    # save resized frames and landmarks
    save_frames(resized, save_folder_path)
    save_landmarks(landmarks, save_folder_path)

    # write into the valid filepath 
    with open(good_filepath, 'a') as f:
        f.write(video_path + '\n')


# code for generating the face landmarks from the face video
def generate_face_landmarks(dirname):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(0))

    video_dir = '/ssd_scratch/cvit/aditya1/dataset_generation/processed_videos_validation'
    # dirname = 'Dmtceb5SMXA'

    bad_filepath = '/ssd_scratch/cvit/aditya1/dataset_generation/processed_videos_validation/{}/bad_files.txt'.format(dirname)
    good_filepath = '/ssd_scratch/cvit/aditya1/dataset_generation/processed_videos_validation/{}/good_files.txt'.format(dirname)

    dir_to_process = osp.join(video_dir, dirname)
    # for the dir to process -- get all the videos inside the dir
    video_files = sorted(glob(dir_to_process + '/*/*.mp4'))

    print(f'Total number of video files to process : {len(video_files)}')
    # print(f'Video files are : {video_files}')

    for video_file in tqdm(video_files):
        generate_image_landmarks(video_file, fa, good_filepath, bad_filepath)

def main(args):
    #### Code for generating the video splits from the larger videos (each video is of duration 8000 frames)
    # video_dir = '/ssd_scratch/cvit/aditya1/dataset_generation/saved_videos/'
    # videos = glob(video_dir + '/*.mp4')

    # WRITE_DIR = '/ssd_scratch/cvit/aditya1/dataset_generation/saved_videos/splits'
    # os.makedirs(WRITE_DIR, exist_ok=True)

    # split_videos(videos, WRITE_DIR)
    
    #### Code for splitting the video according to the faces detected
    # video_dir_path = '/ssd_scratch/cvit/aditya1/dataset_generation/saved_videos/splits'
    # # video_files = sorted(glob(video_dir_path + '/*/*.mp4')) # should not have given the videos directly

    # processed_videos_dir_path = '/ssd_scratch/cvit/aditya1/dataset_generation/processed_videos_validation'    

    # video_dirs = sorted(glob(video_dir_path + '/*'))
    # print(f'Total number of video dirs to process : {len(video_dirs)}')
    # for video_dir in tqdm(video_dirs):
    #     target_video_dir = osp.join(processed_videos_dir_path, osp.basename(video_dir))

    #     os.makedirs(target_video_dir, exist_ok=True)

    #     # send all the video files inside the current directory
    #     video_files = sorted(glob(video_dir + '/*.mp4'))

    #     print(f'Currently processing dir : {video_dir}')
    #     print(f'Number of video files inside the current dir : {len(video_files)}, files are : {video_files}')
    #     print(f'Target dir is : {target_video_dir}')

    #     process_videos(video_files, target_video_dir) # all video files share the same meta directory

    dirname = args.dirname
    # dirname = '-VwgK3V938E'
    generate_face_landmarks(dirname)

    pass

if __name__ == '__main__':
    # read the videos and split based on the number of frames that they have 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str, default=None)
    args = parser.parse_args()
    
    main(args)
