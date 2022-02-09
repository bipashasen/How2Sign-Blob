import cv2
import numpy as np
import os
import json
import func

import glob
import multiprocessing as mp
import tempfile
import time
import warnings
import tqdm

from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

import glob

frame_width, frame_height = 100, 100

dim = (frame_width,frame_height)
fps = 25

output_extension = '.avi'

transcripts = 'transcripts'
os.makedirs(transcripts, exist_ok=True)

def extract_blob(bb, image, kp=None, score=None):
	buff = 10

	confidence_threshold = .1

	if score:
		confidence = score
	else:
		confidence = np.mean(kp[:, 2])
	
	if confidence >= confidence_threshold:
		bb = [ int(x) for x in bb]
	 
		w, h = bb[2]-bb[0], bb[3]-bb[1]
		size = max(w, h) // 2

		# midw, midh = int(np.mean(kp[:, 0])), int(np.mean(kp[:, 1]))
		midw, midh = (bb[2]+bb[0])//2, (bb[3]+bb[1])//2

		startw, starth, endw, endh = midw-size-buff, midh-size-buff, midw+size+buff, midh+size+buff  

		cropped = image[starth:endh, startw:endw]

		# print(cropped.shape)
		
		if cropped.shape[0] > 0 and cropped.shape[1] > 0 and cropped.shape[1] == cropped.shape[0]:
			frame = cv2.resize(cropped, dim)

			return frame
		else:
			return None

	return None

def generate_hands(demo, video_input):
	start_time = time.time()

	video = cv2.VideoCapture(video_input)

	num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	assert os.path.isfile(video_input)

	imgs, boxes, scores = [], [], []

	for img, predictions in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
		instances = predictions["instances"]
		boxes.append(instances.pred_boxes.tensor.detach().cpu().numpy())
		scores.append(instances.scores)

		imgs.append(img)

	print('time elapsed: {}'.format(time.time() - start_time), flush=True)

	video.release()

	return imgs, boxes, scores

def generate_blobs(demo, video, json_base, output_path):
	output_paths = [output_path.replace(output_extension,\
		'_{}_{}'.format(x, output_extension)) for x in ['right', 'left', 'face']]

	if all([os.path.exists(x) for x in output_paths]):
		print('skipping {}'.format(video), flush=True)
		return

	# hands_right, hands_left, face
	outs = [
		cv2.VideoWriter(output_paths[0], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim),\
		cv2.VideoWriter(output_paths[1], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim),\
		cv2.VideoWriter(output_paths[2], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim)
	]

	images, hand_boxes, hand_scores = generate_hands(demo, video)

	total_skipped, face_skipped, hand_skipped = 0, 0, 0

	vname = video.rsplit('/', 1)[-1].split('.')[0]

	valid_frames = []

	for count, image in enumerate(images):
		if hand_boxes[count].shape[0] == 2:
			kp_path = os.path.join(json_base, '{}_{}_keypoints.json'.format(vname, str(count).zfill(12)))

			if os.path.exists(kp_path):
				with open(kp_path) as r:
					kp = json.load(r)['people'][0]

				# x, y, score of face
				face_kp = np.array(kp['face_keypoints_2d']).reshape(-1, 3)
				# x0, y0, x1, y1
				face_bb = np.min(face_kp[:, 0]), np.min(face_kp[:, 1]), np.max(face_kp[:, 0]), np.max(face_kp[:, 1])

				face = extract_blob(face_bb, image, kp=face_kp)
				if face is not None:
					'''
					Do for hands
					'''
					box = hand_boxes[count]
					score = hand_scores[count]
		
					if box[0][2] < box[1][2]:
						rhand, lhand = box[0], box[1]
					else:
						rhand, lhand = box[1], box[0]
					
					hand_bbs = [rhand, lhand]
					
					hands = [extract_blob(hand_bbs[i], image, score=score[i]) for i in range(len(hand_bbs))]

					for i in range(len(hand_bbs)):
						outs[i].write(hands[i])
					outs[-1].write(face)

					valid_frames.append(str(count))
				else:
					face_skipped += 1
					total_skipped += 1
			else:
				face_skipped += 1
				total_skipped += 1
		else:
			hand_skipped +=1 
			total_skipped += 1

	skipped_percent = round(total_skipped/count, 4)*100
	print(f'done with {video}. Total skipped {total_skipped} out of {count} = {skipped_percent}%', flush=True)
	print(f'Hand Skipped {hand_skipped} and face skipped {face_skipped}', flush=True)

	save_txt = '{}/{}.txt'.format(transcripts, vname)

	with open(save_txt, 'w') as w:
		w.write('{}\n'.format(video))
		w.write('{}\n'.format(','.join(valid_frames)))

	with open('skipped.txt', 'a') as w:
		w.write('{}::{}::{}\n'.format(video, total_skipped, count))

	for out in outs:
		out.release()

	if count == 0:
		for output_path in output_paths:
			os.remove(outputs_path)

	cv2.destroyAllWindows()

	return total_skipped, count, hand_skipped, face_skipped

def main():
	'''
	Setup
	'''
	mp.set_start_method("spawn", force=True)

	args = func.Args()

	setup_logger(name="fvcore")
	logger = setup_logger()

	cfg = func.setup_cfg(args)

	demo = VisualizationDemo(cfg)

	'''
	Run
	'''
	videos_path, jsons_path, outputs_path = '../../ds/train-set/train_rgb_front_clips/raw_videos', '../../ds/train-set/train_2D_keypoints/openpose_output/json', 'video_outputs_v2'

	if not os.path.exists(outputs_path):
	    os.makedirs(outputs_path)

	print('Paths exists:', os.path.exists(videos_path), os.path.exists(jsons_path), os.path.exists(outputs_path), flush=True)

	videos = glob.glob('{}/*.mp4'.format(videos_path))

	print(f'Total videos to process: {len(videos)}', flush=True)

	global_count, global_total_skipped = 0, 0
	global_hand_skipped, global_face_skipped = 0, 0

	for i, video in enumerate(sorted(videos)):
	    output_path = video.replace('.mp4', output_extension).replace(videos_path, outputs_path)
	    json_base_path = video.replace('.mp4', '').replace(videos_path, jsons_path)
	    
	    print('starting #{}/{}: {}'.format(i, len(videos), video), flush=True)
	    
	    obj = generate_blobs(demo, video, json_base_path, output_path)
	    
	    if obj:
	        ts, c, hts, fts = obj
	        global_count += c
	        global_total_skipped += ts
	        global_hand_skipped += hts
	        global_face_skipped += fts

	percent = round((global_total_skipped/(global_count+(1e-9)))*100, 2)
	print()
	print('{} skipped out of {} = {}%'.format(global_total_skipped, global_count, percent), flush=True)

	print('{} hands skipeped and {} face skipped'.format(global_hand_skipped, global_face_skipped), flush=True)

if __name__ == '__main__':
	main()