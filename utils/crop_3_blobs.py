import cv2
import numpy as np
import os
import json

import glob

frame_width, frame_height = 100, 100

dim = (frame_width,frame_height)
fps = 25

output_extension = '.avi'

def extract_blob(bb, image, hands=None, score=None):
	buff = 10

	confidence_threshold = .1

	if scores:
		confidence = score
	else:
		confidence = np.mean(hands[:, 2])
	
	if confidence >= confidence_threshold:
		bb = [ int(x) for x in bb]
	 
		w, h = bb[2]-bb[0], bb[3]-bb[1]
		size = max(w, h) // 2

		# midw, midh = int(np.mean(hands[:, 0])), int(np.mean(hands[:, 1]))
		midw, midh = (bb[2]+bb[0])//2, (bb[3]+bb[1])//2

		startw, starth, endw, endh = midw-size-buff, midh-size-buff, midw+size+buff, midh+size+buff  

		cropped = image[starth:endh, startw:endw]

		# print(cropped.shape)

		if cropped.shape[0] > 0 and cropped.shape[1] > 0:
			frame = cv2.resize(cropped, dim)

			return frame
		else:
			return None

	return None

def generate_hand_videos(video, json_base, output_path):
	output_paths = [output_path.replace(output_extension,\
		'_{}_{}'.format(x, output_extension)) for x in ['right', 'left', 'face']]

	if all([os.path.exists(x) for x in output_paths]):
		print('skipping {}'.format(video))
		return

	count = 0
	total_skipped = 0

	vidcap = cv2.VideoCapture(video)
	success, image = vidcap.read()

	# hands_right, hands_left, face
	outs = [
		cv2.VideoWriter(output_paths[0], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim),\
		cv2.VideoWriter(output_paths[1], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim),\
		cv2.VideoWriter(output_paths[2], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim)
	]

	vname = video.rsplit('/', 1)[-1].split('.')[0]

	while success:
		kp_path = os.path.join(json_base, '{}_{}_keypoints.json'.format(vname, str(count).zfill(12)))

		if os.path.exists(kp_path):
			with open(kp_path) as r:
				kp = json.load(r)['people'][0]

			# x, y, score of hands_rights, hands_left, and face
			blob_kps = [np.array(kp['hand_right_keypoints_2d']), np.array(kp['hand_left_keypoints_2d']), np.array(kp['face_keypoints_2d'])]
			blob_kps = [x.reshape(-1, 3) for x in blob_kps]

			bbs = [[np.min(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 0]), np.max(x[:, 1])] for x in blob_kps]
			# x0, y0, x1, y1

			blobs = []

			for i in range(len(bbs)):
				blobs.append(extract_blob(bbs[i], image, hands=blob_kps[i]))

			if all([x is not None for x in blobs]):
				for i in range(len(bbs)):
					outs[i].write(blobs[i])

			else:
				total_skipped += 1
		else:
			total_skipped += 1

		success, image = vidcap.read()
		count += 1

	print('done with {}'.format(video))

	with open('skipped.txt', 'a') as w:
		w.write('{}::{}::{}\n'.format(video, total_skipped, count))

	vidcap.release()
 	for out in outs:
 		out.release()

	if count == 0:
		for output_path in output_paths:
			os.remove(outputs_path)

	cv2.destroyAllWindows()

	return total_skipped, count

'''
This piece of code calls a function to generate - 
1. face blob
2. right and left hand blobs
'''
if __name__ == '__main__':
	videos_path, jsons_path, outputs_path = 'test_rgb_front_clips/raw_videos', 'test_2D_keypoints/openpose_output/json', 'outputs'
	videos = glob.glob('{}/*.mp4'.format(videos_path))

	global_count, global_total_skipped = 0, 0

	if not os.path.exists(outputs_path):
		os.makedirs(outputs_path)

	for i, video in enumerate(sorted(videos)):
		output_path = video.replace('.mp4', output_extension).replace(videos_path, outputs_path)
		json_base_path = video.replace('.mp4', '').replace(videos_path, jsons_path)

		print('starting {}/{}. {}'.format(i, len(videos), video))
		
		ts, c = generate_blobs(video, json_base_path, output_path)

		global_count += c
		global_total_skipped += gts

	print('{} skipped out of {} = {}%'.format(global_total_skipped, global_count, (global_total_skipped/global_count)*100))