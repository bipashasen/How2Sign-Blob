import cv2
import numpy as np
import os
import json

import glob

frame_width, frame_height = 100, 100

dim = (frame_width,frame_height)
fps = 25

output_extension = '.avi'

def extract_hands(hands, bb, out, image):
	buff = 10

	confidence_threshold = .1

	confidence = np.mean(hands[:, 2])
	
	if confidence >= confidence_threshold:
		bb = [ int(x) for x in bb]
	 
		w, h = bb[2]-bb[0], bb[3]-bb[1]
		size = max(w, h) // 2

		midw, midh = int(np.mean(hands[:, 0])), int(np.mean(hands[:, 1]))

		startw, starth, endw, endh = midw-size-buff, midh-size-buff, midw+size+buff, midh+size+buff  

		cropped = image[starth:endh, startw:endw]

		# print(cropped.shape)

		if cropped.shape[0] > 0 and cropped.shape[1] > 0:
			frame = cv2.resize(cropped, dim)

			# cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file   
			out.write(frame)

def generate_hand_videos(video, json_base, output_path):
	output_paths = [output_path.replace(output_extension, '_{}_{}'.format(x, output_extension)) for x in ['right', 'left']]

	if os.path.exists(output_paths[0]) and os.path.exists(output_paths[1]):
		print('skipping {}'.format(video))
		return

	count = 0

	vidcap = cv2.VideoCapture(video)
	success, image = vidcap.read()

	out_right = cv2.VideoWriter(output_paths[0], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim)
	out_left = cv2.VideoWriter(output_paths[1], cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim)

	vname = video.rsplit('/', 1)[-1].split('.')[0]

	while success:
		kp_path = os.path.join(json_base, '{}_{}_keypoints.json'.format(vname, str(count).zfill(12)))

		if os.path.exists(kp_path):
			with open(kp_path) as r:
				kp = json.load(r)['people'][0]

			# x, y, score
			hands_right, hands_left = np.array(kp['hand_right_keypoints_2d']).reshape(-1, 3), np.array(kp['hand_left_keypoints_2d']).reshape(-1, 3)

			bb_right = np.min(hands_right[:, 0]), np.min(hands_right[:, 1]), np.max(hands_right[:, 0]), np.max(hands_right[:, 1]) # x0, y0, x1, y1
			bb_left = np.min(hands_left[:, 0]), np.min(hands_left[:, 1]), np.max(hands_left[:, 0]), np.max(hands_left[:, 1])

			extract_hands(hands_right, bb_right, out_right, image)
			extract_hands(hands_left, bb_left, out_left, image)

		success, image = vidcap.read()
		count += 1

	print('done with {}'.format(video))

	vidcap.release()
	out_right.release()
	out_left.release()

	if count == 0:
		for output_path in output_paths:
			os.remove(outputs_path)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	videos_path, jsons_path, outputs_path = 'test_rgb_front_clips/raw_videos', 'test_2D_keypoints/openpose_output/json', 'outputs'
	videos = glob.glob('{}/*.mp4'.format(videos_path))

	if not os.path.exists(outputs_path):
		os.makedirs(outputs_path)

	for i, video in enumerate(sorted(videos)):
		output_path = video.replace('.mp4', output_extension).replace(videos_path, outputs_path)
		json_base_path = video.replace('.mp4', '').replace(videos_path, jsons_path)

		print('starting {}/{}. {}'.format(i, len(videos), video))
		
		generate_hand_videos(video, json_base_path, output_path)
