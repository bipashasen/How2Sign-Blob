'''
VQVAE-Videos in an non-autoregressive manner. 
'''
import argparse
import sys
import os

import random
import cv2
import numpy as np

import kornia

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils, io

from tqdm import tqdm

from models.jitter_removal_bhalu import PostnetVQ1
from models.stn import Net
from models.temporal_alignment_gru import TemporalAlignment

from dataset import TemporalAlignmentDataset

from ranges import translation_range, rotation_range

torch.autograd.set_detect_anomaly(True)

device = "cuda"

global_step = 0

run_combined_network = True

checkpoint_dir = 'checkpoint_{}'

sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/{}'

def save_frames_as_video(frames, video_path, fps=30):
	os.makedirs(sample_folder, exist_ok=True)
	height, width, layers = frames[0].shape

	video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
	for frame in frames: 
		video.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)) 
	  
	cv2.destroyAllWindows() 
	video.release()

def absolute_loss(x, xrec):
	loss = torch.abs(x - xrec)*5

	total_values = loss.view(-1).shape[0]
	non_zero_values = total_values - (loss == 0.0).sum()

	return loss.sum()/non_zero_values

def run_step(model, data, run_type='train'):
	def process_data(data):
		return [x.to(device) for x in data]

	source_hulls, target_hulls, background, source_images = process_data(data)

	prediction = model(source_hulls, background)

	if not run_type == 'train':
		return source_hulls[0], target_hulls[0], prediction[0], background[0], source_images[0]

	# temporal_diff = abs(
	# 	torch.diff(prediction[:, :, :, 100:200, 100:200].squeeze(0), 
	# 		dim=0)).mean()
	temporal_diff = torch.tensor([0.]).to(device)

	target = target_hulls if not run_combined_network else source_images

	reconstruction_diff = absolute_loss(source_images, prediction)

	return temporal_diff, reconstruction_diff

def validation(model, val_loader, device, epoch):
	for i, data in enumerate(tqdm(val_loader)):
		with torch.no_grad():
			source_hulls, target_hulls, prediction, background, source_images = run_step(model, data, run_type='val')

		saves = {
			'source': source_hulls,
			'target': target_hulls,
			'background': background,
			'predicted_target': prediction,
			'source_images': source_images
		}

		if i % (len(val_loader) // 10) == 0:
			def denormalize(x):
				return (x.clamp(min=-1.0, max=1.0) + 1)/2

			for name in saves:
				saveas = f"{sample_folder}/{epoch + 1}_{global_step}_{i}_{name}.mp4"
				frames = saves[name].detach().cpu()
				frames = [denormalize(x).permute(1, 2, 0).numpy() for x in frames]

				save_frames_as_video(frames, saveas, fps=25)

def train(model, loader, val_loader, optimizer, scheduler, epoch, validate_at):
	global global_step

	loader = tqdm(loader, file=sys.stdout)

	for i, data in enumerate(loader):
		model.zero_grad()

		temporal_diff, reconstruction_diff = run_step(model, data)

		loss = 0.3*temporal_diff + reconstruction_diff

		loss.backward()
		optimizer.step()

		global_step += 1

		loader.set_description(
			(
				f"epoch: {epoch+1} "
				f"step: {i+1} "
				f"global_step: {global_step} "
				f"rec_loss: {reconstruction_diff.item():.3f} "
				f"temp_loss: {temporal_diff.item():.3f}"
			)
		)

		if global_step % validate_at == 1:
			model.eval()

			validation(model, val_loader, device, epoch)

			os.makedirs(checkpoint_dir, exist_ok=True)

			torch.save(model.state_dict(), f"{checkpoint_dir}/temporal_{epoch+1}_{str(i + 1).zfill(3)}.pt")

			model.train()

def get_loaders_and_models(args):
	train_dataset = TemporalAlignmentDataset('train', args.max_frame_len)
	val_dataset = TemporalAlignmentDataset('val', args.max_frame_len)

	train_loader = DataLoader(
		train_dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		num_workers=2)
	val_loader = DataLoader(
		val_dataset, 
		batch_size=args.batch_size, 
		num_workers=2)

	# model = PostnetVQ1().to(device)
	# model = Net().to(device)
	model = TemporalAlignment().to(device)

	return train_loader, val_loader, model

def main(args):
	start = 0
	
	global global_step
	
	loader, val_loader, model = get_loaders_and_models(args)

	model = nn.parallel.DataParallel(model)

	if args.ckpt:
		state_dict = torch.load(args.ckpt)
		state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
		models.module.load_state_dict(state_dict)
		start = int(ckpts[0].split('_')[-1].split('.')[0]) + 1
		global_step = (start-1)*len(loader)

	if args.test:
		validation(model, val_loader, device, 0)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr)

		scheduler = None
		
		if args.sched == "cycle":
			scheduler = CycleScheduler(
				optimizer,
				args.lr,
				n_iter=len(loader) * args.epoch,
				momentum=None,
				warmup_proportion=0.05,
			)

		for i in range(args.epoch)[start:]:
			train(model, loader, val_loader, optimizer, scheduler, i, args.validate_at)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--max_frame_len", type=int, default=512)
	parser.add_argument("--epoch", type=int, default=20000)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--sched", type=str)
	parser.add_argument("--validate_at", type=int, default=128)
	parser.add_argument("--checkpoint_suffix", type=str, default='')
	parser.add_argument("--ckpt", required=False)
	parser.add_argument("--test", action='store_true', required=False)

	args = parser.parse_args()

	args.n_gpu = torch.cuda.device_count()

	args.batch_size = 1 # video-level

	args.max_frame_len = args.max_frame_len * args.n_gpu

	checkpoint_dir = checkpoint_dir.format(args.checkpoint_suffix)

	sample_folder = sample_folder.format(args.checkpoint_suffix)

	pydir = os.path.join(checkpoint_dir, 'py')

	os.makedirs(pydir, exist_ok=True)
	os.system(f'cp train_jitter_removal.py {pydir}') # keep forgetting the state

	if args.test:
		args.max_frame_len *= 20

	print(args, flush=True)

	main(args)

