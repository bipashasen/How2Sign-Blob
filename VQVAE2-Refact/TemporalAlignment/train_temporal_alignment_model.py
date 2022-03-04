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

from models.temporal_alignment import TemporalAlignment

from dataset import TemporalAlignmentDataset

from ranges import translation_range, rotation_range

device = "cuda"

global_step = 0

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
	return torch.abs(x - xrec)

def apply_transformation(x, transformation):
	x = kornia.geometry.transform.rotate(x, transformation[:, 0]*rotation_range)

	translation = torch.cat([
		transformation[:, 1].unsqueeze(1), 
		transformation[:, 2].unsqueeze(1)], 1)
	x = kornia.geometry.transform.translate(x, translation*translation_range)

	# x = kornia.geometry.transform.scale(x, transformation[:, 3].view(-1, 1))

	# shear_factor = torch.cat([
	# 	transformation[:, 4].unsqueeze(1), 
	# 	transformation[:, 5].unsqueeze(1)], 1)
	# x = kornia.geometry.transform.shear(x, shear_factor)

	return x

def get_prediction(source, predicted_transformation):
	target = source.clone()[:, 3:]

	transformed_ = apply_transformation(
		source[:, :3], predicted_transformation)

	# mask = transformed_[..., 0] != 0
	# target[mask] = 0
	
	return transformed_ + target

def run_step(model, data, run_type='train'):
	def process_data(data):
		return [x.to(device).squeeze(0) for x in data]

	source, target, gt_transformation = process_data(data)

	predicted_transformation = model(source)

	predicted_target = get_prediction(source, predicted_transformation)

	if not run_type == 'train':
		return source[:, :3], target, predicted_target

	# gt_flow = compute_flow(target)

	# predicted_flow = compute_flow(predicted_target)

	# transformation_loss = (absolute_loss(
	# 	gt_transformation, predicted_transformation)**2).mean()

	transformation_loss = (absolute_loss(
		gt_transformation[:, 0], predicted_transformation[:, 0]*rotation_range)**2).mean()
	transformation_loss += (absolute_loss(
		gt_transformation[:, 1:], predicted_transformation[:, 1:]*translation_range)**2).mean()

	# most of the surrounding places are black.
	recon_loss = (absolute_loss(
		target[:, :, 100:200, 100:200], 
		predicted_target[:, :, 100:200, 100:200])**2).mean()

	# flow_loss = absolute_loss(gt_flow, predicted_flow).mean()

	losses = {
		'transformation_loss': transformation_loss,
		'recon_loss': recon_loss,
		# 'flow_loss': flow_loss
	}

	prediction_sample = predicted_transformation[0].detach().clone()
	prediction_sample[0] *= rotation_range
	prediction_sample[1:] *= translation_range

	return losses, prediction_sample, gt_transformation[0]

def validation(model, val_loader, device, epoch):
	for i, data in enumerate(tqdm(val_loader)):
		with torch.no_grad():
			source, target, predicted_target = run_step(model, data, run_type='val')

		saves = {
			'source': source,
			'target': target,
			'predicted_target': predicted_target
		}

		if i % (len(val_loader) // 10) == 0:
			def denormalize(x):
				# return (x.clamp(min=-1.0, max=1.0) + 1)/2
				return x

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

		losses, sample_pred, sample_gt = run_step(model, data)

		weights = [1.0, 1.0]#, 1.0]

		description = ""

		loss = 0
		for i, key in enumerate(losses):
			loss += weights[i] * losses[key]
			description += f"{key}: {losses[key].item():.3f} "

		loss.backward()
		optimizer.step()

		global_step += 1

		def tensor_to_list(sample):
			sample = list(sample.detach().cpu().numpy())

			return [round(x, 3) for x in sample]

		loader.set_description(
			(
				f"{description} "
				f"sample: pt: {tensor_to_list(sample_pred)} "
				f"gt: {tensor_to_list(sample_gt)} "
			)
		)

		if global_step % validate_at == 1:
			model.eval()

			validation(model, val_loader, device, epoch)

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

			os.makedirs(checkpoint_dir, exist_ok=True)

			if i % 50 == 0:
				torch.save(model.state_dict(), f"{checkpoint_dir}/temporal_{str(i + 1).zfill(3)}.pt")

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
	os.system(f'cp train_temporal_alignment_model.py {pydir}') # keep forgetting the state

	if args.test:
		args.max_frame_len *= 20

	print(args, flush=True)

	main(args)

