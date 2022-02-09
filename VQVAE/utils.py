import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import distributed as dist
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
from torchvision import transforms
import custom_datasets as cds

def load_cifar():
	train = datasets.CIFAR10(root="data", train=True, download=True,
							 transform=transforms.Compose([
								 transforms.ToTensor(),
								 transforms.Normalize(
									 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							 ]))

	val = datasets.CIFAR10(root="data", train=False, download=True,
						   transform=transforms.Compose([
							   transforms.ToTensor(),
							   transforms.Normalize(
								   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
						   ]))
	return train, val

def load_hangestures(test=False):
	if test:
		return cds.HandGesturesDataset('test')
	return cds.HandGesturesDataset('train'), cds.HandGesturesDataset('val')

def load_blob2full(transform):
	return cds.Blob2Full('train', transform)

def load_block():
	data_folder_path = os.getcwd()
	data_file_path = data_folder_path + \
		'/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

	train = BlockDataset(data_file_path, train=True,
						 transform=transforms.Compose([
							 transforms.ToTensor(),
							 transforms.Normalize(
								 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
						 ]))

	val = BlockDataset(data_file_path, train=False,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(
							   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
					   ]))
	return train, val

def load_latent_block():
	data_folder_path = os.getcwd()
	data_file_path = data_folder_path + \
		'/data/latent_e_indices.npy'

	train = LatentBlockDataset(data_file_path, train=True,
						 transform=None)

	val = LatentBlockDataset(data_file_path, train=False,
					   transform=None)
	return train, val


def data_loaders(train_data, val_data, batch_size):

	train_loader = DataLoader(train_data,
							  batch_size=batch_size,
							  shuffle=True,
							  pin_memory=True)
	val_loader = DataLoader(val_data,
							batch_size=batch_size,
							shuffle=True,
							pin_memory=True)
	return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, distributed=False, test=False):
	if dataset == 'CIFAR10':
		training_data, validation_data = load_cifar()
		training_loader, validation_loader = data_loaders(
			training_data, validation_data, batch_size)
		x_train_var = np.var(training_data.data / 255.0)

	elif dataset == 'BLOCK':
		training_data, validation_data = load_block()
		training_loader, validation_loader = data_loaders(
			training_data, validation_data, batch_size)

		x_train_var = np.var(training_data.data / 255.0)
	elif dataset == 'LATENT_BLOCK':
		training_data, validation_data = load_latent_block()
		training_loader, validation_loader = data_loaders(
			training_data, validation_data, batch_size)

		x_train_var = np.var(training_data.data)

	elif dataset == 'HandGestures':
		if test:
			test_data = load_hangestures(True)

			sampler = dist.data_sampler(test_data, shuffle=True, distributed=distributed)

			return DataLoader(test_data,
								  batch_size=batch_size,
								  sampler=sampler, 
								  num_workers=2)

		training_data, validation_data = load_hangestures()

		training_sampler = dist.data_sampler(training_data, shuffle=True, distributed=distributed)

		training_loader = DataLoader(training_data,
							  batch_size=batch_size,
							  sampler=training_sampler, 
							  num_workers=2)

		val_sampler = dist.data_sampler(validation_data, shuffle=True, distributed=distributed)

		validation_loader = DataLoader(validation_data,
							  batch_size=batch_size,
							  sampler=val_sampler, 
							  num_workers=2)

		x_train_var = 0.5
	elif dataset == 'Blobs2Full':
		transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((256,256)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		training_data = load_blob2full(transform)

		sampler = dist.data_sampler(training_data, shuffle=True, distributed=distributed)

		training_loader = DataLoader(training_data,
							  batch_size=batch_size,
							  sampler=sampler, 
							  num_workers=2)

		validation_data, validation_loader = None, None

		x_train_var = 0.5
	else:
		raise ValueError(
			'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

	return training_data, validation_data, training_loader, validation_loader, x_train_var

def readable_timestamp():
	return time.ctime().replace('  ', ' ').replace(
		' ', '_').replace(':', '_').lower()


def save_model_and_results(model, optimizer, results, hyperparameters, timestamp):
	SAVE_MODEL_PATH = os.getcwd() + '/results'

	results_to_save = {
		'model': model.state_dict(),
		'results': results,
		'optimizer': optimizer.state_dict(),
		'hyperparameters': hyperparameters
	}
	torch.save(results_to_save,
			   SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
