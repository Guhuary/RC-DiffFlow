import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import mdtraj
from pyemma import coordinates
from itertools import chain
import time


def get_optimizer_and_scheduler(model, lr, w_decay, lr_scheduler_step_size, lr_scheduler_gamma):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size,
													gamma=lr_scheduler_gamma)
	return optimizer, scheduler

def rand_projections(z_dim, num_samples=50):
	# This function generates `num_samples` random samples from the latent space's unit sphere
	projections = [w / np.sqrt((w**2).sum())
				   for w in np.random.normal(size=(num_samples, z_dim))]
	projections = torch.from_numpy(np.array(projections)).float()
	return projections

@torch.no_grad()
def embedding_traj(model, traj, device, latent_dim, batch_size = 1024, inverse = False):
	length = traj.shape[0]
	traj_RE = []
	t = 0
	while True:
		actual_size = min(batch_size, length - t)
		tmp_traj = traj[t:t + actual_size].to(device)
		a = model(tmp_traj)[0]
		traj_RE.append(a)
		t += actual_size
		if t >= length:
			break
	out = torch.cat(traj_RE, dim=0).cpu().numpy()
	return out

@torch.no_grad()
def calc_potential_traj(model, traj, device, batch_size = 1024):
	length = traj.shape[0]
	traj_V = []
	forces = []
	t = 0
	while True:
		actual_size = min(batch_size, length - t)
		tmp_traj = torch.FloatTensor(traj[t:t+actual_size]).to(device)
		a = model(tmp_traj)
		forces.append(model.force(tmp_traj))
		traj_V.append(a)
		t += actual_size
		if t >= length:
			break
	out = torch.cat(traj_V, dim=0).cpu().numpy()
	force = torch.cat(forces, dim=0).cpu().numpy()
	return out, force

# Only used for unweighted samples
def sliced_wasserstein_distance(encoded_samples, prior_samples, projection_num=50, p=2, device='cpu'):
	# This function calculates the sliced-Wasserstein distance between the encoded samples and prior samples

	# derive latent space dimension size from random samples drawn from latent prior distribution
	z_dim = prior_samples.size(-1)

	# generate random projections in latent space
	projections = rand_projections(z_dim, projection_num).to(device)

	# calculate projections through the encoded samples
	encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
	# calculate projections through the prior distribution random samples
	prior_projections = (prior_samples.matmul(projections.transpose(0, 1)))
	# calculate the sliced wasserstein distance by
	# sorting the samples per random projection and
	# calculating the difference between the
	# encoded samples and drawn random samples
	# per random projection
	wasserstein_distance = (torch.sort(encoded_projections, dim=0)[0] -
							torch.sort(prior_projections, dim=0)[0])
	# distance between latent space prior and encoded distributions
	# power of 2 by default for Wasserstein-2
	wasserstein_distance = torch.pow(wasserstein_distance, p)
	# approximate mean wasserstein_distance for each projection
	return wasserstein_distance.mean()

class AverageMeter():
	def __init__(self, types, unpooled_metrics=False, intervals=1):
		self.types = types
		self.intervals = intervals
		self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
		self.acc = {t: torch.zeros(intervals) for t in types}
		self.unpooled_metrics = unpooled_metrics

	def add(self, vals, interval_idx=None):
		if self.intervals == 1:
			self.count += 1 if vals[0].dim() == 0 else len(vals[0])
			for type_idx, v in enumerate(vals):
				self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
		else:
			for type_idx, v in enumerate(vals):
				self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
				if not torch.allclose(v, torch.tensor(0.0)):
					self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

	def summary(self):
		if self.intervals == 1:
			out = {k: v.item() / self.count for k, v in self.acc.items()}
			return out
		else:
			out = {}
			for i in range(self.intervals):
				for type_idx, k in enumerate(self.types):
					out['int' + str(i) + '_' + k] = (
							list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
			return out

def disable(module):
	for para in module.parameters():
		para.requires_grad = False

def reable(module):
	for para in module.parameters():
		para.requires_grad = True
		
'''
def construct_dataset_yjy(args):
	"""
	Create data loaders from trajectories. 

	Returns:
	- train_loader (DataLoader): DataLoader for training data.
	- valid_loader (DataLoader): DataLoader for validation data.
	"""
	data_dir = args.data_dir
	tica_data = os.path.join(data_dir, args.tica_data)
	lag = args.lag

	if os.path.exists(tica_data):
		data = np.load(tica_data)
	else:
		structure = os.path.join(data_dir, args.source_struct)
		traj = os.path.join(data_dir, args.source_traj)
		assert os.path.exists(structure) and os.path.exists(traj)
		temp_traj = mdtraj.load(traj, top=structure)
		temp_traj_processed = mdtraj.Trajectory.superpose(temp_traj, temp_traj[0]) # all atom used
		tmp_object = coordinates.tica(temp_traj_processed.xyz.reshape(-1,30)[:300000,], dim=args.tica_dim, lag=lag)
		data = tmp_object.get_output()[0]
		np.save(tica_data, data)

	data = torch.from_numpy(data)
	data_with_lag = torch.cat([data[:-lag], data[lag:]], dim=1)

	train_portion = args.train_portion
	assert train_portion > 0 and train_portion < 1

	n_train = int(len(data_with_lag) * train_portion)

	train_loader = DataLoader(data_with_lag[:n_train], batch_size=args.train_batch_size, shuffle=True)
	valid_loader = DataLoader(data_with_lag[n_train:], batch_size=args.val_batch_size, shuffle=False)

	return data[:-lag], train_loader, valid_loader
'''

