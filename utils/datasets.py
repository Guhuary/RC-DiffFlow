import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import time
import mdtraj, mdshare
import pyemma

class TrajDataset(Dataset):
	def __init__(self, traj, lag1):
		self.X0 = traj[:-lag1 * 2]
		self.X1 = traj[lag1:-lag1]
		self.X2 = traj[2 * lag1:]

	def __len__(self):
		return len(self.X0)

	def __getitem__(self, idx):
		return self.X0[idx], self.X1[idx], self.X2[idx]

class TrajDataset2(Dataset):
	def __init__(self, traj, lag1):
		X0 = traj[:, :-lag1 * 2]
		X1 = traj[:, lag1:-lag1]
		X2 = traj[:, 2 * lag1:]
		dim = X0.shape[-1]
		self.X0 = X0.reshape(-1, dim)
		self.X1 = X1.reshape(-1, dim)
		self.X2 = X2.reshape(-1, dim)

	def __len__(self):
		return len(self.X0)

	def __getitem__(self, idx):
		return self.X0[idx], self.X1[idx], self.X2[idx]

def label_data(dihedral):
	"""
	Label data based on dihedral trajectories. For AD. 
	
	Args:
	- dihedral (numpy.ndarray): Dihedral trajectories.
	
	Returns:
	- label (numpy.ndarray): Assigned labels.
	"""

	label = np.ones([dihedral.shape[0], 1]) * 4
	phi = dihedral[:, 0]
	xi = dihedral[:, 1]

	idx = ((phi >= -2) & (phi <= 0)) & (((xi >= 1) & (xi <= 4)) | ((xi >= -4) & (xi <= -2)))
	label[idx] = 0
	idx = (((phi >= -4) & (phi < -2)) & (((xi >= 1) & (xi <= 4)) | ((xi >= -4) & (xi <= -2)))) | (phi > 2)
	label[idx] = 1
	idx = ((phi >= -2) & (phi <= 0)) & ((xi > -2) & (xi <= 1))
	label[idx] = 2
	idx = ((phi >= -4) & (phi < -2)) & ((xi > -2) & (xi <= 1))
	label[idx] = 2
	idx = ((phi >= 0) & (phi <= 2)) & ((xi > -2) & (xi <= 2))
	label[idx] = 3

	return label

def construct_dataset_ad(args):
	"""
	Create data loaders from trajectories. 

	Returns:
	- train_loader (DataLoader): DataLoader for training data.
	- valid_loader (DataLoader): DataLoader for validation data.
	"""
	data_dir = args.data_dir
	tica_data = os.path.join(data_dir, args.traj)
	dihedral_path = os.path.join(data_dir, args.dihedral)
	lag1 = args.lag1

	if os.path.exists(tica_data) and os.path.exists(dihedral_path):
		data = np.load(tica_data)
		dihedral = np.load(dihedral)
	else:
		pdb = mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='data')
		files = mdshare.fetch('alanine-dipeptide-0-250ns-nowater.xtc', working_directory='data')
		feat = pyemma.coordinates.featurizer(pdb)
		feat.add_backbone_torsions(periodic=False)
		dihedral = pyemma.coordinates.load(files, features=feat)

		feat = pyemma.coordinates.featurizer(pdb)
		feat.add_selection(feat.select_Heavy())
		trajs = pyemma.coordinates.load(files, features=feat)
	
		from deeptime.decomposition import TICA
		estimator = TICA(dim=args.tica_dim, lagtime=args.lag_tica).fit(trajs)
		model_onedim = estimator.fetch_model()
		data = model_onedim.transform(trajs)
		np.save(dihedral_path, dihedral)
		np.save(tica_data, data)
		
	label = label_data(dihedral)
	data = torch.from_numpy(data)
	dataset = TrajDataset(data, lag1)

	train_portion = args.train_portion
	assert train_portion >= 0 and train_portion <= 1

	n_train = int(len(dataset) * train_portion)
	n_valid = len(dataset) - n_train

	train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid])

	train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False)

	return data, dihedral, label, train_loader, valid_loader

def construct_dataset_penta(args):
	lag1 = args.lag1

	pen_struc = mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory='data')
	pen_traj = mdshare.fetch('pentapeptide-*-500ns-impl-solv.xtc', working_directory='data')
	feat = pyemma.coordinates.featurizer(pen_struc)
	# feat.add_backbone_torsions(cossin=True, periodic=False)
	feat.add_selection(feat.select_Heavy())
	trajs = pyemma.coordinates.load(pen_traj, features=feat)

	from deeptime.decomposition import TICA
	estimator = TICA(dim=args.tica_dim, lagtime=args.lag_tica).fit(trajs)
	model_onedim = estimator.fetch_model()
	data = model_onedim.transform(trajs)

	data = torch.from_numpy(data).float()
	# target_class, _ = KMeans(data, K=args.K, Niter=30)
	# target_class = target_class

	dataset = TrajDataset2(data, lag1)

	train_portion = args.train_portion
	assert train_portion >= 0 and train_portion <= 1

	n_train = int(len(dataset) * train_portion)
	n_valid = len(dataset) - n_train

	train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [n_train, n_valid])

	train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False)

	return data.view(-1, args.tica_dim), train_loader, valid_loader

