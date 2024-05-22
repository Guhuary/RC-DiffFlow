import os
import numpy as np
import math
import copy 

import matplotlib.pyplot as plt
import pyemma
from pyemma.plots import plot_free_energy
from pyemma import coordinates
from pyemma import msm
import torch
from torch import nn, optim, autograd
from torch.nn import functional as F

import mdshare
import time
import logging
from colorlog import ColoredFormatter

import yaml

from cfg.parsing import parse_train_args, save_yaml_file
from utils.common import *
from utils.training import *
from utils.visualize import *

from torch.utils.tensorboard import SummaryWriter

from models.RC_DiffFlow import RC_DiffFlow

import warnings
warnings.filterwarnings("ignore")

def main_function():
	def visual(args, experiment_idx=0):
		# Visualization
		rc_latent = embedding_traj(model.embedding_model, traj, device, args.latent_dim)
		visualize_embedding(rc_latent, label, args.visual_results, experiment_idx, args)

	args = parse_train_args()
	print(args)

	# record parameters
	run_dir = os.path.join(args.log_dir, args.run_name)
	if '/' in run_dir and os.path.dirname(run_dir) and not os.path.exists(os.path.dirname(run_dir)):
		os.makedirs(os.path.dirname(run_dir))
		yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
		save_yaml_file(yaml_file_name, args.__dict__)

	import datetime
	now = datetime.datetime.now().strftime("%Y_%m_%d %H:%M")

	# define logger
	logger = logging.getLogger()
	logger.setLevel('INFO')
	BASIC_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
	formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
	chlr = logging.StreamHandler()
	chlr.setFormatter(formatter)
	fhlr = logging.FileHandler('log.log', mode='w')
	fhlr.setFormatter(formatter)
	logger.addHandler(chlr)
	logger.addHandler(fhlr)
	logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

	# Loader
	if args.system == 'ad':
		from utils.datasets import construct_dataset_ad as construct_dataset
		traj, dihedral, label, train_loader, val_loader = construct_dataset(args)
		visualize_dihedral(dihedral, label, args.visual_results, 0)
	else:
		from utils.datasets import construct_dataset_penta as construct_dataset
		traj, train_loader, val_loader = construct_dataset(args)

	model = RC_DiffFlow(args).to(device)
	
	train_embedding(model, train_loader, args, device, logger, zero_potential=True)
	rc_latent = embedding_traj(model.embedding_model, traj, device, args.latent_dim)

	model.Prior.re_init(rc_latent[:, :args.latent_dim], args)
	train_diffusion(model, train_loader, args, device, logger)

	train_joint_12(model, train_loader, args, device, logger)
	rc_latent = embedding_traj(model.embedding_model, traj, device, args.latent_dim)
	torch.save(model.state_dict(), f'ckpts/{args.embeding_type}.ckpt')
	
if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main_function()
