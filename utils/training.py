import copy
import numpy as np
from tqdm import tqdm
import math

import torch
from utils.common import *

def train_embedding(DiffFlow, train_loader, args, device, logger, zero_potential=False):
	'''
		training embedding model only
	'''
	# fix the GMM and set optimizer
	disable(DiffFlow.Prior)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, DiffFlow.parameters()), lr=args.lr) 
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma)

	for epoch in range(args.n_epochs_embedding):
		DiffFlow.train()
		train_loss = 0
		total_num = 0
		for batch_idx, (X0, X1, X2) in enumerate(train_loader):
			X0 = X0.to(device)
			X1 = X1.to(device)
			X2 = X2.to(device)

			R0, E0, logdet0 = DiffFlow.embedding(X0)
			R1, E1, logdet1 = DiffFlow.embedding(X1)
			R2, E2, logdet2 = DiffFlow.embedding(X2)

			loss = - DiffFlow.log_probs1(R0, E0, R1, E1, R2, E2, logdet0, logdet1, logdet2, sample_stepsize=args.sample_stepsize, lag=args.lag1, \
					zero_potential=zero_potential).mean()
			# loss += - args.alpha * DiffFlow.stationary_log_probs1(R1, E1, logdet1).mean()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item() * len(X0)
			# ce_loss += CE.item() * len(X0)
			total_num += len(X0)
			
			if batch_idx > 1 and (batch_idx % 200 == 0 or total_num == len(train_loader.dataset)):
				logger.info('Train Embedding Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, total_num, len(train_loader.dataset),
					100. * total_num / len(train_loader.dataset),
					train_loss / total_num))
				# logger.info('Train NF Epoch: {} [{}/{} ({:.0f}%)]\tCE Loss: {:.6f}'.format(
				# 	epoch, total_num, len(train_loader.dataset),
				# 	100. * total_num / len(train_loader.dataset),
				# 	ce_loss / total_num))
		scheduler.step()
	reable(DiffFlow.Prior)

def train_diffusion(DiffFlow, train_loader, args, device, logger):
	# fix the NF and set optimizer
	disable(DiffFlow.embedding_model)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, DiffFlow.parameters()), lr=args.lr * 10)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma)
	
	for epoch in range(args.n_epochs_latent):
		DiffFlow.train()
		train_loss = 0
		total_num = 0
		for batch_idx, (X0, X1, X2) in enumerate(train_loader):
			X0 = X0.to(device)
			X1 = X1.to(device)
			X2 = X2.to(device)

			R0, E0, logdet0 = DiffFlow.embedding(X0)
			R1, E1, logdet1 = DiffFlow.embedding(X1)
			R2, E2, logdet2 = DiffFlow.embedding(X2)

			BD_loss = - DiffFlow.log_probs1(R0, E0, R1, E1, R2, E2, logdet0, logdet1, logdet2, sample_stepsize=args.sample_stepsize, lag=args.lag1, \
					zero_potential=False)
			distribution_loss = - args.alpha * (DiffFlow.stationary_log_probs1(R0, E0, logdet0) + DiffFlow.stationary_log_probs1(R1, E1, logdet1))

			loss = BD_loss.mean() + distribution_loss.mean()
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item() * len(X0)
			total_num += len(X0)
			
			if batch_idx > 1 and (batch_idx % 200 == 0 or total_num == len(train_loader.dataset)):
				logger.info('Train Latent Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, total_num, len(train_loader.dataset),
					100. * total_num / len(train_loader.dataset),
					train_loss / total_num))

		scheduler.step()
	reable(DiffFlow.embedding_model)

def train_joint_12(DiffFlow, train_loader, args, device, logger):
	# fix the NF and set optimizer
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, DiffFlow.parameters()), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma)
	
	for epoch in range(args.n_epochs_joint):
		DiffFlow.train()
		train_loss = 0
		total_num = 0
		for batch_idx, (X0, X1, X2) in enumerate(train_loader):
			X0 = X0.to(device)
			X1 = X1.to(device)
			X2 = X2.to(device)

			R0, E0, logdet0 = DiffFlow.embedding(X0)
			R1, E1, logdet1 = DiffFlow.embedding(X1)
			R2, E2, logdet2 = DiffFlow.embedding(X2)

			BD_loss = - DiffFlow.log_probs1(R0, E0, R1, E1, R2, E2, logdet0, logdet1, logdet2, sample_stepsize=args.sample_stepsize, lag=args.lag1, \
					zero_potential=False)
			distribution_loss = - args.alpha * (DiffFlow.stationary_log_probs1(R0, E0, logdet0) + DiffFlow.stationary_log_probs1(R1, E1, logdet1))


			loss = BD_loss.mean() + distribution_loss.mean()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item() * len(X0)
			total_num += len(X0)
			
			if batch_idx > 1 and (batch_idx % 200 == 0 or total_num == len(train_loader.dataset)):
				logger.info('Train Joint Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, total_num, len(train_loader.dataset),
					100. * total_num / len(train_loader.dataset),
					train_loss / total_num))

		scheduler.step()
