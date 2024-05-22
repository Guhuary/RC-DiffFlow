import torch
from torch import nn, optim, distributions
from torch.nn import functional as F
import math

import bgflow as bg

class CouplingLayer(nn.Module):
	def __init__(self, input_dim, hid_dim, mask, cond_dim=None, s_tanh_activation=True, smooth_activation=False):
		super().__init__()
		
		if cond_dim is not None:
			total_input_dim = input_dim + cond_dim
		else:
			total_input_dim = input_dim

		self.s_fc1 = nn.Linear(total_input_dim, hid_dim)
		self.s_fc2 = nn.Linear(hid_dim, hid_dim)
		self.s_fc3 = nn.Linear(hid_dim, input_dim)
		self.t_fc1 = nn.Linear(total_input_dim, hid_dim)
		self.t_fc2 = nn.Linear(hid_dim, hid_dim)
		self.t_fc3 = nn.Linear(hid_dim, input_dim)
		# self.mask = nn.Parameter(mask, requires_grad=False)
		self.register_buffer('mask', mask)
		self.s_tanh_activation = s_tanh_activation
		self.smooth_activation = smooth_activation

	def forward(self, x, cond_x=None, mode='direct'):
		x_m = x * self.mask
		if cond_x is not None:
			x_m = torch.cat([x_m, cond_x], -1)
		if self.smooth_activation:
			if self.s_tanh_activation:
				s_out = torch.tanh(self.s_fc3(F.elu(self.s_fc2(F.elu(self.s_fc1(x_m)))))) * (1-self.mask)
			else:
				s_out = self.s_fc3(F.elu(self.s_fc2(F.elu(self.s_fc1(x_m))))) * (1-self.mask)
			t_out = self.t_fc3(F.elu(self.t_fc2(F.elu(self.t_fc1(x_m))))) * (1-self.mask)
		else:
			if self.s_tanh_activation:
				s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m)))))) * (1-self.mask)
			else:
				s_out = self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))) * (1-self.mask)
			t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m))))) * (1-self.mask)
		if mode == 'direct':
			y = x * torch.exp(s_out) + t_out
			log_det_jacobian = s_out.sum(-1, keepdim=True)
		else:
			y = (x - t_out) * torch.exp(-s_out)
			log_det_jacobian = -s_out.sum(-1, keepdim=True)
		return y, log_det_jacobian

class RealNVP(nn.Module):
	def __init__(self, input_dim, hid_dim = 256, n_layers = 6, cond_dim = None, s_tanh_activation = True, smooth_activation=False):
		super().__init__()
		assert n_layers >= 2, 'num of coupling layers should be greater or equal to 2'
		
		self.input_dim = input_dim
		mask = (torch.arange(0, input_dim) // (input_dim / 2)).float()   # % 2
		self.modules = []
		self.modules.append(CouplingLayer(input_dim, hid_dim, mask, cond_dim, s_tanh_activation, smooth_activation))
		for _ in range(n_layers - 2):
			mask = 1 - mask
			self.modules.append(CouplingLayer(input_dim, hid_dim, mask, cond_dim, s_tanh_activation, smooth_activation))
		self.modules.append(CouplingLayer(input_dim, hid_dim, 1 - mask, cond_dim, s_tanh_activation, smooth_activation))
		self.module_list = nn.ModuleList(self.modules)

	def forward(self, x, cond_x=None, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			x: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		logdets = torch.zeros(x.size(0), 1, device=x.device)

		assert mode in ['direct', 'inverse']
		if mode == 'direct':
			for module in self.module_list:
				x, logdet = module(x, cond_x, mode)
				logdets += logdet
		else:
			for module in reversed(self.module_list):
				x, logdet = module(x, cond_x, mode)
				logdets += logdet
		return x, logdets

	def log_probs(self, x, cond_x = None):
		u, log_jacob = self(x, cond_x)
		log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
			-1, keepdim=True)
		return (log_probs + log_jacob).sum(-1, keepdim=True)

	def sample(self, num_samples, noise=None, cond_x=None):
		if noise is None:
			noise = torch.Tensor(num_samples, self.input_dim).normal_()
		device = next(self.parameters()).device
		noise = noise.to(device)
		if cond_x is not None:
			cond_x = cond_x.to(device)
		samples = self.forward(noise, cond_x, mode='inverse')[0]
		return samples

def RealNVP2(input_dim, hid_dim=256, n_layers=6):
    layers = []
    input_dim_indices_1 = list(range(0,input_dim,2))
    input_dim_indices_2 = list(range(1,input_dim,2))
    input_dim_1 = len(input_dim_indices_1)
    input_dim_2 = len(input_dim_indices_2)
    
    layers = []
    layers.append(bg.SplitFlow(input_dim_indices_1, input_dim_indices_2))
    for _ in range(n_layers):
        layers.append(bg.SwapFlow())
        input_dim_1, input_dim_2 = input_dim_2, input_dim_1
        layers.append(bg.CouplingFlow(
            bg.AffineTransformer(
                shift_transformation=bg.DenseNet([input_dim_1, hid_dim, hid_dim, hid_dim, input_dim_2], activation=torch.nn.ReLU()), 
                scale_transformation=bg.DenseNet([input_dim_1, hid_dim, hid_dim, hid_dim, input_dim_2], activation=torch.nn.ReLU())
            )
        ))
    layers.append(bg.InverseFlow(bg.SplitFlow(input_dim_indices_1, input_dim_indices_2)))
    return bg.SequentialFlow(layers)