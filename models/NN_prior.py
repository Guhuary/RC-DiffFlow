import torch
from torch import autograd, nn
from torch.nn import functional as F
import math
import numpy as np

from pyemma import coordinates
import scipy.spatial
import time

from sklearn.mixture import GaussianMixture

class GMM_potential(nn.Module):
	def __init__(self, input_dim, component_num = 50, beta=1.0, gamma=0.9):
		super(GMM_potential, self).__init__()
		self.input_dim = input_dim
		self.component_num = component_num

		self.register_buffer('centers', torch.randn(component_num, input_dim))
		self.para_stds = nn.Parameter(torch.randn(component_num, input_dim) * 10)
		self.para_weights = nn.Parameter(torch.FloatTensor([0.,] * component_num))
		self.beta = beta
		self.gamma = gamma

	def get_device(self):
		return next(self.parameters()).device

	def re_init(self, numpy_data):
		# cluster_obj = coordinates.cluster_kmeans(numpy_data, self.component_num)
		# self.state_dict()["centers"][:] = torch.from_numpy(cluster_obj.clustercenters).float().to(self.centers.device)
		# self.state_dict()["para_stds"][:] = scipy.spatial.distance.pdist(cluster_obj.clustercenters).max()
		device = self.get_device()
		gm = GaussianMixture(n_components=self.component_num, covariance_type='diag').fit(numpy_data)
		self.centers = torch.from_numpy(gm.means_).float().to(device)
		self.para_weights.data = torch.log(torch.from_numpy(gm.weights_).float()).to(device)
		self.para_stds.data = torch.from_numpy(gm.covariances_).float().to(device)

	def forward(self, x0):
		return -self.stationary_log_probs(x0) / self.beta
	
	def force(self, x0):
		device = self.get_device()
		stds = torch.abs(self.para_stds) + 1e-10

		x_minus_centers = x0[..., None, :] - self.centers.unsqueeze(0)  # batch_size * component_num * dim

		log_probs = torch.sum(-x_minus_centers ** 2 / (2 * stds ** 2) - torch.log(stds) - 0.5 * math.log(2 * math.pi), -1)
		log_probs += F.log_softmax(self.para_weights, 0).unsqueeze(0)

		probs_normalized = F.softmax(log_probs, -1)  # batch_size * component_num

		f = - x_minus_centers / stds ** 2

		return torch.sum(f * probs_normalized[..., None], -2) / self.beta
	
	def diffusion_log_probs(self, X0, X1, X2, sample_stepsize, lag, sub_step_num=1, sub_sample_num=20, zero_potential=False):
		
		#	X0: N X 2, X1: n X 2, beta: 1.0, sample_stepsize: 0.01, lag: 1
		
		device = self.get_device()
		tau = sample_stepsize * lag
		gamma = self.gamma
		if sub_step_num == 1 or zero_potential:
			dx_t = X2 - X1
			dx_t2 = X1 - X0

			D = (dx_t - dx_t2) / tau + gamma * dx_t2
			if zero_potential:
				D = D
			else:
				D = D - self.force(X1) * tau 

			Var = 2 * tau / self.beta * gamma
			
			if not np.isscalar(self.beta): # False, go else
				log_probs = (-0.5 * D.pow(2) / Var - 0.5 * (torch.log(Var) + math.log(2 * math.pi))).sum(-1, keepdim=True)
			else:
				log_probs = (-0.5 * D.pow(2) / Var - 0.5 * math.log(2 * math.pi * Var)).sum(-1, keepdim=True)
			return log_probs

	def stationary_log_probs(self, data):
		# data: N x dim
		data = data.unsqueeze(1)   # N x 1 x dim

		stds = (torch.abs(self.para_stds) + 1e-10).unsqueeze(0)
		log_gaussian = - torch.log(stds) - 0.5 * math.log(2 * math.pi) - \
						(data - self.centers.unsqueeze(0)) ** 2 / stds ** 2 / 2  # N x components x dim
		log_gaussian = log_gaussian.sum(-1)
		log_weight = F.log_softmax(self.para_weights, 0).unsqueeze(0)
		out = torch.logsumexp(log_gaussian + log_weight, 1, keepdim=True)
		return out