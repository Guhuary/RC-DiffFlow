import torch
from torch import nn
import torch.nn.functional as F
import math

from models.NF import RealNVP, RealNVP2
from models.NN_prior import GMM_potential
from models.DiffFlow import GET_DiffFlow

from utils.common import disable, reable
# --------------------
# Model
# --------------------

class Embedding_model(nn.Module):
	def __init__(self, args):
		super(Embedding_model, self).__init__()
		self.DiffFlow = GET_DiffFlow(args)
		self.final_nf = RealNVP(input_dim=args.tica_dim, hid_dim=args.hidden_dim_nf, n_layers=args.num_layers_nf)

	def forward(self, X):
		x0, logdet1 = self.DiffFlow(X)
		out2, logdet2 = self.final_nf(x0)
		return out2, logdet1 + logdet2

	def reverse(self, X):
		x, logdet = self.final_nf(X, mode='inverse')
		x = self.DiffFlow.backward_ODE(x)

class Prior_Model(nn.Module):
	def __init__(self, args):
		super(Prior_Model, self).__init__()
		self.beta = args.beta
		self.prior1 = GMM_potential(input_dim=args.latent_dim, beta=args.beta, component_num=args.component_num, gamma=args.gamma)

	def forward(self, x0):
		# potential function
		return self.prior1(x0)

	def force(self, x0):
		# return the force
		return self.prior1.force(x0)

	def log_probs_prior1(self, R0, R1, R2, sample_stepsize, lag, zero_potential):
		return self.prior1.diffusion_log_probs(R0, R1, R2, sample_stepsize, lag, zero_potential=zero_potential)

	def stationary_log_probs_prior1(self, R):
		return self.prior1.stationary_log_probs(R)

	def re_init(self, latent, args):
		if args.prior1_type == 'GMM':
			self.prior1.re_init(latent)

	def init_centers(self):
		self.prior1.init_centers()

class RC_DiffFlow(nn.Module):
	"""docstring for DiffFlow"""
	def __init__(self, args):
		super(RC_DiffFlow, self).__init__()
		self.embedding_dim = args.latent_dim
		self.embedding_model = Embedding_model(args)
		self.Prior = Prior_Model(args)

	def embedding(self, x):
		RE, logdets = self.embedding_model(x)
		R, E = RE[:, :self.embedding_dim], RE[:, self.embedding_dim:]
		return R, E, logdets

	def log_probs1(self, R0, E0, R1, E1, R2, E2, logdet0, logdet1, logdet2, sample_stepsize, lag, zero_potential):
		return self.Prior.log_probs_prior1(R0, R1, R2, sample_stepsize, lag, zero_potential) \
		   + (-0.5 * E2.pow(2)).sum(-1, keepdim=True) \
		   + logdet2

	def stationary_log_probs1(self, R, E, logdets):
		return self.Prior.stationary_log_probs_prior1(R) \
			   + (-0.5 * E.pow(2)).sum(-1, keepdim=True) \
			   + logdets
