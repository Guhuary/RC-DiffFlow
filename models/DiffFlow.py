
import torch
import torch.nn as nn
import torch.nn.functional as F

import attr
import numpy as np
from torch import Tensor
from torch.distributions.normal import Normal
import math
import torch.cuda.amp as amp

from models.layers import * 

@attr.s
class ExpTimer:
	num_steps = attr.ib(30)
	t_start = attr.ib(0.001)
	t_end = attr.ib(0.05)
	exp = attr.ib(0.9)

	def __attrs_post_init__(self):
		self.base = torch.linspace(
			self.t_start ** self.exp, self.t_end ** self.exp, self.num_steps
		)
		self.fix_x_slot = torch.linspace(
			self.t_start ** self.exp, self.t_end ** self.exp, self.num_steps + 1
		)
		self.intervals = self.base[1:] - self.base[:-1]

	def __call__(self):
		value = torch.pow(self.fix_x_slot, 1.0 / self.exp)
		return self.deal_flip(value)

	def deal_flip(self, value):
		if self.exp > 1.0:
			value = self.t_start + self.t_end - value
			value = torch.flip(value, (0,))
		return value

	def rand(self):
		ratio = torch.rand(self.num_steps - 1)
		mid_times = ratio * self.intervals + self.base[:-1]
		times = torch.cat([self.base[:1], mid_times, self.base[-1:]]).flatten()
		value = torch.pow(times, 1.0 / self.exp)
		return self.deal_flip(value)

	def index(self, time):
		if np.isclose(self.t_start, self.t_end):
			return torch.pow(self.base[-1], 1.0 / self.exp) * torch.ones_like(time)
		time = torch.clip(time, self.t_start, self.t_end)
		return time

def batch_noise_square(noise):
	return torch.sum(noise.flatten(start_dim=1) ** 2, dim=1)

# build upon https://arxiv.org/abs/2110.07579
class BaseModel(torch.nn.Module):
	def __init__(self, input_dim, drift_net, score_net):
		super().__init__()
		self.input_dim = input_dim
		self.drift = drift_net
		self.score = score_net
		self._distribution = Normal(torch.zeros(input_dim), torch.ones(input_dim))

	def forward_step(self, x, step_size, cond_f, cond_b, diff_f, diff_b):
		forward_noise = self._distribution.sample_n(x.shape[0]).to(x.device)   # gauss noise
		
		z = (
			self.cal_next_nodiffusion(x, step_size, cond_f)
			+ torch.sqrt(step_size) * diff_f * forward_noise
		)
		backward_noise = self.cal_backnoise(x, z, step_size, cond_b, diff_b)
		delta_s = -0.5 * (
			batch_noise_square(backward_noise) - batch_noise_square(forward_noise)
		)
		return z, delta_s

	def cal_backnoise(self, x, z, step_size, cond_b, diff_b):
		f_backward = self.drift(z, cond_b) - diff_b ** 2 * self.score(z, cond_b)
		return (x - z + f_backward * step_size) / (diff_b * torch.sqrt(step_size))

	def cal_forwardnoise(self, x, z, step_size, cond_f, diff_f):
		f_backward = self.drift(x, cond_f)
		return (z - x - f_backward * step_size) / (diff_f * torch.sqrt(step_size))

	def cal_next_nodiffusion(self, x, step_size, cond_f):
		return x + self.drift(x, cond_f) * step_size

	def cal_prev_nodiffusion(self, z, step_size, cond_b, diff_b):
		return (
			z
			- (self.drift(z, cond_b) - diff_b ** 2 * self.score(z, cond_b)) * step_size
		)

	def cal_prev_ode(self, z, step_size, cond_b, diff_b):
		return (
			z
			- (self.drift(z, cond_b) - diff_b ** 2 * self.score(z, cond_b) / 2) * step_size
		)

	def backward_step(self, z, step_size, cond_f, cond_b, diff_f, diff_b):
		backward_noise = self._distribution.sample_n(z.shape[0])
		x = (
			self.cal_prev_nodiffusion(z, step_size, cond_b, diff_b)
			+ torch.sqrt(step_size) * diff_b * backward_noise
		)
		forward_noise = self.cal_forwardnoise(x, z, step_size, cond_f, diff_f)
		delta_s = -0.5 * (
			batch_noise_square(forward_noise) - batch_noise_square(backward_noise)
		)
		return x, delta_s


	def sample(self, num_samples, timestamps, diffusion, condition):
		z = self._distribution.sample_n(num_samples)
		x, _ = self.backward(z, timestamps, diffusion, condition)
		return x

	def forward(self, x, timestamps, diffusion, condition):
		batch_size = x.shape[0]
		logabsdet = x.new_zeros(batch_size)
		delta_t = timestamps[1:] - timestamps[:-1]
		for i_th, cur_delta_t in enumerate(delta_t):
			x, new_det = self.forward_step(
				x,
				cur_delta_t,
				condition[i_th],
				condition[i_th + 1],
				diffusion[i_th],
				diffusion[i_th + 1],
			)
			logabsdet += new_det
		return x, logabsdet

	def backward(self, z, timestamps, diffusion, condition):
		delta_t = timestamps[1:] - timestamps[:-1]
		logabsdet = z.new_zeros(z.shape[0])
		for i_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
			z, new_det = self.backward_step(
				z,
				cur_delta_t,
				condition[-i_th - 2],
				condition[-i_th - 1],
				diffusion[-i_th - 2],
				diffusion[-i_th - 1],
			)
			logabsdet += new_det
		return z, logabsdet

	def backward_ODE(self, z, timestamps, diffusion, condition):
		delta_t = timestamps[1:] - timestamps[:-1]
		for i_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
			z = self.cal_prev_ode(
				z,
				cur_delta_t,
				condition[-i_th - 1],
				diffusion[-i_th - 1],
			)
		return z

	def forward_list(self, x):
		rtn = [x]
		for i_th, cur_delta_t in enumerate(self.delta_t):
			x, _ = self.forward_step(
				x,
				cur_delta_t,
				self.condition[i_th],
				self.condition[i_th + 1],
				self.diffusion[i_th],
				self.diffusion[i_th + 1],
			)
			rtn.append(x)
		return rtn

	def backward_list(self, z):
		rtn = [z]
		for i_th, cur_delta_t in enumerate(torch.flip(self.delta_t, (0,))):
			z, _ = self.backward_step(
				z,
				cur_delta_t,
				self.condition[-i_th - 2],
				self.condition[-i_th - 1],
				self.diffusion[-i_th - 2],
				self.diffusion[-i_th - 1],
			)
			rtn.append(z)
		return rtn

class DiffFlow(BaseModel):
	def __init__(
		self, input_dim, timestamp, diffusion, condition, drift_net, score_net
	):
		super().__init__(input_dim, drift_net, score_net)
		self.register_buffer("timestamps", timestamp)
		self.register_buffer("diffusion", diffusion)
		self.register_buffer("condition", condition)
		assert self.timestamps.shape == self.diffusion.shape
		self.register_buffer("delta_t", self.timestamps[1:] - self.timestamps[:-1])

	def forward(self, x):
		return super().forward(x, self.timestamps, self.diffusion, self.condition)

	def backward(self, z):
		return super().backward(z, self.timestamps, self.diffusion, self.condition)

	def backward_ODE(self, z):
		return super().backward_ODE(z, self.timestamps, self.diffusion, self.condition)

	def sample(self, n_samples):
		z = self._distribution.sample_n(n_samples)
		x, _ = self.backward(z)
		return x

	def sample_noise(self, n_samples):
		return self._distribution.sample_n(n_samples)

	def noise_log_prob(self, z):
		return self._distribution.log_prob(z)

class SdeF(torch.autograd.Function):
	@staticmethod
	@amp.custom_fwd
	def forward(ctx, x, model, timestamps, diffusion, condition, *model_parameter):
		shapes = [y0_.shape for y0_ in model_parameter]

		def _flatten(parameter):
			# flatten the gradient dict and parameter dict
			return torch.cat(
				[
					param.flatten() if param is not None else x.new_zeros(shape.numel())
					for param, shape in zip(parameter, shapes)
				]
			)

		def _unflatten(tensor, length):
			# return object like parameter groups
			tensor_list = []
			total = 0
			for shape in shapes:
				next_total = total + shape.numel()
				# It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
				tensor_list.append(
					tensor[..., total:next_total].view((*length, *shape))
				)
				total = next_total
			return tuple(tensor_list)

		history_x_state = x.new_zeros(len(timestamps) - 1, *x.shape)
		rtn_logabsdet = x.new_zeros(x.shape[0])
		delta_t = timestamps[1:] - timestamps[:-1]
		new_x = x
		with torch.no_grad():
			for i_th, cur_delta_t in enumerate(delta_t):
				history_x_state[i_th] = new_x
				new_x, new_logabsdet = model.forward_step(
					new_x,
					cur_delta_t,
					condition[i_th],
					condition[i_th + 1],
					diffusion[i_th],
					diffusion[i_th + 1],
				)
				rtn_logabsdet += new_logabsdet
		ctx.model = model
		ctx._flatten = _flatten
		ctx._unflatten = _unflatten
		ctx.nparam = np.sum([shape.numel() for shape in shapes])
		ctx.save_for_backward(
			history_x_state.clone(), new_x.clone(), timestamps, diffusion, condition
		)
		return new_x, rtn_logabsdet

	@staticmethod
	@amp.custom_bwd
	def backward(ctx, dL_dz, dL_logabsdet):
		history_x_state, z, timestamps, diffusion, condition = ctx.saved_tensors
		dL_dparameter = dL_dz.new_zeros((1, ctx.nparam))

		model, _flatten, _unflatten = ctx.model, ctx._flatten, ctx._unflatten
		model_parameter = tuple(model.parameters())
		delta_t = timestamps[1:] - timestamps[:-1]
		b_noise = {}
		with torch.no_grad():
			for bi_th, cur_delta_t in enumerate(torch.flip(delta_t, (0,))):
				bi_th += 1
				with torch.set_grad_enabled(True):
					x = history_x_state[-bi_th].requires_grad_(True)
					z = z.requires_grad_(True)
					noise_b = model.cal_backnoise(
						x, z, cur_delta_t, condition[-bi_th], diffusion[-bi_th]
					)

					cur_delta_s = -0.5 * (
						torch.sum(noise_b.flatten(start_dim=1) ** 2, dim=1)
					)
					dl_dprev_state, dl_dnext_state, *dl_model_b = torch.autograd.grad(
						(cur_delta_s),
						(x, z) + model_parameter,
						grad_outputs=(dL_logabsdet),
						allow_unused=True,
						retain_graph=True,
					)
					dl_dx, *dl_model_f = torch.autograd.grad(
						(
							model.cal_next_nodiffusion(
								x, cur_delta_t, condition[-bi_th - 1]
							)
						),
						(x,) + model_parameter,
						grad_outputs=(dl_dnext_state + dL_dz),
						allow_unused=True,
						retain_graph=True,
					)
					del x, z, dl_dnext_state
				b_noise[f"stat/{bi_th}"] = -1 * cur_delta_s.mean()
				z = history_x_state[-bi_th]
				dL_dz = dl_dx + dl_dprev_state
				dL_dparameter += _flatten(dl_model_b).unsqueeze(0) + _flatten(
					dl_model_f
				).unsqueeze(0)

			# trainer_stat(trainer, as_float(b_noise))

		return (dL_dz, None, None, None, None, *_unflatten(dL_dparameter, (1,)))

class QuickDiffFlow(DiffFlow):
	def forward(self, x):
		return SdeF.apply(
			x,
			self,
			self.timestamps,
			self.diffusion,
			self.condition,
			*tuple(self.parameters())
		)

	def forward_cond(self, x, timestamps, diffusion, condition):
		return SdeF.apply(
			x, self, timestamps, diffusion, condition, *tuple(self.parameters())
		)

def GET_DiffFlow(args):
	input_dim = args.tica_dim
	timesteps_embedding_dim = args.timesteps_embedding_dim
	num_layers = args.diffusion_net_layer
	channels = args.diffusion_net_channels

	timestamp = ExpTimer(args.diffusion_num_steps, args.diffusion_t_start, args.diffusion_t_end, args.diffusion_exp)().unsqueeze(-1)
	diffusion = ExpTimer(args.diffusion_num_steps, args.diffusion_g_start, args.diffusion_g_end, args.diffusion_exp)().unsqueeze(-1) # * args.diffusion_num_steps
	condition = ExpTimer(args.diffusion_num_steps, args.diffusion_t_start, args.diffusion_t_end, args.diffusion_exp)().unsqueeze(-1)
	
	drift_net = FourierMLP(input_dim, timesteps_embedding_dim, num_layers, channels)
	score_net = FourierMLP(input_dim, timesteps_embedding_dim, num_layers, channels)
	return QuickDiffFlow(input_dim, timestamp, diffusion, condition, drift_net, score_net)
