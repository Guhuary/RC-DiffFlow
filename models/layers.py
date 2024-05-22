import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
	""" from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
	assert len(timesteps.shape) == 1
	half_dim = embedding_dim // 2
	emb = math.log(max_positions) / (half_dim - 1)
	emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
	emb = timesteps.float()[:, None] * emb[None, :]
	emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
	if embedding_dim % 2 == 1:  # zero pad
		emb = F.pad(emb, (0, 1), mode='constant')
	assert emb.shape == (timesteps.shape[0], embedding_dim)
	return emb

def get_timestep_embedding(embedding_dim, embedding_scale=1e4):
	emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
	return emb_func


class Mish(nn.Module):
	def forward(self, x):
		return x * torch.tanh(F.softplus(x))


class Downsample(nn.Module):
	def __init__(self, dim_in, dim_out):
		super().__init__()
		self.down = nn.Linear(dim_in, dim_out)

	def forward(self, x):
		return self.down(x)


class Upsample(nn.Module):
	def __init__(self, dim_in, dim_out):
		super().__init__()
		self.up = nn.Linear(dim_in, dim_out)

	def forward(self, x):
		return self.up(x)

class Rezero(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn
		self.g = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		return self.fn(x) * self.g

class Block(nn.Module):
	def __init__(self, dim, dim_out):
		super().__init__()
		self.block = nn.Sequential(
			nn.Linear(dim, dim_out), Mish()
		)

	def forward(self, x):
		return self.block(x)

class ResnetBlock(nn.Module):
	def __init__(self, dim, dim_out, time_emb_dim):
		super().__init__()
		self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))

		self.block1 = Block(dim, dim_out)
		self.block2 = Block(dim_out, dim_out)
		self.res_conv = nn.Linear(dim, dim_out, 1) if dim != dim_out else nn.Identity()

	def forward(self, x, time_emb):
		h = self.block1(x)
		h += self.mlp(time_emb)
		h = self.block2(h)
		return h + self.res_conv(x)

class Unet_score(nn.Module):
	def __init__(self, input_dim, timesteps_embedding_dim=32, num_layers=3):
		super().__init__()

		self.time_pos_emb = get_timestep_embedding(timesteps_embedding_dim)
		self.mlp = nn.Sequential(
			nn.Linear(timesteps_embedding_dim, input_dim * 4), Mish(), nn.Linear(input_dim * 4, input_dim)
		)

		self.input_dim = input_dim
		self.dims = []
		temp_dim = input_dim
		for i in range(num_layers):
			self.dims.append([temp_dim, temp_dim // 2])
			temp_dim //= 2

		self.downs = nn.ModuleList([])
		self.ups = nn.ModuleList([])

		for ind, (dim_in, dim_out) in enumerate(self.dims):
			is_last = (ind == (num_layers - 1))
			self.downs.append(
				nn.ModuleList(
					[
						ResnetBlock(dim_in, dim_out, time_emb_dim=input_dim),
						ResnetBlock(dim_out, dim_out, time_emb_dim=input_dim),
						Downsample(dim_out, dim_out) if not is_last else nn.Identity(),
					]
				)
			)

		mid_dim = self.dims[-1][-1]
		self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=input_dim)
		self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=input_dim)

		for ind, (dim_in, dim_out) in enumerate(reversed(self.dims)):
			is_last = (ind == (num_layers - 1))

			self.ups.append(
				nn.ModuleList(
					[
						ResnetBlock(dim_out * 2, dim_in, time_emb_dim=input_dim),
						ResnetBlock(dim_in, dim_in, time_emb_dim=input_dim),
						Upsample(dim_in, dim_in) if not is_last else nn.Identity(),
					]
				)
			)

		self.final_conv = nn.Sequential(Block(input_dim, input_dim))

	def forward(self, x, time):
		t = self.time_pos_emb(time)
		t = self.mlp(t)

		h = []

		for resnet, resnet2, downsample in self.downs:
			x = resnet(x, t)
			x = resnet2(x, t)
			h.append(x)
			x = downsample(x)

		x = self.mid_block1(x, t)
		x = self.mid_block2(x, t)

		for resnet, resnet2, upsample in self.ups:
			x = torch.cat((x, h.pop()), dim=1)
			x = resnet(x, t)
			x = resnet2(x, t)
			x = upsample(x)

		return self.final_conv(x)

class Unet_drift(nn.Module):
	def __init__(self, input_dim, timesteps_embedding_dim=32, num_layers=3):
		super().__init__()

		self.time_pos_emb = get_timestep_embedding(timesteps_embedding_dim)
		self.mlp = nn.Sequential(
			nn.Linear(timesteps_embedding_dim, input_dim * 4), Mish(), nn.Linear(input_dim * 4, input_dim)
		)

		self.input_dim = input_dim
		self.dims = []
		temp_dim = input_dim
		for i in range(num_layers):
			self.dims.append([temp_dim, temp_dim // 2])
			temp_dim //= 2

		self.downs = nn.ModuleList([])
		self.ups = nn.ModuleList([])

		for ind, (dim_in, dim_out) in enumerate(self.dims):
			is_last = (ind == (num_layers - 1))
			self.downs.append(
				nn.ModuleList(
					[
						ResnetBlock(dim_in, dim_out, time_emb_dim=input_dim),
						ResnetBlock(dim_out, dim_out, time_emb_dim=input_dim),
						Downsample(dim_out, dim_out) if not is_last else nn.Identity(),
					]
				)
			)

		mid_dim = self.dims[-1][-1]
		self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=input_dim)
		self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=input_dim)

		for ind, (dim_in, dim_out) in enumerate(reversed(self.dims)):
			is_last = (ind == (num_layers - 1))

			self.ups.append(
				nn.ModuleList(
					[
						ResnetBlock(dim_out * 2, dim_in, time_emb_dim=input_dim),
						ResnetBlock(dim_in, dim_in, time_emb_dim=input_dim),
						Upsample(dim_in, dim_in) if not is_last else nn.Identity(),
					]
				)
			)

		# self.final_conv = nn.Sequential(Block(input_dim, input_dim))

	def forward(self, x, time, beta):
		t = self.time_pos_emb(time)
		t = self.mlp(t)
		inputs = x
		h = []

		for resnet, resnet2, downsample in self.downs:
			x = resnet(x, t)
			x = resnet2(x, t)
			h.append(x)
			x = downsample(x)

		x = self.mid_block1(x, t)
		x = self.mid_block2(x, t)

		for resnet, resnet2, upsample in self.ups:
			x = torch.cat((x, h.pop()), dim=1)
			x = resnet(x, t)
			x = resnet2(x, t)
			x = upsample(x)

		# self.final_conv(x)
		return x - inputs * (beta ** (2)) / 2

class FourierMLP(nn.Module):
	def __init__(self, input_dim, timesteps_embedding_dim=32, num_layers=2, channels=128):
		super().__init__()
		self.input_dim = input_dim

		self.input_embed = nn.Linear(input_dim, channels)
		self.timestep_embed_function = get_timestep_embedding(timesteps_embedding_dim)
		self.timestep_embed = nn.Sequential(
			nn.Linear(timesteps_embedding_dim, channels),
			nn.GELU(),
			nn.Linear(channels, channels),
		)
		self.layers = nn.Sequential(
			nn.GELU(),
			*[
				nn.Sequential(nn.Linear(channels, channels), nn.GELU())
				for _ in range(num_layers)
			],
			nn.Linear(channels, input_dim),
		)

	def forward(self, inputs, cond, diff_g=None):
		embed_cond = self.timestep_embed_function(cond)
		embed_cond = self.timestep_embed(embed_cond)

		embed_ins = self.input_embed(inputs)
		out = self.layers(embed_ins + embed_cond)
		return out
