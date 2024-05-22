import os
import yaml
from argparse import ArgumentParser,FileType

def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)

def parse_train_args():

	# General arguments
	parser = ArgumentParser()
	parser.add_argument('--config', type=FileType(mode='r'), default=None)
	parser.add_argument('--system', type=str, default='penta', choices=['ad', 'penta'])
	parser.add_argument('--log_dir', type=str, default='workdir', help='Folder in which to save model and logs')
	parser.add_argument('--run_name', type=str, default='try', help='')
	parser.add_argument('--restart_dir', type=str, help='Folder of previous training model from which to restart')
	parser.add_argument('--tica_dim', type=int, default=106, help='dim of tica transform')
	parser.add_argument('--lag_tica', type=int, default=5, help='lag time in tica')
	parser.add_argument('--lag1', type=int, default=1, help='small lag time')
	parser.add_argument('--visual_results', type=str, default='visual', help='visual_results')

	# Old tica data by yjy
	# parser.add_argument('--data_dir', type=str, default='data/', help='Folder containing original structures')
	parser.add_argument('--source_struct', type=str, default='data/pentapeptide-impl-solv.pdb', help='structure data')
	parser.add_argument('--source_traj', type=str, default='Backbone_compact_rot+trans.xtc', help='the souce traj data')
	parser.add_argument('--tica_data', type=str, default='tica_traj_ala0_d3.npy', help='data after tica transform')

	# New tica data
	parser.add_argument('--data_dir', type=str, default='./data', help='Folder containing original structures')
	parser.add_argument('--traj', type=str, default='tica_traj0.npy', help='the souce traj data')
	parser.add_argument('--dihedral', type=str, default='dihedral_traj0.npy', help='the souce dihedral data')

	# Training arguments
	parser.add_argument('--train_portion', type=float, default=0.67, help='portion of training data 0.67')
	parser.add_argument('--train_batch_size', type=int, default=8192, help='Train Batch size')
	parser.add_argument('--val_batch_size', type=int, default=4096, help='Val Batch size')
	parser.add_argument('--n_epochs_embedding', type=int, default=20, help='Number of epochs for training')
	parser.add_argument('--n_epochs_latent', type=int, default=20, help='Number of epochs for training')
	parser.add_argument('--n_epochs_joint', type=int, default=20, help='Number of epochs for training')
	parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
	parser.add_argument('--w_decay', type=float, default=0, help='Weight decay added to loss')
	parser.add_argument('--lr_scheduler_step_size', type=int, default=10, help='')
	parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1, help='')
	# parser.add_argument('--use_ema', type=bool, default=True, help='Whether or not to use ema for the model weights')
	# parser.add_argument('--ema_rate', type=float, default=0.999, help='decay rate for the exponential moving average model parameters ')
	# parser.add_argument('--save_feq', type=int, default=50, help='Save frequence')

	#
	parser.add_argument('--embeding_type', type=str, default='both', choices=['diffusion', 'NF', 'both'], help='timesteps_embedding_dim in diffusion model')
	
	# Diffusion model
	parser.add_argument('--timesteps_embedding_dim', type=int, default=32, help='timesteps_embedding_dim in diffusion model')
	parser.add_argument('--diffusion_net_layer', type=int, default=3, help='Number of interaction layers')
	parser.add_argument('--diffusion_net_channels', type=int, default=128, help='dim of hidden space of diffusion_net')
	parser.add_argument('--diffusion_num_steps', type=int, default=10, help='diffusion_num_steps in diffusion model')
	parser.add_argument('--diffusion_t_start', type=float, default=0.001, help='diffusion_t_start in diffusion model')
	parser.add_argument('--diffusion_t_end', type=float, default=0.999, help='diffusion_t_end in diffusion model')

	parser.add_argument('--diffusion_g_start', type=float, default=1e-3, help='diffusion_g_start in diffusion model 1e-4')
	parser.add_argument('--diffusion_g_end', type=float, default=1e-2, help='diffusion_g_end in diffusion model 2e-2')
	parser.add_argument('--diffusion_exp', type=float, default=0.9, help='diffusion_exp in diffusion model')

	# NF model
	parser.add_argument('--latent_dim', type=int, default=8, help='dim of latent space')
	parser.add_argument('--num_layers_nf', type=int, default=12, help='Number of interaction layers')
	parser.add_argument('--hidden_dim_nf', type=int, default=256, help='dim of hidden space of nf')

	# latent prior model
	parser.add_argument('--prior1_type', type=str, default='GMM', choices=['MLP', 'GMM'], help='prior1_type in LD')
	parser.add_argument('--sample_stepsize', type=float, default=1, help='sample stepsize in LD, 0.01')
	parser.add_argument('--beta', type=float, default=1.0, help='beta in LD')
	parser.add_argument('--gamma', type=float, default=0.9, help='gamma in LD')
	parser.add_argument('--component_num', type=int, default=10, help='beta in LD')
	parser.add_argument('--sub_step_num', type=int, default=10, help='sub step num in LD')
	parser.add_argument('--sub_sample_num', type=int, default=20, help='sub_sample_num in LD')
	parser.add_argument('--alpha', type=float, default=0.1, help='alpha in loss')
	
	# with open('cfg/alanine-dipeptide.yml', 'r') as f:
	# 	default_arg = yaml.load(f, Loader=yaml.FullLoader)
	# parser.set_defaults(**default_arg)
	# args = parser.parse_args()
	args = parser.parse_known_args()[0]

	return args
