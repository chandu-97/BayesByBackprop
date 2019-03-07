import argparse
import torch
import numpy as np

def get_args():
	parser = argparse.ArgumentParser(description="Bayesian CNN")

	# arguments for cnn
	parser.add_argument('--num-epochs', type=int, default=10,
						help='Epochs (100)')
	parser.add_argument('--batch-size', type=int, default=64,
						help='Batch Size(64)')
	parser.add_argument('--resize', type=int, default=28,
						help='Image resize(28x28)')
	parser.add_argument('--is-conv', type=bool, default= False,
						help=' Boolean for has conv structure')
	parser.add_argument('--conv', type=str, default="[[3,6,5],[6,16,5]]",
						help='Convolution Layer Structure[[in_channels_1, \
						out_channels_1, kernel_size_1],...,[in_channels_n,\
						out_channels_n, kernel_size_n]]')
	parser.add_argument('--fc', type=str, default="[120,84]",
						help='FC layer output sizes')
	parser.add_argument('--cuda', action='store_true', default=False,
						help='Cuda is active by default')
	parser.add_argument('--log-interval', type=int, default=1,
						help='Log interval for terminal')
	parser.add_argument('--val-ratio', type=float, default=0.2,
						help='Split ratio for validation')
	parser.add_argument('--num-workers', type=int, default=4,
						help='Number of workers for loading data')
	parser.add_argument('--num-samples', type=int, default=10,
						help='Number of samples to take from distribution')
	parser.add_argument('--beta', type=str, default="Normal",
						help='Type of beta(weight to kl loss)')
	parser.add_argument('--is-normal', type=bool, default=False,
						help='Run it as a Normal MLP')
	parser.add_argument('--sigma-prior', type=float, default=float(np.exp(-3)),
						help='Prior on sigma default(1e-3)')
	parser.add_argument('--is-wandb', action='store_true', default=False,
						help='Use wandb (default:false)')
	parser.add_argument('--is-gaussian', action='store_true', default=True,
						help='Weight initialization - Gaussian')
	parser.add_argument('--is-orthogonal', action='store_true', default=False,
						help='Weight initialization - Orthogonal')
	parser.add_argument('--init-scale', type=float, default=1.0,
						help='Scaling of initialization(default:1.0)')

	# Parse arguments
	args = parser.parse_args()
	args.cuda = args.cuda and torch.cuda.is_available()
	args.fc = list(map(int, args.fc.strip('[]').split(',')))
	# args.conv = list(map(int, args.conv.strip('[]').split(',')))
	print(args)
	print('Conv is not supported yet')
	return args