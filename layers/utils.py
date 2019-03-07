import math 
import torch
from torch.nn.parameter import Parameter
import numpy as np

# Orthogonal Init
def orthogonal(tensor, init_scale=1.0):
	if tensor.ndimension() < 2:
		raise ValueError("Only tensors with 2 or more dimensions are supported")

	rows = tensor.size(0)
	cols = tensor[0].numel()
	flattened = torch.Tensor(rows, cols).normal_(0, 1)

	if rows < cols:
		flattened.t_()

	# Compute the qr factorization
	q, r = torch.qr(flattened)
	# Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
	d = torch.diag(r, 0)
	ph = d.sign()
	q *= ph.expand_as(q)

	if rows < cols:
		q.t_()

	tensor.view_as(q).copy_(q)
	tensor.mul_(init_scale)
	return tensor

# 
def log_gaussian(x, mu, sigma):
	return float(-0.5 * np.log(2*np.pi) - np.log(np.abs(sigma))) - (x-mu)**2 / (2*sigma**2)

# 
def log_gaussian_logsigma(x, mu, logsigma):
	return float(-0.5 * np.log(2*np.pi)) - logsigma - (x-mu)**2 / (2*torch.exp(logsigma)**2)