# https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from layers.utils import log_gaussian, log_gaussian_logsigma, orthogonal
from torch.autograd import Variable
import torch
import numpy as np

# is_normal is no bayes way
class BBBLinear(nn.Module):
	def __init__(self, inp, out, is_cuda=True, var=float(np.exp(-3)),
	 	bias=True, is_normal=False, is_gaussian=True, 
	  	is_orthogonal=False, init_scale=1.0):
		super(BBBLinear, self).__init__()
		self.inp = inp
		self.out = out
		self.var = torch.Tensor([var])
		if is_cuda:
			self.w_mu = Parameter(torch.Tensor(inp, out).cuda())
			self.w_p = Parameter(torch.Tensor(inp, out).cuda())
		else:
			self.w_mu = Parameter(torch.Tensor(inp, out))
			self.w_p = Parameter(torch.Tensor(inp, out))
		self._is_bias = bias
		self.is_normal = is_normal
		if bias:
			if is_cuda:
				self.b_mu = Parameter(torch.Tensor(out).cuda())
				self.b_p = Parameter(torch.Tensor(out).cuda())
			else:
				self.b_mu = Parameter(torch.Tensor(out))
				self.b_p = Parameter(torch.Tensor(out))
		self.is_cuda = is_cuda
		self.is_gaussian = is_gaussian
		self.is_orthogonal = is_orthogonal
		self.init_scale = init_scale
		self.reset_parameters()


	def parameters(self, no_bias=False):
		if no_bias:
			return (self.w_mu, self.w_p)
		else:
			return (self.w_mu, self.w_p, self.b_mu, self.b_p)

	# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
	def reset_parameters(self):
		# Weights initialization

		assert self.is_orthogonal or self.is_gaussian,( 
			'It has to either one orthogonal or gaussian')
		assert not(self.is_orthogonal and self.is_gaussian), (
			'It cannot be both orthogonal or gaussian')
		
		if self.is_gaussian:
			self.w_mu.data.normal_(0,0.01); self.w_mu.data.mul_(self.init_scale)
			self.w_p.data.normal_(0,0.01); self.w_p.data.mul_(self.init_scale)
			if self._is_bias:
				self.b_mu.data.uniform_(-0.01,0.01); self.b_mu.data.mul_(self.init_scale); 
				self.b_p.data.uniform_(-0.01,0.01); self.b_p.data.mul_(self.init_scale); 

		if self.is_orthogonal:
			orthogonal(self.w_mu.data, self.init_scale)
			orthogonal(self.w_p.data, self.init_scale)
			if self._is_bias:
				self.b_mu.data.fill_(0)
				self.b_p.data.fill_(0)

		# Not using kaiming init
		# init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
		# init.kaiming_uniform_(self.w_p, a=math.sqrt(5))
		# if self._is_bias:
		# fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
		# bound = 1 / math.sqrt(fan_in)
		# init.uniform_(self.b_mu, -bound, bound)
		# init.uniform_(self.b_p, -bound, bound)

		# if self.is_cuda:
		# 	self.w_mu = self.w_mu.cuda()
		# 	self.w_p = self.w_p.cuda()
		# 	self.b_mu = self.b_mu.cuda()
		# 	self.b_p = self.b_p.cuda()

	def forward(self,input):
		raise NotImplementedError()	

	# Computes KL loss and forward pass
	def klforward(self,input):
		if self.is_normal:
			return 0, F.linear(input, self.w_mu.t(), self.b_mu)
		epsilon_W, epsilon_b = self.noise()
		w = self.w_mu + torch.log(1 + torch.exp(self.w_p)) * epsilon_W
		var = Variable(self.var)
		if self.is_cuda:
			w = w.cuda()
			var = var.cuda()
		if self._is_bias:
			b = self.b_mu + torch.log(1 + torch.exp(self.b_p)) * epsilon_b
			if self.is_cuda:
				b = b.cuda()
			lpw = log_gaussian(w, 0.0, var).sum() + \
					log_gaussian(b, 0.0, var).sum()
			lqw = log_gaussian_logsigma(w, self.w_mu, self.w_p).sum() + \
					log_gaussian_logsigma(b, self.b_mu, self.b_p).sum()

		else:
			b = None
			lpw = log_gaussian(w, 0, var).sum()
			lqw = log_gaussian_logsigma(w, self.w_mu, self.w_p).sum()
		return lqw-lpw , F.linear(input, w.t(), b)

	# Generates a gaussian noise of same dimension as W and b
	def noise(self):
		epsilon_W = Variable(torch.Tensor(self.inp, self.out).normal_(0,self.var.data[0]))
		if self.is_cuda: 
			epsilon_W = epsilon_W.cuda()
		if self._is_bias:
			epsilon_b = Variable(torch.Tensor(self.out).normal_(0,self.var.data[0]))
			if self.is_cuda:
				epsilon_b = epsilon_b.cuda()
			return (epsilon_W, epsilon_b)
		return (epsilon_W, None)