# Donot use this still in progress

import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from layers.utils import log_gaussian, log_gaussian_logsigma

class _ConvNd(Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride, 
			padding, dilation, transposed, output_padding, groups, bias, var):
		super(_ConvNd, self).__init__()
		if in_channels % groups != 0:
			raise ValueError('in_channels must be divisible by groups')
		if out_channels % groups != 0:
			raise ValueError('out_channels must be divisible by groups')
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.transposed = transposed
		self.output_padding = output_padding
		self.groups = groups
		self._is_bias = bias
		self.var = var

		if self.transposed:
			self.w_mu = Parameter(torch.Tensor(in_channels, \
				out_channels // groups, *kernel_size))
			self.w_p = Parameter(torch.Tensor(in_channels, \
				out_channels // groups, *kernel_size))
		else:
			self.w_mu = Parameter(torch.Tensor(out_channels, \
				in_channels // groups, *kernel_size))
			self.w_p = Parameter(torch.Tensor(out_channels, \
				in_channels // groups, *kernel_size))
		
		if self._is_bias:
			self.b_mu = Parameter(torch.Tensor(out_channels))
			self.b_p = Parameter(torch.Tensor(out_channels))
		else:
			self.b_mu = None
			self.b_p = None
		
		self.reset_parameters()

	def parameters(self):
		return(self.w_mu, self.w_p, self.b_mu, self.b_p)

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / math.sqrt(n)
		self.w_mu.data.uniform_(-stdv, stdv)
		self.w_p.data.uniform_(-stdv, stdv)
		if self.b_mu is not None:
			self.b_mu.data.uniform_(-stdv, stdv)
		if self.b_p is not None:
			self.b_p.data.uniform_(-stdv, stdv)

	def __repr__(self):
		s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'\
			', stride={stride}, variance={var}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.output_padding != (0,) * len(self.output_padding):
			s += ', output_padding={output_padding}'
		if self.groups != 1:
			s += ', groups={groups}'
		if self.bias is None:
			s += ', bias=False'
		s += ')'
		return s.format(name=self.__class__.__name__, **self.__dict__)

	def noise(self):
		"""
		Returns noise with variance of var
		and in weights and bias shape  
		"""
		if self.transposed:
			shape = (self.in_channels, self.out_channels // self.groups, *self.kernel_size)
			if self._is_bias:
				return (Parameter(torch.Tensor(*shape).normal_(0, self.var)),
					Parameter(torch.Tensor(self.out_channels).normal_(0, self.var)) )
			else:
				return ( Parameter(torch.Tensor(*shape).normal_(0, self.var)),
					None)
		else:
			shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
			if self._is_bias:
				return (Parameter(torch.Tensor(*shape).normal_(0, self.var)),
						Parameter(torch.Tensor(self.out_channels).normal_(0, self.var)))
			else:
				return (Parameter(torch.Tensor(*shape).normal_(0, self.var)),
						None) 				

class BBBConv1d(_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True, var=1.0):
		super(BBBConv1d, self).__init__(in_channels, out_channels,
				kernel_size, stride, padding, dilation, 
				False, _single(0), groups, bias, var)
		kernel_size = _single(kernel_size)
		stride = _single(stride)
		padding = _single(padding)
		dilation = _single(dilation)

	def forward(self,input):
		raise NotImplementedError

	def klforward(self, input):
		epsilon_W, epsilon_b = self.noise()
		w = self.w_mu + torch.log(1 + torch.exp(self.w_p)) * epsilon_W
		if self._is_bias:
			b = self.b_mu + torch.log(1 + torch.exp(self.b_p)) * epsilon_b
			lpw = log_gaussian(w, 0, self.var).sum() + \
					log_gaussian(b, 0, self.var).sum()
			lqw = log_gaussian_logsigma(w, self.w_mu, self.w_p).sum() + \
					log_gaussian_logsigma(b, self.b_mu, self.b_p).sum()

		else:
			b = None
			lpw = log_gaussian(w, 0, self.var).sum()
			lqw = log_gaussian_logsigma(w, self.w_mu, self.w_p).sum()

		return (lpw, lqw, F.conv1d(input, w, b, self.stride,
				self.padding, self.dilation, self.groups))

class BBBConv2d(_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True, var=1.0):
		super(BBBConv2d, self).__init__(in_channels, out_channels, 
			kernel_size, stride, padding, dilation, 
			False, _pair(0), groups, bias, var)
		dilation = _pair(dilation)
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)

	def forward(self,input):
		raise NotImplementedError

	def klforward(self, input):
		epsilon_W, epsilon_b = self.noise()
		print(epsilon_b.shape,epsilon_W.shape,self.w_mu.shape,self.b_mu.shape)
		w = self.w_mu + torch.log(1 + torch.exp(self.w_p)) * epsilon_W
		if self._is_bias:
			b = self.b_mu + torch.log(1 + torch.exp(self.b_p)) * epsilon_b
			lpw = logpdf(w, 0, self.var).sum() + \
					logpdf(b, 0, self.var).sum()
			lqw = logpdf(w, self.w_mu, self.w_p).sum() + \
					logpdf(b, self.b_mu, self.b_p).sum()
		else:
			b = None
			lpw = logpdf(w, 0, self.var).sum()
			lqw = logpdf(w, self.w_mu, self.w_p).sum()

		return (lpw, lqw, F.conv2d(input, w, b, self.stride, 
			self.padding, self.dilation, self.groups))
