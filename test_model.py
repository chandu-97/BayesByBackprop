from layers.mlp import BBBLinear
from layers.conv import BBBConv2d
from layers.flatten import FlattenLayer
import torch.nn as nn

class Net(nn.Module):
	"""Architecture for Net"""
	def __init__(self, args, input_size, output_size):
		super(Net, self).__init__()
		layers = []
		if args.is_conv:
			for i in args.conv:
				layers.append(BBBConv2d(i[0],i[1],(i[2],i[2]), args.cuda))
				layers.append(nn.Softplus())
				layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
			layers.append(FlattenLayer(5*5*16))# needs to be automated
			prev = 5*5*16
		else:
			prev = input_size
		for i in args.fc:
			layers.append(BBBLinear(prev,i,var=args.sigma_prior,
				is_normal=args.is_normal, is_cuda=args.cuda,
				is_gaussian=args.is_gaussian, is_orthogonal=args.is_orthogonal,
				init_scale=args.init_scale))
			layers.append(nn.Softplus())
			prev = i
		layers.append(BBBLinear(prev,output_size,var=args.sigma_prior,
			is_normal=args.is_normal, is_cuda=args.cuda,
			is_gaussian=args.is_gaussian, is_orthogonal=args.is_orthogonal,
			init_scale=args.init_scale))
		self.layers = layers

	def forward(self):
		raise NotImplementedError()

	def parameters(self):
		params = []
		for i in self.layers:
			if hasattr(i,'parameters'):
				params.extend(i.parameters())
		return params
		
	def klforward(self,input):
		kl = 0.0
		x = input
		for layer in self.layers:
			if hasattr(layer,'klforward'):
				kl_layer, x = layer.klforward(x)
				kl += kl_layer
			else:
				x = layer.forward(x)

		return kl,x