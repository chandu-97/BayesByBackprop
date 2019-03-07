import torch.nn as nn

class FlattenLayer(nn.Module):
	"""FlattenLayer"""
	def __init__(self, num_feat):
		super(FlattenLayer, self).__init__()
		self.num_feat = num_feat

	def forward(self,input):
		return input.view(-1,self.num_feat)
	