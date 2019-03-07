import torch.nn as nn
import torch
import numpy as np
from layers.utils import log_gaussian
import datetime
import wandb
# import sklearn

# MNIST classes
classes = [str(i) for i in range(10)]

# Repeat the data for num_samples times
def repeat(args,data,target,repeat):
	resize = args.resize
	num_samples = args.num_samples
	if args.is_conv:
		data = data.view(-1, 1, resize, resize)
		if repeat:
			data = data.repeat(num_samples, 1, 1, 1)
	else:
		data = data.view(-1, 1*resize*resize)
		if repeat:
			data = data.repeat(num_samples, 1)
	if repeat:
		target = target.repeat(num_samples)
	if args.cuda:
		return (data.cuda(),target.cuda())
	else:
		return (data,target)

# Load a model
def load(filename):
	return torch.load(filename)

# Different values of beta(changes with batchid in each epoch)
# for KL Scaling types possible are Blundell, Soenderby and Normal
def beta(args, batch_idx, n_batches):
	if args.beta=="Blundell":
		beta = (2**(n_batches-(batch_idx+1))) / float(2**n_batches - 1)
	elif args.beta=="Soenderby":
		beta = min(batch_idx / (n_batches // 4), 1)
	else:
		beta = 1.0 / n_batches
	return beta

# nn.Module class for Loss function
class Loss(nn.Module):
	def __init__(self, is_cuda=False,loss=nn.CrossEntropyLoss()):
		super(Loss,self).__init__()
		self.loss = loss
		self.is_cuda = is_cuda

	def forward(self, output, target, kl, n_batches, batch_size, sigma_prior, beta):
		target = np.squeeze(np.eye(len(classes))[target])
		if self.is_cuda:
			target = torch.Tensor(target).cuda()
		else:
			target = torch.Tensor(target)
		l_likelihood = log_gaussian(output, target, sigma_prior).sum()
		loss = (beta*kl - l_likelihood).sum() / float(batch_size)
		return loss

# Logging the output
def log(args, train_loss, val_loss, epoch, val_acc, entropy_val):
	if args.is_wandb:
		wandb.log({
				"Training Loss vs Epoch" : train_loss,
				"Validation Loss vs Epoch" : val_loss,
				"Validation Accuracy" : val_acc,
				"Entropy Validation": entropy_val,
			})
	if epoch%args.log_interval==0:
		template = ("Epoch : {}\n \t Training Loss : {}, "
			"Validation Loss : {}, Validation Accuracy : {}, "
			"Entropy Validation : {}")
		print(template.format(epoch, train_loss,
		 	val_loss, val_acc, entropy_val))

# Converts an array to string
def array_to_str(arr):
	s = ''
	if not len(arr): return s
	for i in arr:
		if type(i)==int:
			s += str(i)+'-'
		else:
			s += '_'+array_to_str(i)+'_'
	return s[:-1]

# Weight inittialization types
def name_init(args):
	if args.is_orthogonal:
		return "ORTH"
	else:
		return "GAUS"

# Filename for saving the current best model
def filename(args):
	if args.is_conv:
		name = 'models/best_model.conv.{}.fc{}.batch_size.{}.pt'.format(
			args.conv, str(args.fc), args.batch_size)
	else:
		name = ('models/best_model.fc.{}.num_epoch{}'+
		'.beta{}.sigma_prior.{}.init.{}.init_scale.{}'+
		'.val_ratio.{}.batch_size.{}.pt').format(
			args.fc, args.num_epochs,  
			args.beta, args.sigma_prior, 
			name_init(args), args.init_scale,
			args.val_ratio, args.batch_size,)
	return name

# The name of run in Wandb
def run_name(args):
	name = ''
	if args.is_conv:
		name +='CONV-'
		name += array_to_str(args.conv)+'.'
	if args.fc:
		name += 'FC-'
		name += array_to_str(args.fc)+'.'
	if args.is_gaussian:
		name += 'GAUSS.'
	if args.is_orthogonal:
		name += 'ORTHO.'
	name += 'Scale-'
	name += str(args.init_scale)+'.'
	name += 'Beta-'
	name += str(args.beta)+'.'
	name += 'Epochs-'
	name += str(args.num_epochs)+'.'
	name += 'Time-'
	name += str(datetime.datetime.now())
	return name

# updates the best now model
def update(args,model,val_loss_min,val_loss):
	if val_loss_min >= val_loss:
		name = filename(args)
		torch.save(model, name)
		return val_loss
	else:
		return val_loss_min

# to convert a torch tensor to numpy
def to_numpy(tensor, args):
	if args.cuda:
		return(tensor.cpu().numpy())
	else:
		return(tensor.numpy())

# Compares the target and prediction 
def compare(args, target, pred):
	correct = pred.eq(target.data.view_as(pred))
	return np.squeeze(to_numpy(correct, args))

# Counts the label info
# Example :- 
# 	Input := [0,0,0,1,1,3,3]
# 	Output := [3,2,0,2] Count array
def count(pred):
	count = np.array([(pred==i).sum() for i in range(len(classes))])
	return count

# Log the test accuracy
def log_test_info(args, class_total, class_correct, entropy_test):
	test_acc = float(np.sum(class_correct))/np.sum(class_total)
	print("Test Accuracy:{}\t Entropy Test:{}".format(
		test_acc, entropy_test))
	if args.is_wandb:
		wandb.log({
			"Test Accuracy ": test_acc,
			"Entropy Test": entropy_test,
			})