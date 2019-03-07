from torchvision import datasets
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import math

# Transformations on MNIST
transform = transforms.Compose([
	transforms.RandomHorizontalFlip(), # randomly flip and rotate
	transforms.RandomRotation(10),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

# Loaders for Train, Val and Test set
def loaders(args,is_train=True):
	if is_train:
		train_data = datasets.MNIST('data', train=True,
								  download=True, transform=transform)
		num_train = len(train_data)
		indices = list(range(num_train))
		np.random.shuffle(indices)
		split = int(np.floor(args.val_ratio * num_train))
		train_idx, valid_idx = indices[split:], indices[:split]

		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)

		train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
			sampler=train_sampler, num_workers=args.num_workers)
		val_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
			sampler=valid_sampler, num_workers=args.num_workers)
		batch_epoch_train = math.ceil(len(train_data)*(1-args.val_ratio)/args.batch_size)
		batch_epoch_val = math.ceil(len(train_data)*(args.val_ratio)/args.batch_size)

		return train_loader, val_loader, batch_epoch_train, batch_epoch_val, args.val_ratio

	else:
		test_data = datasets.MNIST('data', train=False,
								 download=True, transform=transform)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, 
			num_workers=args.num_workers)
		batch_epoch_test = math.ceil(len(test_data) / args.batch_size)
		return test_loader, batch_epoch_test