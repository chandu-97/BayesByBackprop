# Importing Libraries
import test_args
import test_data
import test_model
import test_utils as utils
import wandb
import numpy as np
import torch.optim as optim
import torch
from scipy.stats import entropy

# Train
def train(args):
	# Load the train val loaders
	(train_loader, val_loader, batch_epoch_train, batch_epoch_val, 
		valratio) = test_data.loaders(args,is_train=True)
	resize = args.resize

	# Conv feature needs to be implemented yet
	# Model creation
	if args.is_conv:
		model = test_model.Net(args, (1,resize,resize), len(utils.classes))
	else:
		model = test_model.Net(args, 1*resize*resize, len(utils.classes))

	if args.cuda: model = model.cuda()
	num_samples = args.num_samples
	batch_size = args.batch_size

	# Wandb Logging
	if args.is_wandb:
		wandb.init(project='Week 5 BBB Abalations')
		wandb.config.update(args)
		wandb.run.description = utils.run_name(args)
		wandb.run.save()

	val_loss_min = np.Inf

	# Optimizer setup
	optimizer = optim.Adam(model.parameters())


	# Run for these number of epochs
	for epoch in range(1,args.num_epochs+1):
		train_loss = val_loss = 0.0
		entropy_val = 0.0
		# Train
		model.train()
		loss_cl = utils.Loss(is_cuda=args.cuda)
		for batch_idx,(data,target) in enumerate(train_loader):
			data,target = utils.repeat(args,data,target,repeat=True) # check even cuda here
			beta = utils.beta(args, batch_idx, batch_epoch_train)
			optimizer.zero_grad()
			kl,output = model.klforward(data)
			loss = loss_cl(output, target, kl, batch_epoch_train, 
				batch_size, args.sigma_prior, beta)
			loss.backward()
			optimizer.step()
			train_loss += float(loss.item()*data.size(0)) / (num_samples) # Train loss

		# Validation
		model.eval()
		loss_cl = utils.Loss(is_cuda=args.cuda)
		class_correct = [0.0]*len(utils.classes)
		class_total  = [0.0]*len(utils.classes)
		for batch_idx,(data,target) in enumerate(val_loader):
			batch_size = target.shape[0]
			pred = torch.Tensor(num_samples*batch_size)
			if args.cuda: pred = pred.cuda()
			data,target = utils.repeat(args,data,target,repeat=False)
			for i in range(num_samples):
				beta = utils.beta(args, batch_idx, batch_epoch_val)
				kl,output = model.klforward(data)
				loss = loss_cl(output, target, kl, batch_epoch_val, 
					batch_size, args.sigma_prior, beta)
				val_loss += float(loss.item()*data.size(0)) / num_samples
				_, temp = torch.max(output,1)
				pred[i*batch_size:(i+1)*batch_size] = temp

			# Taking argmax over predictions and calculating entropy
			unique_samples = target.shape[0]
			preds = np.zeros(unique_samples)
			for i in range(unique_samples):
				idxs = [j*unique_samples+i for j in range(args.num_samples)]
				temp = pred[idxs]
				counts = utils.count(temp)
				entropy_val += entropy(counts/float(np.sum(counts)))
				preds[i] = np.argmax(counts)

			# Calculating correct labels
			preds = torch.tensor(preds,dtype=torch.long)
			if args.cuda: preds=preds.cuda()
			correct = utils.compare(args, target, preds)
			for i in range(unique_samples):
				label = target.data[i]
				class_correct[label] += correct[i].item()
				class_total[label] += 1

		entropy_val = float(entropy_val)/(args.batch_size * batch_epoch_val)
		val_acc = float(np.sum(class_correct))/np.sum(class_total)
		train_loss = train_loss/(len(train_loader.dataset)*(1-valratio))
		val_loss = val_loss/(len(train_loader.dataset)*valratio)
		utils.log(args, train_loss, val_loss, epoch, val_acc, entropy_val)
		val_loss_min = utils.update(args,model,val_loss_min,val_loss)

# Testing
def test(args):
	# Loading test loader
	test_loader, batch_epoch_test = test_data.loaders(args,is_train=False)
	resize = args.resize

	# Model creation
	if args.is_conv:
		model = test_model.Net(args, (3,resize,resize), len(utils.classes))
	else:
		model = test_model.Net(args, 1*resize*resize, len(utils.classes))
	if args.cuda:model.cuda()
	filename = utils.filename(args)
	model = utils.load(filename)
	test_loss = 0.0
	class_correct = [0.0]*len(utils.classes)
	class_total  = [0.0]*len(utils.classes)
	num_samples = args.num_samples
	
	# Iterate over test set
	loss_cl = utils.Loss(is_cuda=args.cuda)
	entropy_test = 0.0
	for batch_idx,(data,target) in enumerate(test_loader):
		batch_size = target.shape[0]
		pred = torch.Tensor(num_samples*batch_size)
		if args.cuda: pred = pred.cuda()
		data, target = utils.repeat(args,data,target,repeat=False)
		for i in range(num_samples):
			beta = utils.beta(args, batch_idx, batch_epoch_test)
			kl, output = model.klforward(data)
			loss = loss_cl(output, target, kl, batch_epoch_test, 
				args.batch_size, args.sigma_prior, beta)
			test_loss += loss.item()*data.size(0)/num_samples
			_, temp = torch.max(output,1)
			pred[i*batch_size:(i+1)*batch_size] = temp

		# Taking argmax over predictions and calculating entropy
		unique_samples = target.shape[0]
		preds = np.zeros(unique_samples)
		for i in range(unique_samples):
			idxs = [j*unique_samples+i for j in range(args.num_samples)]
			temp = pred[idxs]
			counts = utils.count(temp)
			entropy_test += entropy( counts/float(np.sum(counts)) )
			preds[i] = np.argmax(counts)

		# Calculating correct labels
		preds = torch.tensor(preds,dtype=torch.long)
		if args.cuda: preds=preds.cuda()
		correct = utils.compare(args, target, preds)
		for i in range(unique_samples):
			label = target.data[i]
			class_correct[label] += correct[i].item()
			class_total[label] += 1

	entropy_test = float(entropy_test)/(args.batch_size * batch_epoch_test)
	utils.log_test_info(args, class_total, class_correct, entropy_test)

if __name__ == '__main__':
	args = test_args.get_args()
	train(args)
	test(args)