import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import ipdb
import wandb

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from calibration_loss import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

scaler = torch.cuda.amp.GradScaler()
model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
					help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--lamda', default=0.1, type=float, help='lambda for calibration loss')
parser.add_argument('--loss', type=str, required=True, choices=['ce', 'ce+mmce', 'ce+sbece' ,'ce+esd'])
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--calibration', action='store_true', help="use calibration")
parser.add_argument('--calset_ratio', default=0, type=float, help='ratio of calibration set')
parser.add_argument('--theta', default=0.4, type=float, help='kernel width for mmce')
parser.add_argument('--T', default=0.01, type=float, help='temperature for sbece')
parser.add_argument('--loss_bin', default=15, type=int, help='number of bins for sbece')
parser.add_argument('--experiment_name', type=str, required=True)

args = parser.parse_args()
#for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic=True
cudnn.benchmark = False
from torch.utils.data import Dataset

class CustomImageDataset_(Dataset):
	def __init__(self, files_dir):
		self.files_dir = files_dir
		self.all_imgs = os.listdir(self.files_dir)
	def __len__(self):
		return len(self.all_imgs)

	def __getitem__(self, idx):
		image, label = torch.load(os.path.join(self.files_dir, str(idx) + '_batch.pt'))
		return image, label

from pathlib import Path
def main():		

	wandb.init(
		project="imagenet_calibration",
		name=args.experiment_name
		)
	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	if torch.cuda.is_available():
		ngpus_per_node = torch.cuda.device_count()
	else:
		ngpus_per_node = 1
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
	global best_acc1
	args.gpu = gpu

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
	# create model
	if args.pretrained:
		print("=> using pre-trained model '{}'".format(args.arch))
		model = models.__dict__[args.arch](pretrained=True)
	else:
		print("=> creating model '{}'".format(args.arch))
		model = models.__dict__[args.arch](num_classes=100)

	if not torch.cuda.is_available() and not torch.backends.mps.is_available():
		print('using CPU, this will be slow')
	elif args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if torch.cuda.is_available():
			if args.gpu is not None:
				torch.cuda.set_device(args.gpu)
				model.cuda(args.gpu)
				# When using a single GPU per process and per
				# DistributedDataParallel, we need to divide the batch size
				# ourselves based on the total number of GPUs of the current node.
				args.batch_size = int(args.batch_size / ngpus_per_node)
				args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
				model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
			else:
				model.cuda()
				# DistributedDataParallel will divide and allocate batch_size to all
				# available GPUs if device_ids are not set
				model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None and torch.cuda.is_available():
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)

	else:
		# DataParallel will divide and allocate batch_size to all available GPUs
		if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
			model.features = torch.nn.DataParallel(model.features)
			model.cuda()
		else:
			model = torch.nn.DataParallel(model).cuda()

	if torch.cuda.is_available():
		if args.gpu:
			device = torch.device('cuda:{}'.format(args.gpu))
		else:
			device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")
	# define loss function (criterion), optimizer, and learning rate scheduler
	criterion = nn.CrossEntropyLoss().to(device)

	# Data loading code
	if args.dummy:
		print("=> Dummy data is used!")
		train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
		val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
	else:
		traindir = os.path.join(args.data, 'train')
		valdir = os.path.join(args.data, 'val')
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

		if not args.calibration:
			train_dataset = datasets.ImageFolder(
				traindir,
				transforms.Compose([
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					normalize,
				]))

			val_dataset = datasets.ImageFolder(
				traindir,
				transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					normalize,
				]))
			
		else:
			train_dataset = datasets.ImageFolder(
				traindir,
				transforms.Compose([
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					normalize,
				]))

			val_dataset = datasets.ImageFolder(
				traindir,
				transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					normalize,
				]))

			cal_dataset = datasets.ImageFolder(
				traindir,
				transforms.Compose([
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					normalize,
				]))

		test_dataset = datasets.ImageFolder(
			valdir,
			transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
			]))

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
	else:
		train_sampler = None
		val_sampler = None

	from torch.utils.data.sampler import SubsetRandomSampler
	import numpy as np
	valid_size = 0.1
	num_train = len(train_dataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.seed(args.seed)
	np.random.shuffle(indices)
	train_idx, valid_idx = indices[split:], indices[:split]
	if args.calibration:
		num_train = len(train_idx)
		split = int(np.floor(args.calset_ratio * num_train))
		train_idx, cal_idx = train_idx[split:], train_idx[:split]
		cal_sampler = SubsetRandomSampler(cal_idx)
		
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	lamda = args.lamda
	cal_dataloader = None

	if not args.calibration:
		train_dataloader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True, sampler=train_sampler, prefetch_factor=2, persistent_workers=True)

		val_dataloader = torch.utils.data.DataLoader(
			val_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True, sampler=valid_sampler, prefetch_factor=2, persistent_workers=True)

		test_dataloader = torch.utils.data.DataLoader(
			test_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True, sampler=None, prefetch_factor=2, persistent_workers=True)
		cal_dataloader = None
	else:
		import numpy as np
		train_dataloader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=4, pin_memory=True, sampler=train_sampler, prefetch_factor=2)

		cal_dataloader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=4, pin_memory=True, sampler=cal_sampler, prefetch_factor=2)

		val_dataloader = torch.utils.data.DataLoader(
			val_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=4, pin_memory=True, sampler=valid_sampler, prefetch_factor=2)
			
		test_dataloader = torch.utils.data.DataLoader(
			test_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=4, pin_memory=True, sampler=None, prefetch_factor=2)

	loss_criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = args.weight_decay)

	softmax_layer = torch.nn.Softmax(dim=1)
	model = model.to(device)
	scaler = torch.cuda.amp.GradScaler()
	
	for epoch in tqdm(range(args.epochs)):
		model.train()
		if args.calibration and args.calset_ratio > 0:
			cal_dataloader_iterator = iter(cal_dataloader)
		for batch_id, (image,label) in enumerate(tqdm(train_dataloader)):
			with torch.cuda.amp.autocast():
				optimizer.zero_grad()
				image = image.to(device, non_blocking=True)
				label = label.to(device, non_blocking=True)
				
				logit = model(image)
				output = softmax_layer(logit)
				loss = loss_criterion(logit,label)

				if args.calibration and args.calset_ratio > 0:  
					try:
						image_calset, label_calset = next(cal_dataloader_iterator)
					except StopIteration:
						cal_dataloader_iterator = iter(cal_dataloader)
						image_calset, label_calset = next(cal_dataloader_iterator)
					image_calset = image_calset.to(device)
					label_calset = label_calset.to(device)
					
					logit_calset = model(image_calset)
					output_calset = softmax_layer(logit_calset)

					confidence_calset, prediction_calset = torch.max(output_calset, dim = 1)
					correct_calset = label_calset.eq(prediction_calset)
					
					if args.loss == 'ce+esd':
						calterm = ESD(device, confidence_calset, correct_calset)
						loss += lamda * torch.sqrt(torch.nn.functional.relu(calterm))
					elif args.loss == 'ce+sbece':
						calterm = SBECE(args.loss_bin, confidence_calset, correct_calset, args.T, device)
						loss += lamda * calterm
					elif args.loss == 'ce+mmce':
						calterm = MMCE_unweighted(device, confidence_calset, correct_calset, kernel_theta=args.theta)
						loss += lamda * torch.sqrt(calterm)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

		num_class = output.size(1)
		if epoch % 1 == 0:
			print(epoch)
			print('start logging')
			logits_list, labels_list = postprocess_preprocess(args, model, device, val_dataloader)
			T = tempscale_train(args, model, device, logits_list, labels_list)
			w,b = plattscale_train(args, model, device, logits_list, labels_list, num_class = num_class)
			with torch.no_grad():
				test_accuracy, test_accuracy_vs, ece_test, ece_test_tempscale, ece_test_vs, val_accuracy, ece_val = log_wandb_imagenet(args, model, train_dataloader, cal_dataloader, val_dataloader, test_dataloader, device, T, w, b)
			wandb.log({"Epoch": epoch, "Test Accuracy": test_accuracy, "Test Accuracy VS": test_accuracy_vs, "ECE Test": ece_test, "ECE Test TempScale": ece_test_tempscale, "ECE Test VS": ece_test_vs, "Val Accuracy": val_accuracy, "ECE Val": ece_val})

			print('end logging')
		
def tempscale_train(args, model, device, logits_list, labels_list):

	#load saved model
	def T_scaling(logits, temperature):
		return torch.div(logits, temperature)    

	temperature = torch.nn.Parameter(torch.ones(1).cuda())
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

	def _eval():
		loss = criterion(T_scaling(logits_list, temperature), labels_list)
		loss.backward()
		return loss
	optimizer.step(_eval)

	return temperature

def postprocess_preprocess(args, model, device, val_dataloader):
	logits_list = []
	labels_list = []

	for i, data in enumerate(val_dataloader, 0):
		images, labels = torch.squeeze(data[0]).to(device), torch.squeeze(data[1]).to(device)
		model.eval()
		with torch.no_grad():
			logits_list.append(model(images))
			labels_list.append(labels)
	
	#create tensors
	logits_list = torch.cat(logits_list).to(device)
	labels_list = torch.cat(labels_list).to(device)

	return logits_list, labels_list

def plattscale_train(args, model, device, logits_list, labels_list, num_class):

	#load saved model
	def T_scaling(logits, w, b):
		return torch.bmm((torch.diag(w).unsqueeze(0)).expand(logits.size(0),num_class,num_class),logits.unsqueeze(2)).view(logits.size(0),-1) + b.unsqueeze(0).expand(logits.size(0),-1)
	w_parameter = torch.nn.Parameter(torch.ones(num_class).cuda())
	b_parameter = torch.nn.Parameter(torch.ones(num_class).cuda())
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.LBFGS([w_parameter,b_parameter], lr=0.01, max_iter=10000, line_search_fn='strong_wolfe')

	def _eval():
		loss = criterion(T_scaling(logits_list, w_parameter, b_parameter), labels_list)
		loss.backward()
		return loss
	optimizer.step(_eval)
	return w_parameter, b_parameter
	

if __name__ == '__main__':
	main()
