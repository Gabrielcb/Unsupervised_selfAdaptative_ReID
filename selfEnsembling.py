### Made by Gabriel Bertocco ###
import torch
import torchreid
import torchvision
from torchvision.transforms import ToTensor, Compose, TenCrop, Normalize, Resize, ToPILImage
from torch.nn import Module, Dropout, BatchNorm1d, Linear, AdaptiveAvgPool2d
from torch.nn import functional as F
from torch import nn


import os

from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import h5py
import argparse
import joblib
import h5py

from random import shuffle
import matplotlib.pyplot as plt

from load_sets_from_datasets import load_dataset

from featureExtraction import extractFeatures
from load_sets_from_datasets import load_set_from_market_duke, load_set_from_MSMT17
from torch.backends import cudnn

transform = Compose([Resize((256, 128), interpolation=3), ToTensor(), 
						Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True


# Market -> Duke (osnet_x1_0): 64.6/79.1 (0.1), 64.8/78.8 (0.09), 65.5/79.4 (0.08), 65.6/79.9(0.07), 65.7/79.6(0.06), 65.8/80.0 (0.05)
# 64.8/79.3 (0.04), 63.9/79.1 (0.03), 63.6/78.6 (0.02), 61.1/77.0 (0.01)

# Duke -> Market (osnet_x1_0): 	


def main(gpu_ids, source, target, dir_name, model_name, reliability_path, version, rerank):

	############================ CHANGED ON SIGMOID ON TMUX A -T 2 ================############ 
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
	print("Number of GPU's: ", torch.cuda.device_count())

	if len(gpu_ids) > 1:
		gpu_index = 1
	else:
		gpu_index = 0

	print("GPU index:", gpu_index)

	if source == "Duke":
		num_source_classes = 702
	elif source == "Market":
		num_source_classes = 751
	elif source == "None":
		num_source_classes = 1000 # ImageNet
	else:
		print("Dataset not implemented yet")
		exit()

	model_result = torchreid.models.build_model(
	    name=model_name,
	    num_classes=num_source_classes,
	    pretrained=True
	)

	t0 = time.time()
	trained_models = []

	number_of_models = 50 # Best for Duke -> Market
	count_models = 0

	model_filenames = []
	for file in os.listdir(dir_name):
		if file.endswith(version):
			#print(file)
			iter_number = int(file.split("_")[1])
			model_filenames.append([file, iter_number])
			
	model_filenames = sorted(model_filenames, key=lambda elem: elem[1])

	for file, iter_number in model_filenames:
		print(file)
		full_path = os.path.join(dir_name, file)
		state_dict = torch.load(full_path, map_location='cuda:0')
		trained_models.append(state_dict)

		count_models += 1
		if count_models >= number_of_models:
			break

	if model_name == "osnet_x1_0":
		model_result = OSNETReID(model_result)
	elif model_name == "resnet50":
		model_result = ResNet50ReID(model_result)
	elif model_name == "densenet121":
		model_result = DenseNet121ReID(model_result)

	state_dict_result = model_result.state_dict()
	keys = list(trained_models[0].keys())

	size = len(keys) 

	print(size)
	#idx = np.arange(0, 150, 5)
	reliability_progress = np.array(joblib.load(reliability_path)) #[idx]
	
	number_of_models = len(trained_models)
	print("Number of models: %d" % number_of_models)
	


	for i in range(size):

		sum_weights = None
		j = 0
		total_reability = 0
		for state_dict in trained_models:
			#print("Reliability: %f" % reliability_progress[j])
			if sum_weights == None:
				sum_weights = reliability_progress[j]*state_dict[keys[i]].clone()
			else:
				#print(i)
				sum_weights += reliability_progress[j]*state_dict[keys[i]].clone()
		
			total_reability += reliability_progress[j]
			j += 1

		mean_weights = sum_weights/np.sum(reliability_progress[:j])
		state_dict_result[keys[i]] = mean_weights.clone()

	model_result.load_state_dict(state_dict_result)
	model_result = model_result.cuda(gpu_index)
	model_result = model_result.eval()
			
	train_images_target, gallery_images_target, queries_images_target = load_dataset(target)
	tf = time.time()
	dt = tf - t0
	print("Self-Ensebling performed in %.2f seconds" % dt)

	print("Validating Model Result on %s ..." % target)
	validate(queries_images_target, gallery_images_target, model_result, gpu_index=gpu_index)

	torch.save(model_result.state_dict(), "model_%s_selfEnsembleLearning.h5" % version[:-3])
		



def calculateMetrics(distmat, queries_images, gallery_images):

	# Compute Ranks
	ranks=[1, 5, 10, 20]
	print('Computing CMC and mAP ...')
	cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries_images[:,1], gallery_images[:,1], 
														queries_images[:,2], gallery_images[:,2], 
														use_metric_cuhk03=False)

	print('** Results **')
	print('mAP: {:.2%}'.format(mAP))
	print('CMC curve')
	for r in ranks:
		print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))



		
def validate(queries, gallery, model, rerank=False, gpu_index=0):
    model.eval()
    queries_fvs = extractFeatures(queries, model, 500, gpu_index)
    gallery_fvs = extractFeatures(gallery, model, 500, gpu_index)

    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)


    # Compute Ranks
    ranks=[1, 5, 10, 20]
    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
        
    
    return cmc[:20], mAP


## New Model Definition for ResNet50
class ResNet50ReID(Module):
    
	def __init__(self, model_base):
		super(ResNet50ReID, self).__init__()


		self.conv1 = model_base.conv1
		self.bn1 = model_base.bn1
		self.maxpool = model_base.maxpool
		self.layer1 = model_base.layer1
		self.layer2 = model_base.layer2
		self.layer3 = model_base.layer3
		self.layer4 = model_base.layer4

		self.layer4[0].conv2.stride = (1,1)
		self.layer4[0].downsample[0].stride = (1,1)

		self.global_avgpool = model_base.global_avgpool
		self.last_bn = BatchNorm1d(2048)
		

	def forward(self, x):
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.global_avgpool(x)

		x = x.view(x.size(0), -1)
		output = self.last_bn(x)

		return output 

## New Model Definition for DenseNet121
class DenseNet121ReID(Module):
    
	def __init__(self, model_base):
		super(DenseNet121ReID, self).__init__()

		self.model_base = model_base.features
		self.gap = AdaptiveAvgPool2d(1)
		self.last_bn = BatchNorm1d(2048)
			
		
	def forward(self, x):
		
		x = self.model_base(x)
		x = F.relu(x, inplace=True)
		x = self.gap(x)
		x = torch.cat([x,x], dim=1)

		x = x.view(x.size(0), -1)
		output = self.last_bn(x)
		
		return output 

## New Definition for OSNET
class OSNETReID(Module):
    
	def __init__(self, model_base):
		super(OSNETReID, self).__init__()

		self.conv1 = model_base.conv1
		self.maxpool = model_base.maxpool
		self.conv2 = model_base.conv2
		self.conv3 = model_base.conv3
		self.conv4 = model_base.conv4
		self.conv5 = model_base.conv5
		self.avgpool = model_base.global_avgpool
		self.fc = model_base.fc

	def forward(self, x):
		
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		v = self.avgpool(x)
		v = v.view(v.size(0), -1)

		output = self.fc(v)
		return output



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Define the UDA parameters')
	parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
	parser.add_argument('--source', type=str, help='Name of source dataset')
	parser.add_argument('--target', type=str, help='Name of target dataset')
	parser.add_argument('--dir_name', type=str, help='Directory with models')
	parser.add_argument('--model_name', type=str, help='Name of the model architecture')
	parser.add_argument('--reliability_path', type=str, help='Path to the reliability of the model along the adaptation')
	parser.add_argument('--version', type=str, help='Model version name')
	parser.add_argument('--rerank', type=str, help='Indicates if Re-Ranking must or not be applied')
	
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	source = args.source
	target = args.target
	dir_name = args.dir_name
	model_name = args.model_name
	reliability_path = args.reliability_path
	version = args.version
	rerank = args.rerank

	if rerank == "True":
		rerank = True
	else:
		rerank = False
	
	main(gpu_ids, source, target, dir_name, model_name, reliability_path, version, rerank)




