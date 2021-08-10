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
import argparse
import joblib

from sklearn.metrics import pairwise_distances
from load_sets_from_datasets import load_dataset

from featureExtraction import extractFeatures
from load_sets_from_datasets import load_set_from_market_duke, load_set_from_MSMT17
from torch.backends import cudnn

transform = Compose([Resize((256, 128), interpolation=3), ToTensor(), 
						Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

def main(gpu_ids, source, target, rerank):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
	print(torch.cuda.device_count())

	if len(gpu_ids) > 1:
		gpu_index = 1
	else:
		gpu_index = 0

	print(gpu_index)

	if source == "Duke":
		num_source_classes = 702
	elif source == "Market":
		num_source_classes = 751
	elif source == "None":
		num_source_classes = 1000
	else:
		print("Dataset not implemented yet")
		exit()


	model01 = torchreid.models.build_model(
	    name="resnet50",
	    num_classes=num_source_classes,
	    loss='softmax',
	    pretrained=True
	)

	model02 = torchreid.models.build_model(
	    name="osnet_x1_0",
	    num_classes=num_source_classes,
	    loss='softmax',
	    pretrained=True
	)

	model03 = torchreid.models.build_model(
	    name="densenet121",
	    num_classes=num_source_classes,
	    loss='softmax',
	    pretrained=True
	)


	###============ Loading ResNet50 ============###     
	path_to_source_model01 = "model_%sTo%s_resnet50_example_selfEnsembleLearning.h5" % (source, target)
	
	model01 = ResNet50ReID(model01)
	model01.load_state_dict(torch.load(path_to_source_model01, map_location='cuda:0'))
	
	###============ Loading OSNet ============###
	path_to_source_model02 = "model_%sTo%s_osnet_x1_0_example_selfEnsembleLearning.h5" % (source, target)

	state_dict02 = torch.load(path_to_source_model02, map_location='cuda:0')
	model02 = OSNETReID(model02)
	model02.load_state_dict(state_dict02)

	###============ Loading Denset121 ============###
	path_to_source_model03 = "model_%sTo%s_densenet121_example_selfEnsembleLearning.h5" % (source, target)

	model03 = DenseNet121ReID(model03)
	model03.load_state_dict(torch.load(path_to_source_model03, map_location='cuda:0'))
	
	model01 = model01.cuda(gpu_index)
	model01 = model01.eval()

	model02 = model02.cuda(gpu_index)
	model02 = model02.eval()

	model03 = model03.cuda(gpu_index)
	model03 = model03.eval()

	train_images_target, gallery_images_target, queries_images_target = load_dataset(target)

	
	print("Validating ResNet50 on %s ..." % target)
	distmat_model01, dt_mean_by_query_resnet = validate(queries_images_target, gallery_images_target, model01, 
															rerank=rerank, gpu_index=gpu_index)

	print("Validating OSNet on %s ..." % target)
	distmat_model02, dt_mean_by_query_osnet = validate(queries_images_target, gallery_images_target, model02, 
															rerank=rerank, gpu_index=gpu_index)

	print("Validating DenseNet121 on %s ..." % target)
	distmat_model03, dt_mean_by_query_densenet = validate(queries_images_target, gallery_images_target, model03, 
															rerank=rerank, gpu_index=gpu_index)

	n_models = 3

	t0 = time.time()
	distmat_ensemble = (distmat_model01 + distmat_model02 + distmat_model03)/n_models
	calculateMetrics(distmat_ensemble, queries_images_target, gallery_images_target)
	tf = time.time()
	dt = tf - t0

	dt_mean_by_query_ensemble = dt/queries_images_target.shape[0]
	dt_mean_total = max(dt_mean_by_query_resnet, dt_mean_by_query_osnet, dt_mean_by_query_densenet) + dt_mean_by_query_ensemble

	print("Total mean single query prediction:", dt_mean_total)

		

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

    dt_total_queries = 0
    
    t0 = time.time()
    queries_fvs = extractFeatures(queries, model, 500, gpu_index)
    tf = time.time()
    dt = tf - t0
    dt_total_queries += dt

    gallery_fvs = extractFeatures(gallery, model, 500, gpu_index)

    t0 = time.time()
    queries_fvs = queries_fvs/torch.norm(queries_fvs, dim=1, keepdim=True)
    tf = time.time()
    dt = tf - t0
    dt_total_queries += dt

    gallery_fvs = gallery_fvs/torch.norm(gallery_fvs, dim=1, keepdim=True)

    t0 = time.time()
    distmat = torchreid.metrics.compute_distance_matrix(queries_fvs, gallery_fvs, metric="euclidean")
    distmat = distmat.numpy()
    tf = time.time()
    dt = tf - t0
    dt_total_queries += dt

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torchreid.metrics.compute_distance_matrix(queries_fvs, queries_fvs, metric="euclidean")
        distmat_gg = torchreid.metrics.compute_distance_matrix(gallery_fvs, gallery_fvs, metric="euclidean")
        distmat = torchreid.utils.re_ranking(distmat, distmat_qq, distmat_gg)


    # Compute Ranks
    t0 = time.time()
    ranks=[1, 5, 10, 20]
    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.metrics.evaluate_rank(distmat, queries[:,1], gallery[:,1], 
                                                queries[:,2], gallery[:,2], use_metric_cuhk03=False)
    print('** Results **')
    print('mAP: {:.2%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

    tf = time.time()
    dt = tf - t0
    dt_total_queries += dt

    dt_mean_by_query = dt_total_queries/queries.shape[0]
    print("Mean time to a single query prediction:", dt_mean_by_query)
        
    
    return distmat, dt_mean_by_query

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
	parser.add_argument('--source', type=str, help='Name of source model')
	parser.add_argument('--target', type=str, help='Name of target model')
	parser.add_argument('--rerank', type=str, help='Indicates if Re-Ranking must or not be applied')
	
	args = parser.parse_args()
	gpu_ids = args.gpu_ids
	source = args.source
	target = args.target
	rerank = args.rerank

	if rerank == "True":
		rerank = True
	else:
		rerank = False
	
	main(gpu_ids, source, target, rerank)




