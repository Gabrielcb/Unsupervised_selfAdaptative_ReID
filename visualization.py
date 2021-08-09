### Made by Gabriel Bertocco ###

import torch
import torchreid

from torch.nn import Module, BatchNorm1d, AdaptiveAvgPool2d
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, ToPILImage, Pad
from torchvision.utils import save_image

from PIL import Image

import numpy as np 
import matplotlib.pyplot as plt

import os
import argparse

from featureExtraction import extractFeatureMaps, extractFeatures
from load_sets_from_datasets import load_dataset

from Backbones import validate


transform_original = Compose([Resize((256, 128), interpolation=3), ToTensor()])
transform = Compose([Resize((256, 128), interpolation=3), ToTensor(), 
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def main(gpu_ids, source, target, model_name, model_path):

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
	print(torch.cuda.device_count())

	cmap = plt.get_cmap('jet')

	if source == "Duke":
		num_source_classes = 702
	elif source == "Market":
		num_source_classes = 751
	elif source == "None":
		num_source_classes = 1000 #ImageNet
	else:
		print("Dataset not implemented yet")
		exit()
	
	model = torchreid.models.build_model(name=model_name, num_classes=num_source_classes, pretrained=True)
	
	model = ResNet50ReID_XAI(model)
	model.load_state_dict(torch.load(model_path))
	
	# Get Parameters of the last BN layer before final embedding
	bn_bias = model.last_bn.bias.unsqueeze(dim=1).unsqueeze(dim=1).detach()
	bn_weights = model.last_bn.weight.unsqueeze(dim=1).unsqueeze(dim=1).detach()
	bn_mean = model.last_bn.running_mean.unsqueeze(dim=1).unsqueeze(dim=1)
	bn_var = model.last_bn.running_var.unsqueeze(dim=1).unsqueeze(dim=1)

	model = model.cuda()
	model = model.eval()

	train_images_target, gallery_images_target, queries_images_target = load_dataset(target)

	all_cameras = np.unique(queries_images_target[:,2])

	correct_per_camera = {key:0 for key in all_cameras}
	failure_per_camera = {key:0 for key in all_cameras}

	correct_query_cases = {}
	correct_gallery_cases = {}

	failure_query_cases = {}
	failure_gallery_cases = {}

	print("Extracting Query Feature Maps ...")
	featmap_queries, fvs_before_bn_queries, fvs_after_bn_queries = extractFeatureMaps(queries_images_target, model, batch_size=500)
	print(featmap_queries.shape, fvs_before_bn_queries.shape, fvs_after_bn_queries.shape)

	print("Extracting Gallery Feature Maps ...")
	featmap_gallery, fvs_before_bn_gallery, fvs_after_bn_gallery = extractFeatureMaps(gallery_images_target, model, batch_size=500)
	print(featmap_gallery.shape, fvs_before_bn_gallery.shape, fvs_after_bn_gallery.shape)

	fvs_norm_queries = fvs_after_bn_queries/torch.norm(fvs_after_bn_queries, dim=1, keepdim=True)
	fvs_norm_galleries = fvs_after_bn_gallery/torch.norm(fvs_after_bn_gallery, dim=1, keepdim=True)

	distances = torch.cdist(fvs_norm_queries, fvs_norm_galleries, p=2.0)
	sorted_distances = torch.argsort(distances, dim=1)

	queries_size = queries_images_target.shape[0]
	total_R01 = 0.0
	total_R05 = 0.0
	total_R10 = 0.0
	total_R20 = 0.0

	topk = 10


	correct_seen_ids = []
	failure_seen_ids = []

	for i in range(queries_size):

		closest_gallery_images = gallery_images_target[sorted_distances[i]]

		query_camera = queries_images_target[i][2]
		gallery_camera = closest_gallery_images[:,2]

		query_pid = queries_images_target[i][1]
		gallery_pid = closest_gallery_images[:,1]

		selected_gallery_idx = np.invert((query_camera == gallery_camera) & (query_pid == gallery_pid))
		cross_camera_closest_gallery_images = closest_gallery_images[selected_gallery_idx]

		cross_camera_pids = cross_camera_closest_gallery_images[:,1]

		if query_pid in cross_camera_pids[:1]:

			if correct_per_camera[query_camera] == 0 and query_pid not in correct_seen_ids:
				correct_query_cases[query_camera] = i
				correct_gallery_cases[query_camera] = sorted_distances[i][selected_gallery_idx][:topk]
				correct_per_camera[query_camera] = 1
				correct_seen_ids.append(query_pid)
			
			total_R01 += 1
			total_R05 += 1
			total_R10 += 1
			total_R20 += 1

		elif query_pid in cross_camera_pids[:5]:
			total_R05 += 1
			total_R10 += 1
			total_R20 += 1

		elif query_pid in cross_camera_pids[:10]:

			if failure_per_camera[query_camera] == 0 and query_pid not in failure_seen_ids:
				failure_query_cases[query_camera] = i
				failure_gallery_cases[query_camera] = sorted_distances[i][selected_gallery_idx][:topk]
				failure_per_camera[query_camera] = 1
				failure_seen_ids.append(query_pid)			

			total_R10 += 1
			total_R20 += 1

		elif query_pid in cross_camera_pids[:20]:
			total_R20 += 1

	R01 = total_R01/queries_size
	R05 = total_R05/queries_size
	R10 = total_R10/queries_size
	R20 = total_R20/queries_size

	print("R1: %.2f" % (100*R01))
	print("R5: %.2f" % (100*R05))
	print("R10: %.2f" % (100*R10))
	print("R20: %.2f" % (100*R20))

	print(correct_query_cases)
	print(correct_gallery_cases)

	h = featmap_queries.shape[2]
	w = featmap_queries.shape[3]

	h_image = 276 # Original height (256) + 5 colored pixels upper and lower + 5 white pixels upper and lower
	w_image = 148 # Original height (128) + 5 colored pixels on right and left + 5 white pixels on right and left

	visualization = ReIDVisualization(bn_mean, bn_var, bn_weights, bn_bias, h, w)

	query_keys = correct_query_cases.keys()

	print("Obtaining Visualizations for R1 correct matches ...")
	for key in query_keys:

		grid = Image.new("RGB", size=((topk+1)*w_image, 1*h_image))

		print("Camera: %s" % key)
		query_idx = correct_query_cases[key]
		imgPIL_query = torchreid.utils.tools.read_image(queries_images_target[query_idx][0])
		imgPIL_query = Resize((256,128))(imgPIL_query)

		# Transformations to create the grid of images
		imgPIL_query_pad_color = Pad(5, fill=(0,0,200))(imgPIL_query) # Blue around query image
		imgPIL_query_pad_for_space = Pad(5, fill=(255, 255, 255))(imgPIL_query_pad_color)
		grid.paste(imgPIL_query_pad_for_space, box=(0,0))

		Fi = featmap_queries[query_idx]
		fv_i = fvs_after_bn_queries[query_idx:query_idx+1]
		#imgPIL_query.save("images/corrects/query_%sTo%s_camera_%s_%s.jpg" % (source, target, key, 
		#																		queries_images_target[query_idx][1]))
		
		gallery_indexes = correct_gallery_cases[key]
		rank = 1
		
		for gal_idx in gallery_indexes:
			imgPIL_gallery = torchreid.utils.tools.read_image(gallery_images_target[gal_idx][0])
						
			Fj = featmap_gallery[gal_idx]
			fv_j = fvs_after_bn_gallery[gal_idx:gal_idx+1]
						
			heatmap = visualization.getHeatmap(Fj, Fi, fv_j, fv_i)
			activated_regions = visualization.blendImageHeatmap(imgPIL_gallery, heatmap)
			#activated_regions.save("images/corrects/heatmap_%sTo%s_camera_%s_%s_%s_%d.jpg" % (source, target, 
			#																		gallery_images_target[gal_idx][2], 
			#																		queries_images_target[query_idx][1], 
			#																		gallery_images_target[gal_idx][1], rank))

			correct_match = queries_images_target[query_idx][1] == gallery_images_target[gal_idx][1]

			if correct_match:
				# Pad correct matches with green 
				activated_regions_pad_by_color_match = Pad(5, fill=(0,190,0))(activated_regions)
				activated_regions_pad_for_space = Pad(5, fill=(255, 255, 255))(activated_regions_pad_by_color_match)
			else:
				# Pad incorrect matches with green 
				activated_regions_pad_by_color_match = Pad(5, fill=(197,0,0))(activated_regions)
				activated_regions_pad_for_space = Pad(5, fill=(255, 255, 255))(activated_regions_pad_by_color_match)

			grid.paste(activated_regions_pad_for_space, box=(rank*w_image, 0))
			rank += 1 

		grid.save("images/%sTo%s_grid_heatmaps_query_%s_camera_%s_corrects.jpg" % (source, target, 
																					queries_images_target[query_idx][1], key))

	query_keys = failure_query_cases.keys()

	print("Obtaining Visualizations for R5 false matches ...")
	for key in query_keys:

		grid = Image.new("RGB", size=((topk+1)*w_image, 1*h_image))

		print("Camera: %s" % key)
		query_idx = failure_query_cases[key]
		imgPIL_query = torchreid.utils.tools.read_image(queries_images_target[query_idx][0])
		imgPIL_query = Resize((256,128))(imgPIL_query)

		# Transformations to create the grid of images
		imgPIL_query_pad_color = Pad(5, fill=(0,0,200))(imgPIL_query) # Blue around query image
		imgPIL_query_pad_for_space = Pad(5, fill=(255, 255, 255))(imgPIL_query_pad_color)
		grid.paste(imgPIL_query_pad_for_space, box=(0,0))

		Fi = featmap_queries[query_idx]
		fv_i = fvs_after_bn_queries[query_idx:query_idx+1]
		#imgPIL_query.save("images/failures/query_%sTo%s_camera_%s_%s.jpg" % (source, target, key, 
		#																		queries_images_target[query_idx][1]))
		
		gallery_indexes = failure_gallery_cases[key]
		rank = 1
		
		for gal_idx in gallery_indexes:
			imgPIL_gallery = torchreid.utils.tools.read_image(gallery_images_target[gal_idx][0])
						
			Fj = featmap_gallery[gal_idx]
			fv_j = fvs_after_bn_gallery[gal_idx:gal_idx+1]
						
			heatmap = visualization.getHeatmap(Fj, Fi, fv_j, fv_i)
			activated_regions = visualization.blendImageHeatmap(imgPIL_gallery, heatmap)
			#activated_regions.save("images/failures/heatmap_%sTo%s_camera_%s_%s_%s_%d.jpg" % (source, target, 
			#																		gallery_images_target[gal_idx][2], 
			#																		queries_images_target[query_idx][1], 
			#																		gallery_images_target[gal_idx][1], rank))

			correct_match = queries_images_target[query_idx][1] == gallery_images_target[gal_idx][1]

			if correct_match:
				# Pad correct matches with green 
				activated_regions_pad_by_color_match = Pad(5, fill=(0,190,0))(activated_regions)
				activated_regions_pad_for_space = Pad(5, fill=(255, 255, 255))(activated_regions_pad_by_color_match)
			else:
				# Pad incorrect matches with green 
				activated_regions_pad_by_color_match = Pad(5, fill=(197,0,0))(activated_regions)
				activated_regions_pad_for_space = Pad(5, fill=(255, 255, 255))(activated_regions_pad_by_color_match)

			grid.paste(activated_regions_pad_for_space, box=(rank*w_image, 0))
			rank += 1

		grid.save("images/%sTo%s_grid_heatmaps_query_%s_camera_%s_failures.jpg" % (source, target, 
							 														queries_images_target[query_idx][1], key))


class ReIDVisualization(object):

    def __init__(self, bn_mean, bn_var, bn_weights, bn_bias, height, width):
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.bn_weights = bn_weights
        self.bn_bias = bn_bias
        self.height = height
        self.width = width
        self.cmap = cmap = plt.get_cmap('jet')

    def getHeatmap(self, FeatureMap_i, FeatureMap_j, fv_i, fv_j):

        normalized_fv_i = torch.norm(fv_i, dim=1)
        normalized_fv_j = torch.norm(fv_j, dim=1)
        Z = normalized_fv_i[0]*normalized_fv_j[0]*self.height*self.width

        response = ((((FeatureMap_i - self.bn_mean)/self.bn_var)*self.bn_weights) + self.bn_bias)*fv_j.T.unsqueeze(dim=1)
        response = response/Z

        heatmap = torch.sum(response, dim=0)

        min_value, max_value = heatmap.min(), heatmap.max()
        heatmap = (heatmap - min_value)/(max_value - min_value)
        heatmap = torch.Tensor(np.moveaxis(self.cmap(heatmap.cpu().numpy())[:,:,:3], -1, 0))
        heatmap = Resize((256,128))(ToPILImage()(heatmap))

        return heatmap

    def blendImageHeatmap(self, img, heatmap, alpha=0.55):
        img = Resize((256,128))(img)
        final_image = Image.blend(img, heatmap, alpha=alpha)
        return final_image

	
## New Model Definition for ResNet50
class ResNet50ReID_XAI(Module):
    
	def __init__(self, model_base):
		super(ResNet50ReID_XAI, self).__init__()


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
		feature_map = self.layer4(x)

		x = self.global_avgpool(feature_map)
		beta_before_bn = x.view(x.size(0), -1)
		beta_after_bn = self.last_bn(beta_before_bn)

		return feature_map, beta_before_bn, beta_after_bn
		#return beta_after_bn 


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Define the UDA parameters')
	parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
	parser.add_argument('--source', type=str, help='Name of source model')
	parser.add_argument('--target', type=str, help='Name of target model')
	parser.add_argument('--model_name', type=str, help='Name of the desired model')
	parser.add_argument('--model_path', type=str, help='Path to the model weights')
	
	args = parser.parse_args()
	gpu_ids = args.gpu_ids
	source = args.source
	target = args.target
	model_name = args.model_name
	model_path = args.model_path
		
	main(gpu_ids, source, target, model_name, model_path)