### Made by Gabriel Bertocco ###

import torch
import torchreid
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, RandomErasing, RandomHorizontalFlip
from torch import nn

import os
import copy

from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import argparse
import joblib

from featureExtraction import extractFeatures
from load_sets_from_datasets import load_set_from_market_duke, load_set_from_MSMT17, load_dataset
from torch.backends import cudnn

from Triplet import createTriplets, distMatrices, getTripletsQuality
from Clustering import GPUOptics, filterClusters
from Backbones import getBackbone, validate

transform = Compose([RandomHorizontalFlip(), Resize((256, 128), interpolation=3), ToTensor(), 
						Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), RandomErasing()])

np.random.seed(12)
torch.manual_seed(12)
cudnn.deterministic = True

def main(gpu_ids, current_lr, xi_value, source, target, source_model_name, dir_to_save, 
											dir_to_save_metrics, version, print_validation_performance):


	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
	print(torch.cuda.device_count())

	if len(gpu_ids) > 1:
		gpu_index = 1
	else:
		gpu_index = 0

	model_source = getBackbone(source, source_model_name)
	model_source = model_source.cuda(gpu_index)
	model_source = model_source.eval()

	train_images_target, gallery_images_target, queries_images_target = load_dataset(target)

	print("Training Size:", train_images_target.shape)
	print("Gallery Size:", gallery_images_target.shape)
	print("Query Size:", queries_images_target.shape)
	
	print("Validating on %s ..." % target)
	validate(queries_images_target, gallery_images_target, model_source, gpu_index=gpu_index)
	
	cmc_progress = []
	mAP_progress = []
	reliable_data_progress = []
	percentage_of_removed_clusters_along_training = []

		
	version_optics = source + "To" + target + "_" + source_model_name + "_" + version 

	verbose = 0
	margin = 0.3

	batch_size = 30
	K_batch = 2
	number_of_iterations = 50 #51

	total_clustering_time = 0.0
	total_finetuning_time = 0.0
	total_iteration_time = 0.0

	t0_pipeline = time.time()
	for pipeline_iter in range(number_of_iterations):

		t0_iteration = time.time()

		print("###============ Iteration number %d/%d ============###" % (pipeline_iter+1, number_of_iterations))
		train_fvs_target = extractFeatures(train_images_target, model_source, 500, gpu_index=gpu_index)
		
		# removing duplicates
		train_fvs_target, train_images_target = removeDuplicates(train_fvs_target, train_images_target) 
		train_fvs_target = train_fvs_target/torch.norm(train_fvs_target, dim=1, keepdim=True)

		distance_matrix = torch.cdist(train_fvs_target, train_fvs_target, p=2.0)
		max_distance = distance_matrix.max()**2

		t0_clustering = time.time()
		proposed_labels = GPUOptics(train_fvs_target, max_distance, version_optics, xi=xi_value, min_samples=5)		
		selectedIdx, ratio_of_removed_clusters = filterClusters(train_images_target, train_fvs_target, proposed_labels, verbose=0)
		percentage_of_removed_clusters_along_training.append(ratio_of_removed_clusters)
		tf_clustering = time.time()

		print("%.2f clusters with one camera were removed" % ratio_of_removed_clusters)

		dt_clustering = tf_clustering - t0_clustering
		total_clustering_time += dt_clustering
		print("Clustering and Filtering time: %.2f seconds" % dt_clustering)
		
		selectedTargetFvs = train_fvs_target[selectedIdx]
		labelsTargetFvs = train_images_target[selectedIdx]
		pseudo_labels = proposed_labels[selectedIdx]

		ratio_of_reliable_data = np.count_nonzero(selectedIdx == True)/selectedIdx.shape[0]
		print("Ratio of reliable data from target: %f" % ratio_of_reliable_data)
		
		if pipeline_iter == 30:
			current_lr = current_lr/10

		adam = torch.optim.Adam(model_source.parameters(), lr=current_lr, weight_decay=0.0)

		t0_finetuning = time.time()
		model_source = finetunning(model_source, labelsTargetFvs, pseudo_labels, 
									gpu_index, adam, queries_images_target, gallery_images_target, batch_size=30, 
									margin=0.3)
		tf_finetuning = time.time()

		dt_finetunning = tf_finetuning - t0_finetuning
		total_finetuning_time += dt_finetunning
		print("Finetuning time: %.2f seconds" % dt_finetunning)
		
		if print_validation_performance:
			cmc, mAP = validate(queries_images_target, gallery_images_target, model_source, gpu_index=gpu_index)
			cmc_progress.append(cmc)
			mAP_progress.append(mAP)

			joblib.dump(cmc_progress, "%s/CMC_%s_%s_%s" % (dir_to_save_metrics, source + "To" + target, source_model_name, version))
			joblib.dump(mAP_progress, "%s/mAP_%s_%s_%s" % (dir_to_save_metrics, source + "To" + target, source_model_name, version))

		reliable_data_progress.append(ratio_of_reliable_data)
		joblib.dump(reliable_data_progress, "%s/reliability_progress_%s_%s_%s" % (dir_to_save_metrics, source + "To" + target, 
																					source_model_name, version))
		joblib.dump(percentage_of_removed_clusters_along_training, "%s/percentage_cluster_remotion_%s_%s_%s" % (dir_to_save_metrics, 
																					source + "To" + target, 
																					source_model_name, version))
		
		torch.save(model_source.state_dict(), "%s/model_%d_%s_%s_%s.h5" % (dir_to_save, pipeline_iter, 
																			source + "To" + target,
																			source_model_name, version))

		tf_iteration = time.time()
		dt_iteration = tf_iteration - t0_iteration
		total_iteration_time += dt_iteration
		print("Iteration time: %.2f seconds" % dt_iteration)

	tf_pipeline = time.time()

	total_adaptation_time = tf_pipeline - t0_pipeline

	mean_clustering_time = total_clustering_time/number_of_iterations
	mean_finetuning_time = total_finetuning_time/number_of_iterations
	mean_iteration_time = total_iteration_time/number_of_iterations

	print("Mean Time to cluster features and filter clusters: %.2f seconds" % mean_clustering_time)
	print("Mean Time to finetune model on an iteration: %.2f seconds" % mean_finetuning_time)
	print("Mean Time to take one iteration: %.2f seconds" % mean_iteration_time)
	print("Total Adaptation time: %.2f seconds" % total_adaptation_time)

	validate(queries_images_target, gallery_images_target, model_source, gpu_index=gpu_index)

	
		

def finetunning(model_source, labelsTargetFvs, pseudo_labels, gpu_index, optmizer, queries_images_target, gallery_images_target,
												batch_size=30, margin=0.3, verbose=0):

	trained_models = []
	number_of_epoch_finetunning = 0	
	fine_tuning = True

	unique_pseudo_labels = np.unique(pseudo_labels)
	number_of_classes = unique_pseudo_labels.shape[0]

	for number_of_epoch_finetunning in range(5):
	
		print("#======= Fine-tuning on epoch %d ======= #" % number_of_epoch_finetunning)

		model_source.eval()
		selectedTargetFvs = extractFeatures(labelsTargetFvs, model_source, 500, gpu_index=gpu_index)
		selectedTargetFvs = selectedTargetFvs/torch.norm(selectedTargetFvs, dim=1, keepdim=True)

		### Creating Triplets and updating model ###
		distFvs = distMatrices(selectedTargetFvs, selectedTargetFvs)
		triplets_idx = createTriplets(labelsTargetFvs, pseudo_labels, distFvs)

		print("Number of classes: %d" % number_of_classes)

		datasetTriplets = sampleTriplets(triplets_idx, labelsTargetFvs)
		TripletsLoader = DataLoader(datasetTriplets, batch_size=batch_size, num_workers=8, 
										pin_memory=True, shuffle=True, collate_fn=collate_fn)

		model_source.train()

		num_iter = 0
		total_loss = 0

		for batch_idx, batch in enumerate(TripletsLoader):

			imgs_triplets = batch.cuda(gpu_index)
			
			batch_fvs = model_source(imgs_triplets)
			batch_fvs = batch_fvs/torch.norm(batch_fvs, dim=1, keepdim=True)

			anchors = batch_fvs[::3]
			positives = batch_fvs[1::3]
			negatives = batch_fvs[2::3]

			loss = TripletLoss(anchors, positives, negatives, margin=margin)
			total_loss += loss

			if verbose >= 1:
				print("#======= For batch %d =======#" % batch_idx)
				print("Loss Value: %f" % loss.item())

			optmizer.zero_grad()
			loss.backward()
			optmizer.step() 

			num_iter += 1

		mean_loss = total_loss/num_iter
		print("Final Loss Value on Epoch: %f" % mean_loss)

	model_source.eval()
	return model_source

def removeDuplicates(train_fvs, train_images):

	number_of_samples_before_removing = train_fvs.shape[0]
	train_fvs, mapping = torch.unique(train_fvs, dim=0, return_inverse=True)
	number_of_samples_after_removing = train_fvs.shape[0]

	train_images_no_duplicated = train_fvs.shape[0]*[[]]

	for i in range(train_images.shape[0]):
		train_images_no_duplicated[mapping[i]] = train_images[i].copy()

	train_images = np.array(train_images_no_duplicated).copy()
	return train_fvs, train_images

def collate_fn(TripletsBatch):
    return torch.cat(TripletsBatch, dim=0)

class sampleTriplets(Dataset):
    
    def __init__(self, triplets, images_names):
        self.triplets = triplets
        self.images_name = images_names
        
    def __getitem__(self, idx):
       
        anchor_idx = int(self.triplets[idx][0])
        anchor_name = self.images_name[anchor_idx][0]
        imgPIL = torchreid.utils.tools.read_image(anchor_name)
        img_anchor = torch.stack([transform(imgPIL)])
       
        img_idx = int(self.triplets[idx][1][0])
        pos_name = self.images_name[img_idx][0]
        imgPIL = torchreid.utils.tools.read_image(pos_name)
        img_pos = torch.stack([transform(imgPIL)])
            
        img_idx = int(self.triplets[idx][2][0])
        neg_name = self.images_name[img_idx][0]
        imgPIL = torchreid.utils.tools.read_image(neg_name)
        img_neg = torch.stack([transform(imgPIL)])

        return torch.cat((img_anchor, img_pos, img_neg), dim=0)
                
        
    def __len__(self):
        return len(self.triplets)


def TripletLoss(anchors, positives, negatives, margin=0.1):
    dist_anchor_pos = torch.sum((anchors - positives)**2, dim=1)
    dist_anchor_neg = torch.sum((anchors - negatives)**2, dim=1)   
    
    distances = dist_anchor_pos - dist_anchor_neg + margin
    distances = torch.clamp(distances, min=0)
    loss = torch.mean(distances)
    return loss


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define the UDA parameters')
	
	parser.add_argument('--gpu_ids', type=str, default="7", help='GPU IDs')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
	parser.add_argument('--xi_value', type=float, default=0.05, help='ki-value on OPTICS Clustering')
	parser.add_argument('--source', type=str, help='Name of source dataset')
	parser.add_argument('--target', type=str, help='Name of target dataset')
	parser.add_argument('--source_model_name', type=str, help='Name of the model')
	parser.add_argument('--path_to_save_models', type=str, help='Path to save models')
	parser.add_argument('--path_to_save_metrics', type=str, help='Path to save metrics (mAP, CMC, ...)')
	parser.add_argument('--version', type=str, help='Name of current version')
	parser.add_argument('--print_validation_performance', type=str, help='If the model must print performance on validation after each iteration')
	
	args = parser.parse_args()

	gpu_ids = args.gpu_ids
	current_lr = args.lr
	xi_value = args.xi_value
	source = args.source
	target = args.target
	source_model_name = args.source_model_name
	dir_to_save = args.path_to_save_models
	dir_to_save_metrics = args.path_to_save_metrics
	version = args.version
	print_validation_performance = args.print_validation_performance

	if print_validation_performance == "True":
		print_validation_performance = True
	else:
		print_validation_performance = False

	main(gpu_ids, current_lr, xi_value, source, target, source_model_name, 
		dir_to_save, dir_to_save_metrics, version, print_validation_performance)




