### Made by Gabriel Bertocco ###
import os
import torch
import numpy as np

from sklearn.cluster import cluster_optics_xi

def GPUOptics(train_fvs, max_dist, version_optics, xi=0.05, min_samples=5):

	num_samples = train_fvs.shape[0]
	dim = train_fvs.shape[1]
	

	train_filename = "train_file_%s.txt" % version_optics
	orderedPoints_filename = "orderedPoints_%s.txt" % version_optics

	np.savetxt(train_filename, train_fvs.numpy())

	# Executing OPTCS algortithm based on GPU and heap
	print('Running OPTICS script ...')
	bash_command = "./optics/goptics %f %d %s %d %d 1 %s" % (max_dist, min_samples, train_filename, 
																	num_samples, dim, orderedPoints_filename)
	os.system(bash_command)
	
	# Load the ordered clusters with the reachability values
	clustered_fvs = np.loadtxt(orderedPoints_filename, skiprows=1)
	read_fvs = clustered_fvs[:, :dim]
	reachabilities = clustered_fvs[:, dim+1]

	#### Obtaining "ordering_" atributtte
	dists = torch.cdist(torch.Tensor(read_fvs), torch.Tensor(train_fvs), p=2)
	fvs_mapping = np.argmin(dists, axis=1).numpy()

	#### Obtaining "predecessor_" and "reachability_" atributttes
	predecessor = []
	reachability = []

	for index in range(num_samples):
		order_position = np.where(fvs_mapping == index)[0][0]
	
		if order_position > 0:
			pred = fvs_mapping[order_position - 1]
		else:
			pred = -1

		predecessor.append(pred)
		reachability.append(reachabilities[order_position])

	predecessor = np.array(predecessor)
	reachability = np.array(reachability)

	labels, clusters = cluster_optics_xi(reachability=reachability, predecessor=predecessor, ordering=fvs_mapping, 
											min_samples=min_samples, xi=xi, predecessor_correction=True)
	
	# Removing files after running the algorithm
	bash_command = "rm -rf %s %s" % (train_filename, orderedPoints_filename) 
	os.system(bash_command)

	return labels
	

def filterClusters(train_images, train_fvs, pseudo_labels, verbose=0):

	selectedIdx = np.array(train_fvs.shape[0]*[False])
	selected_pids = []
	quality_values = []
	
	label_levels = np.unique(pseudo_labels)

	if label_levels[0] == -1:
		i = 1
	else:
		i = 0

	total_number_of_clusters = 0
	total_number_of_removed_clusters = 0	

	for pid in label_levels[i:]:
		cameras, cameras_freqs = np.unique(train_images[pseudo_labels == pid,2], return_counts=True)
		num_cameras = cameras.shape[0]
		#print(train_images[pseudo_labels == pid])
		
		if num_cameras >= 2:
			selectedIdx[pseudo_labels == pid] = True
		else:
			total_number_of_removed_clusters += 1

		total_number_of_clusters += 1

	ratio_of_removed_clusters = total_number_of_removed_clusters/total_number_of_clusters
			
	return selectedIdx, ratio_of_removed_clusters