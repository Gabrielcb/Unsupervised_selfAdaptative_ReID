### Made by Gabriel Bertocco ###
import torch
import torchreid
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize

np.random.seed(12)

def distMatrices(a, b):
     
    dot_ab = -2*torch.matmul(a, b.T)
    sum_a = torch.sum(a**2, dim=1)
    sum_b = torch.sum(b**2, dim=1)
    
    repeated_sum_b = sum_b.repeat(sum_a.shape[0], 1)
    sum_ab = sum_a[:,None] + repeated_sum_b
    
    res = dot_ab + sum_ab
    return res

def createTriplets(labelsTargetFvs, pseudo_labels, distFvs, K_batch=2):

    triplets_idx = []
    
    unique_pseudo_labels = np.unique(pseudo_labels)
    num_clusters_only_one_camera = 0

    print("-----------")
    print("Creating Triplets ...")

    for pid in unique_pseudo_labels:

        cameras = np.unique(labelsTargetFvs[pseudo_labels == pid][:,2])
        num_cameras = cameras.shape[0]

        if num_cameras > 1:

            for camid in cameras:

                # Select anchors based on a pid and camera
                anchors = np.logical_and(pseudo_labels == pid, labelsTargetFvs[:,2] == camid)

                # "anchors" is a boolean vector with TRUE where there is an anchorand FALSE where 
                # it is not referring to an anchor
                number_of_anchors = np.sum(anchors)
                anchor_indexes = np.random.choice(np.where(anchors == True)[0], 
                                                    size=min(number_of_anchors, K_batch), 
                                                    replace=False)

                # Select negative to have different pid but same camera
                negatives = np.logical_and(pseudo_labels != pid, labelsTargetFvs[:,2] == camid)


                for anchor_idx in anchor_indexes:

                    d_a_n = distFvs[anchor_idx, negatives]
                    ordered_idx = np.argsort(d_a_n)
                    # ordered_idx is PyTorch tensor. It must be numpy array
                    hardest_negatives = np.where(negatives == True)[0][ordered_idx.numpy()]
                    k_neg = 0

                    for camid_pos in cameras:

                        if camid_pos != camid:

                            # Select positive to have same pid but different camera
                            positives = np.logical_and(pseudo_labels == pid, labelsTargetFvs[:,2] == camid_pos)
        
                            # Selecting top-k hardest positives and negatives
                            d_a_p = distFvs[anchor_idx, positives]
                            k_pos = d_a_p.shape[0]//2 # Get the meadin hardest positive to avoid false positives 
                            ordered_idx = np.argsort(d_a_p)[k_pos:k_pos+1]
                            # ordered_idx is PyTorch tensor. It must be numpy array
                            hardest_positive_idx = np.where(positives == True)[0][ordered_idx.numpy()]
                            hardest_negative_idx = hardest_negatives[k_neg:k_neg+1]

                            if len(hardest_positive_idx) != 0 and len(hardest_negative_idx) != 0:
                                triplets_idx.append([anchor_idx, hardest_positive_idx, hardest_negative_idx])
                              
                            k_neg += 1

     
    return triplets_idx


def getTripletsQuality(triplets_idx, labelsTargetFvs):

    num_correct_pos = 0
    num_correct_neg = 0

    total_pos = 0
    total_neg = 0

    for tri_idx in range(len(triplets_idx)):

        anchor_idx = int(triplets_idx[tri_idx][0])

        for pos_i in range(len(triplets_idx[tri_idx][1])):
            pos_idx = int(triplets_idx[tri_idx][1][pos_i])
            if(labelsTargetFvs[anchor_idx][1] == labelsTargetFvs[pos_idx][1]):
                num_correct_pos += 1 

            total_pos += 1

        for neg_i in range(len(triplets_idx[tri_idx][2])):
            neg_idx = int(triplets_idx[tri_idx][2][neg_i])
            if(labelsTargetFvs[anchor_idx][1] != labelsTargetFvs[neg_idx][1]):
                num_correct_neg += 1

            total_neg += 1

    positive_rate = num_correct_pos/total_pos
    negative_rate = num_correct_neg/total_neg

    print("Percentage of correct positives: %.2f" % positive_rate)
    print("Percentage of correct negatives: %.2f" % negative_rate)

