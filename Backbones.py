### Made by Gabriel Bertocco ###
import torch
import torchreid
from featureExtraction import extractFeatures
from torch.nn import Module, BatchNorm1d, AdaptiveAvgPool2d
from torch.nn import functional as F

def getBackbone(source, source_model_name):

	if source == "Duke":
		num_source_classes = 702
	elif source == "Market":
		num_source_classes = 751
	elif source == "None":
		num_source_classes = 1000 #ImageNet
	else:
		print("Dataset not implemented yet")
		exit()
	
	model_source = torchreid.models.build_model(name=source_model_name, num_classes=num_source_classes, pretrained=True)
	
	if source_model_name == 'resnet50':
		model_source = ResNet50ReID(model_source)

		if source == "Market":
			state_dict_meb = torch.load("pretrained_models/resnet50_Market.pth.tar")["state_dict"]
		elif source == "Duke":
			state_dict_meb = torch.load("pretrained_models/resnet50_Duke.pth.tar")["state_dict"]
		else:
			return model_source

		state_dict_torchreid = model_source.state_dict()

		meb_keys = list(state_dict_meb.keys())
		torchreid_keys = list(state_dict_torchreid.keys())
		min_size = min(len(meb_keys), len(torchreid_keys))

		for i in range(min_size):
			if state_dict_meb[meb_keys[i]].shape == state_dict_torchreid[torchreid_keys[i]].shape:
				state_dict_torchreid[torchreid_keys[i]] = state_dict_meb[meb_keys[i]].clone()

		model_source.load_state_dict(state_dict_torchreid)

	elif source_model_name == 'densenet121': 
		model_source = DenseNet121ReID(model_source)
		
		if source == "Market":
			state_dict_meb = torch.load("pretrained_models/densenet121_Market.pth.tar")["state_dict"]
		elif source == "Duke":
			state_dict_meb = torch.load("pretrained_models/densenet121_Duke.pth.tar")["state_dict"]
		else:
			return model_source

		state_dict_torchreid = model_source.state_dict()

		meb_keys = list(state_dict_meb.keys())
		torchreid_keys = list(state_dict_torchreid.keys())
		min_size = min(len(meb_keys), len(torchreid_keys))

		for i in range(min_size):
			if state_dict_meb[meb_keys[i]].shape == state_dict_torchreid[torchreid_keys[i]].shape:
				state_dict_torchreid[torchreid_keys[i]] = state_dict_meb[meb_keys[i]].clone()

		model_source.load_state_dict(state_dict_torchreid)

	elif source_model_name == 'osnet_x1_0': 

		if source == "Market":
			path_to_source_model = "pretrained_models/osnet_Market.pth.tar"
		elif source == "Duke":
			path_to_source_model = "pretrained_models/osnet_Duke.pth.tar"
		else:
			model_source = OSNETReID(model_source)
			return model_source

		state_dict = torch.load(path_to_source_model)["state_dict"]
		model_source.load_state_dict(state_dict)
		model_source = OSNETReID(model_source)

	else:
		print("Model name not defined. Please, selected a valid one.")
		exit()

	return model_source


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