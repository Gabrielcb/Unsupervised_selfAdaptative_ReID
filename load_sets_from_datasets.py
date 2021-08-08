### Made by Gabriel Bertocco ###
import os
import numpy as np

## Load target dataset
def load_dataset(dataset_name):
	
	if dataset_name == "Market":
	
		train_images = load_set_from_market_duke("/home/gbertocco/Doctorate/dejavu/reid-data/market1501/Market-1501-v15.09.15/bounding_box_train")
		gallery_images = load_set_from_market_duke("/home/gbertocco/Doctorate/dejavu/reid-data/market1501/Market-1501-v15.09.15/bounding_box_test")
		queries_images = load_set_from_market_duke("/home/gbertocco/Doctorate/dejavu/reid-data/market1501/Market-1501-v15.09.15/query")

	elif dataset_name == "Duke":

		train_images = load_set_from_market_duke("/home/gbertocco/Doctorate/dejavu/reid-data/dukemtmc/DukeMTMC-reID/bounding_box_train")
		gallery_images = load_set_from_market_duke("/home/gbertocco/Doctorate/dejavu/reid-data/dukemtmc/DukeMTMC-reID/bounding_box_test")
		queries_images = load_set_from_market_duke("/home/gbertocco/Doctorate/dejavu/reid-data/dukemtmc/DukeMTMC-reID/query")

	elif dataset_name == "MSMT17":

		base_name_train = "/home/gbertocco/Doctorate/dejavu/reid-data/MSMT17_V2/mask_train_v2"
		# list_train_uda.txt was created by me fusing list_train.txt and list_val.txt to get a unique training set
		train_images = load_set_from_MSMT17("/home/gbertocco/Doctorate/dejavu/reid-data/MSMT17_V2/list_train_uda.txt", base_name_train)

		base_name_test = "/home/gbertocco/Doctorate/dejavu/reid-data/MSMT17_V2/mask_test_v2"
		gallery_images = load_set_from_MSMT17("/home/gbertocco/Doctorate/dejavu/reid-data/MSMT17_V2/list_gallery.txt", base_name_test)
		queries_images = load_set_from_MSMT17("/home/gbertocco/Doctorate/dejavu/reid-data/MSMT17_V2/list_query.txt", base_name_test)


	return train_images, gallery_images, queries_images
	
def load_set_from_market_duke(directory):
	
	images_names = []
	for filename in os.listdir(directory):
	    if filename.endswith(".jpg"):
	        camid = int(filename.split("_")[1][1])
	        pid = int(filename.split("_")[0])
	        if(pid != -1):
	            img_path = os.path.join(directory, filename)
	            images_names.append([img_path, pid, camid])
	            
	images_names = np.array(images_names)

	return images_names

def load_set_from_MSMT17(PATH, base_name):
	
	images_names = []
	train_file = open(PATH, "r")
	for line in train_file.readlines():
		img_name, pid_name = line.split(" ")

		pid = int(pid_name[:-1])
		camid = img_name.split("_")[2]
		
		img_path = os.path.join(base_name, img_name)
		images_names.append([img_path, pid, camid])

	images_names = np.array(images_names)
	return images_names