### Made by Gabriel Bertocco ###
import torch
import time
import torchreid
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.utils.data import Dataset, DataLoader

transform = Compose([Resize((256, 128), interpolation=3), ToTensor(), 
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class sample(Dataset):
    
    def __init__(self, Set):
        self.set = Set        
            
    def __getitem__(self, idx):
        
        sample = self.set[idx]
        imgPIL = torchreid.utils.tools.read_image(sample[0])
        img = torch.stack([transform(imgPIL)])
        return img[0]
                 
    def __len__(self):
        return self.set.shape[0]


@torch.no_grad()
def getFVs(batch, model, gpu_index, multiscale):
    batch_gpu = batch.cuda(gpu_index)
    fv = model(batch_gpu)
    fv_cpu = fv.data.cpu()
    return fv_cpu


def extractFeatures(subset, model, batch_size, gpu_index=0, eval_mode=True, multiscale=False):

    if eval_mode:
        model.eval()
    else:
        model.train()
        
    dataSubset = sample(subset)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()
    subset_fvs = []
    for batch_idx, batch in enumerate(loader):

        fvs = getFVs(batch, model, gpu_index, multiscale)
        if len(subset_fvs) == 0:
            subset_fvs = fvs
        else:
            subset_fvs = torch.cat((subset_fvs, fvs), 0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))

    return subset_fvs

def extractFeatureMaps(subset, model, batch_size=500):
    
    dataSubset = sample(subset)
    loader = DataLoader(dataSubset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    start = time.time()
    initialized = False

    for batch_idx, batch in enumerate(loader):

        with torch.no_grad():
            batch_gpu = batch.cuda()
            feature_maps, beta_before_bn, beta_after_bn = model(batch_gpu)

            feature_maps_cpu = feature_maps.data.cpu()
            beta_before_bn_cpu = beta_before_bn.data.cpu()
            beta_after_bn_cpu = beta_after_bn.data.cpu()

        if not initialized:
            featmaps = feature_maps_cpu
            fvs_before_bn = beta_before_bn_cpu
            fvs_after_bn = beta_after_bn_cpu
            initialized = True
        else:
            featmaps = torch.cat((featmaps, feature_maps_cpu), dim=0)
            fvs_before_bn = torch.cat((fvs_before_bn, beta_before_bn_cpu), dim=0)
            fvs_after_bn = torch.cat((fvs_after_bn, beta_after_bn_cpu), dim=0)
            
    end = time.time()
    print("Features extracted in %.2f seconds" % (end-start))
    return featmaps, fvs_before_bn, fvs_after_bn