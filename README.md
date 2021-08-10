# Unsupervised and self-adaptative techniques for cross-domain person re-identification

The following libraries must be downloaded:

* Torchreid 1.3.2 - this library can be found in this [link](https://github.com/KaiyangZhou/deep-person-reid)
* Scikit-learn 0.24.2
* Numpy 1.17.4
* Pytorch 1.4.0
* Torchvision 0.5.0

Hardware requirements:
* GPU: GTX 1080 Ti
* Driver Version: 465.19.01    
* CUDA Version: 11.3

OBS: Different hardware or library versions might result in different results published on the paper

## Environment setup 

1. Download the Torchreid library from [here](https://github.com/KaiyangZhou/deep-person-reid)
2. Go to "deep-person-reid" directory: `cd deep-person-reid/`
3. Execute the following command to meet libraries requirements: `pip install -r requirements.txt`
4. Execute `python setup.py install` to install the library

The other libraries can be installed by `pip` or directly from the respective website

## Training one model on adaptation scenario

We provide here one example on how to run our pipeline using OSNet backbone on Market to Duke adaptation scenario:
`python main.py --gpu_ids=5,7 --lr=1e-4 --xi_value=0.05 --source=Market --target=Duke --source_model_name=osnet_x1_0 --path_to_save_models=model_checkpoints --path_to_save_metrics=metric_checkpoints --version=example --print_validation_performance=True > log_training_MarketToDuke_osnet.txt`

The model will use gpu 5 to perform clustering and gpu 7 to train update backbone paramenters. The expected log is shown on 
the provided file `log_training_MarketToDuke_osnet.txt`

## Self-Ensembling
One of our main contributions is the algorithm to summarize the different knowledges from different backbones obtained along the training. To do so we 
must save the backnbone checkpoints after an iteration of our pipeline. The checkpoints obtained after the above training are provided on `model_checkpoints`. 
Moreover our algorithm also requires the percentage of reliable data on each iteration which were stored on `metric_checkpoints/reliability_progress_MarketToDuke_osnet_x1_0_example`.  Now we can run the proposed self-Ensembling algorithm:

`python selfEnsembling.py --gpu_ids=7 --source=Market --target=Duke --dir_name=model_checkpoints --model_name=osnet_x1_0 --reliability_path=metric_checkpoints/reliability_progress_MarketToDuke_osnet_x1_0_example --version=MarketToDuke_osnet_x1_0_example.h5 --rerank=False > log_selfEnsembling_MarketToDuke_osnet.txt`

To run our code for other datasets as source or target, you must only change `--source` and `--target` flags. To change the backbone, you must also change 
the name on `--source_model_name` flag. The current available backbones are ResNet50 (`resnet50`), OSNet (`osnet_x1_0`) and DenseNet121 (`densenet121`). 
The pretrained models provided on `pretrained_models` holds knowledge over or Market or Duke, since MSMT17 is used only as target dataset. 

## Ensembling of different backbones
After training the three backbones on the same source -> target scenario, we perform the ensembling of the results by averaging the three distances
(one for each backbone) between an query image and an gallery image. After calculating all averaged distance between each pair of query/gallery the 
metrics are reported. To run the code you must execute:

`python EnsembleEvaluation.py --gpu_ids=7 --source=Market --target=Duke --rerank=False`

It will result on the final metrics reported on the original paper. 




