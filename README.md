# Unsupervised and self-adaptative techniques for cross-domain person re-identification
This repository implements the published algorithm on IEEE Transaction on Information Forensics and Security Journal about Cross-Domain Person Re-Identification.

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

1. Download the available docker image and start the container on iteration mode by following command:
2. After that, download the Torchreid library from [here](https://github.com/KaiyangZhou/deep-person-reid)
3. Go to "deep-person-reid" directory: `cd deep-person-reid/`
4. Execute the following command to meet libraries requirements: `pip install -r requirements.txt`
5. Execute `python setup.py install` to install the library

The other libraries can be installed by `pip` or directly from the respective website


