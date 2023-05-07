# Multi-View Compressed Representation (MVCR)
Source code of "Towards Robust Low-Resource Fine-Tuning with Multi-View Compressed Representation" (ACL23)


## Python requirements
This code is tested on:
- Python 3.6.12
- transformers  3.3.1
- pytorch 1.10.1

  
## Parameters in the code 
* ```DATA_SEED``` Seed used to sample data for low-resource scenario
* ```NUM_SAMPLES``` Number of training and dev data
* ```TASK``` Name of the task e.g. mnli
* ```NUM_DIM_SET``` Number of dimension sets
* ```DIM_SET``` MVCR dimesion set e.g. 128,256,512
* ```LAYER_SET``` Layers that MVCR adds to e.g. 2,12
* ```GPU``` CUDA GPU
* ```SEED``` Runnning seed
* ```RECON_LOSS``` Ratio used for reconstruction loss
* ```DROPOUT``` Dropout rate for all layers
* ```MIXOUT``` Mixout rate
* to run the model on the subsampled datasets, add ```--sample_train``` option.

## Usage
We provide the following sample scripts.


1. To Train MVCR on sentence-level tasks:
```
bash aebert.sh bert-base-uncased DATA_SEED NUM_SAMPLES TASK NUM_DIM_SET DIM_SET LAYER_SET GPU SEED RECON_LOSS DROPOUT MIXOUT AE2TK
```

2. To Train MVCR on token-level tasks:
```
bash panx.sh
```