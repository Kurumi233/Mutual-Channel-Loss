## Mutual-Channel Loss for Fine-Grained Image Classification

Reimplementation of MCLoss on CUB_200_2011 dataset. 

The code of mcloss is integrated into a class.

#### Requirements:

- python 3
- PyTorch 1.3.0
- torchvision

#### Training:

- Extract files to CUB_200_2011, run `gen_train_test.py` to get the label files, which are used by `dataload.py`, or modify the `Dataset` in `train.py` .
- Train: `python train.py` , parameters are set as default.

#### Results:

- The experiment is conduct with 2 RTX 2080Ti GPUs, and the batchsize is set to 32.
- Trained from scratch:
  - Init_lr: 0.05 for all
  - lr_scheduler: MultiStepLR
  
| Model |cnums|cgroups|p|alpha|img_size|feat_dim|Acc@1|
| ----| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|VGG16|[3]|[200]|0.5|1.5| 256->224 |600\*7\*7|67.4|
|VGG16|[5]|[200]|0.5|1.5| 256->224 |1000\*7\*7|66.4|
- Using Imagenet pretrained model
  - Init_lr: 0.005 for conv layers, 0.05 for dense layers
  - lr_scheduler: StepLR

| Model |cnums|cgroups|p|alpha|img_size|feat_dim|Acc@1|
| ----| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ResNet50  |[10, 11]|[152, 48]|0.4|0.0005| 600->448 | 2048 |86.9|

#### Reference:

- The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification (TIP 2020) [DOI](https://doi.org/10.1109/TIP.2020.2973812)

- https://github.com/dongliangchang/Mutual-Channel-Loss





