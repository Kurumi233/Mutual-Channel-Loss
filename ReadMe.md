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
  - Init_lr: 0.1 for all
  - lr_scheduler: MultiStepLR (total-300, milestones-[150, 225], lr_gamma-0.1)
  - ***The random seed is set for reproducibility.***
  
| Model |cnums|cgroups|p|alpha|img_size|feat_dim|seed|Acc@1|
| ----| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|VGG16|[3]|[200]|0.5|1.5| 224->224 |600\*7\*7|None|64.88|
|VGG16|[3]|[200]|0.5|2.0| 224->224 |600\*7\*7|None|66.66|
|VGG16|[3]|[200]|0.5|1.5| 224->224 |600\*7\*7|8|66.17|
|VGG16|[3]|[200]|0.5|2.0| 224->224 |600\*7\*7|8|66.03|
- Using Imagenet pretrained model
  - Init_lr: 0.005 for conv layers, 0.05 for dense layers
  - lr_scheduler: StepLR (total-150, lr_step-30, lr_gamma-0.8)

| Model |cnums|cgroups|p|alpha|img_size|feat_dim|Acc@1|
| ----| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ResNet50  |[10, 11]|[152, 48]|0.4|0.0005| 600->448 | 2048 |86.93|

***PS: the results maybe influenced by random initial, using batch size 64 can be closer to the results of the paper.***



#### Reference:

- The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification (TIP 2020) [DOI](https://doi.org/10.1109/TIP.2020.2973812)

- Official code: https://github.com/PRIS-CV/Mutual-Channel-Loss





