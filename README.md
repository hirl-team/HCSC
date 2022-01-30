
## Introduction
<p align="center">
  <img src="docs/framework.png" /> 
</p>

This is the implementation of *Hierarchical Contrastive Selective Coding* in PyTorch.

## Installation

Install dependencies (python3.7 with pip installed):
```
pip3 install -r requirement.txt
```

If having trouble installing PyTorch, follow the original guidance (https://pytorch.org/).
Notably, the code is tested with cudatoolkit version 10.2.

## Catalog
- [ ] Release 800 epochs models
- [ ] Vision transformer backbone support 

## Model Zoo

We release our pre-trained HCSC models for reproduction:  

| Backbone | Method | Crop | Epoch | Batch size | Lincls top-1 Acc. | KNN top-1 Acc. | url |
|---------------------|-----------------|---------------------|--------------------|--------------------|----------------------|----------------------|-------------------|
| ResNet-50 | HCSC | Single | 200 | 256 | 69.2 | 60.7 | [model](https://hcscpretrained.s3.us-east-2.amazonaws.com/hcsc_200eps.pth) |
| ResNet-50 | HCSC | Multi | 200 | 256 | 73.7 | 66.8 | [model](https://hcscpretrained.s3.us-east-2.amazonaws.com/hcsc_multicrop_200eps.pth) |
| ResNet-50 | HCSC | Single | 400 | 256 | 70.6 | 63.4 | [model](https://hcscpretrained.s3.us-east-2.amazonaws.com/hcsc_400eps.pth) |


## Usage
### Pre-training on ImageNet

Download [ImageNet](https://image-net.org/challenges/LSVRC/2012/) dataset under [ImageNet Folder]. Go to the path "[ImageNet Folder]/val" and use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to build sub-folders.


Single Crop Pre-training:
```
python3 -m torch.distributed.launch --master_port 29212 --nproc_per_node=1 \
pretrain.py [your ImageNet Folder]
```

Multi Crop Pre-training:
```
python3 -m torch.distributed.launch --master_port 29212 --nproc_per_node=1 \
pretrain.py --multicrop [your ImageNet Folder]
```

### Linear Classification on ImageNet

```
python3 eval_lincls_imagenet.py --data [your ImageNet Folder] \
--dist-url tcp://localhost:10205 --world-size 1 --rank 0 \
--pretrained [your pre-trained model (example:out.pth)]
```

### KNN Evaluation on ImageNet

```
python3 -m torch.distributed.launch --master_port 29212 --nproc_per_node=1 eval_knn.py \
--checkpoint_key state_dict \
--pretrained [your pre-trained model] \
--data [your ImageNet Folder]
```

### Semi-supervised Learning on ImageNet

#### 1% of labels:
```
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 29212 eval_semisup.py \
--labels_perc 1 \
--pretrained [your pretrained weights] \
[your ImageNet Folder]
```

#### 10% of labels:
```
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 29212 eval_semisup.py \
--labels_perc 10 \
--pretrained [your pretrained weights] \
[your ImageNet Folder]
```

### Transfer Learning - Classification on VOC / Places205

#### VOC

##### 1. Download the [VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) dataset.

##### 2. Finetune and evaluation on PASCAL VOC:
```
cd voc_cls/ python3 main.py --data [your voc data folder] \
--pretrained [your pretrained weights]
```

#### Places205

##### 1. Download the [Places205](http://places.csail.mit.edu/user/index.php) dataset (resized 256x256 version)

##### 2. Linear Classification on Places205:
```
python3 eval_lincls_places.py --data [your places205 data folder] \
--data-url tcp://localhost:10025 \
--pretrained [your pretrained weights]
```

### Transfer Learning - Object Detection on VOC / COCO

#### 1. Download [VOC](http://places.csail.mit.edu/user/index.php) and [COCO](https://cocodataset.org/#download) Dataset (under ./detection/datasets).

#### 2. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

#### 3. Convert a pre-trained model to the format of detectron2:
```
cd detection
python3 convert-pretrain-to-detectron2.py [your pretrained weight] out.pkl
```

#### 4. Train on PASCAL VOC/COCO:

##### VOC:
```
cd detection
python3 train_net.py --config-file ./configs/pascal_voc_R_50_C4_24k_hcsc.yaml \
--num-gpus 8 MODEL.WEIGHTS out.pkl
```

##### COCO:
```
cd detection
python3 train_net.py --config-file ./configs/coco_R_50_C4_2x_adco.yaml \
--num-gpus 8 MODEL.WEIGHTS out.pkl
```

### Clustering Evaluation on ImageNet
```
python3 eval_clustering.py --dist-url tcp://localhost:10205 \
--multiprocessing-distributed --world-size 1 --rank 0 \
--num-cluster [target num cluster] \
--pretrained [your pretrained model weights] \
[your ImageNet Folder]
```