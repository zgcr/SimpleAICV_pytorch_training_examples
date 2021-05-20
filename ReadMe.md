   * [My ZhiHu column](#my-zhihu-column)
   * [Update log](#Update-log)
   * [Requirements](#requirements)
   * [How to prepare dataset](#how-to-prepare-dataset)
   * [How to download my pretrained models](#how-to-download-my-pretrained-models)
   * [How to reproduce my model](#how-to-reproduce-my-model)
   * [How to inference single image](#how-to-inference-single-image)
   * [COCO2017 detection training results](#coco2017-detection-training-results)
      * [RetinaNet](#retinanet)
      * [FCOS](#fcos)
      * [CenterNet(Objects as Points)](#centernetobjects-as-points)
      * [YOLO series](#YOLO-series)
   * [VOC2007 2012 detection training results](#voc20072012-detection-training-results)
   * [CIFAR100 classification training results](#cifar100-classification-training-results)
   * [ILSVRC2012(ImageNet) classification training results](#ilsvrc2012imagenet-classification-training-results)
      * [Training in nn.parallel mode results](#training-in-nnparallel-mode-results)
      * [Training in nn.DistributedDataParallel mode results](#training-in-nndistributeddataparallel-mode-results)
   * [Citation](#citation)

# My ZhiHu column

https://www.zhihu.com/column/c_1249719688055193600

# Update log

**2020.12.1:**
1. Modify RetinaNet/FCOS loss calculation method.Training time is reduced by 40% and model performance is improved.

**2021.5.18:**
1. All classification/detection/segmentation model have a public train.py and test.py file in tools/.
2. For training and testing, train.info.log and test.info.log files are generated in the work directory respectively.
3. Build repvgg net in simpleAICV/classification/backbones/repvgg.py.

# Requirements

Platform:Ubuntu 18.04

```
python==3.7.7
torch==1.8.0
torchvision==0.9.0
torchaudio==0.8.0
pycocotools==2.0.2
numpy
Cython
matplotlib
opencv-python
tqdm
thop
```

use python -m pip or conda command to install those packages:

```
python -m pip install -r requirement.txt
conda install --yes --file requirements.txt
```

**How to install apex?**

apex needs to be installed separately.Please use the following orders to install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If the above command fails to install apex，you can use the following orders to install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```
Using apex to train can reduce video memory usage by 25%-30%, but the training speed will be slower, the trained model has the same performance as not using apex.

**How to use DCNv2 with apex mixed precision training opt_level='O1' (for CenterNet:Objects as Points training)?**

I write DCNv2 by using torchvision.ops.deform_conv2d function in simpleAICV/detection/models/dcnv2.py. It doesn't need to install DCNv2 in https://github.com/CharlesShang/DCNv2.git, just make sure your torchvision version>=0.9.0.

torchvision.ops.deform_conv2d function can't use apex mixed precision training,so I register this function as a float function in tools/utils.py in build_training_mode function. If you use apex mixed precision training for centernet training,the torchvision.ops.deform_conv2d function actually do single precision float point computation(other layers do half precision float point computation or single precision float point computation due to apex ops rule).

# How to prepare dataset

If you want to reproduce my imagenet pretrained models,you need download ILSVRC2012 dataset,and make sure the folder architecture as follows:
```
ILSVRC2012
|
|-----train----1000 sub classes folders
|
|-----val------1000 sub classes folders
Please make sure the same class has same class folder name in train and val folders.
```

If you want to reproduce my COCO pretrained models,you need download COCO2017 dataset,and make sure the folder architecture as follows:
```
COCO2017
|
|-----annotations----all label jsons
|                 
|                |----train2017
|----images------|----val2017
                 |----test2017
```

If you want to reproduce my VOC pretrained models,you need download VOC2007+VOC2012 dataset,and make sure the folder architecture as follows:
```
VOCdataset
|                 |----Annotations
|                 |----ImageSets
|----VOC2007------|----JPEGImages
|                 |----SegmentationClass
|                 |----SegmentationObject
|        
|                 |----Annotations
|                 |----ImageSets
|----VOC2012------|----JPEGImages
|                 |----SegmentationClass
|                 |----SegmentationObject
```

# How to download my pretrained models

You can download all my pretrained models from here:
```
https://drive.google.com/drive/folders/1t8vmuxy_rTNczJo_Ej5zFd84fFMYap-I?usp=sharing
```

If you are in China,you can download from here:
```
链接: https://pan.baidu.com/s/1leeoHAUZtnxc9ing38E3Nw
提取码: 4epf
```

# How to reproduce my model

If you want to reproduce my model,you need enter a training folder directory,then run train.sh and test.sh.

For example,you can enter classification_training/imagenet/resnet_vovnet_darknet_example.
If you want to train this model,run train.sh:
```
# DataParallel mode config.distributed=False
CUDA_VISIBLE_DEVICES=0,1 python ../../../tools/train_classification_model.py --work-dir ./
# DistributedDataParallel mode config.distributed=True
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 20001 ../../../tools/train_classification_model.py --work-dir ./
```

CUDA_VISIBLE_DEVICES is used to specify the gpu ids for this training.Please make sure the nproc_per_node number is correct and master_addr/master_port are different from other experiments.You can modify training super parameters in train_config.py.

if you want to test this model,run test.sh:
```
# DataParallel mode config.distributed=False
CUDA_VISIBLE_DEVICES=0,1 python ../../../tools/test_classification_model.py --work-dir ./
# DistributedDataParallel mode config.distributed=True
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 20001 ../../../tools/test_classification_model.py --work-dir ./
```

# How to inference single image

I provide classification and detection scripts for testing single image in ./inference_demo/.

classification testing example:
```
./run_classify_single_image.sh
```

detection testing example:
```
./run_detect_single_image.sh
```

# COCO2017 detection training results

Trained on COCO2017_train dataset, tested on COCO2017_val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).
mAP50 is IoU=0.5,area=all,maxDets=100,mAP(COCOeval,stats[1]).
mAP75 is IoU=0.75,area=all,maxDets=100,mAP(COCOeval,stats[2]).

You can find more model training details in detection_training/.

## RetinaNet

Paper:https://arxiv.org/abs/1708.02002

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAP50-mAP75 |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-RetinaNet | RetinaStyleResize-400 | 32 | 2 RTX3090 | yes | no | 12 | 0.321,0.482,0.340 |
| ResNet50-RetinaNet | RetinaStyleResize-800 | 8 | 2 RTX3090 | yes | no | 12 | 0.355,0.526,0.380 |

## FCOS

Paper:https://arxiv.org/abs/1904.01355 

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAP50-mAP75 |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-FCOS | RetinaStyleResize-400 | 32 | 2 RTX3090 | yes | no | 12 | 0.346,0.527,0.366 |
| ResNet50-FCOS | RetinaStyleResize-800 | 8 | 2 RTX3090 | yes | no | 12 | 0.379,0.562,0.410 |

## CenterNet(Objects as Points)

Paper:https://arxiv.org/abs/1904.07850

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAP50-mAP75 |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet18DCNv2-CenterNet | YoloStyleResize-512 | 128 | 2 RTX3090 | yes | no | 140 | |

## YOLO series

Paper:https://arxiv.org/abs/1804.02767

**How to use yolov3 anchor clustering method to generate a set of custom anchors for your own dataset?**

I provide a script in simpleAICV/detection/yolov3_anchor_cluster.py,and I give two examples for generate anchors on COCO2017 and VOC2007+2012 datasets.If you want to generate anchors for your dataset,just modify the part of input code,get width and height of all annotaion boxes,then use the script to compute anchors.The anchors size will change with different datasets or different input resizes.

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAP50-mAP75 |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| YOLOv3backbone-YOLOv4loss | YoloStyleResize-416 | 128 | 2 RTX3090 | yes | no | 500 | |


# VOC2007+2012 detection training results

Trained on VOC2007 trainval + VOC2012 trainval, tested on VOC2007,using 11-point interpolated AP.

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-RetinaNet | RetinaStyleResize-400 | 32 | 2 RTX3090 | yes | no | 12 | 0.769 |

# CIFAR100 classification training results

Training in nn.parallel mode result:

| Network       | gpu-num | warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- |  --- | 
| ResNet-18     | 1 RTX2080Ti | no | multistep | 200 | 21.59 | 
| ResNet-34     | 1 RTX2080Ti | no | multistep | 200 | 21.16 | 
| ResNet-50     | 1 RTX2080Ti | no | multistep | 200 | 22.12 | 
| ResNet-101    | 1 RTX2080Ti | no | multistep | 200 | 19.84 | 
| ResNet-152    | 1 RTX2080Ti | no | multistep | 200 | 19.01 | 

You can find more model training details in cifar100_experiments/resnet50cifar/.

# ILSVRC2012(ImageNet) classification training results

##  Training in nn.parallel mode results

| Network       | gpu-num | warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- |  --- | 
| ResNet-18     | 4 RTX2080Ti | no | multistep | 100 | 29.684 | 
| ResNet-34-half     | 4 RTX2080Ti | no | multistep | 100 | 32.528 | 
| ResNet-34     | 4 RTX2080Ti | no | multistep | 100 | 26.264 | 
| ResNet-50-half     | 4 RTX2080Ti | no | multistep | 100 | 27.934 | 
| ResNet-50     | 4 RTX2080Ti | no | multistep | 100 | 23.488 | 
| ResNet-101    | 4 RTX2080Ti | no | multistep | 100 | 22.276 | 
| ResNet-152    | 8 RTX2080Ti | no | multistep | 100 | 21.436 |
| EfficientNet-b0    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 24.492 |
| EfficientNet-b1    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 23.092 |
| EfficientNet-b2    | 8 RTX2080Ti | yes,5 epochs | consine | 100 | 22.224 |
| EfficientNet-b3    | 8 RTX2080Ti | yes,5 epochs | consine | 100 | 21.884 |
| DarkNet-19  | 4 RTX2080Ti | no | multistep | 100 | 26.132 | 
| DarkNet-53  | 4 RTX2080Ti | no | multistep | 100 | 22.992 | 
| VovNet-19-slim-depthwise-se  | 4 RTX2080Ti | no | multistep | 100 | 33.276 | 
| VovNet-19-slim-se  | 4 RTX2080Ti | no | multistep | 100 | 30.646 | 
| VovNet-19-se  | 4 RTX2080Ti | no | multistep | 100 | 25.364 | 
| VovNet-39-se  | 4 RTX2080Ti | no | multistep | 100 | 22.662 | 
| VovNet-57-se  | 4 RTX2080Ti | no | multistep | 100 | 22.014 | 
| VovNet-99-se  | 8 RTX2080Ti | no | multistep | 100 | 21.608 | 
| RegNetY-200MF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 29.904 |
| RegNetY-400MF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 26.210 |
| RegNetY-600MF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 25.276 |
| RegNetY-800MF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 24.006 |
| RegNetY-1.6GF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 22.692 |
| RegNetY-3.2GF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 21.092 |
| RegNetY-4.0GF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 21.684 |
| RegNetY-6.4GF    | 4 RTX2080Ti | yes,5 epochs | consine | 100 | 21.230 |


All nets are trained by input_size=224x224 except DarkNet(input size 256x256) and EfficientNet.

For training resnet50 with batch_size=256,you need at least 4 2080ti gpus,and need about three or four days.

## Training in nn.DistributedDataParallel mode results

| Network       | gpu-num | sync-BN |warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- |  --- |  --- | 
| ResNet-50     | 4 RTX2080Ti | no | no | multistep | 100 | 23.72 |
| ResNet-50     | 4 RTX2080Ti | yes | no | multistep | 100 | 25.44 |  

You can find more model training details in imagenet_experiments/experiment_folder/.

# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={pytorch-ImageNet-CIFAR-COCO-VOC-training},
 author={zgcr},
 year={2020}
}
```