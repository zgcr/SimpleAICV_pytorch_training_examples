   * [My ZhiHu column](#my-zhihu-column)
   * [Update log](#Update-log)
   * [Requirements](#requirements)
   * [How to download my pretrained models](#how-to-download-my-pretrained-models)
   * [How to prepare dataset directory structure for training and testing](#how-to-prepare-dataset-directory-structure-for-training-and-testing)
   * [How to reproduce my experiment results](#how-to-reproduce-my-experiment-results)
   * [How to test my pretrained models](#how-to-test-my-pretrained-models)
   * [How to use a object detection pretrained model to detect a single image?](#how-to-use-a-object-detection-pretrained-model-to-detect-a-single-image)
   * [COCO2017 detection training results](#coco2017-detection-training-results)
      * [RetinaNet](#retinanet)
      * [FCOS](#fcos)
      * [CenterNet(Objects as Points)](#centernetobjects-as-points)
      * [YOLOv3](#YOLOv3)
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
2. Add CenterNet multi scale training method.

# Requirements

Platform:Ubuntu 18.04.4
```
1.torch==1.4.0
2.torchvision==0.5.0
3.python==3.6.9
4.numpy==1.17.0
5.opencv-python==4.1.1.26
6.tqdm==4.46.0
7.thop==0.0.31
8.Cython==0.29.19
9.matplotlib==3.2.1
10.pycocotools==2.0.0
11.apex==0.1
12.DCNV2==0.1
```

Please use the following orders to install apex:
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

How to use DCNv2 with apex mixed precision training opt_level='O1' (for CenterNet:Objects as Points training)?

I trained and tested centernet on 4 RTX2080Ti graphics card, and no errors were reported during the training process and testing process.

Download DCNv2 from here: https://github.com/CharlesShang/DCNv2 (master branch).

Unzip and modify dcn_v2.py:
```
# add:
try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# for class _DCNv2(Function),add:
@amp.float_function
# at the previous line of the "def forward()" and "def backward()"
```
build DCNv2:
```
sudo -s
./make.sh
```
**Attention:**

Please make sure your Python environment is not an Anaconda virtual environment,and your torch/torchvision packages installed by pip.Otherwise, the following mistake may happened:
```
RuntimeError: Not compiled with GPU support
```
Related issue:https://github.com/CharlesShang/DCNv2/issues/82 

# How to download my pretrained models

You can download all my pretrained models from here:
```
https://drive.google.com/drive/folders/1rewWULfXsvE0voA-A_ooTWwadq9lsk3X?usp=sharing
```

If you are in China,you can download from here:
```
链接: https://pan.baidu.com/s/1nrlexTJ1mdBo0zKv1mqTGg
提取码: yph6
```

# How to prepare dataset directory structure for training and testing

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

# How to reproduce my experiment results

If you want to reproduce my experiment result,just enter a category experiments folder,then enter a specific experiment folder.Each experiment folder has it's own config.py and train.py.

If the experiment use nn.parallel mode to train,just run this command:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

CUDA_VISIBLE_DEVICES is used to specify the gpu ids for this training.

If the experiment use nn.DistributedDataParallel mode to train,just run this command:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 20001 train.py
```

Please make sure the nproc_per_node number is correct and master_addr/master_port are different from other experiments.

# How to test my pretrained models

I provide two scripts for testing on COCO2017 and ILSVRC2012.You can find testing codes in public/test_scripts/.I give two example to run testing on COCO2017 and ILSVRC2012.

COCO2017 testing example:
```
# enter in detection_experiments/resnet50_retinanet_coco_distributed_apex_resize667/
./test.sh
```

VOC2007 testing example:
```
# enter in detection_experiments/resnet50_retinanet_voc_distributed_apex_resize667_usecocopre/
./test.sh
```

ILSVRC2012 testing example:
```
# enter in imagenet_experiments/resnet_imagenet_DataParallel_train_example/
./test.sh
```

# How to use a object detection pretrained model to detect a single image?

I provided an example in public/test_scripts/detect_single_image.py.run this command to detect a single image and save the detected image.
```
./detect_single_image.sh
```
You can read the codes in detect_single_image.py to find more details.

# COCO2017 detection training results

Trained on COCO2017_train dataset, tested on COCO2017_val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).
mAR is IoU=0.5:0.95,area=all,maxDets=100,mAR(COCOeval,stats[8]).

You can find more model training details in detection_experiments/experiment_folder/.

## RetinaNet

Paper:https://arxiv.org/abs/1708.02002

For RetinaNet training,I use yolov3 resize method,this method resize=667 has same flops as the resize=400 method proposed in the RetinaNet paper,resize=1000 has same flops as the resize=600 method proposed in the RetinaNet paper,resize=1333 has same flops as the resize=800 method proposed in the RetinaNet paper.

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAR-loss | training-time(hours) |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-RetinaNet | 667 | 20 | 2 RTX2080Ti | yes | no | 12 | 0.305,0.421,0.56 | 17.43 |
| ResNet101-RetinaNet  | 667 | 16 | 2 RTX2080Ti | yes | no | 12 | 0.306,0.420,0.55 | 22.06 |

For ResNet50-RetinaNet resize=1000 training,I use ResNet50-RetinaNet resize=667 trained model(mAP=0.305) as pretrained model parameters to initialize the ResNet50-RetinaNet resize=1000 model.

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAR-loss | training-time(hours) |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-RetinaNet | 1000 | 16 | 4 RTX2080Ti | yes | no | 12 | 0.332,0.458,0.57 | 26.25 |

**Inference time**:

Using one RTX2080Ti to test RetinaNet model inference speed.The test is performed COCO2017_val dataset,compute average per image inference time(ms).Testing num_workers=8.

| Network | resize | batch |per image inference time(ms)|
| --- | --- |  --- |  --- |
| ResNet50-RetinaNet | 667 | 1 | 102 |
| ResNet50-RetinaNet | 667 | 4 | 47 |
| ResNet50-RetinaNet | 667 | 8 | 38 |
| ResNet50-RetinaNet | 667 | 16 | 30 |
| ResNet50-RetinaNet | 667 | 32 | 29 |
| ResNet50-RetinaNet | 1000 | 1 | 205 |
| ResNet50-RetinaNet | 1000 | 4 | 93 |
| ResNet50-RetinaNet | 1000 | 8 | 76 |
| ResNet50-RetinaNet | 1000 | 16 | 64 |

## FCOS

Paper:https://arxiv.org/abs/1904.01355 

For FCOS training,I use yolov3 resize method,this method resize=667 has same flops as the resize=400 method proposed in the FCOS paper,resize=1000 has same flops as the resize=600 method proposed in the FCOS paper,resize=1333 has same flops as the resize=800 method proposed in the FCOS paper.

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAR-loss | training-time(hours) |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-FCOS | 667 | 24 | 2 RTX2080Ti | yes | no | 12 | 0.318,0.452,1.09 | 14.17 |
| ResNet101-FCOS | 667 | 20 | 2 RTX2080Ti | yes | no | 12 | 0.342,0.475,1.07 | 19.20 |

For ResNet50-FCOS resize=1000/ResNet50-FCOS resize=1333 training,I use ResNet50-FCOS resize=667 trained model(mAP=0.318) as pretrained model parameters to initialize the ResNet50-FCOS resize=1000/ResNet50-FCOS resize=1333 model.

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAR-loss | training-time(hours) |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-FCOS | 1000 | 20 | 4 RTX2080Ti | yes | no | 12 | 0.361,0.502,1.10 | 18.92 |
| ResNet50-FCOS | 1333 | 12 | 4 RTX2080Ti | yes | no | 24 | 0.381,0.534,1.03 | 37.73 |

**Inference time**:

Using one RTX2080Ti to test FCOS model inference speed.The test is performed COCO2017_val dataset,compute average per image inference time(ms).Testing num_workers=8.

| Network | resize | batch |per image inference time(ms)|
| --- | --- |  --- |  --- |
| ResNet50-FCOS | 667 | 1 | 90 |
| ResNet50-FCOS | 667 | 4 | 42 |
| ResNet50-FCOS | 667 | 8 | 36 |
| ResNet50-FCOS | 667 | 16 | 32 |
| ResNet50-FCOS | 667 | 32 | 33 |
| ResNet50-FCOS | 1000 | 1 | 192 |
| ResNet50-FCOS | 1000 | 4 | 87 |
| ResNet50-FCOS | 1000 | 8 | 73 |
| ResNet50-FCOS | 1000 | 16 | 64 |

## CenterNet(Objects as Points)

Paper:https://arxiv.org/abs/1904.07850

In CenterNet paper,the author use yolov3 resize method,my resize method is same as yolov3 resize method.

| Network | resize | batch | gpu-num | apex | syncbn | epoch | mAP-mAR-loss | training-time(hours) |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet18DCN-CenterNet | 512 | 128 | 4 RTX2080Ti | yes | no | 140 | 0.248,0.366,1.41 | 55.71 |
| ResNet18DCN-CenterNet-MultiScale | 512 | 96 | 4 RTX2080Ti | yes | no | 140 | 0.266,0.401,1.86 | 57.26 |

**Inference time**:

Using one RTX2080Ti to test CenterNet model inference speed.The test is performed COCO2017_val dataset,compute average per image inference time(ms).Testing num_workers=8.
| Network | resize | batch |per image inference time(ms)|
| --- | --- |  --- |  --- |
| ResNet18DCN-CenterNet | 512 | 1 | 15 |
| ResNet18DCN-CenterNet | 512 | 4 | 9 |
| ResNet18DCN-CenterNet | 512 | 8 | 10 |
| ResNet18DCN-CenterNet | 512 | 16 | 10 |
| ResNet18DCN-CenterNet | 512 | 32 | 10 |
| ResNet18DCN-CenterNet | 512 | 64 | 10 |

## YOLOv3

Paper:https://arxiv.org/abs/1804.02767

**How to use yolov3 anchor clustering method to generate a set of custom anchors for your own dataset?**

I provide a script in public/detection/yolov3_anchor_cluster.py,and I give two examples for generate anchors on COCO2017 and VOC2007+2012 datasets.

If you want to generate anchors for your dataset,just modify the part of input code,get width and height of all annotaion boxes,then use the script to compute anchors.The anchors size will change with different datasets or different input resizes.


# VOC2007+2012 detection training results

Trained on VOC2007 trainval + VOC2012 trainval, tested on VOC2007,using 11-point interpolated AP.

| Network | resize | batch | gpu-num | apex | syncbn | epoch5-mAP-loss | epoch10-mAP-loss | epoch15-mAP-loss | epoch20-mAP-loss |  training-time(hours) |
| --- | --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |
| ResNet50-RetinaNet | 667 | 24 | 2 RTX2080Ti | yes | no | 0.697,0.69 | 0.729,0.50 | 0.741,0.40 | 0.730,0.34 | 3.75 |
| ResNet50-RetinaNet-usecocopre | 667 | 24 | 2 RTX2080Ti | yes | no | 0.797,0.41 | 0.797,0.31 | 0.789,0.26 | 0.784,0.22 | 3.75 |

You can find more model training details in detection_experiments/experiment_folder/.

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