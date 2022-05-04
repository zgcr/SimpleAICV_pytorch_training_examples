- [My ZhiHu column](#my-zhihu-column)
- [Environments](#environments)
- [Prepare datasets](#prepare-datasets)
- [Download my pretrained models](#download-my-pretrained-models)
- [Train and test model](#train-and-test-model)
- [Classification training results](#classification-training-results)
  - [ILSVRC2012(ImageNet) training results](#ilsvrc2012imagenet-training-results)
  - [CIFAR100 training results](#cifar100-training-results)
- [Detection training results](#detection-training-results)
  - [COCO2017 training results](#coco2017-training-results)
  - [VOC2007 and VOC2012 training results](#voc2007-and-voc2012-training-results)
- [Distillation training results](#distillation-training-results)
  - [ImageNet training results](#imagenet-training-results)
- [Citation](#citation)


# My ZhiHu column

https://www.zhihu.com/column/c_1249719688055193600

# Environments

**This repository only support one server one gpu card/one server multi gpu cards.**

**environments:**
Ubuntu 20.04.3 LTS,30 core AMD EPYC 7543 32-Core Processor, 2*RTX A5000, Python Version:3.8, CUDA Version:11.3

Please make sure your Python version>=3.7.
**Use pip or conda to install those Packages:**
```
torch==1.10.0
torchvision==0.11.1
torchaudio==0.10.0
onnx==1.11.0
onnx-simplifier==0.3.6
numpy
Cython
pycocotools
opencv-python
tqdm
thop
yapf
apex
```

**How to install apex?**

apex needs to be installed separately.For torch1.10,modify apex/apex/amp/utils.py:
```
if cached_x.grad_fn.next_functions[1][0].variable is not x:
```
to
```
if cached_x.grad_fn.next_functions[0][0].variable is not x:
```

Then use the following orders to install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```
Using apex to train can reduce video memory usage by 25%-30%, but the training speed will be slower, the trained model has the same performance as not using apex.

# Prepare datasets

If you want to reproduce my imagenet pretrained models,you need download ILSVRC2012 dataset,and make sure the folder architecture as follows:
```
ILSVRC2012
|
|-----train----1000 sub classes folders
|-----val------1000 sub classes folders
Please make sure the same class has same class folder name in train and val folders.
```

If you want to reproduce my cifar100 pretrained models,you need download cifar100 dataset,and make sure the folder architecture as follows:
```
CIFAR100
|
|-----train unzip from cifar-100-python.tar.gz
|-----test  unzip from cifar-100-python.tar.gz
|-----meta  unzip from cifar-100-python.tar.gz
```

If you want to reproduce my COCO pretrained models,you need download COCO2017 dataset,and make sure the folder architecture as follows:
```
COCO2017
|
|-----annotations----all .json file (label file)
|                 
|                |----train2017
|----images------|----val2017
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

# Download my pretrained models

You can download all my pretrained models from google drive or BAIDUWANGPAN:
```
https://drive.google.com/drive/folders/1oif1oma3BvJ54bEB_487U8mmbToNI4Jh?usp=sharing

链接：https://pan.baidu.com/s/1IN81YQWkfVGq2bg6IhFztw 
提取码：ruzk
```

# Train and test model

If you want to train or test model,you need enter a training folder directory,then run train.sh and test.sh.

For example,you can enter classification_training/imagenet/resnet50.
If you want to train this model from scratch,please delete checkpoints and log folders first,then run train.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../../tools/train_classification_model.py --work-dir ./
```

CUDA_VISIBLE_DEVICES is used to specify the gpu ids for this training.Please make sure the number of nproc_per_node equal to the number of gpu cards.
Make sure master_addr/master_port are unique for each training.

if you want to test this model,you need have a pretrained model first,modify trained_model_path in test_config.py,then run test.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../../tools/test_classification_model.py --work-dir ./
```
Also, You can modify super parameters in train_config.py/test_config.py.

# Classification training results

## ILSVRC2012(ImageNet) training results

| Network              | macs     | params      | input size | gpu num      | batch     | warm up | lr decay  | apex | syncbn | epochs | Top-1  |
| -------------        | -------- | ----------- | ---------- | ------------ | --------- | ------- | --------  | ---- | ------ | ------ | ------ |
| ResNet18             | 1.819G   | 11.690M     | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 70.490 |
| ResNet34half         | 949.323M | 5.585M      | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 67.690 |
| ResNet34             | 3.671G   | 21.798M     | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 73.950 |
| ResNet50half         | 1.063G   | 6.918M      | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 72.048 |
| ResNet50             | 4.112G   | 25.557M     | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 76.334 |
| ResNet101            | 7.834G   | 44.549M     | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 77.716 |
| ResNet152            | 11.559G  | 60.193M     | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 78.318 |
| ResNet50-200epoch    | 4.112G   | 25.557M     | 224x224    | 2 RTX A5000  | 256       | 5       | cosinelr  | True | False  | 200    | 77.326 |
| ResNet50-autoaugment | 4.112G   | 25.557M     | 224x224    | 2 RTX A5000  | 256       | 5       | cosinelr  | True | False  | 200    | 77.692 |
| ResNet50-randaugment | 4.112G   | 25.557M     | 224x224    | 2 RTX A5000  | 256       | 5       | cosinelr  | True | False  | 200    | 77.578 |
| DarkNetTiny          | 412.537M | 2.087M      | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 54.720 |
| DarkNet19            | 3.663G   | 20.842M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 73.830 |
| DarkNet53            | 9.322G   | 41.610M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 76.796 |
| Yolov4CspDarkNetTiny | 977.589M | 4.143M      | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 64.340 |
| Yolov4CspDarkNet53   | 6.584G   | 27.642M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 77.418 |
| Yolov5nBackbone      | 205.613M | 937.480K    | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 55.474 |
| Yolov5sBackbone      | 759.354M | 3.225M      | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 66.486 |
| Yolov5mBackbone      | 2.230G   | 7.556M      | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 72.090 |
| Yolov5lBackbone      | 4.932G   | 14.315M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 73.186 |
| Yolov5xBackbone      | 9.243G   | 23.961M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 73.618 |
| YoloxnBackbone       | 104.508M | 716.968K    | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 57.350 |
| YoloxtBackbone       | 504.979M | 2.757M      | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 66.246 |
| YoloxsBackbone       | 876.729M | 4.726M      | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 69.092 |
| YoloxmBackbone       | 2.683G   | 13.122M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 72.378 |
| YoloxlBackbone       | 6.072G   | 28.101M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 73.976 |
| YoloxxBackbone       | 11.548G  | 51.583M     | 256x256    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 74.484 |

You can find more model training details in classification_training/imagenet/.

## CIFAR100 training results

| Network              | macs     | params      | input size | gpu num      | batch     | warm up | lr decay  | apex | syncbn | epochs | Top-1  |
| -------------        | -------- | ----------- | ---------- | ------------ | --------- | ------- | --------  | ---- | ------ | ------ | ------ |
| ResNet18Cifar        | 556.706M | 11.220M     | 32x32      | 1 RTX A5000  | 128       | 0       | multistep | True | False  | 200    | 78.180 |
| ResNet34halfCifar    | 291.346M | 5.350M      | 32x32      | 1 RTX A5000  | 128       | 0       | multistep | True | False  | 200    | 76.690 |
| ResNet34Cifar        | 1.162G   | 21.328M     | 32x32      | 1 RTX A5000  | 128       | 0       | multistep | True | False  | 200    | 79.310 |
| ResNet50halfCifar    | 328.447M | 5.991M      | 32x32      | 1 RTX A5000  | 128       | 0       | multistep | True | False  | 200    | 77.170 |
| ResNet50Cifar        | 1.305G   | 23.705M     | 32x32      | 1 RTX A5000  | 128       | 0       | multistep | True | False  | 200    | 76.950 |
| ResNet101Cifar       | 2.520G   | 42.697M     | 32x32      | 1 RTX A5000  | 128       | 0       | multistep | True | False  | 200    | 78.270 |
| ResNet152Cifar       | 3.737G   | 58.341M     | 32x32      | 1 RTX A5000  | 128       | 0       | multistep | True | False  | 200    | 78.700 |

You can find more model training details in classification_training/cifar100/.

# Detection training results

## COCO2017 training results

Trained on COCO2017_train dataset, tested on COCO2017_val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

**RetinaNet**
Paper:https://arxiv.org/abs/1708.02002

**FCOS**
Paper:https://arxiv.org/abs/1904.01355

**CenterNet**
Paper:https://arxiv.org/abs/1904.07850

**TTFNet**
Paper:https://arxiv.org/abs/1909.00700

**YOLOv3**
Paper:https://arxiv.org/abs/1804.02767

**YOLOv4**
Paper:https://arxiv.org/abs/2004.10934

**YOLOv5**
Code:https://github.com/ultralytics/yolov5

**YOLOX**
Paper:https://arxiv.org/abs/2107.08430

**How to use yolov3 anchor clustering method to generate a set of custom anchors for your own dataset?**

I provide a script in simpleAICV/detection/yolov3_anchor_cluster.py,and I give two examples for generate anchors on COCO2017 and VOC2007+2012 datasets.If you want to generate anchors for your dataset,just modify the part of input code,get width and height of all annotaion boxes,then use the script to compute anchors.The anchors size will change with different datasets or different input resizes.

| Network               | resize-style    | input size | macs     | params   | gpu num      | batch     | warm up | lr decay  | apex | syncbn | epochs | mAP    |
| -------------         | ------------    | ---------- | -------- | -------- | ------------ | --------- | ------- | --------  | ---- | ------ | ------ | ------ |
| ResNet50-RetinaNet    | RetinaStyle-400 | 400x667    | 63.093G  | 37.969M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 32.067 |
| ResNet50-RetinaNet    | RetinaStyle-800 | 800x1333   | 250.069G | 37.969M  | 2 RTX A5000  | 8         | 0       | multistep | True | False  | 13     | 35.647 |
| ResNet50-RetinaNet    | YoloStyle-640   | 640x640    | 95.558G  | 37.969M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 32.971 |
| ResNet50-FCOS         | RetinaStyle-400 | 400x667    | 54.066G  | 32.291M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 34.046 |
| ResNet50-FCOS         | RetinaStyle-800 | 800x1333   | 214.406G | 32.291M  | 2 RTX A5000  | 8         | 0       | multistep | True | False  | 13     | 37.857 |
| ResNet50-FCOS         | YoloStyle-640   | 640x640    | 81.943G  | 32.291M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 35.055 |
| ResNet18DCN-CenterNet | YoloStyle-512   | 512x512    | 14.854G  | 12.889M  | 2 RTX A5000  | 64        | 0       | multistep | True | False  | 140    | 27.813 |
| ResNet18DCN-TTFNet-3x | YoloStyle-512   | 512x512    | 16.063G  | 13.737M  | 2 RTX A5000  | 64        | 0       | multistep | True | False  | 39     | 28.155 |
| ResNet18DCN-TTFNet-70 | YoloStyle-512   | 512x512    | 16.063G  | 13.737M  | 2 RTX A5000  | 64        | 0       | multistep | True | False  | 70     | 29.675 |

You can find more model training details in detection_training/coco/.

## VOC2007 and VOC2012 training results

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Network               | resize-style    | input size | macs     | params   | gpu num      | batch     | warm up | lr decay  | apex | syncbn | epochs | mAP    |
| -------------         | ------------    | ---------- | -------- | -------- | ------------ | --------- | ------- | --------  | ---- | ------ | ------ | ------ |
| ResNet50-RetinaNet    | RetinaStyle-400 | 400x667    | 56.093G  | 36.724M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 79.804 |
| ResNet50-RetinaNet    | YoloStyle-640   | 640x640    | 84.947G  | 36.724M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 80.565 |
| ResNet50-FCOS         | RetinaStyle-400 | 400x667    | 53.288G  | 32.153M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 79.894 |
| ResNet50-FCOS         | YoloStyle-640   | 640x640    | 80.764G  | 32.153M  | 2 RTX A5000  | 32        | 0       | multistep | True | False  | 13     | 80.510 |

You can find more model training details in detection_training/voc/.

# Distillation training results

## ImageNet training results

**KD loss**
Paper:https://arxiv.org/abs/1503.02531

**DKD loss**
Paper:https://arxiv.org/abs/2203.08679

**DML loss**
Paper:https://arxiv.org/abs/1706.00384

| Teacher Network  | Student Network  | method  | Freeze Teacher | input size | gpu num      | batch     | warm up | lr decay  | apex | syncbn | epochs | Teacher Top-1  | Student Top-1  |
| ---------------- | ---------------- | ------- | -------------- | ---------- | ------------ | --------- | ------- | --------  | ---- | ------ | ------ | -------------- | -------------- |
| ResNet34         | ResNet18         | CE+KD   | True           | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | /              | 71.848         |
| ResNet34         | ResNet18         | CE+DKD  | True           | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | /              | 71.856         |
| ResNet34         | ResNet18         | CE+DML  | False          | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 74.318         | 71.678         |
| ResNet152        | ResNet50         | CE+KD   | True           | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | /              | 76.830         |
| ResNet152        | ResNet50         | CE+DKD  | True           | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | /              | 77.692         |
| ResNet152        | ResNet50         | CE+DML  | False          | 224x224    | 2 RTX A5000  | 256       | 0       | multistep | True | False  | 100    | 79.462         | 77.618         |

You can find more model training details in distillation_training/imagenet/.

# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={SimpleAICV-ImageNet-CIFAR-COCO-VOC-training},
 author={zgcr},
 year={2022}
}
```