- [My column](#my-column)
- [📢 News!](#-news)
- [Introduction](#introduction)
- [All task training results](#all-task-training-results)
- [Environments](#environments)
- [Download my pretrained models and experiments records](#download-my-pretrained-models-and-experiments-records)
- [Prepare datasets](#prepare-datasets)
  - [CIFAR10](#cifar10)
  - [CIFAR100](#cifar100)
  - [ImageNet 1K(ILSVRC2012)](#imagenet-1kilsvrc2012)
  - [ImageNet 21K(Winter 2021 release)](#imagenet-21kwinter-2021-release)
  - [ACCV2022](#accv2022)
  - [VOC2007 and VOC2012](#voc2007-and-voc2012)
  - [COCO2017](#coco2017)
  - [SAMACOCO](#samacoco)
  - [Objects365(v2,2020)](#objects365v22020)
  - [ADE20K](#ade20k)
  - [CelebA-HQ](#celeba-hq)
  - [FFHQ](#ffhq)
  - [Places365-standard/challenge](#places365-standardchallenge)
- [How to train and test model](#how-to-train-and-test-model)
- [How to use gradio demo](#how-to-use-gradio-demo)
- [Reference](#reference)
- [Citation](#citation)


# My column

https://www.zhihu.com/column/c_1692623656205897728

# 📢 News!

* 2024/04/15: support segment-anything model training/testing/jupyter example/gradio demo.

# Introduction

**This repository provides simple training and testing examples for the following tasks:**

| task                          | support dataset                                                                    | support network                                               |
| ----------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Image classification task     | CIFAR100<br>ImageNet1K(ILSVRC2012)<br>ImageNet21K(Winter 2021 release)<br>ACCV2022 | ResNet<br>DarkNet<br>RepVGG<br>RegNetX<br>ViT<br>VAN          |
| Object detection task         | VOC2007 and VOC2012<br>COCO2017<br>Objects365(v2,2020)                             | RetinaNet<br>FCOS<br>CenterNet<br>TTFNet<br>DETR<br>DINO-DETR |
| Semantic segmentation task    | ADE20K                                                                             | DeepLabv3+<br>U2Net                                           |
| Instance segmentation task    | COCO2017                                                                           | YOLACT<br>SOLOv2                                              |
| Knowledge distillation task   | ImageNet1K(ILSVRC2012)                                                             | KD loss(for ResNet)<br>DML loss(for ResNet)                   |
| Contrastive learning task     | ImageNet1K(ILSVRC2012)                                                             | DINO(for ResNet)                                              |
| Masked image modeling task    | ImageNet1K(ILSVRC2012)<br>ACCV2022                                                 | MAE(for ViT)                                                  |
| OCR text detection task       | /                                                                                  | DBNet                                                         |
| OCR text recognition task     | /                                                                                  | CTC Model                                                     |
| Human matting task            | /                                                                                  | PFAN Matting model                                            |
| Salient object detection task | /                                                                                  | PFAN Segmentation model                                       |
| Face detection task           | /                                                                                  | RetinaFace                                                    |
| Interactive segmentation task | /                                                                                  | SAM(segment-anything)                                         |
| Image inpainting task         | CelebA-HQ<br>Places365-standard<br>Places365-challenge                             | AOT-GAN<br>TRANSX-LKA-AOT-GAN                                 |
| Diffusion model task          | CIFAR10<br>CIFAR100<br>CelebA-HQ<br>FFHQ                                           | DDPM<br>DDIM                                                  |


# All task training results

**See all task training results in [results.md](results.md).**

# Environments

**1、This repository only supports running on ubuntu(verison>=18.04 LTS).**

**2、This repository only support one node one gpu/one node multi gpus mode with pytorch DDP training.**

**3、Please make sure your Python environment version>=3.7.**

**4、Please make sure your pytorch version>=1.10.**

**5、If you want to use torch.complie() function,please make sure your pytorch version>=2.0.Using pytorch2.0/2.2/2.3,don't use pytorch2.1.**

**Use pip or conda to install those Packages in your Python environment:**
```
torch
torchvision
pillow
numpy
Cython
colormath
pycocotools
opencv-python
scipy
einops
scikit-image
pyclipper
shapely
imagesize
nltk
tqdm
yapf
onnx
onnxruntime
onnxsim
thop
gradio==4.26.0
transformers==4.41.2
open-clip-torch==2.24.0
```


**If you want to use xformers,install xformers Packge from offical github repository:**

https://github.com/facebookresearch/xformers


**If you want to use dino-detr model,install MultiScaleDeformableAttention Packge in your Python environment:**

cd to simpleAICV/detection/compile_multiscale_deformable_attention,then run commands:
```
chmod +x make.sh
./make.sh
```

# Download my pretrained models and experiments records

You can download all my pretrained models and all my experiments records/checkpoints from huggingface or Baidu-Netdisk.

If you only want to download all my pretrained models(model.state_dict()),you can download pretrained_models folder.

```
# huggingface
https://huggingface.co/zgcr654321/classification_training/tree/main
https://huggingface.co/zgcr654321/contrastive_learning_training/tree/main
https://huggingface.co/zgcr654321/detection_training/tree/main
https://huggingface.co/zgcr654321/image_inpainting_training/tree/main
https://huggingface.co/zgcr654321/diffusion_model_training/tree/main
https://huggingface.co/zgcr654321/distillation_training/tree/main
https://huggingface.co/zgcr654321/instance_segmentation_training/tree/main
https://huggingface.co/zgcr654321/masked_image_modeling_training/tree/main
https://huggingface.co/zgcr654321/ocr_text_detection_training/tree/main
https://huggingface.co/zgcr654321/ocr_text_recognition_training/tree/main
https://huggingface.co/zgcr654321/human_matting_training/tree/main
https://huggingface.co/zgcr654321/salient_object_detection_training/tree/main
https://huggingface.co/zgcr654321/face_detection_training/tree/main
https://huggingface.co/zgcr654321/interactive_segmentation_training/tree/main
https://huggingface.co/zgcr654321/semantic_segmentation_training/tree/main
https://huggingface.co/zgcr654321/pretrained_models/tree/main

# Baidu-Netdisk
链接：https://pan.baidu.com/s/1yhEwaZhrb2NZRpJ5eEqHBw 
提取码：rgdo
```

# Prepare datasets

## CIFAR10

Make sure the folder architecture as follows:
```
CIFAR10
|
|-----batches.meta  unzip from cifar-10-python.tar.gz
|-----data_batch_1  unzip from cifar-10-python.tar.gz
|-----data_batch_2  unzip from cifar-10-python.tar.gz
|-----data_batch_3  unzip from cifar-10-python.tar.gz
|-----data_batch_4  unzip from cifar-10-python.tar.gz
|-----data_batch_5  unzip from cifar-10-python.tar.gz
|-----readme.html   unzip from cifar-10-python.tar.gz
|-----test_batch    unzip from cifar-10-python.tar.gz
```

## CIFAR100

Make sure the folder architecture as follows:
```
CIFAR100
|
|-----train unzip from cifar-100-python.tar.gz
|-----test  unzip from cifar-100-python.tar.gz
|-----meta  unzip from cifar-100-python.tar.gz
```

## ImageNet 1K(ILSVRC2012)

Make sure the folder architecture as follows:
```
ILSVRC2012
|
|-----train----1000 sub classes folders
|-----val------1000 sub classes folders
Please make sure the same class has same class folder name in train and val folders.
```

## ImageNet 21K(Winter 2021 release)

Make sure the folder architecture as follows:
```
ImageNet21K
|
|-----train-----------10450 sub classes folders
|-----val-------------10450 sub classes folders
|-----small_classes---10450 sub classes folders
|-----imagenet21k_miil_tree.pth
Please make sure the same class has same class folder name in train and val folders.
```

## ACCV2022

Make sure the folder architecture as follows:
```
ACCV2022
|
|-----train-------------5000 sub classes folders
|-----testa-------------60000 images
|-----accv2022_broken_list.json
```

## VOC2007 and VOC2012

Make sure the folder architecture as follows:
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

## COCO2017

Make sure the folder architecture as follows:
```
COCO2017
|                |----captions_train2017.json
|                |----captions_val2017.json
|--annotations---|----instances_train2017.json
|                |----instances_val2017.json
|                |----person_keypoints_train2017.json
|                |----person_keypoints_val2017.json
|                 
|                |----train2017
|----images------|----val2017
```

## SAMACOCO

Make sure the folder architecture as follows:
```
SAMA-COCO
|                |----sama_coco_train.json
|                |----sama_coco_validation.json
|--annotations---|----train_labels.json
|                |----validation_labels.json
|                |----test_labels.json
|                |----image_info_test2017.json
|                |----image_info_test-dev2017.json
|                 
|                |----train
|----images------|----validation
```

## Objects365(v2,2020)

Make sure the folder architecture as follows:
```
objects365_2020
|
|                |----zhiyuan_objv2_train.json
|--annotations---|----zhiyuan_objv2_val.json
|                |----sample_2020.json
|                 
|                |----train all train patch folders
|----images------|----val   all val patch folders
                 |----test  all test patch folders
```

## ADE20K

Make sure the folder architecture as follows:
```
ADE20K
|                 |----training
|---images--------|----validation
|                 |----testing
|        
|                 |----training
|---annotations---|----validation
```

## CelebA-HQ

Make sure the folder architecture as follows:
```
CelebA-HQ
|                 |----female
|---train---------|----male
|        
|                 |----female
|---val-----------|----male
```

## FFHQ

Make sure the folder architecture as follows:
```
FFHQ
|
|---images
|---ffhq-dataset-v1.json
|---ffhq-dataset-v2.json
```

## Places365-standard/challenge

Make sure the folder architecture as follows:
```
Places365-standard/challenge
|
|                            |---train_large all sub folders
|---high_resolution_images---|---val_large   all images
|                            |---test_large  all images
```

# How to train and test model

**If you want to train or test model,you need enter a training experiment folder directory,then run train.sh or test.sh.**

For example,you can enter in folder classification_training/imagenet/resnet50.

If you want to restart train this model,please delete checkpoints and log folders first,then run train.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../../tools/train_classification_model.py --work-dir ./
```

if you want to test this model,you need have a pretrained model first,modify trained_model_path in test_config.py,then run test.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../../tools/test_classification_model.py --work-dir ./
```

**CUDA_VISIBLE_DEVICES is used to specify the gpu ids for this training.Please make sure the number of nproc_per_node equal to the number of using gpu cards.Make sure master_addr/master_port are unique for each training.**

**All checkpoints/log are saved in your executing training/testing experiment folder directory.**

**Also, You can modify super parameters in train_config.py/test_config.py.**

# How to use gradio demo

cd to gradio_demo,we have:
```
classification demo
detection demo
semantic_segmentation demo
instance_segmentation demo
text_detection demo
text_recognition demo
human_matting demo
salient_object_detection demo
face_detection demo
segment_anything demo
```

For example,you can run detection gradio demo(please prepare trained model weight first):
```
python gradio_detect_single_image.py
```

# Reference

https://github.com/facebookresearch/segment-anything


# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={SimpleAICV-pytorch-training-examples},
 author={zgcr},
 year={2020-2024}
}
```