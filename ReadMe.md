<div align="center">
      <h1>SimpleAICV</h1>
</div>

<div align="center">
    <p align="center">
          <em> Open-source / Simple / Lightweight / Easy-to-use / Extensible </em>
    </p>
</div>

<hr>

- [📢 News!](#-news)
- [Simplicity](#simplicity)
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
- [How to train or test a model](#how-to-train-or-test-a-model)
- [How to use gradio demo](#how-to-use-gradio-demo)
- [Reference](#reference)
- [My column](#my-column)
- [Citation](#citation)


# 📢 News!

* 2025/02/16: train light segment-anything model with bf16.

# Simplicity
This repository maintains a lightweight codebase.It requiring only Python and PyTorch as core dependencies(no third-party frameworks like MMCV).

# Introduction

**This repository provides simple training and testing examples for following tasks:**

| task                          | support dataset                                                        | support network                                                                    |
| ----------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Image classification task     | CIFAR100<br>ImageNet1K(ILSVRC2012)<br>ImageNet21K(Winter 2021 release) | Convformer<br>DarkNet<br>ResNet<br>VAN<br>ViT                                      |
| Knowledge distillation task   | ImageNet1K(ILSVRC2012)                                                 | DML loss(ResNet)<br>KD loss(ResNet)                                                |
| Masked image modeling task    | ImageNet1K(ILSVRC2012)                                                 | MAE(ViT)                                                                           |
| Object detection task         | COCO2017<br>Objects365(v2,2020)<br>VOC2007 and VOC2012                 | DETR<br>DINO-DETR<br>RetinaNet<br>FCOS                                             |
| Semantic segmentation task    | ADE20K<br>COCO2017                                                     | DeepLabv3+                                                                         |
| Instance segmentation task    | COCO2017                                                               | SOLOv2<br>YOLACT                                                                   |
| Salient object detection task | combine dataset                                                        | pfan-segmentation                                                                  |
| Human matting task            | combine dataset                                                        | pfan-matting                                                                       |
| OCR text detection task       | combine dataset                                                        | DBNet                                                                              |
| OCR text recognition task     | combine dataset                                                        | CTC Model                                                                          |
| Face detection task           | combine dataset                                                        | RetinaFace                                                                         |
| Face parsing task             | FaceSynthetics<br>CelebAMask-HQ                                        | pfan-face-parsing     <br>sapiens_face_parsing                                     |
| Human parsing task            | LIP<br>CIHP                                                            | pfan-human-parsing <br>sapiens_human_parsing                                       |
| Interactive segmentation task | combine dataset                                                        | SAM(segment-anything)<br>light_sam<br>light_sam_matting |

# All task training results

**Most experiments were trained on 2-8 RTX4090D GPUs, pytorch2.3, ubuntu22.04.**

**See all task training results in [results.md](results.md).**

# Environments

**1、This repository only supports running on ubuntu(verison>=22.04 LTS).**

**2、Most Experiments only support Single-Node Single-GPU training/Single-Node Multi-GPU DDP training, but segment-anything also support Multi-Node Multi-GPU DDP training(Requires RDMA/IB network support).**

**3、Please make sure your Python version>=3.9 and pytorch version>=2.0.**

**4、Create a conda environment:**
```
conda create -n simpleAICV python=3.12
```

**5、Using commands to install PyTorch:**
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
To install a different PyTorch version, select command from here:

https://pytorch.org/get-started/previous-versions/

**6、Install other Packages:**
```
pip install -r requirements.txt
```

**7、(optional) If you want to use xformers,install xformers Packge from offical github repository:**

https://github.com/facebookresearch/xformers


**8、(optional) If you want to use dino-detr model,install MultiScaleDeformableAttention Packge in your Python environment:**

cd to simpleAICV/detection/compile_multiscale_deformable_attention,then run commands:
```
chmod +x make.sh
./make.sh
```

# Download my pretrained models and experiments records

You can download all my pretrained models and experiments records/checkpoints from huggingface or Baidu-Netdisk.

If you only want to download all my pretrained models(model.state_dict()),you can download pretrained_models folder.

```
# huggingface
https://huggingface.co/zgcr654321/00.classification_training/tree/main
https://huggingface.co/zgcr654321/01.distillation_training/tree/main
https://huggingface.co/zgcr654321/02.masked_image_modeling_training/tree/main
https://huggingface.co/zgcr654321/03.detection_training/tree/main
https://huggingface.co/zgcr654321/04.semantic_segmentation_training/tree/main
https://huggingface.co/zgcr654321/05.instance_segmentation_training/tree/main
https://huggingface.co/zgcr654321/06.salient_object_detection_training/tree/main
https://huggingface.co/zgcr654321/07.human_matting_training/tree/main
https://huggingface.co/zgcr654321/08.ocr_text_detection_training/tree/main
https://huggingface.co/zgcr654321/09.ocr_text_recognition_training/tree/main
https://huggingface.co/zgcr654321/10.face_detection_training/tree/main
https://huggingface.co/zgcr654321/11.face_parsing_training/tree/main
https://huggingface.co/zgcr654321/12.human_parsing_training/tree/main
https://huggingface.co/zgcr654321/13.interactive_segmentation_training/tree/main
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

# How to train or test a model

**If you want to train or test a model,you need enter a training experiment folder directory,then run train.sh or test.sh.**

For example,you can enter in folder classification_training/imagenet/resnet50.

If you want to restart train this model,please delete checkpoints and log folders first,then run train.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../../tools/train_classification_model.py --work-dir ./
```

if you want to test this model,you need have a pretrained model first,modify trained_model_path in test_config.py,then run test.sh:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_addr 127.0.1.0 --master_port 10000 ../../../tools/test_classification_model.py --work-dir ./
```

**CUDA_VISIBLE_DEVICES is used to specify the gpu ids for this training.Please make sure the number of nproc_per_node equal to the number of using gpu cards.Make sure master_addr/master_port are unique for each training.**

**Checkpoints/log folders are saved in your executing training/testing experiment folder directory.**

**Also, You can modify super parameters in train_config.py/test_config.py.**

# How to use gradio demo

cd to gradio_demo,we have:
```
classification demo
detection demo
semantic_segmentation demo
instance_segmentation demo
salient_object_detection demo
human_matting demo
text_detection demo
text_recognition demo
face_detection demo
face_parsing demo
human_parsing demo
point target segment_anything demo
circle target segment_anything demo
```

For example,you can run detection gradio demo(please prepare trained model weight first and modify model weight load path):
```
python gradio_detect_single_image.py
```

# Reference

```
https://github.com/facebookresearch/segment-anything
https://github.com/facebookresearch/sam2
```

# My column

https://www.zhihu.com/column/c_1692623656205897728

# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={SimpleAICV-pytorch-training-examples},
 author={zgcr},
 year={2020-2025}
}
```