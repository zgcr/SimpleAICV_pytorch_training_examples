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
- [Training GPU server](#training-gpu-server)
- [Environments](#environments)
- [Download pretrained models and experiments checkpoints/logs](#download-pretrained-models-and-experiments-checkpointslogs)
- [How to use gradio demo](#how-to-use-gradio-demo)
- [How to use inference demo](#how-to-use-inference-demo)
- [How to train or test model](#how-to-train-or-test-model)
- [Prepare datasets](#prepare-datasets)
  - [CIFAR100](#cifar100)
  - [ImageNet 1K(ILSVRC2012)](#imagenet-1kilsvrc2012)
  - [ImageNet 21K(Winter 2021 release)](#imagenet-21kwinter-2021-release)
  - [COCO2017](#coco2017)
  - [Objects365(v2,2020)](#objects365v22020)
  - [VOC2007\&VOC2012](#voc2007voc2012)
  - [ADE20K](#ade20k)
  - [SAMACOCO](#samacoco)
- [Reference](#reference)
- [My column](#my-column)
- [Citation](#citation)

# 📢 News!

* 2026/02/02: update dinov3 backbone implementation in SimpleAICV/detection/models/backbones.
* 2026/02/02: update SAM(segment_anything)/SAM_Matting model training pipeline and jupyter example in 13.interactive_segmentation_training.
* 2026/02/02: update SAM2(segment_anything2)/SAM2_Matting model training pipeline and jupyter example in 14.video_interactive_segmentation_training.
* 2026/02/02: update universal_segmentation/universal_matting model training pipeline in 16.universal_segmentation_training.
* 2026/02/02: updata all task gradio demo in gradio_demo.
* 2026/02/02: updata all task inference demo in inference_demo.

# Simplicity

This repository maintains a lightweight codebase.It requiring only Python and PyTorch as core dependencies(no third-party frameworks like MMCV).

# Introduction

**This repository provides simple training and testing examples for following tasks:**

| task                                       | support dataset                                                        | support model                                 |
| ------------------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------- |
| 00.classification_training                 | CIFAR100<br>ImageNet1K(ILSVRC2012)<br>ImageNet21K(Winter 2021 release) | DarkNet<br>ResNet<br>Convformer<br>VAN<br>ViT |
| 01.distillation_training                   | ImageNet1K(ILSVRC2012)                                                 | DML loss(ResNet)<br>KD loss(ResNet)           |
| 02.masked_image_modeling_training          | ImageNet1K(ILSVRC2012)                                                 | MAE(ViT)                                      |
| 03.detection_training                      | COCO2017<br>Objects365(v2,2020)<br>VOC2007&VOC2012                     | RetinaNet<br>FCOS<br>DETR                     |
| 04.semantic_segmentation_training          | ADE20K<br>COCO2017                                                     | pfan_semantic_segmentation                    |
| 05.instance_segmentation_training          | COCO2017                                                               | SOLOv2<br>YOLACT                              |
| 06.salient_object_detection_training       | combine dataset                                                        | pfan_segmentation                             |
| 07.human_matting_training                  | combine dataset                                                        | pfan_matting                                  |
| 08.ocr_text_detection_training             | combine dataset                                                        | DBNet                                         |
| 09.ocr_text_recognition_training           | combine dataset                                                        | CTC_Model                                     |
| 10.face_detection_training                 | combine dataset                                                        | RetinaFace                                    |
| 11.face_parsing_training                   | CelebAMask-HQ<br>FaceSynthetics                                        | pfan_face_parsing                             |
| 12.human_parsing_training                  | CIHP<br>LIP                                                            | pfan_human_parsing                            |
| 13.interactive_segmentation_training       | combine dataset                                                        | SAM(segment_anything)<br>SAM_Matting          |
| 14.video_interactive_segmentation_training | combine dataset                                                        | SAM2(segment_anything2)<br>SAM2_Matting       |
| 16.universal_segmentation_training         | combine dataset                                                        | universal_segmentation<br>universal_matting   |

# All task training results

**See all task training results in [RESULTS.md](RESULTS.md).**

# Training GPU server

1、1-8 RTX 4090D(24GB) GPUs, Python3.12, Pytorch2.5.1, CUDA12.4, Ubuntu22.04(for most experiments).

2、8 RTX PRO 6000(96GB) GPUs, Python3.12, Pytorch2.8.0, CUDA12.8, Ubuntu22.04(for 13.interactive_segmentation_training/14.video_interactive_segmentation_training/16.universal_segmentation_training).

# Environments

**1、Python and Pytorch Supported Version: Python>=3.12, Pytorch>=2.5.1.**

**2、Most Experiments only support Single-Node Single-GPU training/Single-Node Multi-GPU DDP training, but 13.interactive_segmentation_training/14.video_interactive_segmentation_training also support Multi-Node Multi-GPU DDP training(Requires InfiniBand/RoCE).**

**3、Create a conda environment:**
```
conda create -n SimpleAICV python=3.12
```

**4、Install PyTorch:**
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
To install a different PyTorch version, find command from here:

https://pytorch.org/get-started/previous-versions/

**5、Install other Packages:**
```
pip install -r requirements.txt
```

# Download pretrained models and experiments checkpoints/logs

You can download all my pretrained models and experiments checkpoints/logs from huggingface or Baidu-Netdisk.

If you only need the pretrained models (model.state_dict()), you can download the pretrained_models folder.
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
https://huggingface.co/zgcr654321/14.video_interactive_segmentation_training/tree/main
https://huggingface.co/zgcr654321/16.universal_segmentation_training/tree/main
https://huggingface.co/zgcr654321/pretrained_models/tree/main

# Baidu-Netdisk
链接:https://pan.baidu.com/s/17oSFXgIy1vxUdPUhTzRkdw?pwd=3l99
提取码：3l99
```

# How to use gradio demo

cd to gradio_demo folder,we have:
```
00.gradio_classify_single_image.py
03.gradio_detect_single_image.py
04.gradio_semantic_segment_single_image.py
05.gradio_instance_segment_single_image.py
06.gradio_salient_object_detection_single_image.py
07.gradio_human_matting_single_image.py
08.gradio_ocr_text_detect_single_image.py
09.gradio_ocr_text_recognition_single_image.py
10.gradio_face_detect_single_image.py
11.gradio_face_parsing_single_image.py
12.gradio_human_parsing_single_image.py
13.0.0.gradio_sam_point_target_single_image.py
13.0.1.gradio_sam_circle_target_single_image.py
16.0.gradio_universal_segment_single_image.py
16.1.gradio_universal_matting_single_image.py
```

For example,you can run 03.gradio_detect_single_image.py(please prepare pretrained model weight first and modify pretrained model load path):
```
python 03.gradio_detect_single_image.py
```

# How to use inference demo

cd to inference_demo folder,we have:
```
00.inference_classify_single_image.py
03.inference_detect_single_image.py
04.inference_semantic_segment_single_image.py
05.inference_instance_segment_single_image.py
06.inference_salient_object_detection_single_image.py
07.inference_human_matting_single_image.py
08.inference_ocr_text_detect_single_image.py
09.inference_ocr_text_recognition_single_image.py
10.inference_face_detect_single_image.py
11.inference_face_parsing_single_image.py
12.inference_human_parsing_single_image.py
13.0.inference_sam_single_image.py
16.0.inference_universal_segment_single_image.py
16.1.inference_universal_matting_single_image.py
```

For example,you can run 03.inference_detect_single_image.py(please prepare pretrained model weight first and modify pretrained model load path):
```
python 03.inference_detect_single_image.py
```

# How to train or test model

**If you want to train or test model, you need enter a training experiment folder directory, then run train.sh or test.sh.**

For example,you can enter in folder 00.classification_training/imagenet/resnet50.

If you want to train model from scratch,please delete checkpoints and log folders first,then run train.sh:
```
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_addr 127.0.1.0 \
    --master_port 10000 \
    ../../../tools/train_classification_model.py \
    --work-dir ./
```

if you want to test model,you need have a pretrained model first,modify trained_model_path in test_config.py,then run test.sh:
```
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_addr 127.0.1.1 \
    --master_port 10001 \
    ../../../tools/test_classification_model.py \
    --work-dir ./
```

**CUDA_VISIBLE_DEVICES is used to specify gpu_ids for training.Please make sure the number of nproc_per_node equal to the number of gpus.Make sure master_addr/master_port are unique for each training.**

**Checkpoints/log folders are saved in your training/testing experiment folder directory.Also, You can modify super parameters in train_config.py/test_config.py.**

# Prepare datasets

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

## VOC2007&VOC2012

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

# Reference

```
https://github.com/facebookresearch/dinov3
https://github.com/facebookresearch/segment-anything
https://github.com/facebookresearch/sam2
https://github.com/tue-mps/EoMT
```

# My column

https://www.zhihu.com/column/c_1692623656205897728

# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={SimpleAICV-pytorch-training-examples},
 author={zgcr},
 year={2020-2030}
}
```