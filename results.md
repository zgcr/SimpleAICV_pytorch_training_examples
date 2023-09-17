- [Image classification task results](#image-classification-task-results)
  - [ResNetCifar training from scratch on CIFAR100](#resnetcifar-training-from-scratch-on-cifar100)
  - [ResNet training from scratch on ImageNet1K(ILSVRC2012)](#resnet-training-from-scratch-on-imagenet1kilsvrc2012)
  - [DarkNet training from scratch on ImageNet1K(ILSVRC2012)](#darknet-training-from-scratch-on-imagenet1kilsvrc2012)
  - [RepVGG training from scratch on ImageNet1K(ILSVRC2012)](#repvgg-training-from-scratch-on-imagenet1kilsvrc2012)
  - [RegNet training from scratch on ImageNet1K(ILSVRC2012)](#regnet-training-from-scratch-on-imagenet1kilsvrc2012)
  - [U2NetBackbone training from scratch on ImageNet1K(ILSVRC2012)](#u2netbackbone-training-from-scratch-on-imagenet1kilsvrc2012)
  - [ResNet finetune from ImageNet21k pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-finetune-from-imagenet21k-pretrain-weight-on-imagenet1kilsvrc2012)
  - [ResNet finetune from DINO pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-finetune-from-dino-pretrain-weight-on-imagenet1kilsvrc2012)
  - [ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-self-trained-mae-pretrain-weight400epoch-on-imagenet1kilsvrc2012)
  - [ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-offical-mae-pretrain-weight800-epoch-on-imagenet1kilsvrc2012)
  - [ResNet train from ImageNet1K pretrain weight on ImageNet21K(Winter 2021 release)](#resnet-train-from-imagenet1k-pretrain-weight-on-imagenet21kwinter-2021-release)
- [Object detection task results](#object-detection-task-results)
  - [All detection models training from scratch on COCO2017](#all-detection-models-training-from-scratch-on-coco2017)
  - [All detection models finetune from objects365 pretrain weight on COCO2017](#all-detection-models-finetune-from-objects365-pretrain-weight-on-coco2017)
  - [All detection models train from COCO2017 pretrain weight on Objects365(v2,2020)](#all-detection-models-train-from-coco2017-pretrain-weight-on-objects365v22020)
  - [All detection models training from scratch on VOC2007 and VOC2012](#all-detection-models-training-from-scratch-on-voc2007-and-voc2012)
  - [All detection models finetune from objects365 pretrain weight on VOC2007 and VOC2012](#all-detection-models-finetune-from-objects365-pretrain-weight-on-voc2007-and-voc2012)
- [Semantic Segmentation task results](#semantic-segmentation-task-results)
  - [All semantic segmentation models training from scratch on ADE20K](#all-semantic-segmentation-models-training-from-scratch-on-ade20k)
  - [All semantic segmentation models training from scratch on COCO2017](#all-semantic-segmentation-models-training-from-scratch-on-coco2017)
- [Knowledge distillation task results](#knowledge-distillation-task-results)
  - [ResNet training from pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-training-from-pretrain-weight-on-imagenet1kilsvrc2012)
- [Contrastive learning task results](#contrastive-learning-task-results)
  - [ResNet DINO pretrain on ImageNet1K(ILSVRC2012)](#resnet-dino-pretrain-on-imagenet1kilsvrc2012)
  - [ResNet finetune from DINO pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-finetune-from-dino-pretrain-weight-on-imagenet1kilsvrc2012-1)
- [Masked image modeling task results](#masked-image-modeling-task-results)
  - [ViT MAE pretrain on ImageNet1K(ILSVRC2012)](#vit-mae-pretrain-on-imagenet1kilsvrc2012)
  - [ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-self-trained-mae-pretrain-weight400epoch-on-imagenet1kilsvrc2012-1)
  - [ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-offical-mae-pretrain-weight800-epoch-on-imagenet1kilsvrc2012-1)
- [Image inpainting model task results](#image-inpainting-model-task-results)
  - [All image inpainting model training from scratch on CelebA-HQ](#all-image-inpainting-model-training-from-scratch-on-celeba-hq)
  - [All image inpainting model training from scratch on Places365-standard](#all-image-inpainting-model-training-from-scratch-on-places365-standard)
  - [All image inpainting model training from scratch on Places365-challenge](#all-image-inpainting-model-training-from-scratch-on-places365-challenge)
- [Diffusion model task results](#diffusion-model-task-results)
  - [All diffusion model with different sampling methods on CIFAR10](#all-diffusion-model-with-different-sampling-methods-on-cifar10)
  - [All diffusion model with different sampling methods on CIFAR100](#all-diffusion-model-with-different-sampling-methods-on-cifar100)
  - [All diffusion model with different sampling methods on CelebA-HQ](#all-diffusion-model-with-different-sampling-methods-on-celeba-hq)
  - [All diffusion model with different sampling methods on FFHQ](#all-diffusion-model-with-different-sampling-methods-on-ffhq)





# Image classification task results

**ResNet**

Paper:https://arxiv.org/abs/1512.03385

**DarkNet**

Paper:https://arxiv.org/abs/1804.02767?e05802c1_page=1

**RepVGG**

Paper:https://arxiv.org/abs/2101.03697

**RegNet**

Paper:https://arxiv.org/abs/2003.13678

**ViT**

Paper:https://arxiv.org/abs/2010.11929

## ResNetCifar training from scratch on CIFAR100 

**ResNetCifar is different from ResNet in the first few layers.**

| Network        | macs     | params  | input size | gpu num     | batch | epochs | Top-1  |
| -------------- | -------- | ------- | ---------- | ----------- | ----- | ------ | ------ |
| ResNet18Cifar  | 557.935M | 11.220M | 32x32      | 1 RTX A5000 | 128   | 200    | 77.110 |
| ResNet34Cifar  | 1.164G   | 21.328M | 32x32      | 1 RTX A5000 | 128   | 200    | 78.140 |
| ResNet50Cifar  | 1.312G   | 23.705M | 32x32      | 1 RTX A5000 | 128   | 200    | 75.610 |
| ResNet101Cifar | 2.531G   | 42.697M | 32x32      | 1 RTX A5000 | 128   | 200    | 76.970 |
| ResNet152Cifar | 3.751G   | 58.341M | 32x32      | 1 RTX A5000 | 128   | 200    | 77.710 |

You can find more model training details in classification_training/cifar100/.

## ResNet training from scratch on ImageNet1K(ILSVRC2012)

| Network   | macs    | params  | input size | gpu num     | batch | epochs | Top-1  |
| --------- | ------- | ------- | ---------- | ----------- | ----- | ------ | ------ |
| ResNet18  | 1.819G  | 11.690M | 224x224    | 2 RTX A5000 | 256   | 100    | 70.512 |
| ResNet34  | 3.671G  | 21.798M | 224x224    | 2 RTX A5000 | 256   | 100    | 73.680 |
| ResNet50  | 4.112G  | 25.557M | 224x224    | 2 RTX A5000 | 256   | 100    | 76.300 |
| ResNet101 | 7.834G  | 44.549M | 224x224    | 2 RTX A5000 | 256   | 100    | 77.380 |
| ResNet152 | 11.559G | 60.193M | 224x224    | 2 RTX A5000 | 256   | 100    | 77.542 |

You can find more model training details in classification_training/imagenet/.

## DarkNet training from scratch on ImageNet1K(ILSVRC2012)

| Network     | macs     | params  | input size | gpu num     | batch | epochs | Top-1  |
| ----------- | -------- | ------- | ---------- | ----------- | ----- | ------ | ------ |
| DarkNetTiny | 412.537M | 2.087M  | 256x256    | 2 RTX A5000 | 256   | 100    | 57.786 |
| DarkNet19   | 3.663G   | 20.842M | 256x256    | 2 RTX A5000 | 256   | 100    | 74.248 |
| DarkNet53   | 9.322G   | 41.610M | 256x256    | 2 RTX A5000 | 256   | 100    | 76.352 |

You can find more model training details in classification_training/imagenet/.

## RepVGG training from scratch on ImageNet1K(ILSVRC2012)

| Network          | macs    | params  | input size | gpu num     | batch | epochs | Top-1  |
| ---------------- | ------- | ------- | ---------- | ----------- | ----- | ------ | ------ |
| RepVGG_A0_deploy | 1.362G  | 8.309M  | 224x224    | 2 RTX A5000 | 256   | 120    | 72.010 |
| RepVGG_A1_deploy | 2.364G  | 12.790M | 224x224    | 2 RTX A5000 | 256   | 120    | 74.032 |
| RepVGG_A2_deploy | 5.117G  | 25.500M | 224x224    | 2 RTX A5000 | 256   | 120    | 76.078 |
| RepVGG_B0_deploy | 3.058G  | 14.339M | 224x224    | 2 RTX A5000 | 256   | 120    | 74.880 |
| RepVGG_B1_deploy | 11.816G | 51.829M | 224x224    | 2 RTX A5000 | 256   | 120    | 77.790 |
| RepVGG_B2_deploy | 18.377G | 80.315M | 224x224    | 2 RTX A5000 | 256   | 120    | 78.120 |

You can find more model training details in classification_training/imagenet/.

## RegNet training from scratch on ImageNet1K(ILSVRC2012)

| Network       | macs     | params  | input size | gpu num     | batch | epochs | Top-1  |
| ------------- | -------- | ------- | ---------- | ----------- | ----- | ------ | ------ |
| RegNetX_400MF | 410.266M | 5.158M  | 224x224    | 2 RTX A5000 | 4096  | 300    | 69.466 |
| RegNetX_600MF | 616.813M | 6.196M  | 224x224    | 2 RTX A5000 | 4096  | 300    | 71.754 |
| RegNetX_800MF | 820.324M | 7.260M  | 224x224    | 2 RTX A5000 | 4096  | 300    | 73.148 |
| RegNetX_1_6GF | 1.635G   | 9.190M  | 224x224    | 2 RTX A5000 | 4096  | 300    | 76.142 |
| RegNetX_3_2GF | 3.222G   | 15.297M | 224x224    | 2 RTX A5000 | 4096  | 300    | 78.244 |
| RegNetX_4_0GF | 4.013G   | 22.118M | 224x224    | 2 RTX A5000 | 4096  | 300    | 78.916 |

You can find more model training details in classification_training/imagenet/.

## U2NetBackbone training from scratch on ImageNet1K(ILSVRC2012)

| Network       | macs    | params  | input size | gpu num     | batch | epochs | Top-1  |
| ------------- | ------- | ------- | ---------- | ----------- | ----- | ------ | ------ |
| U2NetBackbone | 13.097G | 26.181M | 224x224    | 2 RTX A5000 | 256   | 100    | 76.038 |

You can find more model training details in classification_training/imagenet/.

## ResNet finetune from ImageNet21k pretrain weight on ImageNet1K(ILSVRC2012)

| Network   | macs    | params  | input size | gpu num     | batch | epochs | Top-1  |
| --------- | ------- | ------- | ---------- | ----------- | ----- | ------ | ------ |
| ResNet18  | 1.819G  | 11.690M | 224x224    | 2 RTX A5000 | 4096  | 300    | 71.580 |
| ResNet34  | 3.671G  | 21.798M | 224x224    | 2 RTX A5000 | 4096  | 300    | 76.316 |
| ResNet50  | 4.112G  | 25.557M | 224x224    | 2 RTX A5000 | 4096  | 300    | 79.484 |
| ResNet101 | 7.834G  | 44.549M | 224x224    | 2 RTX A5000 | 4096  | 300    | 80.940 |
| ResNet152 | 11.559G | 60.193M | 224x224    | 2 RTX A5000 | 4096  | 300    | 81.236 |

You can find more model training details in classification_training/imagenet/.

## ResNet finetune from DINO pretrain weight on ImageNet1K(ILSVRC2012)

| Network  | macs   | params  | input size | gpu num     | batch | epochs | Top-1  |
| -------- | ------ | ------- | ---------- | ----------- | ----- | ------ | ------ |
| ResNet50 | 4.112G | 25.557M | 224x224    | 2 RTX A5000 | 256   | 100    | 77.114 |
| ResNet50 | 4.112G | 25.557M | 224x224    | 2 RTX A5000 | 4096  | 300    | 79.418 |

You can find more model training details in classification_training/imagenet/.

## ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs    | params   | input size | gpu num     | batch | epochs | Top-1  |
| ----------------- | ------- | -------- | ---------- | ----------- | ----- | ------ | ------ |
| ViT-Tiny-Patch16  | 1.075G  | 5.670M   | 224x224    | 1 RTX A5000 | 4096  | 100    | 68.614 |
| ViT-Small-Patch16 | 4.241G  | 21.955M  | 224x224    | 2 RTX A5000 | 4096  | 100    | 79.006 |
| ViT-Base-Patch16  | 16.849G | 86.377M  | 224x224    | 2 RTX A5000 | 4096  | 100    | 83.204 |
| ViT-Large-Patch16 | 59.647G | 304.024M | 224x224    | 2 RTX A5000 | 4096  | 100    | 85.020 |

You can find more model training details in classification_training/imagenet/.

## ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs    | params   | input size | gpu num     | batch | epochs | Top-1  |
| ----------------- | ------- | -------- | ---------- | ----------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 16.849G | 86.377M  | 224x224    | 2 RTX A5000 | 4096  | 100    | 83.290 |
| ViT-Large-Patch16 | 59.647G | 304.024M | 224x224    | 2 RTX A5000 | 4096  | 100    | 85.876 |

You can find more model training details in classification_training/imagenet/.

## ResNet train from ImageNet1K pretrain weight on ImageNet21K(Winter 2021 release)

| Network   | macs    | params  | input size | gpu num     | batch | epochs | Semantic Softmax Acc |
| --------- | ------- | ------- | ---------- | ----------- | ----- | ------ | -------------------- |
| ResNet18  | 1.819G  | 11.690M | 224x224    | 2 RTX A5000 | 4096  | 80     | 68.639               |
| ResNet34  | 3.671G  | 21.798M | 224x224    | 2 RTX A5000 | 4096  | 80     | 71.873               |
| ResNet50  | 4.112G  | 25.557M | 224x224    | 2 RTX A5000 | 4096  | 80     | 74.664               |
| ResNet101 | 7.834G  | 44.549M | 224x224    | 2 RTX A5000 | 4096  | 80     | 76.136               |
| ResNet152 | 11.559G | 60.193M | 224x224    | 2 RTX A5000 | 4096  | 80     | 75.731               |

You can find more model training details in classification_training/imagenet21k/.

# Object detection task results

**RetinaNet**

Paper:https://arxiv.org/abs/1708.02002

**FCOS**

Paper:https://arxiv.org/abs/1904.01355

**CenterNet**

Paper:https://arxiv.org/abs/1904.07850

**TTFNet**

Paper:https://arxiv.org/abs/1909.00700

**DETR**

Paper:https://arxiv.org/abs/2005.12872

**DINO-DETR**

Paper:https://arxiv.org/abs/2203.03605

## All detection models training from scratch on COCO2017

ResNet50 backbone using DINO pretrained weight on ImageNet1K.

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Network               | resize-style    | input size | macs     | params  | gpu num     | batch | epochs | mAP    |
| --------------------- | --------------- | ---------- | -------- | ------- | ----------- | ----- | ------ | ------ |
| ResNet50-RetinaNet    | YoloStyle-640   | 640x640    | 95.558G  | 37.969M | 2 RTX A5000 | 32    | 13     | 34.459 |
| ResNet50-RetinaNet    | YoloStyle-800   | 800x800    | 149.522G | 37.969M | 2 RTX A5000 | 32    | 13     | 36.023 |
| ResNet50-RetinaNet    | RetinaStyle-800 | 800x1333   | 250.069G | 37.969M | 2 RTX A5000 | 8     | 13     | 35.434 |
| ResNet50-FCOS         | YoloStyle-640   | 640x640    | 81.943G  | 32.291M | 2 RTX A5000 | 32    | 13     | 37.176 |
| ResNet50-FCOS         | YoloStyle-800   | 800x800    | 128.160G | 32.291M | 2 RTX A5000 | 32    | 13     | 38.745 |
| ResNet50-FCOS         | RetinaStyle-800 | 800x1333   | 214.406G | 32.291M | 2 RTX A5000 | 8     | 13     | 39.649 |
| ResNet18DCN-CenterNet | YoloStyle-512   | 512x512    | 14.854G  | 12.889M | 2 RTX A5000 | 64    | 140    | 26.209 |
| ResNet18DCN-TTFNet-3x | YoloStyle-512   | 512x512    | 16.063G  | 13.737M | 2 RTX A5000 | 64    | 39     | 27.054 |
| ResNet50-DETR         | YoloStyle-1024  | 1024x1024  | 89.577G  | 30.440M | 8 RTX A5000 | 64    | 500    | 36.941 |
| ResNet50-DINO-DETR    | YoloStyle-1024  | 1024x1024  | 844.204G | 47.082M | 8 RTX A5000 | 16    | 13     | 42.870 |
| ResNet50-DINO-DETR    | YoloStyle-1024  | 1024x1024  | 844.204G | 47.082M | 8 RTX A5000 | 16    | 39     | 45.445 |

You can find more model training details in detection_training/coco/.

## All detection models finetune from objects365 pretrain weight on COCO2017

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Network            | resize-style    | input size | macs     | params  | gpu num     | batch | epochs | mAP    |
| ------------------ | --------------- | ---------- | -------- | ------- | ----------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640   | 640x640    | 95.558G  | 37.969M | 2 RTX A5000 | 32    | 13     | 38.930 |
| ResNet50-RetinaNet | YoloStyle-800   | 800x800    | 149.522G | 37.969M | 2 RTX A5000 | 32    | 13     | 40.483 |
| ResNet50-RetinaNet | RetinaStyle-800 | 800x1333   | 250.069G | 37.969M | 2 RTX A5000 | 8     | 13     | 40.424 |
| ResNet50-FCOS      | YoloStyle-640   | 640x640    | 81.943G  | 32.291M | 2 RTX A5000 | 32    | 13     | 42.871 |
| ResNet50-FCOS      | YoloStyle-800   | 800x800    | 128.160G | 32.291M | 2 RTX A5000 | 32    | 13     | 44.526 |
| ResNet50-FCOS      | RetinaStyle-800 | 800x1333   | 214.406G | 32.291M | 2 RTX A5000 | 8     | 13     | 42.848 |

You can find more model training details in detection_training/coco/.

## All detection models train from COCO2017 pretrain weight on Objects365(v2,2020)

Trained on objects365 train dataset, tested on objects365 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Network            | resize-style    | input size | macs     | params  | gpu num     | batch | epochs | mAP    |
| ------------------ | --------------- | ---------- | -------- | ------- | ----------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-800   | 800x800    | 149.522G | 37.969M | 8 RTX A5000 | 32    | 13     | 16.360 |
| ResNet50-FCOS      | RetinaStyle-800 | 800x1333   | 214.406G | 32.291M | 8 RTX A5000 | 32    | 13     | 17.068 |

## All detection models training from scratch on VOC2007 and VOC2012

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Network            | resize-style  | input size | macs    | params  | gpu num     | batch | epochs | mAP    |
| ------------------ | ------------- | ---------- | ------- | ------- | ----------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640 | 640x640    | 84.947G | 36.724M | 2 RTX A5000 | 32    | 13     | 81.948 |
| ResNet50-FCOS      | YoloStyle-640 | 640x640    | 80.764G | 32.153M | 2 RTX A5000 | 32    | 13     | 81.624 |

You can find more model training details in detection_training/voc/.

## All detection models finetune from objects365 pretrain weight on VOC2007 and VOC2012

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Network            | resize-style  | input size | macs    | params  | gpu num     | batch | epochs | mAP    |
| ------------------ | ------------- | ---------- | ------- | ------- | ----------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640 | 640x640    | 84.947G | 36.724M | 2 RTX A5000 | 32    | 13     | 90.220 |
| ResNet50-FCOS      | YoloStyle-640 | 640x640    | 80.764G | 32.153M | 2 RTX A5000 | 32    | 13     | 90.371 |

You can find more model training details in detection_training/voc/.

# Semantic Segmentation task results

**DeepLabv3+**

Paper:https://arxiv.org/abs/1802.02611

**U2Net**

Paper:https://arxiv.org/abs/2005.09007

## All semantic segmentation models training from scratch on ADE20K

| Network             | input size | macs     | params  | gpu num     | batch | epochs | miou   |
| ------------------- | ---------- | -------- | ------- | ----------- | ----- | ------ | ------ |
| ResNet50-DeepLabv3+ | 512x512    | 25.548G  | 26.738M | 2 RTX A5000 | 8     | 128    | 34.659 |
| U2Net               | 512x512    | 219.012G | 46.191M | 2 RTX A5000 | 8     | 128    | 39.046 |

You can find more model training details in semantic_segmentation_training/ade20k/.

## All semantic segmentation models training from scratch on COCO2017

| Network             | input size | macs     | params  | gpu num     | batch | epochs | miou   |
| ------------------- | ---------- | -------- | ------- | ----------- | ----- | ------ | ------ |
| ResNet50-DeepLabv3+ | 512x512    | 25.548G  | 26.738M | 2 RTX A5000 | 32    | 64     | 64.176 |
| U2Net               | 512x512    | 219.012G | 46.191M | 4 RTX A5000 | 32    | 64     | 66.529 |

You can find more model training details in semantic_segmentation_training/coco/.

# Knowledge distillation task results

**KD loss**

Paper:https://arxiv.org/abs/1503.02531

**DML loss**

Paper:https://arxiv.org/abs/1706.00384

## ResNet training from pretrain weight on ImageNet1K(ILSVRC2012)

| Teacher Network | Student Network | method         | Freeze Teacher | input size | gpu num     | batch | epochs | Teacher Top-1 | Student Top-1 |
| --------------- | --------------- | -------------- | -------------- | ---------- | ----------- | ----- | ------ | ------------- | ------------- |
| ResNet152       | ResNet50        | CE+KD          | True           | 224x224    | 2 RTX A5000 | 256   | 100    | /             | 77.352        |
| ResNet152       | ResNet50        | CE+DML         | False          | 224x224    | 2 RTX A5000 | 256   | 100    | 79.274        | 78.122        |
| ResNet152       | ResNet50        | CE+KD+Vit Aug  | True           | 224x224    | 2 RTX A5000 | 4096  | 300    | /             | 80.168        |
| ResNet152       | ResNet50        | CE+DML+Vit Aug | False          | 224x224    | 2 RTX A5000 | 4096  | 300    | 81.508        | 79.810        |

You can find more model training details in distillation_training/imagenet/.

# Contrastive learning task results

**DINO:Emerging Properties in Self-Supervised Vision Transformers**

Paper:https://arxiv.org/abs/2104.14294

## ResNet DINO pretrain on ImageNet1K(ILSVRC2012)

| Network       | input size | gpu num     | batch | epochs | Loss  |
| ------------- | ---------- | ----------- | ----- | ------ | ----- |
| ResNet50-DINO | 224x224    | 4 RTX A5000 | 256   | 400    | 1.997 |

You can find more model training details in contrastive_learning_training/imagenet/.

## ResNet finetune from DINO pretrain weight on ImageNet1K(ILSVRC2012)

| Network  | macs   | params  | input size | gpu num     | batch | epochs | Top-1  |
| -------- | ------ | ------- | ---------- | ----------- | ----- | ------ | ------ |
| ResNet50 | 4.112G | 25.557M | 224x224    | 2 RTX A5000 | 256   | 100    | 77.114 |
| ResNet50 | 4.112G | 25.557M | 224x224    | 2 RTX A5000 | 4096  | 300    | 79.418 |

You can find more model training details in classification_training/imagenet/.

# Masked image modeling task results

**MAE:Masked Autoencoders Are Scalable Vision Learners**

Paper:https://arxiv.org/abs/2111.06377

## ViT MAE pretrain on ImageNet1K(ILSVRC2012)

| Network           | input size | gpu num     | batch | epochs | Loss  |
| ----------------- | ---------- | ----------- | ----- | ------ | ----- |
| ViT-Tiny-Patch16  | 224x224    | 1 RTX A5000 | 256   | 400    | 0.427 |
| ViT-Small-Patch16 | 224x224    | 2 RTX A5000 | 256   | 400    | 0.414 |
| ViT-Base-Patch16  | 224x224    | 2 RTX A5000 | 256   | 400    | 0.388 |
| ViT-Large-Patch16 | 224x224    | 2 RTX A5000 | 256   | 400    | 0.378 |

You can find more model training details in masked_image_modeling_training/imagenet/.

## ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs    | params   | input size | gpu num     | batch | epochs | Top-1  |
| ----------------- | ------- | -------- | ---------- | ----------- | ----- | ------ | ------ |
| ViT-Tiny-Patch16  | 1.075G  | 5.670M   | 224x224    | 1 RTX A5000 | 4096  | 100    | 68.614 |
| ViT-Small-Patch16 | 4.241G  | 21.955M  | 224x224    | 2 RTX A5000 | 4096  | 100    | 79.006 |
| ViT-Base-Patch16  | 16.849G | 86.377M  | 224x224    | 2 RTX A5000 | 4096  | 100    | 83.204 |
| ViT-Large-Patch16 | 59.647G | 304.024M | 224x224    | 2 RTX A5000 | 4096  | 100    | 85.020 |

You can find more model training details in classification_training/imagenet/.

## ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs    | params   | input size | gpu num     | batch | epochs | Top-1  |
| ----------------- | ------- | -------- | ---------- | ----------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 16.849G | 86.377M  | 224x224    | 2 RTX A5000 | 4096  | 100    | 83.290 |
| ViT-Large-Patch16 | 59.647G | 304.024M | 224x224    | 2 RTX A5000 | 4096  | 100    | 85.876 |

You can find more model training details in classification_training/imagenet/.

# Image inpainting model task results

**Aggregated Contextual Transformations for High-Resolution Image Inpainting**

Paper:https://arxiv.org/abs/2104.01431

## All image inpainting model training from scratch on CelebA-HQ

Trained image inpainting model on CelebA-HQ dataset.Test image num=2000.

| Network | input size | epochs | Mask     | mae    | psnr   | ssim   | fid    |
| ------- | ---------- | ------ | -------- | ------ | ------ | ------ | ------ |
| AOT-GAN | 512x512    | 100    | 0.01-0.1 | 0.0023 | 40.329 | 0.9858 | 0.7849 |
| AOT-GAN | 512x512    | 100    | 0.1-0.2  | 0.0064 | 33.598 | 0.9599 | 2.1839 |
| AOT-GAN | 512x512    | 100    | 0.2-0.3  | 0.0124 | 29.861 | 0.9253 | 3.8720 |
| AOT-GAN | 512x512    | 100    | 0.3-0.4  | 0.0195 | 27.198 | 0.8866 | 5.7180 |
| AOT-GAN | 512x512    | 100    | 0.4-0.5  | 0.0285 | 24.970 | 0.8424 | 8.7763 |
| AOT-GAN | 512x512    | 100    | 0.5-0.6  | 0.0497 | 21.422 | 0.7682 | 15.476 |

You can find more model training details in image_inpainting_training/celebahq/.

## All image inpainting model training from scratch on Places365-standard

Trained image inpainting model on Places365-standard dataset.Test image num=36500.

| Network | input size | epochs | Mask     | mae    | psnr   | ssim   | fid    |
| ------- | ---------- | ------ | -------- | ------ | ------ | ------ | ------ |
| AOT-GAN | 512x512    | 100    | 0.01-0.1 | 0.0041 | 35.479 | 0.9772 | 0.1518 |
| AOT-GAN | 512x512    | 100    | 0.1-0.2  | 0.0114 | 29.276 | 0.9378 | 0.5504 |
| AOT-GAN | 512x512    | 100    | 0.2-0.3  | 0.0213 | 25.858 | 0.8866 | 1.4065 |
| AOT-GAN | 512x512    | 100    | 0.3-0.4  | 0.0328 | 23.470 | 0.8314 | 2.9936 |
| AOT-GAN | 512x512    | 100    | 0.4-0.5  | 0.0463 | 21.590 | 0.7711 | 5.8776 |
| AOT-GAN | 512x512    | 100    | 0.5-0.6  | 0.0720 | 19.029 | 0.6851 | 16.803 |


| Network       | input size | epochs | Mask     | mae    | psnr   | ssim   | fid    |
| ------------- | ---------- | ------ | -------- | ------ | ------ | ------ | ------ |
| AOT-GAN-light | 512x512    | 100    | 0.01-0.1 | 0.0044 | 34.932 | 0.9754 | 0.1946 |
| AOT-GAN-light | 512x512    | 100    | 0.1-0.2  | 0.0122 | 28.858 | 0.9336 | 0.7815 |
| AOT-GAN-light | 512x512    | 100    | 0.2-0.3  | 0.0225 | 25.526 | 0.8805 | 2.1669 |
| AOT-GAN-light | 512x512    | 100    | 0.3-0.4  | 0.0345 | 23.196 | 0.8238 | 4.8066 |
| AOT-GAN-light | 512x512    | 100    | 0.4-0.5  | 0.0485 | 21.362 | 0.7625 | 9.5062 |
| AOT-GAN-light | 512x512    | 100    | 0.5-0.6  | 0.0747 | 18.879 | 0.6771 | 24.950 |

You can find more model training details in image_inpainting_training/places365_standard.


## All image inpainting model training from scratch on Places365-challenge

Trained image inpainting model on Places365-challenge dataset.Test image num=36500.

| Network | input size | epochs | Mask     | mae    | psnr   | ssim   | fid    |
| ------- | ---------- | ------ | -------- | ------ | ------ | ------ | ------ |
| AOT-GAN | 512x512    | 100    | 0.01-0.1 | 0.0040 | 35.596 | 0.9775 | 0.1396 |
| AOT-GAN | 512x512    | 100    | 0.1-0.2  | 0.0113 | 29.338 | 0.9382 | 0.4866 |
| AOT-GAN | 512x512    | 100    | 0.2-0.3  | 0.0212 | 25.882 | 0.8868 | 1.2023 |
| AOT-GAN | 512x512    | 100    | 0.3-0.4  | 0.0327 | 23.464 | 0.8309 | 2.4987 |
| AOT-GAN | 512x512    | 100    | 0.4-0.5  | 0.0462 | 21.566 | 0.7696 | 4.9181 |
| AOT-GAN | 512x512    | 100    | 0.5-0.6  | 0.0721 | 18.974 | 0.6804 | 14.437 |


| Network       | input size | epochs | Mask     | mae    | psnr   | ssim   | fid    |
| ------------- | ---------- | ------ | -------- | ------ | ------ | ------ | ------ |
| AOT-GAN-light | 512x512    | 100    | 0.01-0.1 | 0.0042 | 35.217 | 0.9763 | 0.1695 |
| AOT-GAN-light | 512x512    | 100    | 0.1-0.2  | 0.0118 | 29.055 | 0.9356 | 0.6544 |
| AOT-GAN-light | 512x512    | 100    | 0.2-0.3  | 0.0219 | 25.659 | 0.8835 | 1.7514 |
| AOT-GAN-light | 512x512    | 100    | 0.3-0.4  | 0.0336 | 23.281 | 0.8275 | 3.8301 |
| AOT-GAN-light | 512x512    | 100    | 0.4-0.5  | 0.0474 | 21.409 | 0.7665 | 7.5940 |
| AOT-GAN-light | 512x512    | 100    | 0.5-0.6  | 0.0739 | 18.853 | 0.6803 | 20.025 |

You can find more model training details in image_inpainting_training/places365_challenge.

# Diffusion model task results

**Denoising Diffusion Probabilistic Models**

Paper:https://arxiv.org/abs/2006.11239

**Denoising Diffusion Implicit Models**

Paper:https://arxiv.org/abs/2010.02502

**High-Resolution Image Synthesis with Latent Diffusion Models**

Paper:https://arxiv.org/abs/2112.10752

## All diffusion model with different sampling methods on CIFAR10

Trained diffusion unet on CIFAR10 dataset(DDPM method).Test image num=50000.

| sampling method | input size | steps | condition label(train/test) | FID   | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ----- | ------------------ |
| DDPM            | 32x32      | 1000  | False/False                 | 5.394 | 8.684/0.169        |
| DDIM            | 32x32      | 50    | False/False                 | 7.644 | 8.642/0.129        |
| PLMS            | 32x32      | 20    | False/False                 | 7.027 | 8.834/0.200        |
| DDPM            | 32x32      | 1000  | True/True                   | 3.949 | 8.985/0.139        |

You can find more model training details in diffusion_model_training/cifar10/.

## All diffusion model with different sampling methods on CIFAR100

Trained diffusion unet on CIFAR100 dataset(DDPM method).Test image num=50000.

| sampling method | input size | steps | condition label(train/test) | FID    | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ------ | ------------------ |
| DDPM            | 32x32      | 1000  | False/False                 | 9.620  | 9.399/0.138        |
| DDIM            | 32x32      | 50    | False/False                 | 13.250 | 8.946/0.150        |
| PLMS            | 32x32      | 20    | False/False                 | 11.854 | 9.391/0.202        |
| DDPM            | 32x32      | 1000  | True/True                   | 5.209  | 10.880/0.180       |

You can find more model training details in diffusion_model_training/cifar100/.

## All diffusion model with different sampling methods on CelebA-HQ

Trained diffusion unet on CelebA-HQ dataset(DDPM method).Test image num=28000.

| sampling method | input size | steps | condition label(train/test) | FID    | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ------ | ------------------ |
| DDPM            | 64x64      | 1000  | False/False                 | 6.491  | 2.577/0.035        |
| DDIM            | 64x64      | 50    | False/False                 | 15.195 | 2.625/0.028        |
| PLMS            | 64x64      | 20    | False/False                 | 18.061 | 2.701/0.040        |

You can find more model training details in diffusion_model_training/celebahq/.

## All diffusion model with different sampling methods on FFHQ

Trained diffusion unet on FFHQ dataset(DDPM method).Test image num=60000.

| sampling method | input size | steps | condition label(train/test) | FID    | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ------ | ------------------ |
| DDPM            | 64x64      | 1000  | False/False                 | 6.671  | 3.399/0.055        |
| DDIM            | 64x64      | 50    | False/False                 | 10.479 | 3.431/0.044        |
| PLMS            | 64x64      | 20    | False/False                 | 12.387 | 3.462/0.034        |

You can find more model training details in diffusion_model_training/ffhq/.