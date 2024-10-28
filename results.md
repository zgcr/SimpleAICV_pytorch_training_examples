- [Image classification task results](#image-classification-task-results)
  - [ResNetCifar training from scratch on CIFAR100](#resnetcifar-training-from-scratch-on-cifar100)
  - [Convformer finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)](#convformer-finetune-from-offical-pretrain-weight-on-imagenet1kilsvrc2012)
  - [DarkNet training from scratch on ImageNet1K(ILSVRC2012)](#darknet-training-from-scratch-on-imagenet1kilsvrc2012)
  - [ResNet training from scratch on ImageNet1K(ILSVRC2012)](#resnet-training-from-scratch-on-imagenet1kilsvrc2012)
  - [ResNet finetune from ImageNet21k pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-finetune-from-imagenet21k-pretrain-weight-on-imagenet1kilsvrc2012)
  - [VAN finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)](#van-finetune-from-offical-pretrain-weight-on-imagenet1kilsvrc2012)
  - [ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-self-trained-mae-pretrain-weight400epoch-on-imagenet1kilsvrc2012)
  - [ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-offical-mae-pretrain-weight800-epoch-on-imagenet1kilsvrc2012)
  - [ResNet train from ImageNet1K pretrain weight on ImageNet21K(Winter 2021 release)](#resnet-train-from-imagenet1k-pretrain-weight-on-imagenet21kwinter-2021-release)
- [Knowledge distillation task results](#knowledge-distillation-task-results)
  - [ResNet distill from pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-distill-from-pretrain-weight-on-imagenet1kilsvrc2012)
- [Masked image modeling task results](#masked-image-modeling-task-results)
  - [ViT MAE pretrain on ImageNet1K(ILSVRC2012)](#vit-mae-pretrain-on-imagenet1kilsvrc2012)
- [Object detection task results](#object-detection-task-results)
  - [All detection models training from scratch on COCO2017](#all-detection-models-training-from-scratch-on-coco2017)
  - [All detection models finetune from objects365 pretrain weight on COCO2017](#all-detection-models-finetune-from-objects365-pretrain-weight-on-coco2017)
  - [All detection models train on Objects365(v2,2020) from COCO2017 pretrain weight](#all-detection-models-train-on-objects365v22020-from-coco2017-pretrain-weight)
  - [All detection models training from scratch on VOC2007 and VOC2012](#all-detection-models-training-from-scratch-on-voc2007-and-voc2012)
  - [All detection models finetune from objects365 pretrain weight on VOC2007 and VOC2012](#all-detection-models-finetune-from-objects365-pretrain-weight-on-voc2007-and-voc2012)
- [Semantic Segmentation task results](#semantic-segmentation-task-results)
  - [All semantic segmentation models training from scratch on ADE20K](#all-semantic-segmentation-models-training-from-scratch-on-ade20k)
  - [All semantic segmentation models training from scratch on COCO2017](#all-semantic-segmentation-models-training-from-scratch-on-coco2017)
- [Instance Segmentation task results](#instance-segmentation-task-results)
  - [All instance segmentation models training from scratch on COCO2017](#all-instance-segmentation-models-training-from-scratch-on-coco2017)
- [Salient object detection task results](#salient-object-detection-task-results)
- [Human matting task results](#human-matting-task-results)
- [OCR text detection task results](#ocr-text-detection-task-results)
- [OCR text recognition task results](#ocr-text-recognition-task-results)
- [Face detection task results](#face-detection-task-results)
- [Diffusion model task results](#diffusion-model-task-results)
  - [All diffusion model with different sampling methods on CelebA-HQ](#all-diffusion-model-with-different-sampling-methods-on-celeba-hq)
  - [All diffusion model with different sampling methods on CIFAR10](#all-diffusion-model-with-different-sampling-methods-on-cifar10)
  - [All diffusion model with different sampling methods on CIFAR100](#all-diffusion-model-with-different-sampling-methods-on-cifar100)
  - [All diffusion model with different sampling methods on FFHQ](#all-diffusion-model-with-different-sampling-methods-on-ffhq)


# Image classification task results

**Convformer**

Paper:https://arxiv.org/pdf/2210.13452

**DarkNet**

Paper:https://arxiv.org/abs/1804.02767?e05802c1_page=1

**ResNet**

Paper:https://arxiv.org/abs/1512.03385

**VAN**

Paper:https://arxiv.org/abs/2202.09741

**ViT**

Paper:https://arxiv.org/abs/2010.11929

## ResNetCifar training from scratch on CIFAR100 

**ResNetCifar is different from ResNet in the first few layers.**

| Network        | macs     | params  | input size | batch | epochs | Top-1  |
| -------------- | -------- | ------- | ---------- | ----- | ------ | ------ |
| ResNet18Cifar  | 557.935M | 11.220M | 32x32      | 128   | 200    | 76.890 |
| ResNet34Cifar  | 1.164G   | 21.328M | 32x32      | 128   | 200    | 78.010 |
| ResNet50Cifar  | 1.312G   | 23.705M | 32x32      | 128   | 200    | 75.360 |
| ResNet101Cifar | 2.531G   | 42.697M | 32x32      | 128   | 200    | 77.180 |
| ResNet152Cifar | 3.751G   | 58.341M | 32x32      | 128   | 200    | 77.340 |

You can find more model training details in classification_training/cifar100/.

## Convformer finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)

| Network        | macs    | params  | input size | batch | epochs | Top-1  |
| -------------- | ------- | ------- | ---------- | ----- | ------ | ------ |
| convformer-s18 | 3.953G  | 24.184M | 224x224    | 2048  | 300    | 82.018 |
| convformer-s36 | 7.663G  | 37.424M | 224x224    | 1024  | 300    | 83.290 |
| convformer-m36 | 12.876G | 53.994M | 224x224    | 1024  | 300    | 84.000 |
| convformer-b36 | 22.673G | 95.216M | 224x224    | 1024  | 300    | 84.480 |

You can find more model training details in classification_training/imagenet/.

## DarkNet training from scratch on ImageNet1K(ILSVRC2012)

| Network     | macs     | params  | input size | batch | epochs | Top-1  |
| ----------- | -------- | ------- | ---------- | ----- | ------ | ------ |
| DarkNetTiny | 414.602M | 2.087M  | 256x256    | 256   | 100    | 57.858 |
| DarkNet19   | 3.669G   | 20.842M | 256x256    | 256   | 100    | 74.364 |
| DarkNet53   | 9.335G   | 41.610M | 256x256    | 256   | 100    | 76.250 |

You can find more model training details in classification_training/imagenet/.

## ResNet training from scratch on ImageNet1K(ILSVRC2012)

| Network   | macs    | params  | input size | batch | epochs | Top-1  |
| --------- | ------- | ------- | ---------- | ----- | ------ | ------ |
| ResNet18  | 1.824G  | 11.690M | 224x224    | 256   | 100    | 70.594 |
| ResNet34  | 3.679G  | 21.798M | 224x224    | 256   | 100    | 73.622 |
| ResNet50  | 4.134G  | 25.557M | 224x224    | 256   | 100    | 76.182 |
| ResNet101 | 7.866G  | 44.549M | 224x224    | 256   | 100    | 77.242 |
| ResNet152 | 11.604G | 60.193M | 224x224    | 256   | 100    | 77.772 |

You can find more model training details in classification_training/imagenet/.

## ResNet finetune from ImageNet21k pretrain weight on ImageNet1K(ILSVRC2012)

| Network   | macs    | params  | input size | batch | epochs | Top-1  |
| --------- | ------- | ------- | ---------- | ----- | ------ | ------ |
| ResNet50  | 4.134G  | 25.557M | 224x224    | 2048  | 300    | 80.258 |
| ResNet101 | 7.866G  | 44.549M | 224x224    | 1024  | 300    | 81.668 |
| ResNet152 | 11.604G | 60.193M | 224x224    | 1024  | 300    | 81.934 |

You can find more model training details in classification_training/imagenet/.

## VAN finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)

| Network | macs     | params  | input size | batch | epochs | Top-1  |
| ------- | -------- | ------- | ---------- | ----- | ------ | ------ |
| van-b0  | 870.860M | 4.103M  | 224x224    | 2048  | 300    | 75.424 |
| van-b1  | 2.506G   | 13.856M | 224x224    | 2048  | 300    | 80.740 |
| van-b2  | 5.010G   | 26.567M | 224x224    | 1024  | 300    | 82.592 |
| van-b3  | 8.951G   | 26.567M | 224x224    | 1024  | 300    | 83.202 |

You can find more model training details in classification_training/imagenet/.

## ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs     | params   | input size | batch | epochs | Top-1  |
| ----------------- | -------- | -------- | ---------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 16.880G  | 86.416M  | 224x224    | 256   | 100    | 82.676 |
| ViT-Large-Patch16 | 59.731G  | 304.124M | 224x224    | 128   | 50     | 84.978 |
| ViT-Huge-Patch14  | 162.071G | 631.716M | 224x224    | 128   | 50     | 85.966 |

You can find more model training details in classification_training/imagenet/.

## ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs     | params   | input size | batch | epochs | Top-1  |
| ----------------- | -------- | -------- | ---------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 16.880G  | 86.416M  | 224x224    | 256   | 100    | 83.404 |
| ViT-Large-Patch16 | 59.731G  | 304.124M | 224x224    | 128   | 50     | 85.672 |
| ViT-Huge-Patch14  | 162.071G | 631.716M | 224x224    | 128   | 50     | 86.608 |

You can find more model training details in classification_training/imagenet/.

## ResNet train from ImageNet1K pretrain weight on ImageNet21K(Winter 2021 release)

| Network   | macs    | params  | input size | batch | epochs | Semantic Softmax Acc |
| --------- | ------- | ------- | ---------- | ----- | ------ | -------------------- |
| ResNet50  | 4.134G  | 25.557M | 224x224    | 2048  | 80     | 75.319               |
| ResNet101 | 7.866G  | 44.549M | 224x224    | 2048  | 80     | 76.795               |
| ResNet152 | 11.604G | 60.193M | 224x224    | 1024  | 80     | 77.345               |

You can find more model training details in classification_training/imagenet21k/.

# Knowledge distillation task results

**DML loss**

Paper:https://arxiv.org/abs/1706.00384

**KD loss**

Paper:https://arxiv.org/abs/1503.02531

## ResNet distill from pretrain weight on ImageNet1K(ILSVRC2012)

| Teacher Network | Student Network | method         | Freeze Teacher | input size | batch | epochs | Teacher Top-1 | Student Top-1 |
| --------------- | --------------- | -------------- | -------------- | ---------- | ----- | ------ | ------------- | ------------- |
| ResNet152       | ResNet50        | CE+DML         | False          | 224x224    | 256   | 100    | 79.246        | 78.168        |
| ResNet152       | ResNet50        | CE+DML+Vit Aug | False          | 224x224    | 1024  | 300    | 82.760        | 80.798        |
| ResNet152       | ResNet50        | CE+KD          | True           | 224x224    | 256   | 100    | 77.764        | 77.566        |
| ResNet152       | ResNet50        | CE+KD+Vit Aug  | True           | 224x224    | 2048  | 300    | 81.936        | 80.806        |


You can find more model training details in distillation_training/imagenet/.

# Masked image modeling task results

**MAE:Masked Autoencoders Are Scalable Vision Learners**

Paper:https://arxiv.org/abs/2111.06377

## ViT MAE pretrain on ImageNet1K(ILSVRC2012)

| Network           | input size | batch | epochs | Loss  |
| ----------------- | ---------- | ----- | ------ | ----- |
| ViT-Base-Patch16  | 224x224    | 1024  | 400    | 0.388 |
| ViT-Large-Patch16 | 224x224    | 1024  | 400    | 0.378 |
| ViT-Huge-Patch14  | 224x224    | 1024  | 400    | 0.350 |

You can find more model training details in masked_image_modeling_training/imagenet/.

# Object detection task results

**DETR**

Paper:https://arxiv.org/abs/2005.12872

**DINO-DETR**

Paper:https://arxiv.org/abs/2203.03605

**RetinaNet**

Paper:https://arxiv.org/abs/1708.02002

**FCOS**

Paper:https://arxiv.org/abs/1904.01355

## All detection models training from scratch on COCO2017

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Network            | resize-style    | input size | macs     | params  | batch | epochs | mAP    |
| ------------------ | --------------- | ---------- | -------- | ------- | ----- | ------ | ------ |
| ResNet50-DETR      | YoloStyle-1024  | 1024x1024  | 89.577G  | 30.440M | 64    | 500    | 38.609 |
| ResNet50-DINO-DETR | YoloStyle-1024  | 1024x1024  | 844.204G | 47.082M | 16    | 39     | 47.396 |
| ResNet50-RetinaNet | RetinaStyle-800 | 800x1333   | 250.069G | 37.969M | 16    | 13     | 37.281 |
| ResNet50-FCOS      | RetinaStyle-800 | 800x1333   | 214.406G | 32.291M | 16    | 13     | 41.071 |

You can find more model training details in detection_training/coco/.

## All detection models finetune from objects365 pretrain weight on COCO2017

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Network            | resize-style    | input size | macs     | params  | batch | epochs | mAP    |
| ------------------ | --------------- | ---------- | -------- | ------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | RetinaStyle-800 | 800x1333   | 250.069G | 37.969M | 16    | 13     | 40.947 |
| ResNet50-FCOS      | RetinaStyle-800 | 800x1333   | 214.406G | 32.291M | 16    | 13     | 46.511 |

You can find more model training details in detection_training/coco/.

## All detection models train on Objects365(v2,2020) from COCO2017 pretrain weight

Trained on objects365(v2,2020) train dataset, tested on objects365(v2,2020) val dataset.

| Network            | resize-style   | input size | batch | epochs | loss  |
| ------------------ | -------------- | ---------- | ----- | ------ | ----- |
| ResNet50-RetinaNet | YoloStyle-1024 | 1024x1024  | 32    | 13     | 0.355 |
| ResNet50-FCOS      | YoloStyle-1024 | 1024x1024  | 64    | 13     | 0.968 |

## All detection models training from scratch on VOC2007 and VOC2012

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Network            | resize-style  | input size | macs    | params  | batch | epochs | mAP    |
| ------------------ | ------------- | ---------- | ------- | ------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640 | 640x640    | 84.947G | 36.724M | 32    | 13     | 83.765 |
| ResNet50-FCOS      | YoloStyle-640 | 640x640    | 80.764G | 32.153M | 32    | 13     | 83.250 |

You can find more model training details in detection_training/voc/.

## All detection models finetune from objects365 pretrain weight on VOC2007 and VOC2012

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Network            | resize-style  | input size | macs    | params  | batch | epochs | mAP    |
| ------------------ | ------------- | ---------- | ------- | ------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640 | 640x640    | 84.947G | 36.724M | 32    | 13     | 90.082 |
| ResNet50-FCOS      | YoloStyle-640 | 640x640    | 80.764G | 32.153M | 32    | 13     | 90.585 |

You can find more model training details in detection_training/voc/.

# Semantic Segmentation task results

**DeepLabv3+**

Paper:https://arxiv.org/abs/1802.02611

## All semantic segmentation models training from scratch on ADE20K

| Network                     | input size | macs    | params  | batch | epochs | mean_iou |
| --------------------------- | ---------- | ------- | ------- | ----- | ------ | -------- |
| resnet50_deeplabv3plus      | 512x512    | 43.500G | 30.254M | 32    | 100    | 40.462   |
| convformerm36_deeplabv3plus | 512x512    | 83.898G | 56.760M | 32    | 100    | 47.826   |

You can find more model training details in semantic_segmentation_training/ade20k/.

## All semantic segmentation models training from scratch on COCO2017

| Network                     | input size | macs    | params  | batch | epochs | mean_iou |
| --------------------------- | ---------- | ------- | ------- | ----- | ------ | -------- |
| resnet50_deeplabv3plus      | 512x512    | 43.500G | 30.254M | 64    | 100    | 68.975   |
| convformerm36_deeplabv3plus | 512x512    | 83.898G | 56.760M | 64    | 100    | 74.214   |

You can find more model training details in semantic_segmentation_training/coco/.

# Instance Segmentation task results

**YOLACT**

Paper:https://arxiv.org/abs/1904.02689

**SOLOv2**

Paper:https://arxiv.org/abs/2003.10152

## All instance segmentation models training from scratch on COCO2017

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Network              | resize-style   | input size | macs     | params  | batch | epochs | mAP    |
| -------------------- | -------------- | ---------- | -------- | ------- | ----- | ------ | ------ |
| resnet50_yolact      | YoloStyle-1024 | 1024x1024  | 202.012G | 31.165M | 64    | 39     | 26.342 |
| convformerm36_yolact | YoloStyle-1024 | 1024x1024  | 382.336G | 60.452M | 64    | 39     | 34.047 |
| resnet50_solov2      | YoloStyle-1024 | 1024x1024  | 248.965G | 46.582M | 32    | 39     | 37.807 |
| convformerm36_solov2 | YoloStyle-1024 | 1024x1024  | 426.605G | 75.828M | 32    | 39     | 40.296 |

You can find more model training details in instance_segmentation_training/coco/.

# Salient object detection task results

**PFAN+Segmentation**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2202.09741

Use combine dataset DIS5K/HRS10K/HRSOD/UHRSD to train and test.

| Network                         | macs     | params  | input size | batch | epochs | iou    | precision | recall | f_squared_beta |
| ------------------------------- | -------- | ------- | ---------- | ----- | ------ | ------ | --------- | ------ | -------------- |
| resnet50_pfan_segmentation      | 71.303G  | 26.580M | 832x832    | 96    | 100    | 0.8461 | 0.8970    | 0.9346 | 0.9053         |
| convformerm36_pfan_segmentation | 186.496G | 54.459M | 832x832    | 96    | 100    | 0.8865 | 0.9263    | 0.9517 | 0.9319         |

You can find more model training details in salient_object_detection_training/.

# Human matting task results

**PFAN+Matting**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2104.14222

Paper3:https://arxiv.org/abs/2202.09741

Use combine dataset Deep_Automatic_Portrait_Matting/RealWorldPortrait636/P3M10K to train and test.

| Network                    | macs     | params  | input size | batch | epochs | iou    | precision | recall | sad    | mae    | mse    | grad   | conn   |
| -------------------------- | -------- | ------- | ---------- | ----- | ------ | ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| resnet50_pfan_matting      | 86.093G  | 29.654M | 832x832    | 96    | 100    | 0.9809 | 0.9873    | 0.9932 | 6.0597 | 0.0087 | 0.0051 | 7.9799 | 5.8297 |
| convformerm36_pfan_matting | 195.854G | 55.503M | 832x832    | 96    | 100    | 0.9843 | 0.9901    | 0.9939 | 5.0884 | 0.0073 | 0.0038 | 6.4283 | 4.8397 |

You can find more model training details in human_matting_training/.

# OCR text detection task results

**DBNet**

Paper:https://arxiv.org/abs/1911.08947

Use combine dataset include ICDAR2017RCTW/ICDAR2019ART/ICDAR2019LSVT/ICDAR2019MLT to train and test.

| Network             | macs     | params  | input size | batch | epochs | precision | recall | f1     |
| ------------------- | -------- | ------- | ---------- | ----- | ------ | --------- | ------ | ------ |
| resnet50_dbnet      | 158.914G | 24.784M | 1024x1024  | 128   | 100    | 92.072    | 86.595 | 89.249 |
| convformerm36_dbnet | 340.367G | 54.528M | 1024x1024  | 64    | 100    | 92.748    | 89.947 | 91.326 |

You can find more model training details in ocr_text_detection_training/.

# OCR text recognition task results

**CRNN+LSTM+CTC**

Paper:https://arxiv.org/abs/1507.05717

Use combine dataset aistudio_baidu_street/chinese_dataset/synthetic_chinese_string_dataset/meta_self_learning_dataset to train and test.

| Network                 | macs    | params   | input size | batch | epochs | lcs_precision | lcs_recall |
| ----------------------- | ------- | -------- | ---------- | ----- | ------ | ------------- | ---------- |
| resnet50_ctc_model      | 12.509G | 179.870M | 32x512     | 1024  | 50     | 99.498        | 99.212     |
| convformerm36_ctc_model | 8.051G  | 70.121M  | 32x512     | 1024  | 50     | 99.452        | 99.201     |

You can find more model training details in ocr_text_recognition_training/.

# Face detection task results

**RetinaFace**

Paper:https://arxiv.org/pdf/1905.00641

Use WiderFace train and UFDD val datasets to train, WiderFace val dataset to test.

| Network             | macs     | params  | input size | batch | epochs | Easy AP | Medium AP | Hard AP |
| ------------------- | -------- | ------- | ---------- | ----- | ------ | ------- | --------- | ------- |
| resnet50_retinaface | 114.229G | 27.280M | 1024x1024  | 16    | 100    | 0.9369  | 0.9148    | 0.7801  |

You can find more model training details in face_detection_training/.

# Diffusion model task results

**Denoising Diffusion Probabilistic Models**

Paper:https://arxiv.org/abs/2006.11239

**Denoising Diffusion Implicit Models**

Paper:https://arxiv.org/abs/2010.02502

## All diffusion model with different sampling methods on CelebA-HQ

Trained diffusion unet on CelebA-HQ dataset(DDPM method).Test image num=28000.

| sampling method | input size | steps | condition label(train/test) | FID    | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ------ | ------------------ |
| DDPM            | 64x64      | 1000  | False/False                 | 6.409  | 2.486/0.082        |
| DDIM            | 64x64      | 50    | False/False                 | 14.623 | 2.622/0.073        |

You can find more model training details in diffusion_model_training/celebahq/.

## All diffusion model with different sampling methods on CIFAR10

Trained diffusion unet on CIFAR10 dataset(DDPM method).Test image num=50000.

| sampling method | input size | steps | condition label(train/test) | FID    | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ------ | ------------------ |
| DDPM            | 32x32      | 1000  | False/False                 | 10.302 | 8.213/0.257        |
| DDIM            | 32x32      | 50    | False/False                 | 12.440 | 8.318/0.408        |
| DDPM            | 32x32      | 1000  | True/True                   | 5.049  | 8.654/0.112        |

You can find more model training details in diffusion_model_training/cifar10/.

## All diffusion model with different sampling methods on CIFAR100

Trained diffusion unet on CIFAR100 dataset(DDPM method).Test image num=50000.

| sampling method | input size | steps | condition label(train/test) | FID    | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ------ | ------------------ |
| DDPM            | 32x32      | 1000  | False/False                 | 16.298 | 8.398/0.281        |
| DDIM            | 32x32      | 50    | False/False                 | 21.402 | 8.344/0.192        |
| DDPM            | 32x32      | 1000  | True/True                   | 6.953  | 10.344/0.150       |

You can find more model training details in diffusion_model_training/cifar100/.

## All diffusion model with different sampling methods on FFHQ

Trained diffusion unet on FFHQ dataset(DDPM method).Test image num=60000.

| sampling method | input size | steps | condition label(train/test) | FID    | IS score(mean/std) |
| --------------- | ---------- | ----- | --------------------------- | ------ | ------------------ |
| DDPM            | 64x64      | 1000  | False/False                 | 7.758  | 3.283/0.124        |
| DDIM            | 64x64      | 50    | False/False                 | 11.328 | 3.417/0.071        |

You can find more model training details in diffusion_model_training/ffhq/.