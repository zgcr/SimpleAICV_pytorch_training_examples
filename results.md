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
- [Face parsing task results](#face-parsing-task-results)
- [Human parsing task results](#human-parsing-task-results)
- [Interactive segmentation task results](#interactive-segmentation-task-results)
  - [light sam distill from pretrain weight on sa\_1b\_11w](#light-sam-distill-from-pretrain-weight-on-sa_1b_11w)
  - [light sam train on combine salient object detection and human matting dataset](#light-sam-train-on-combine-salient-object-detection-and-human-matting-dataset)
  - [light sam matting train on combine human matting dataset](#light-sam-matting-train-on-combine-human-matting-dataset)
  - [light sam matting train on combine salient object detection dataset](#light-sam-matting-train-on-combine-salient-object-detection-dataset)


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

You can find more model training details in 00.classification_training/cifar100/.

## Convformer finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)

| Network        | macs    | params  | input size | batch | epochs | Top-1  |
| -------------- | ------- | ------- | ---------- | ----- | ------ | ------ |
| convformer-s18 | 3.953G  | 24.184M | 224x224    | 2048  | 300    | 82.018 |
| convformer-s36 | 7.663G  | 37.424M | 224x224    | 1024  | 300    | 83.290 |
| convformer-m36 | 12.876G | 53.994M | 224x224    | 1024  | 300    | 84.000 |
| convformer-b36 | 22.673G | 95.216M | 224x224    | 1024  | 300    | 84.480 |

You can find more model training details in 00.classification_training/imagenet/.

## DarkNet training from scratch on ImageNet1K(ILSVRC2012)

| Network     | macs     | params  | input size | batch | epochs | Top-1  |
| ----------- | -------- | ------- | ---------- | ----- | ------ | ------ |
| DarkNetTiny | 414.602M | 2.087M  | 256x256    | 256   | 100    | 57.858 |
| DarkNet19   | 3.669G   | 20.842M | 256x256    | 256   | 100    | 74.364 |
| DarkNet53   | 9.335G   | 41.610M | 256x256    | 256   | 100    | 76.250 |

You can find more model training details in 00.classification_training/imagenet/.

## ResNet training from scratch on ImageNet1K(ILSVRC2012)

| Network   | macs    | params  | input size | batch | epochs | Top-1  |
| --------- | ------- | ------- | ---------- | ----- | ------ | ------ |
| ResNet18  | 1.824G  | 11.690M | 224x224    | 256   | 100    | 70.594 |
| ResNet34  | 3.679G  | 21.798M | 224x224    | 256   | 100    | 73.622 |
| ResNet50  | 4.134G  | 25.557M | 224x224    | 256   | 100    | 76.182 |
| ResNet101 | 7.866G  | 44.549M | 224x224    | 256   | 100    | 77.242 |
| ResNet152 | 11.604G | 60.193M | 224x224    | 256   | 100    | 77.772 |

You can find more model training details in 00.classification_training/imagenet/.

## ResNet finetune from ImageNet21k pretrain weight on ImageNet1K(ILSVRC2012)

| Network   | macs    | params  | input size | batch | epochs | Top-1  |
| --------- | ------- | ------- | ---------- | ----- | ------ | ------ |
| ResNet50  | 4.134G  | 25.557M | 224x224    | 2048  | 300    | 80.258 |
| ResNet101 | 7.866G  | 44.549M | 224x224    | 1024  | 300    | 81.668 |
| ResNet152 | 11.604G | 60.193M | 224x224    | 1024  | 300    | 81.934 |

You can find more model training details in 00.classification_training/imagenet/.

## VAN finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)

| Network | macs     | params  | input size | batch | epochs | Top-1  |
| ------- | -------- | ------- | ---------- | ----- | ------ | ------ |
| van-b0  | 870.860M | 4.103M  | 224x224    | 2048  | 300    | 75.424 |
| van-b1  | 2.506G   | 13.856M | 224x224    | 2048  | 300    | 80.740 |
| van-b2  | 5.010G   | 26.567M | 224x224    | 1024  | 300    | 82.592 |
| van-b3  | 8.951G   | 26.567M | 224x224    | 1024  | 300    | 83.202 |

You can find more model training details in 00.classification_training/imagenet/.

## ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs     | params   | input size | batch | epochs | Top-1  |
| ----------------- | -------- | -------- | ---------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 16.880G  | 86.416M  | 224x224    | 256   | 100    | 82.676 |
| ViT-Large-Patch16 | 59.731G  | 304.124M | 224x224    | 128   | 50     | 84.978 |
| ViT-Huge-Patch14  | 162.071G | 631.716M | 224x224    | 128   | 50     | 85.966 |

You can find more model training details in 00.classification_training/imagenet/.

## ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)

| Network           | macs     | params   | input size | batch | epochs | Top-1  |
| ----------------- | -------- | -------- | ---------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 16.880G  | 86.416M  | 224x224    | 256   | 100    | 83.404 |
| ViT-Large-Patch16 | 59.731G  | 304.124M | 224x224    | 128   | 50     | 85.672 |
| ViT-Huge-Patch14  | 162.071G | 631.716M | 224x224    | 128   | 50     | 86.608 |

You can find more model training details in 00.classification_training/imagenet/.

## ResNet train from ImageNet1K pretrain weight on ImageNet21K(Winter 2021 release)

| Network   | macs    | params  | input size | batch | epochs | Semantic Softmax Acc |
| --------- | ------- | ------- | ---------- | ----- | ------ | -------------------- |
| ResNet50  | 4.134G  | 25.557M | 224x224    | 2048  | 80     | 75.319               |
| ResNet101 | 7.866G  | 44.549M | 224x224    | 2048  | 80     | 76.795               |
| ResNet152 | 11.604G | 60.193M | 224x224    | 1024  | 80     | 77.345               |

You can find more model training details in 00.classification_training/imagenet21k/.

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


You can find more model training details in 01.distillation_training/imagenet/.

# Masked image modeling task results

**MAE:Masked Autoencoders Are Scalable Vision Learners**

Paper:https://arxiv.org/abs/2111.06377

## ViT MAE pretrain on ImageNet1K(ILSVRC2012)

| Network           | input size | batch | epochs | Loss  |
| ----------------- | ---------- | ----- | ------ | ----- |
| ViT-Base-Patch16  | 224x224    | 1024  | 400    | 0.388 |
| ViT-Large-Patch16 | 224x224    | 1024  | 400    | 0.378 |
| ViT-Huge-Patch14  | 224x224    | 1024  | 400    | 0.350 |

You can find more model training details in 02.masked_image_modeling_training/imagenet/.

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

You can find more model training details in 03.detection_training/coco/.

## All detection models finetune from objects365 pretrain weight on COCO2017

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Network            | resize-style    | input size | macs     | params  | batch | epochs | mAP    |
| ------------------ | --------------- | ---------- | -------- | ------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | RetinaStyle-800 | 800x1333   | 250.069G | 37.969M | 16    | 13     | 40.947 |
| ResNet50-FCOS      | RetinaStyle-800 | 800x1333   | 214.406G | 32.291M | 16    | 13     | 46.511 |

You can find more model training details in 03.detection_training/coco/.

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

You can find more model training details in 03.detection_training/voc/.

## All detection models finetune from objects365 pretrain weight on VOC2007 and VOC2012

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Network            | resize-style  | input size | macs    | params  | batch | epochs | mAP    |
| ------------------ | ------------- | ---------- | ------- | ------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640 | 640x640    | 84.947G | 36.724M | 32    | 13     | 90.082 |
| ResNet50-FCOS      | YoloStyle-640 | 640x640    | 80.764G | 32.153M | 32    | 13     | 90.585 |

You can find more model training details in 03.detection_training/voc/.

# Semantic Segmentation task results

**DeepLabv3+**

Paper:https://arxiv.org/abs/1802.02611

## All semantic segmentation models training from scratch on ADE20K

| Network                     | input size | macs    | params  | batch | epochs | mean_iou |
| --------------------------- | ---------- | ------- | ------- | ----- | ------ | -------- |
| resnet50_deeplabv3plus      | 512x512    | 43.500G | 30.254M | 32    | 100    | 40.462   |
| convformerm36_deeplabv3plus | 512x512    | 83.898G | 56.760M | 32    | 100    | 47.826   |

You can find more model training details in 04.semantic_segmentation_training/ade20k/.

## All semantic segmentation models training from scratch on COCO2017

| Network                     | input size | macs    | params  | batch | epochs | mean_iou |
| --------------------------- | ---------- | ------- | ------- | ----- | ------ | -------- |
| resnet50_deeplabv3plus      | 512x512    | 43.500G | 30.254M | 64    | 100    | 68.975   |
| convformerm36_deeplabv3plus | 512x512    | 83.898G | 56.760M | 64    | 100    | 74.214   |

You can find more model training details in 04.semantic_segmentation_training/coco/.

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

You can find more model training details in 05.instance_segmentation_training/coco/.

# Salient object detection task results

**PFAN+Segmentation**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2202.09741

Use combine dataset DIS5K/HRS10K/HRSOD/UHRSD to train and test.

| Network                         | macs     | params  | input size | batch | epochs | iou    | precision | recall | f_squared_beta |
| ------------------------------- | -------- | ------- | ---------- | ----- | ------ | ------ | --------- | ------ | -------------- |
| resnet50_pfan_segmentation      | 71.303G  | 26.580M | 832x832    | 96    | 100    | 0.8461 | 0.8970    | 0.9346 | 0.9053         |
| convformerm36_pfan_segmentation | 186.496G | 54.459M | 832x832    | 96    | 100    | 0.8865 | 0.9263    | 0.9517 | 0.9319         |

You can find more model training details in 06.salient_object_detection_training/.

# Human matting task results

**PFAN+Matting**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2104.14222

Paper3:https://arxiv.org/abs/2202.09741

Use combine dataset Deep_Automatic_Portrait_Matting/RealWorldPortrait636/P3M10K to train and test.

| Network                    | macs     | params  | input size | batch | epochs | iou    | precision | recall | sad    | mae    | mse    | grad   | conn   |
| -------------------------- | -------- | ------- | ---------- | ----- | ------ | ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| resnet50_pfan_matting      | 86.093G  | 29.654M | 832x832    | 96    | 100    | 0.9824 | 0.9884    | 0.9937 | 5.7071 | 0.0082 | 0.0047 | 6.7001 | 5.4373 |
| convformerm36_pfan_matting | 195.854G | 55.503M | 832x832    | 96    | 100    | 0.9865 | 0.9912    | 0.9951 | 4.5806 | 0.0066 | 0.0033 | 5.0129 | 4.2882 |

You can find more model training details in 07.human_matting_training/.

# OCR text detection task results

**DBNet**

Paper:https://arxiv.org/abs/1911.08947

Use combine dataset include ICDAR2017RCTW/ICDAR2019ART/ICDAR2019LSVT/ICDAR2019MLT to train and test.

| Network             | macs     | params  | input size | batch | epochs | precision | recall | f1     |
| ------------------- | -------- | ------- | ---------- | ----- | ------ | --------- | ------ | ------ |
| resnet50_dbnet      | 158.914G | 24.784M | 1024x1024  | 128   | 100    | 92.072    | 86.595 | 89.249 |
| convformerm36_dbnet | 340.367G | 54.528M | 1024x1024  | 64    | 100    | 92.748    | 89.947 | 91.326 |

You can find more model training details in 08.ocr_text_detection_training/.

# OCR text recognition task results

**CRNN+LSTM+CTC**

Paper:https://arxiv.org/abs/1507.05717

Use combine dataset aistudio_baidu_street/chinese_dataset/synthetic_chinese_string_dataset/meta_self_learning_dataset to train and test.

| Network                 | macs    | params   | input size | batch | epochs | lcs_precision | lcs_recall |
| ----------------------- | ------- | -------- | ---------- | ----- | ------ | ------------- | ---------- |
| resnet50_ctc_model      | 12.509G | 179.870M | 32x512     | 1024  | 50     | 99.498        | 99.212     |
| convformerm36_ctc_model | 8.051G  | 70.121M  | 32x512     | 1024  | 50     | 99.452        | 99.201     |

You can find more model training details in 09.ocr_text_recognition_training/.

# Face detection task results

**RetinaFace**

Paper:https://arxiv.org/pdf/1905.00641

Use WiderFace train and UFDD val datasets to train, WiderFace val dataset to test.

| Network             | macs     | params  | input size | batch | epochs | Easy AP | Medium AP | Hard AP |
| ------------------- | -------- | ------- | ---------- | ----- | ------ | ------- | --------- | ------- |
| resnet50_retinaface | 114.229G | 27.280M | 1024x1024  | 16    | 100    | 0.9369  | 0.9148    | 0.7801  |

You can find more model training details in 10.face_detection_training/.

# Face parsing task results

**PFAN face parsing**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2202.09741

**Sapiens**

Paper:https://arxiv.org/pdf/2408.12569

Use FaceSynthetics and CelebAMask-HQ dataset to train and test.

| Network                         | dataset        | macs     | params   | input size | batch | epochs | precision | recall  | iou     | dice    |
| ------------------------------- | -------------- | -------- | -------- | ---------- | ----- | ------ | --------- | ------- | ------- | ------- |
| resnet50_pfan_face_parsing      | FaceSynthetics | 28.361G  | 26.585M  | 512x512    | 192   | 100    | 95.4084   | 95.0583 | 91.1481 | 95.2320 |
| convformerm36_pfan_face_parsing | FaceSynthetics | 71.985G  | 54.464M  | 512x512    | 192   | 100    | 96.2895   | 96.2122 | 92.9436 | 96.2506 |
| sapiens_0_3b_face_parsing       | FaceSynthetics | 452.167G | 314.250M | 512x512    | 160   | 100    | 97.0999   | 96.9897 | 94.3823 | 97.0446 |
| resnet50_pfan_face_parsing      | CelebAMask-HQ  | 28.361G  | 26.585M  | 512x512    | 192   | 100    | 82.0985   | 77.9908 | 69.3835 | 79.7142 |
| convformerm36_pfan_face_parsing | CelebAMask-HQ  | 71.985G  | 54.464M  | 512x512    | 192   | 100    | 83.4664   | 81.1791 | 72.6132 | 82.1953 |
| sapiens_0_3b_face_parsing       | CelebAMask-HQ  | 452.167G | 314.250M | 512x512    | 160   | 100    | 86.0223   | 84.0680 | 76.2724 | 84.9471 |

You can find more model training details in 11.face_parsing_training/.

# Human parsing task results

**PFAN human parsing**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2202.09741

**Sapiens**

Paper:https://arxiv.org/pdf/2408.12569

Use LIP and CIHP dataset to train and test.

| Network                          | dataset | macs     | params   | input size | batch | epochs | precision | recall  | iou     | dice    |
| -------------------------------- | ------- | -------- | -------- | ---------- | ----- | ------ | --------- | ------- | ------- | ------- |
| resnet50_pfan_human_parsing      | LIP     | 28.437G  | 26.585M  | 512x512    | 192   | 100    | 57.5257   | 50.6568 | 39.2989 | 53.2604 |
| convformerm36_pfan_human_parsing | LIP     | 72.060G  | 54.464M  | 512x512    | 192   | 100    | 60.6652   | 57.3280 | 44.3857 | 58.7892 |
| sapiens_0_3b_human_parsing       | LIP     | 452.175G | 314.250M | 512x512    | 160   | 100    | 57.0063   | 51.9517 | 39.8993 | 54.0054 |
| resnet50_pfan_human_parsing      | CIHP    | 28.437G  | 26.585M  | 512x512    | 192   | 100    | 61.9748   | 55.4004 | 44.7195 | 57.8736 |
| convformerm36_pfan_human_parsing | CIHP    | 72.060G  | 54.464M  | 512x512    | 192   | 100    | 67.4147   | 62.6415 | 51.0651 | 64.6072 |
| sapiens_0_3b_human_parsing       | CIHP    | 452.175G | 314.250M | 512x512    | 160   | 100    | 65.0747   | 57.9976 | 47.1512 | 60.7108 |

You can find more model training details in 12.human_parsing_training/.

# Interactive segmentation task results

**SAM**

Paper:https://arxiv.org/pdf/2304.02643

Use sa_1b_11w dataset, combine salient object detection dataset,combine human matting dataset to train and test.

You can find all jupyter notebook examples in 13.interactive_segmentation_training/sam_predict_example/.

## light sam distill from pretrain weight on sa_1b_11w

| Network                            | dataset   | input size | batch | epochs | loss   |
| ---------------------------------- | --------- | ---------- | ----- | ------ | ------ |
| convformer_m36_sam_encoder_distill | sa_1b_11w | 1024x1024  | 48    | 40     | 0.0034 |
| convformer_m36_sam_distill         | sa_1b_11w | 1024x1024  | 32    | 5      | 0.1010 |
| convformer_m36_sam                 | sa_1b_11w | 1024x1024  | 64    | 5      | 0.1214 |

You can find more model training details in 13.interactive_segmentation_training/sa_1b/.

## light sam train on combine salient object detection and human matting dataset

| Network            | dataset         | input size | batch | epochs | precision | recall | iou    |
| ------------------ | --------------- | ---------- | ----- | ------ | --------- | ------ | ------ |
| convformer_m36_sam | combine dataset | 1024x1024  | 64    | 100    | 0.9492    | 0.9508 | 0.9088 |

You can find more model training details in 13.interactive_segmentation_training/salient_object_detection_human_matting_pretrain/.

## light sam matting train on combine human matting dataset

| Network                     | dataset         | input size | batch | epochs | iou    | precision | recall | sad    | mae    | mse    | grad   | conn   |
| --------------------------- | --------------- | ---------- | ----- | ------ | ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| convformer_m36_sam_matting1 | combine dataset | 1024x1024  | 48    | 200    | 0.9792 | 0.9854    | 0.9936 | 7.0763 | 0.0094 | 0.0058 | 7.0546 | 6.8581 |
| convformer_m36_sam_matting2 | combine dataset | 1024x1024  | 32    | 200    | 0.9792 | 0.9860    | 0.9925 | 7.2788 | 0.0096 | 0.0061 | 6.9827 | 7.0555 |

You can find more model training details in 13.interactive_segmentation_training/human_matting/.

## light sam matting train on combine salient object detection dataset

| Network                     | dataset         | input size | batch | epochs | iou    | precision | recall | sad     | mae    | mse    | grad    | conn    |
| --------------------------- | --------------- | ---------- | ----- | ------ | ------ | --------- | ------ | ------- | ------ | ------ | ------- | ------- |
| convformer_m36_sam_matting1 | combine dataset | 1024x1024  | 48    | 200    | 0.8647 | 0.9188    | 0.9348 | 22.2654 | 0.0304 | 0.0294 | 40.5077 | 22.2860 |
| convformer_m36_sam_matting2 | combine dataset | 1024x1024  | 32    | 200    | 0.8690 | 0.9244    | 0.9343 | 21.1366 | 0.0289 | 0.0279 | 39.2742 | 21.1572 |

You can find more model training details in 13.interactive_segmentation_training/salient_object_detection/.