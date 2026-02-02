- [00.classification\_training results](#00classification_training-results)
  - [ResNetCifar training from scratch on CIFAR100](#resnetcifar-training-from-scratch-on-cifar100)
  - [DarkNet training from scratch on ImageNet1K(ILSVRC2012)](#darknet-training-from-scratch-on-imagenet1kilsvrc2012)
  - [ResNet training from scratch on ImageNet1K(ILSVRC2012)](#resnet-training-from-scratch-on-imagenet1kilsvrc2012)
  - [ResNet finetune from ImageNet21k pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-finetune-from-imagenet21k-pretrain-weight-on-imagenet1kilsvrc2012)
  - [Convformer finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)](#convformer-finetune-from-offical-pretrain-weight-on-imagenet1kilsvrc2012)
  - [VAN finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)](#van-finetune-from-offical-pretrain-weight-on-imagenet1kilsvrc2012)
  - [ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-self-trained-mae-pretrain-weight400epoch-on-imagenet1kilsvrc2012)
  - [ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)](#vit-finetune-from-offical-mae-pretrain-weight800-epoch-on-imagenet1kilsvrc2012)
  - [ResNet train from pytorch official weight on ImageNet21K(Winter 2021 release)](#resnet-train-from-pytorch-official-weight-on-imagenet21kwinter-2021-release)
- [01.distillation\_training results](#01distillation_training-results)
  - [ResNet distill from pretrain weight on ImageNet1K(ILSVRC2012)](#resnet-distill-from-pretrain-weight-on-imagenet1kilsvrc2012)
- [02.masked\_image\_modeling\_training results](#02masked_image_modeling_training-results)
  - [ViT MAE pretrain on ImageNet1K(ILSVRC2012)](#vit-mae-pretrain-on-imagenet1kilsvrc2012)
- [03.detection\_training results](#03detection_training-results)
  - [All detection models training from scratch on COCO2017](#all-detection-models-training-from-scratch-on-coco2017)
  - [All detection models finetune from objects365 pretrain weight on COCO2017](#all-detection-models-finetune-from-objects365-pretrain-weight-on-coco2017)
  - [All detection models training from scratch on Objects365(v2,2020)](#all-detection-models-training-from-scratch-on-objects365v22020)
  - [All detection models training from scratch on VOC2007\&VOC2012](#all-detection-models-training-from-scratch-on-voc2007voc2012)
  - [All detection models finetune from objects365 pretrain weight on VOC2007\&VOC2012](#all-detection-models-finetune-from-objects365-pretrain-weight-on-voc2007voc2012)
- [04.semantic\_segmentation\_training results](#04semantic_segmentation_training-results)
- [05.instance\_segmentation\_training results](#05instance_segmentation_training-results)
- [06.salient\_object\_detection\_training results](#06salient_object_detection_training-results)
- [07.human\_matting\_training results](#07human_matting_training-results)
- [08.ocr\_text\_detection\_training results](#08ocr_text_detection_training-results)
- [09.ocr\_text\_recognition\_training results](#09ocr_text_recognition_training-results)
- [10.face\_detection\_training results](#10face_detection_training-results)
- [11.face\_parsing\_training results](#11face_parsing_training-results)
- [12.human\_parsing\_training results](#12human_parsing_training-results)
- [13.interactive\_segmentation\_training results](#13interactive_segmentation_training-results)
- [14.video\_interactive\_segmentation\_training results](#14video_interactive_segmentation_training-results)
- [16.universal\_segmentation\_training](#16universal_segmentation_training)
  - [universal\_segmentation semantic\_segmentation\_training results](#universal_segmentation-semantic_segmentation_training-results)
  - [universal\_segmentation instance\_segmentation\_training results](#universal_segmentation-instance_segmentation_training-results)
  - [universal\_segmentation salient\_object\_detection\_training results](#universal_segmentation-salient_object_detection_training-results)
  - [universal\_matting human\_matting\_training results](#universal_matting-human_matting_training-results)
  - [universal\_matting human\_instance\_matting\_training results](#universal_matting-human_instance_matting_training-results)
  - [universal\_segmentation face\_parsing\_training results](#universal_segmentation-face_parsing_training-results)
  - [universal\_segmentation human\_parsing\_training results](#universal_segmentation-human_parsing_training-results)

# 00.classification_training results

**DarkNet**

Paper:https://arxiv.org/abs/1804.02767?e05802c1_page=1

**ResNet**

Paper:https://arxiv.org/abs/1512.03385

**Convformer**

Paper:https://arxiv.org/abs/2210.13452

**VAN**

Paper:https://arxiv.org/abs/2202.09741

**ViT**

Paper:https://arxiv.org/abs/2010.11929

## ResNetCifar training from scratch on CIFAR100 

**ResNetCifar is different from ResNet in the first few layers.**

| Model          | input size | batch | epochs | Top-1  |
| -------------- | ---------- | ----- | ------ | ------ |
| ResNet18Cifar  | 32x32      | 128   | 200    | 76.990 |
| ResNet34Cifar  | 32x32      | 128   | 200    | 77.710 |
| ResNet50Cifar  | 32x32      | 128   | 200    | 77.300 |
| ResNet101Cifar | 32x32      | 128   | 200    | 77.450 |
| ResNet152Cifar | 32x32      | 128   | 200    | 77.950 |

You can find more model training details in 00.classification_training/cifar100/.

## DarkNet training from scratch on ImageNet1K(ILSVRC2012)

| Model       | input size | batch | epochs | Top-1  |
| ----------- | ---------- | ----- | ------ | ------ |
| DarkNetTiny | 256x256    | 256   | 100    | 58.074 |
| DarkNet19   | 256x256    | 256   | 100    | 74.040 |
| DarkNet53   | 256x256    | 256   | 100    | 76.366 |

You can find more model training details in 00.classification_training/imagenet/.

## ResNet training from scratch on ImageNet1K(ILSVRC2012)

| Model     | input size | batch | epochs | Top-1  |
| --------- | ---------- | ----- | ------ | ------ |
| ResNet18  | 224x224    | 256   | 100    | 70.520 |
| ResNet34  | 224x224    | 256   | 100    | 73.796 |
| ResNet50  | 224x224    | 256   | 100    | 76.242 |
| ResNet101 | 224x224    | 256   | 100    | 77.436 |
| ResNet152 | 224x224    | 256   | 100    | 77.834 |

You can find more model training details in 00.classification_training/imagenet/.

## ResNet finetune from ImageNet21k pretrain weight on ImageNet1K(ILSVRC2012)

| Model     | input size | batch | epochs | Top-1  |
| --------- | ---------- | ----- | ------ | ------ |
| ResNet50  | 224x224    | 2048  | 300    | 80.110 |
| ResNet101 | 224x224    | 1024  | 300    | 81.586 |
| ResNet152 | 224x224    | 1024  | 300    | 81.712 |

You can find more model training details in 00.classification_training/imagenet/.

## Convformer finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)

| Model          | input size | batch | epochs | Top-1  |
| -------------- | ---------- | ----- | ------ | ------ |
| convformer-s18 | 224x224    | 2048  | 300    | 81.914 |
| convformer-s36 | 224x224    | 2048  | 300    | 83.210 |
| convformer-m36 | 224x224    | 1024  | 300    | 83.980 |
| convformer-b36 | 224x224    | 1024  | 300    | 84.424 |

You can find more model training details in 00.classification_training/imagenet/.

## VAN finetune from offical pretrain weight on ImageNet1K(ILSVRC2012)

| Model  | input size | batch | epochs | Top-1  |
| ------ | ---------- | ----- | ------ | ------ |
| van-b0 | 224x224    | 2048  | 300    | 75.216 |
| van-b1 | 224x224    | 2048  | 300    | 80.608 |
| van-b2 | 224x224    | 1024  | 300    | 82.540 |
| van-b3 | 224x224    | 1024  | 300    | 83.240 |

You can find more model training details in 00.classification_training/imagenet/.

## ViT finetune from self-trained MAE pretrain weight(400epoch) on ImageNet1K(ILSVRC2012)

| Model             | input size | batch | epochs | Top-1  |
| ----------------- | ---------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 224x224    | 256   | 100    | 82.794 |
| ViT-Large-Patch16 | 224x224    | 128   | 50     | 84.842 |
| ViT-Huge-Patch14  | 224x224    | 128   | 50     | 85.816 |

You can find more model training details in 00.classification_training/imagenet/.

## ViT finetune from offical MAE pretrain weight(800 epoch) on ImageNet1K(ILSVRC2012)

| Model             | input size | batch | epochs | Top-1  |
| ----------------- | ---------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 224x224    | 256   | 100    | 83.152 |
| ViT-Large-Patch16 | 224x224    | 128   | 50     | 85.870 |
| ViT-Huge-Patch14  | 224x224    | 128   | 50     | 86.608 |

You can find more model training details in 00.classification_training/imagenet/.

## ResNet train from pytorch official weight on ImageNet21K(Winter 2021 release)

| Model     | input size | batch | epochs | Semantic Softmax Acc |
| --------- | ---------- | ----- | ------ | -------------------- |
| ResNet50  | 224x224    | 2048  | 80     | 75.354               |
| ResNet101 | 224x224    | 2048  | 80     | 76.842               |
| ResNet152 | 224x224    | 1024  | 80     | 77.342               |

You can find more model training details in 00.classification_training/imagenet21k/.

# 01.distillation_training results

**DML loss**

Paper:https://arxiv.org/abs/1706.00384

**KD loss**

Paper:https://arxiv.org/abs/1503.02531

## ResNet distill from pretrain weight on ImageNet1K(ILSVRC2012)

| Teacher Model | Student Model | method         | Freeze Teacher | input size | batch | epochs | Teacher Top-1 | Student Top-1 |
| ------------- | ------------- | -------------- | -------------- | ---------- | ----- | ------ | ------------- | ------------- |
| ResNet152     | ResNet50      | CE+DML         | False          | 224x224    | 256   | 100    | 79.370        | 78.086        |
| ResNet152     | ResNet50      | CE+DML+Vit Aug | False          | 224x224    | 1024  | 300    | 82.722        | 80.830        |
| ResNet152     | ResNet50      | CE+KD          | True           | 224x224    | 256   | 100    | 77.836        | 77.578        |
| ResNet152     | ResNet50      | CE+KD+Vit Aug  | True           | 224x224    | 2048  | 300    | 81.712        | 80.672        |

You can find more model training details in 01.distillation_training/imagenet/.

# 02.masked_image_modeling_training results

**MAE**

Paper:https://arxiv.org/abs/2111.06377

## ViT MAE pretrain on ImageNet1K(ILSVRC2012)

| Model             | input size | batch | epochs | Loss   |
| ----------------- | ---------- | ----- | ------ | ------ |
| ViT-Base-Patch16  | 224x224    | 1024  | 400    | 0.3876 |
| ViT-Large-Patch16 | 224x224    | 1024  | 400    | 0.3784 |
| ViT-Huge-Patch14  | 224x224    | 1024  | 400    | 0.3502 |

You can find more model training details in 02.masked_image_modeling_training/imagenet/.

# 03.detection_training results

**RetinaNet**

Paper:https://arxiv.org/abs/1708.02002

**FCOS**

Paper:https://arxiv.org/abs/1904.01355

**DETR**

Paper:https://arxiv.org/abs/2005.12872

## All detection models training from scratch on COCO2017

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Model              | resize-style   | input size | batch | epochs | mAP    |
| ------------------ | -------------- | ---------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-1024 | 1024x1024  | 32    | 13     | 36.893 |
| ResNet50-FCOS      | YoloStyle-1024 | 1024x1024  | 32    | 13     | 40.155 |
| ResNet50-DETR      | YoloStyle-1024 | 1024x1024  | 64    | 500    | 38.735 |

You can find more model training details in 03.detection_training/coco/.

## All detection models finetune from objects365 pretrain weight on COCO2017

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Model              | resize-style   | input size | batch | epochs | mAP    |
| ------------------ | -------------- | ---------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-1024 | 1024x1024  | 32    | 13     | 41.259 |
| ResNet50-FCOS      | YoloStyle-1024 | 1024x1024  | 32    | 13     | 45.249 |

You can find more model training details in 03.detection_training/coco/.

## All detection models training from scratch on Objects365(v2,2020)

Trained on objects365(v2,2020) train dataset.

| Model              | resize-style   | input size | batch | epochs | loss   |
| ------------------ | -------------- | ---------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-1024 | 1024x1024  | 128   | 13     | 0.3237 |
| ResNet50-FCOS      | YoloStyle-1024 | 1024x1024  | 128   | 13     | 0.9669 |

You can find more model training details in 03.detection_training/objects365/.

## All detection models training from scratch on VOC2007&VOC2012

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Model              | resize-style  | input size | batch | epochs | mAP    |
| ------------------ | ------------- | ---------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640 | 640x640    | 32    | 13     | 83.460 |
| ResNet50-FCOS      | YoloStyle-640 | 640x640    | 32    | 13     | 83.320 |

You can find more model training details in 03.detection_training/voc/.

## All detection models finetune from objects365 pretrain weight on VOC2007&VOC2012

Trained on VOC2007 trainval dataset + VOC2012 trainval dataset, tested on VOC2007 test dataset.

mAP is IoU=0.50,area=all,maxDets=100,mAP.

| Model              | resize-style  | input size | batch | epochs | mAP    |
| ------------------ | ------------- | ---------- | ----- | ------ | ------ |
| ResNet50-RetinaNet | YoloStyle-640 | 640x640    | 32    | 13     | 90.034 |
| ResNet50-FCOS      | YoloStyle-640 | 640x640    | 32    | 13     | 89.900 |

You can find more model training details in 03.detection_training/voc/.

# 04.semantic_segmentation_training results

**pfan_semantic_segmentation**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2210.13452

Paper3:https://arxiv.org/abs/2508.10104

Use ADE20K and COCO2017 dataset to train and test.

| Model                                              | dataset  | input size | batch | epochs | mean_iou |
| -------------------------------------------------- | -------- | ---------- | ----- | ------ | -------- |
| resnet50_pfan_semantic_segmentation                | ADE20K   | 512x512    | 32    | 100    | 30.326   |
| convformerm36_pfan_semantic_segmentation           | ADE20K   | 512x512    | 32    | 100    | 40.281   |
| dinov3_vit_base_patch16_pfan_semantic_segmentation | ADE20K   | 512x512    | 32    | 100    | 45.964   |
| resnet50_pfan_semantic_segmentation                | COCO2017 | 512x512    | 64    | 100    | 53.238   |
| convformerm36_pfan_semantic_segmentation           | COCO2017 | 512x512    | 64    | 100    | 61.187   |
| dinov3_vit_base_patch16_pfan_semantic_segmentation | COCO2017 | 512x512    | 64    | 100    | 64.774   |

You can find more model training details in 04.semantic_segmentation_training/.

# 05.instance_segmentation_training results

**SOLOv2**

Paper:https://arxiv.org/abs/2003.10152

**YOLACT**

Paper:https://arxiv.org/abs/1904.02689

Trained on COCO2017 train dataset, tested on COCO2017 val dataset.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).

| Model                          | resize-style   | input size | batch | epochs | mAP    |
| ------------------------------ | -------------- | ---------- | ----- | ------ | ------ |
| resnet50_yolact                | YoloStyle-1024 | 1024x1024  | 64    | 39     | 29.211 |
| convformerm36_yolact           | YoloStyle-1024 | 1024x1024  | 64    | 39     | 33.046 |
| dinov3_vit_base_patch16_yolact | YoloStyle-1024 | 1024x1024  | 64    | 39     | 36.085 |
| resnet50_solov2                | YoloStyle-1024 | 1024x1024  | 32    | 39     | 37.661 |
| convformerm36_solov2           | YoloStyle-1024 | 1024x1024  | 32    | 39     | 40.501 |
| dinov3_vit_base_patch16_solov2 | YoloStyle-1024 | 1024x1024  | 32    | 39     | 43.591 |

You can find more model training details in 05.instance_segmentation_training/coco/.

# 06.salient_object_detection_training results

**pfan_segmentation**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2210.13452

Paper3:https://arxiv.org/abs/2508.10104

Use combine dataset to train and test.

| Model                                     | input size | batch | epochs | iou    | precision | recall | f_squared_beta |
| ----------------------------------------- | ---------- | ----- | ------ | ------ | --------- | ------ | -------------- |
| resnet50_pfan_segmentation                | 1024x1024  | 64    | 100    | 0.8444 | 0.8954    | 0.9335 | 0.9039         |
| convformerm36_pfan_segmentation           | 1024x1024  | 64    | 100    | 0.8916 | 0.9290    | 0.9549 | 0.9348         |
| dinov3_vit_base_patch16_pfan_segmentation | 1024x1024  | 64    | 100    | 0.9065 | 0.9439    | 0.9566 | 0.9467         |

You can find more model training details in 06.salient_object_detection_training/.

# 07.human_matting_training results

**pfan_matting**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2104.14222

Paper3:https://arxiv.org/abs/2210.13452

Paper4:https://arxiv.org/abs/2508.10104

Use combine dataset to train and test.

| Model                                | input size | batch | epochs | iou    | precision | recall | sad    | mae    | mse    | grad    | conn   |
| ------------------------------------ | ---------- | ----- | ------ | ------ | --------- | ------ | ------ | ------ | ------ | ------- | ------ |
| resnet50_pfan_matting                | 1024x1024  | 32    | 100    | 0.9823 | 0.9874    | 0.9948 | 6.5496 | 0.0062 | 0.0040 | 10.7192 | 6.5801 |
| convformerm36_pfan_matting           | 1024x1024  | 32    | 100    | 0.9881 | 0.9910    | 0.9970 | 4.4842 | 0.0042 | 0.0022 | 8.0214  | 4.4843 |
| dinov3_vit_base_patch16_pfan_matting | 1024x1024  | 32    | 100    | 0.9871 | 0.9914    | 0.9955 | 5.0023 | 0.0047 | 0.0026 | 8.7974  | 5.0621 |

You can find more model training details in 07.human_matting_training/.

# 08.ocr_text_detection_training results

**DBNet**

Paper:https://arxiv.org/abs/1911.08947

Use combine dataset to train and test.

| Model               | input size | batch | epochs | precision | recall  | f1      |
| ------------------- | ---------- | ----- | ------ | --------- | ------- | ------- |
| resnet50_dbnet      | 1024x1024  | 64    | 100    | 92.3463   | 87.1304 | 89.6626 |
| convformerm36_dbnet | 1024x1024  | 64    | 100    | 93.1819   | 89.5183 | 91.3134 |

You can find more model training details in 08.ocr_text_detection_training/.

# 09.ocr_text_recognition_training results

**CTC_Model**

Paper:https://arxiv.org/abs/1507.05717

Use combine dataset to train and test.

| Model                   | input size | batch | epochs | lcs_precision | lcs_recall |
| ----------------------- | ---------- | ----- | ------ | ------------- | ---------- |
| resnet50_ctc_model      | 32x512     | 1024  | 50     | 99.1379       | 98.8073    |
| convformerm36_ctc_model | 32x512     | 1024  | 50     | 99.4651       | 99.2434    |

You can find more model training details in 09.ocr_text_recognition_training/.

# 10.face_detection_training results

**RetinaFace**

Paper:https://arxiv.org/abs/1905.00641

Use combine dataset to train and test.

| Model               | input size | batch | epochs | Easy AP | Medium AP | Hard AP |
| ------------------- | ---------- | ----- | ------ | ------- | --------- | ------- |
| resnet50_retinaface | 1024x1024  | 16    | 100    | 0.9375  | 0.9148    | 0.7804  |

You can find more model training details in 10.face_detection_training/.

# 11.face_parsing_training results

**pfan_face_parsing**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2210.13452

Paper3:https://arxiv.org/abs/2508.10104

Use CelebAMask-HQ and FaceSynthetics dataset to train and test.

| Model                                     | dataset        | input size | batch | epochs | precision | recall  | iou     | dice    |
| ----------------------------------------- | -------------- | ---------- | ----- | ------ | --------- | ------- | ------- | ------- |
| resnet50_pfan_face_parsing                | CelebAMask-HQ  | 512x512    | 192   | 100    | 81.4427   | 77.5129 | 68.9136 | 79.2088 |
| convformerm36_pfan_face_parsing           | CelebAMask-HQ  | 512x512    | 192   | 100    | 84.2701   | 81.5477 | 72.9179 | 82.7265 |
| dinov3_vit_base_patch16_pfan_face_parsing | CelebAMask-HQ  | 512x512    | 192   | 100    | 86.1822   | 83.7555 | 75.3506 | 84.8245 |
| resnet50_pfan_face_parsing                | FaceSynthetics | 512x512    | 192   | 100    | 95.3781   | 95.1519 | 91.2068 | 95.2643 |
| convformerm36_pfan_face_parsing           | FaceSynthetics | 512x512    | 192   | 100    | 96.2706   | 96.1944 | 92.9115 | 96.2323 |
| dinov3_vit_base_patch16_pfan_face_parsing | FaceSynthetics | 512x512    | 192   | 100    | 95.9629   | 95.7920 | 92.2981 | 95.8769 |

You can find more model training details in 11.face_parsing_training/.

# 12.human_parsing_training results

**pfan_human_parsing**

Paper1:https://arxiv.org/abs/1903.00179

Paper2:https://arxiv.org/abs/2210.13452

Paper3:https://arxiv.org/abs/2508.10104

Use CIHP and LIP dataset to train and test.

| Model                                      | dataset | input size | batch | epochs | precision | recall  | iou     | dice    |
| ------------------------------------------ | ------- | ---------- | ----- | ------ | --------- | ------- | ------- | ------- |
| resnet50_pfan_human_parsing                | CIHP    | 512x512    | 192   | 100    | 62.2381   | 56.0526 | 45.2076 | 58.4858 |
| convformerm36_pfan_human_parsing           | CIHP    | 512x512    | 192   | 100    | 67.9648   | 63.0336 | 51.5180 | 65.0746 |
| dinov3_vit_base_patch16_pfan_human_parsing | CIHP    | 512x512    | 192   | 100    | 73.1447   | 70.3957 | 58.2466 | 71.6496 |
| resnet50_pfan_human_parsing                | LIP     | 512x512    | 192   | 100    | 56.1325   | 50.6626 | 38.8464 | 52.7264 |
| convformerm36_pfan_human_parsing           | LIP     | 512x512    | 192   | 100    | 61.5202   | 57.3418 | 44.6563 | 59.0827 |
| dinov3_vit_base_patch16_pfan_human_parsing | LIP     | 512x512    | 192   | 100    | 64.7237   | 62.3281 | 48.8433 | 63.3994 |

You can find more model training details in 12.human_parsing_training/.

# 13.interactive_segmentation_training results

**SAM**

Paper1:https://arxiv.org/abs/2304.02643

Paper2:https://arxiv.org/abs/2508.10104

Use combine dataset to train and test.

You can find all jupyter example in 13.interactive_segmentation_training/sam_predict_example/.

| Model                                                 | input size | batch | epochs | loss   |
| ----------------------------------------------------- | ---------- | ----- | ------ | ------ |
| sam_h_encoder_distill_dinov3_vit_base_patch16_encoder | 1024x1024  | 128   | 5      | 0.0013 |
| sam_b                                                 | 1024x1024  | 160   | 2      | 0.0954 |
| sam_b_multilevel                                      | 1024x1024  | 160   | 2      | 0.1413 |

You can find more model training details in 13.interactive_segmentation_training/.

# 14.video_interactive_segmentation_training results

**SAM2**

Paper1:https://arxiv.org/abs/2408.00714

Paper2:https://arxiv.org/abs/2508.10104

Use combine dataset to train and test.

You can find all jupyter example in 14.video_interactive_segmentation_training/sam2_predict_example/.

| Model                                                   | input size | batch | frame_num | epochs | loss   |
| ------------------------------------------------------- | ---------- | ----- | --------- | ------ | ------ |
| hiera_l_encoder_distill_dinov3_vit_base_patch16_encoder | 1024x1024  | 24    | 8         | 20     | 0.0438 |
| hiera_b_plus_sam2video_stage1                           | 1024x1024  | 160   | 1         | 2      | 0.1315 |
| hiera_b_plus_sam2video_stage2                           | 1024x1024  | 16    | 8         | 40     | 0.4212 |
| hiera_b_plus_sam2video_stage3                           | 1024x1024  | 16    | 16        | 20     | 0.9382 |
| hiera_b_plus_sam2video_multilevel_stage1                | 1024x1024  | 160   | 1         | 2      | 0.1839 |
| hiera_b_plus_sam2video_multilevel_stage2                | 1024x1024  | 16    | 8         | 40     | 0.5131 |
| hiera_b_plus_sam2video_multilevel_stage3                | 1024x1024  | 16    | 16        | 20     | 0.9516 |

You can find more model training details in 14.video_interactive_segmentation_training/.

# 16.universal_segmentation_training

**universal_segmentation**

Paper:https://arxiv.org/abs/2503.19108

## universal_segmentation semantic_segmentation_training results

| Model                                           | dataset  | input size | batch | epochs | mean_iou |
| ----------------------------------------------- | -------- | ---------- | ----- | ------ | -------- |
| dinov3_vit_large_patch16_universal_segmentation | ADE20K   | 512x512    | 128   | 100    | 47.8155  |
| dinov3_vit_large_patch16_universal_segmentation | COCO2017 | 512x512    | 256   | 100    | 64.7959  |

You can find more model training details in 16.universal_segmentation_training/16.0.semantic_segmentation_training/.

## universal_segmentation instance_segmentation_training results

| Model                                           | dataset  | resize-style   | input size | batch | epochs | mAP     |
| ----------------------------------------------- | -------- | -------------- | ---------- | ----- | ------ | ------- |
| dinov3_vit_large_patch16_universal_segmentation | COCO2017 | YoloStyle-1024 | 1024x1024  | 64    | 50     | 45.3113 |

You can find more model training details in 16.universal_segmentation_training/16.1.instance_segmentation_training/.

## universal_segmentation salient_object_detection_training results

| Model                                           | input size | batch | epochs | iou    | precision | recall | f_squared_beta |
| ----------------------------------------------- | ---------- | ----- | ------ | ------ | --------- | ------ | -------------- |
| dinov3_vit_large_patch16_universal_segmentation | 1024x1024  | 64    | 50     | 0.9079 | 0.9369    | 0.9651 | 0.9432         |

You can find more model training details in 16.universal_segmentation_training/16.2.salient_object_detection_training/.

## universal_matting human_matting_training results

| Model                                      | input size | batch | epochs | iou    | precision | recall | sad    | mae    | mse    | grad   | conn   |
| ------------------------------------------ | ---------- | ----- | ------ | ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| dinov3_vit_large_patch16_universal_matting | 1024x1024  | 32    | 50     | 0.9886 | 0.9913    | 0.9973 | 4.1426 | 0.0039 | 0.0018 | 7.7149 | 4.1218 |

You can find more model training details in 16.universal_segmentation_training/16.3.human_matting_training/.

## universal_matting human_instance_matting_training results

| Model                                      | input size | batch | epochs | loss   |
| ------------------------------------------ | ---------- | ----- | ------ | ------ |
| dinov3_vit_large_patch16_universal_matting | 1024x1024  | 32    | 50     | 0.0746 |

You can find more model training details in 16.universal_segmentation_training/16.4.human_instance_matting_training/.

## universal_segmentation face_parsing_training results

| Model                                           | dataset        | input size | batch | epochs | precision | recall  | iou     | dice    |
| ----------------------------------------------- | -------------- | ---------- | ----- | ------ | --------- | ------- | ------- | ------- |
| dinov3_vit_large_patch16_universal_segmentation | CelebAMask-HQ  | 512x512    | 256   | 100    | 86.6002   | 84.5362 | 76.0747 | 85.5090 |
| dinov3_vit_large_patch16_universal_segmentation | FaceSynthetics | 512x512    | 256   | 100    | 97.3316   | 97.2978 | 94.8875 | 97.3139 |

You can find more model training details in 16.universal_segmentation_training/16.5.face_parsing_training/.

## universal_segmentation human_parsing_training results

| Model                                           | dataset | input size | batch | epochs | precision | recall  | iou     | dice    |
| ----------------------------------------------- | ------- | ---------- | ----- | ------ | --------- | ------- | ------- | ------- |
| dinov3_vit_large_patch16_universal_segmentation | CIHP    | 512x512    | 256   | 100    | 80.6561   | 77.1104 | 66.0162 | 78.7259 |
| dinov3_vit_large_patch16_universal_segmentation | LIP     | 512x512    | 256   | 100    | 67.2514   | 64.3616 | 50.9822 | 65.6268 |


You can find more model training details in 16.universal_segmentation_training/16.6.human_parsing_training/.