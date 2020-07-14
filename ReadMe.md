# Requirements
Platform:Ubuntu 18.04.4
```
1.pytorch==1.4.0
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
```

If you use python3.7,please use the following orders to install apex:
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

# My pretrained models
You can download all my pretrained models from here:https://drive.google.com/drive/folders/1rewWULfXsvE0voA-A_ooTWwadq9lsk3X?usp=sharing .


# Preparing the dataset
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

# COCO training results
Trained on COCO2017_train, tested on COCO2017_val,using IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).
All experiments input_size=667,which is equal to resize=400 in RetinaNet paper(https://arxiv.org/pdf/1708.02002.pdf).

| Network | batch | gpu-num | apex | syncbn | epoch5-mAP-loss | epoch10-mAP-loss | epoch12-mAP-loss |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- | 
| ResNet18-RetinaNet | 24 | 2 | yes | no |   |   |   | 
| ResNet34-RetinaNet | 24 | 2 | yes | no |   |   |   | 
| ResNet50-RetinaNet | 24 | 2 | yes | no | 0.253,0.61 | 0.287,0.51 | 0.293,0.49 | 
| ResNet101-RetinaNet  | 16 | 2 | yes | no |   | |  | 

# VOC training results
Trained on VOC2007 trainval + VOC2012 trainval, tested on VOC2007,using 11-point interpolated AP.

# CIFAR100 training results
Training in nn.parallel mode result:
| Network       | warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- | 
| ResNet-18     | no | multistep | 200 | 21.59 | 
| ResNet-34     | no | multistep | 200 | 21.16 | 
| ResNet-50     | no | multistep | 200 | 22.12 | 
| ResNet-101    | no | multistep | 200 | 19.84 | 
| ResNet-152    | no | multistep | 200 | 19.01 | 

You can see model training details in cifar100_experiments/resnet50cifar/.


# ImageNet training results
##  Training in nn.parallel mode results
| Network       | warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- | 
| ResNet-18     | no | multistep | 100 | 29.684 | 
| ResNet-34-half     | no | multistep | 100 | 32.528 | 
| ResNet-34     | no | multistep | 100 | 26.264 | 
| ResNet-50-half     | no | multistep | 100 | 27.934 | 
| ResNet-50     | no | multistep | 100 | 23.488 | 
| ResNet-101    | no | multistep | 100 | 22.276 | 
| ResNet-152    | no | multistep | 100 | 21.436 |
| EfficientNet-b0    | yes,5 epochs | consine | 100 | 24.492 |
| EfficientNet-b1    | yes,5 epochs | consine | 100 | 23.092 |
| EfficientNet-b2    | yes,5 epochs | consine | 100 | 22.224 |
| EfficientNet-b3    | yes,5 epochs | consine | 100 | 21.884 |
| DarkNet-19  | no | multistep | 100 | 26.132 | 
| DarkNet-53  | no | multistep | 100 | 22.992 | 
| VovNet-19-slim-depthwise-se  | no | multistep | 100 | 33.276 | 
| VovNet-19-slim-se  | no | multistep | 100 | 30.646 | 
| VovNet-19-se  | no | multistep | 100 | 25.364 | 
| VovNet-39-se  | no | multistep | 100 | 22.662 | 
| VovNet-57-se  | no | multistep | 100 | 22.014 | 
| VovNet-99-se  | no | multistep | 100 | 21.608 | 
| RegNetY-200MF    | yes,5 epochs | consine | 100 | 29.904 |
| RegNetY-400MF    | yes,5 epochs | consine | 100 | 26.210 |
| RegNetY-600MF    | yes,5 epochs | consine | 100 | 25.276 |
| RegNetY-800MF    | yes,5 epochs | consine | 100 | 24.006 |
| RegNetY-1.6GF    | yes,5 epochs | consine | 100 | 22.692 |
| RegNetY-3.2GF    | yes,5 epochs | consine | 100 | 21.092 |
| RegNetY-4.0GF    | yes,5 epochs | consine | 100 | 21.684 |
| RegNetY-6.4GF    | yes,5 epochs | consine | 100 | 21.230 |


All nets are trained by input size 224x224 except DarkNet(input size 256x256) and EfficientNet.

For training resnet50 with batch_size=256,you need at least 4 2080ti gpus,and need about three or four days.

## Training in nn.DistributedDataParallel mode results
| Network       | sync-BN |warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- |  --- | 
| ResNet-50     | no | no | multistep | 100 | 23.72 |
| ResNet-50     | yes | no | multistep | 100 | 25.44 |  

You can see model training details in imagenet_experiments/experiment_folder/.

# Citation
If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={pytorch-ImageNet-CIFAR-COCO-VOC-training},
 author={Chaoran Zhuge},
 year={2020}
}
```
