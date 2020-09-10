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

If you are in China,you can download from here:
```
链接：https://pan.baidu.com/s/1b6m70EQclE8aG-A2tkWrhQ 
提取码：aieg 
```

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

# How to reproduce my results
If you want to reproduce my experiment result,just enter a category experiments folder,then enter a specific experiment folder.Each experiment folder has it's own config.py and train.py.

If the experiment use nn.parallel to train,you should add this in train.py to specify the GPU for training:
```
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```
then run this command to train:
```
python train.py
```

If the experiment use nn.DistributedDataParallel to train,you should add this in train.py to specify the GPU for training:
```
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```
then run this command to train:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 20001 train.py
```
Please make sure the nproc_per_node number is correct and master_addr/master_port are different from other experiments.


# COCO training results
Trained on COCO2017_train, tested on COCO2017_val.

mAP is IoU=0.5:0.95,area=all,maxDets=100,mAP(COCOeval,stats[0]).
mAR is IoU=0.5:0.95,area=all,maxDets=100,mAR(COCOeval,stats[8]).

My size=667 is equal to resize=400 in RetinaNet paper(https://arxiv.org/pdf/1708.02002.pdf) ,my resize=1000 is equal to resize=600 in RetinaNet paper.

| Network | resize | batch | gpu-num | apex | syncbn | epoch5-mAP-mAR-loss | epoch10-mAP-mAR-loss | epoch12-mAP-mAR-loss |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- | 
| ResNet50-RetinaNet | 667 | 24 | 2 | yes | no | 0.253,0.361,0.61 | 0.287,0.398,0.51 | 0.293,0.401,0.49 | 
| ResNet101-RetinaNet  | 667 | 16 | 2 | yes | no | 0.254,0.362,0.60 | 0.290,0.398,0.51 | 0.296,0.402,0.48 |
| ResNet50-RetinaNet  | 1000 | 16 | 4 | yes | no | 0.305,0.425,0.55 | 0.306,0.429,0.55 | 0.333,0.456,0.46 | 

For ResNet50-RetinaNet-resize1000 training,I use ResNet50-RetinaNet-resize667 as a pretrained model parameters to initialize the  ResNet50-RetinaNet-resize1000.

For ResNet50-RetinaNet-resize667,the per image inference time = 116 ms(batch=1,use one GTX 1070 Max-Q).

| Network | resize | batch | gpu-num | apex | syncbn | epoch5-mAP-mAR-loss | epoch10-mAP-mAR-loss | epoch12-mAP-mAR-loss | epoch15-mAP-mAR-loss | epoch20-mAP-mAR-loss | epoch24-mAP-mAR-loss |
| --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- |  
| ResNet50-FCOS | 667 | 32 | 2 | yes | no | 0.162,0.289,1.31 | 0.226,0.342,1.21 | 0.248,0.370,1.20 | 0.217,0.343,1.17 | 0.282,0.409,1.14 | 0.286,0.409,1.12 | 
| ResNet101-FCOS | 667 | 24 | 2 | yes | no | 0.206,0.325,1.29 | 0.237,0.359,1.20 | 0.263,0.380,1.18 | 0.277,0.400,1.15 | 0.260,0.385,1.13 | 0.291,0.416,1.10 | 
| ResNet50-FCOS  | 1000 | 32 | 4 | yes | no | 0.305,0.443,1.15 | 0.315,0.451,1.14 | / | / | / | / |

My size=667 is equal to resize=400 in FCOS paper(https://arxiv.org/pdf/1904.01355.pdf) ,my resize=1000 is equal to resize=600 in FCOS paper.

This FCOS implementation doesn't contains GN and CenterSample.

For ResNet50-FCOS-resize1000 training,I use ResNet50-FCOS-resize667 as a pretrained model parameters to initialize the  ResNet50-FCOS-resize1000.

For ResNet50-FCOS-resize667,the per image inference time = 103 ms(batch=1,use one GTX 1070 Max-Q).


You can see more model training details in detection_experiments/experiment_folder/.


# VOC training results
Trained on VOC2007 trainval + VOC2012 trainval, tested on VOC2007,using 11-point interpolated AP.

| Network | resize | batch | gpu-num | apex | syncbn | epoch5-mAP-loss | epoch10-mAP-loss | epoch15-mAP-loss | epoch20-mAP-loss |
| --- | --- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  --- | 
| ResNet50-RetinaNet | 667 | 24 | 2 | yes | no | 0.660,0.62 | 0.705,0.44 | 0.723,0.35 | 0.732,0.30 | 
| ResNet50-RetinaNet-usecocopre | 667 | 24 | 2 | yes | no | 0.789,0.34 | 0.780,0.26 | 0.776,0.22 | 0.770,0.19 | 

You can see more model training details in detection_experiments/experiment_folder/.

# CIFAR100 training results
Training in nn.parallel mode result:
| Network       | warm up | lr decay | total epochs | Top-1 error |
| --- | --- |  --- |  --- |  --- | 
| ResNet-18     | no | multistep | 200 | 21.59 | 
| ResNet-34     | no | multistep | 200 | 21.16 | 
| ResNet-50     | no | multistep | 200 | 22.12 | 
| ResNet-101    | no | multistep | 200 | 19.84 | 
| ResNet-152    | no | multistep | 200 | 19.01 | 

You can see more model training details in cifar100_experiments/resnet50cifar/.


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

You can see more model training details in imagenet_experiments/experiment_folder/.

# Citation
If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={pytorch-ImageNet-CIFAR-COCO-VOC-training},
 author={Chaoran Zhuge},
 year={2020}
}
```
