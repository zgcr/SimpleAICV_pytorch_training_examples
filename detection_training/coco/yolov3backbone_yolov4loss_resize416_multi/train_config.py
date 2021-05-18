import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path
from simpleAICV.detection.models import yolov3
from simpleAICV.detection.losses import Yolov4Loss
from simpleAICV.detection.decode import Yolov4Decoder
from simpleAICV.datasets.cocodataset import CocoDetection
from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'COCO'
    network = 'yolov3'
    pretrained = False
    num_classes = 80
    input_image_size = 416

    model = yolov3.__dict__[network](**{
        'pretrained':
        pretrained,
        'anchor_sizes': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                         [59, 119], [116, 90], [156, 198], [373, 326]],
        'strides': [8, 16, 32],
        'num_classes':
        num_classes,
    })

    criterion = Yolov4Loss(
        anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                      [59, 119], [116, 90], [156, 198], [373, 326]],
        strides=[8, 16, 32],
    )
    decoder = Yolov4Decoder()

    train_dataset = CocoDetection(COCO2017_path,
                                  set_name='train2017',
                                  transform=transforms.Compose([
                                      RandomHorizontalFlip(flip_prob=0.5),
                                      Normalize(),
                                      YoloStyleResize(
                                          resize=input_image_size,
                                          multi_scale=True,
                                          multi_scale_range=[0.8, 1.0]),
                                  ]))

    val_dataset = CocoDetection(COCO2017_path,
                                set_name='val2017',
                                transform=transforms.Compose([
                                    Normalize(),
                                    YoloStyleResize(
                                        resize=input_image_size,
                                        multi_scale=False,
                                        multi_scale_range=[0.8, 1.0]),
                                ]))

    seed = 0
    # batch_size is total size in DataParallel mode
    # batch_size is per gpu node size in DistributedDataParallel mode
    batch_size = 64
    num_workers = 16

    # choose 'SGD' or 'AdamW'
    optimizer = 'AdamW'
    # 'AdamW' doesn't need gamma and momentum variable
    gamma = 0.1
    momentum = 0.9
    # choose 'MultiStepLR' or 'CosineLR'
    # milestones only use in 'MultiStepLR'
    scheduler = 'MultiStepLR'
    lr = 1e-4
    weight_decay = 1e-3
    milestones = [400]
    warm_up_epochs = 0

    epochs = 400
    eval_epoch = [i * 5 for i in range(epochs // 5)]
    print_interval = 20

    # only in DistributedDataParallel mode can use sync_bn
    distributed = True
    sync_bn = False
    apex = True
