import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path
from simpleAICV.segmentation.models import solov2
from simpleAICV.segmentation.losses import Solov2Loss
from simpleAICV.segmentation.decode import Solov2Decoder
from simpleAICV.datasets.cocodataset import CocoSegmentation
from simpleAICV.segmentation.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'COCO'
    network = 'resnet50_solov2'
    pretrained = False
    num_classes = 80
    input_image_size = 400

    model = solov2.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })

    criterion = Solov2Loss(mask_loss_weight=3.0)
    decoder = Solov2Decoder()

    train_dataset = CocoSegmentation(COCO2017_path,
                                     set_name='train2017',
                                     transform=transforms.Compose([
                                         RandomHorizontalFlip(flip_prob=0.5),
                                         Normalize(),
                                         RetinaStyleResize(
                                             resize=input_image_size,
                                             multi_scale=True,
                                             multi_scale_range=[0.8, 1.0]),
                                     ]))

    val_dataset = CocoSegmentation(COCO2017_path,
                                   set_name='val2017',
                                   transform=transforms.Compose([
                                       Normalize(),
                                       RetinaStyleResize(
                                           resize=input_image_size,
                                           multi_scale=False,
                                           multi_scale_range=[0.8, 1.0]),
                                   ]))

    seed = 0
    # batch_size is total size in DataParallel mode
    # batch_size is per gpu node size in DistributedDataParallel mode
    batch_size = 16
    num_workers = 2

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
    milestones = [36]
    warm_up_epochs = 0

    epochs = 36
    eval_epoch = [5, 10, 15, 20, 21, 22, 23, 24, 25, 30, 35]
    print_interval = 100

    # only in DistributedDataParallel mode can use sync_bn
    distributed = True
    sync_bn = False
    apex = True