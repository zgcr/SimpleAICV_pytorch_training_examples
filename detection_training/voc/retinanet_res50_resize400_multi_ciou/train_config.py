import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import VOCdataset_path
from simpleAICV.detection.models import retinanet
from simpleAICV.detection.losses import RetinaLoss
from simpleAICV.detection.decode import RetinaDecoder
from simpleAICV.datasets.vocdataset import VocDetection
from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'VOC'
    network = 'resnet50_retinanet'
    pretrained = False
    num_classes = 20
    input_image_size = 400

    model = retinanet.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })
    criterion = RetinaLoss()
    decoder = RetinaDecoder()

    train_dataset = VocDetection(root_dir=VOCdataset_path,
                                 image_sets=[('2007', 'trainval'),
                                             ('2012', 'trainval')],
                                 transform=transforms.Compose([
                                     RandomHorizontalFlip(flip_prob=0.5),
                                     Normalize(),
                                     RetinaStyleResize(
                                         resize=input_image_size,
                                         multi_scale=True,
                                         multi_scale_range=[0.8, 1.0]),
                                 ]),
                                 keep_difficult=False)
    val_dataset = VocDetection(root_dir=VOCdataset_path,
                               image_sets=[('2007', 'test')],
                               transform=transforms.Compose([
                                   Normalize(),
                                   RetinaStyleResize(
                                       resize=input_image_size,
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
                               ]),
                               keep_difficult=False)

    seed = 0
    # batch_size is total size in DataParallel mode
    # batch_size is per gpu node size in DistributedDataParallel mode
    batch_size = 16
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
    milestones = [8, 11]
    warm_up_epochs = 0

    epochs = 12
    eval_epoch = [1, 3, 5, 8, 11, 12]
    eval_iou_threshold = 0.5
    print_interval = 100

    # only in DistributedDataParallel mode can use sync_bn
    distributed = True
    sync_bn = False
    apex = True
