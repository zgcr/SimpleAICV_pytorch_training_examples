import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path

from simpleAICV.detection import models
from simpleAICV.detection import losses
from simpleAICV.detection import decode
from simpleAICV.detection.datasets.cocodataset import CocoDetection
from simpleAICV.detection.common import RetinaStyleResize, YoloStyleResize, RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet18_centernet'
    num_classes = 80
    input_image_size = [512, 512]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/pretrained_models/resnet/resnet18-acc70.712.pth'
    # backbone_pretrained_path = os.path.join(BASE_DIR, '')
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['CenterNetLoss'](**{
        'alpha': 2.,
        'beta': 4.,
        'heatmap_loss_weight': 1.0,
        'offset_loss_weight': 1.0,
        'wh_loss_weight': 0.1,
        'min_overlap': 0.7,
        'max_object_num': 100,
    })

    test_criterion = losses.__dict__['CenterNetLoss'](**{
        'alpha': 2.,
        'beta': 4.,
        'heatmap_loss_weight': 1.0,
        'offset_loss_weight': 1.0,
        'wh_loss_weight': 0.1,
        'min_overlap': 0.7,
        'max_object_num': 100,
    })

    decoder = decode.__dict__['CenterNetDecoder'](**{
        'topk': 100,
        'stride': 4,
        'min_score_threshold': 0.05,
        'max_object_num': 100,
    }).cuda()

    train_dataset = CocoDetection(COCO2017_path,
                                  set_name='train2017',
                                  transform=transforms.Compose([
                                      RandomHorizontalFlip(prob=0.5),
                                      RandomCrop(prob=0.5),
                                      RandomTranslate(prob=0.5),
                                      YoloStyleResize(
                                          resize=input_image_size[0],
                                          divisor=32,
                                          stride=32,
                                          multi_scale=True,
                                          multi_scale_range=[0.6, 1.4]),
                                      Normalize(),
                                  ]))

    test_dataset = CocoDetection(COCO2017_path,
                                 set_name='val2017',
                                 transform=transforms.Compose([
                                     YoloStyleResize(
                                         resize=input_image_size[0],
                                         divisor=32,
                                         stride=32,
                                         multi_scale=False,
                                         multi_scale_range=[0.6, 1.4]),
                                     Normalize(),
                                 ]))
    train_collater = DetectionCollater()
    test_collater = DetectionCollater()

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 16

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-3,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [90, 120],
        },
    )

    epochs = 140
    print_interval = 100
    accumulation_steps = 1

    # 'COCO' or 'VOC'
    eval_type = 'COCO'
    eval_epoch = [1] + [i * 5 for i in range(epochs // 5)]
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]
    save_model_metric = 'IoU=0.50:0.95,area=all,maxDets=100,mAP'

    sync_bn = False
    apex = True

    use_ema_model = False
    ema_model_decay = 0.9999