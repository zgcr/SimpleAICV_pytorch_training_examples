import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import VOCdataset_path

from simpleAICV.detection import models
from simpleAICV.detection import losses
from simpleAICV.detection import decode
from simpleAICV.detection.datasets.vocdataset import VocDetection
from simpleAICV.detection.common import RetinaStyleResize, YoloStyleResize, RandomHorizontalFlip, Normalize, DetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_fcos'
    num_classes = 20
    input_image_size = [640, 640]

    model = models.__dict__[network](**{
        'backbone_pretrained_path': '',
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = '/root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/detection_training/voc/res50_fcos_yoloresize640/checkpoints/resnet50_fcos-metric79.898.pth'
    # trained_model_path = os.path.join(BASE_DIR, '')
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['FCOSLoss'](
        **{
            'strides': [8, 16, 32, 64, 128],
            'mi': [[-1, 64], [64, 128], [128, 256], [256, 512],
                   [512, 100000000]],
            'alpha': 0.25,
            'gamma': 2.,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'center_ness_loss_weight': 1.,
            'box_loss_iou_type': 'CIoU',
            'center_sample_radius': 1.5,
            'use_center_sample': True,
        })

    decoder = decode.__dict__['FCOSDecoder'](**{
        'strides': [8, 16, 32, 64, 128],
        'max_object_num': 100,
        'min_score_threshold': 0.05,
        'topn': 1000,
        'nms_type': 'python_nms',
        'nms_threshold': 0.6,
    })

    test_dataset = VocDetection(root_dir=VOCdataset_path,
                                image_sets=[('2007', 'test')],
                                transform=transforms.Compose([
                                    YoloStyleResize(
                                        resize=input_image_size[0],
                                        divisor=32,
                                        stride=32,
                                        multi_scale=False,
                                        multi_scale_range=[0.5, 1.0]),
                                    Normalize(),
                                ]),
                                keep_difficult=False)

    test_collater = DetectionCollater()

    eval_type = 'VOC'
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 16