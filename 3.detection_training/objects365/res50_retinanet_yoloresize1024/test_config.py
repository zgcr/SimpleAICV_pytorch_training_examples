import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import Objects365_path

from simpleAICV.detection import models
from simpleAICV.detection import losses
from simpleAICV.detection import decode
from simpleAICV.detection.datasets.objects365dataset import Objects365Detection
from simpleAICV.detection.common import DetectionResize, RandomHorizontalFlip, Normalize, DetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_retinanet'
    num_classes = 365
    input_image_size = [1024, 1024]

    model = models.__dict__[network](**{
        'backbone_pretrained_path': '',
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['RetinaLoss'](
        **{
            'areas': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
            'ratios': [0.5, 1, 2],
            'scales': [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
            'strides': [8, 16, 32, 64, 128],
            'alpha': 0.25,
            'gamma': 2,
            'beta': 1.0 / 9.0,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'box_loss_type': 'SmoothL1',
        })

    decoder = decode.__dict__['RetinaDecoder'](
        **{
            'areas': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
            'ratios': [0.5, 1, 2],
            'scales': [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
            'strides': [8, 16, 32, 64, 128],
            'max_object_num': 100,
            'min_score_threshold': 0.05,
            'topn': 1000,
            'nms_type': 'python_nms',
            'nms_threshold': 0.5,
        })

    test_dataset = Objects365Detection(Objects365_path,
                                       set_name='val',
                                       max_annots_num=150,
                                       filter_no_object_image=True,
                                       transform=transforms.Compose([
                                           DetectionResize(
                                               resize=input_image_size[0],
                                               stride=32,
                                               resize_type='yolo_style',
                                               multi_scale=False,
                                               multi_scale_range=[0.8, 1.0]),
                                           Normalize(),
                                       ]))

    test_collater = DetectionCollater(resize=input_image_size[0],
                                      resize_type='yolo_style',
                                      max_annots_num=150)

    # 'COCO' or 'VOC'
    eval_type = 'COCO'
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]

    seed = 0
    # batch_size is total size
    batch_size = 8
    # num_workers is total workers
    num_workers = 30
