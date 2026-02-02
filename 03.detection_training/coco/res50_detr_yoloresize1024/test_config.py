import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path

from SimpleAICV.detection import models
from SimpleAICV.detection import losses
from SimpleAICV.detection import decode
from SimpleAICV.detection.datasets.cocodataset import CocoDetection
from SimpleAICV.detection.common import DetectionResize, RandomHorizontalFlip, Normalize, DETRDetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_detr'
    num_classes = 80
    input_image_size = [1024, 1024]

    model = models.__dict__[network](**{
        'backbone_pretrained_path': '',
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['DETRLoss'](**{
        'cls_match_cost': 1.0,
        'box_match_cost': 5.0,
        'giou_match_cost': 2.0,
        'cls_loss_weight': 1.0,
        'box_l1_loss_weight': 5.0,
        'iou_loss_weight': 2.0,
        'no_object_cls_weight': 0.1,
        'num_classes': num_classes,
    })
    decoder = decode.__dict__['DETRDecoder'](**{
        'num_classes': num_classes,
        'max_object_num': 100,
        'min_score_threshold': 0.05,
        'topn': 100,
        'nms_type': None,
        'nms_threshold': 0.5,
    })

    test_dataset = CocoDetection(COCO2017_path,
                                 set_name='val2017',
                                 transform=transforms.Compose([
                                     DetectionResize(
                                         resize=input_image_size[0],
                                         stride=32,
                                         resize_type='yolo_style',
                                         multi_scale=False,
                                         multi_scale_range=[0.8, 1.0]),
                                     Normalize(),
                                 ]))

    test_collater = DETRDetectionCollater(resize=input_image_size[0],
                                          resize_type='yolo_style',
                                          max_annots_num=100)

    # 'COCO' or 'VOC'
    eval_type = 'COCO'
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 30
