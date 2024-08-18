import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import salient_object_detection_dataset_path

from simpleAICV.salient_object_detection import models
from simpleAICV.salient_object_detection import losses
from simpleAICV.salient_object_detection.datasets.salient_object_detection_dataset import SalientObjectDetectionDataset
from simpleAICV.salient_object_detection.common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, ResizeSegmentationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    input_image_size = [832, 832]
    network = 'van_b2_pfan_segmentation'

    model = models.__dict__[network](**{})

    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['BCELoss']()

    # 完整数据集必须在list中第0个位置
    val_dataset_name_list = [
        [
            'DIS5K',
            'HRS10K',
            'HRSOD',
            'UHRSD',
        ],
    ]

    val_dataset_list = []
    for per_sub_dataset_list in val_dataset_name_list:
        per_sub_val_dataset = SalientObjectDetectionDataset(
            salient_object_detection_dataset_path,
            set_name_list=per_sub_dataset_list,
            set_type='val',
            transform=transforms.Compose([
                YoloStyleResize(resize=input_image_size[0],
                                divisor=64,
                                stride=64),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)
    val_collater = ResizeSegmentationCollater(resize=input_image_size[0],
                                              stride=64)

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 4

    thresh = [0.2]
    squared_beta = 0.3
