import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import human_matting_dataset_path

from SimpleAICV.human_matting import models
from SimpleAICV.human_matting import losses
from SimpleAICV.human_matting.datasets.human_matting_dataset import HumanMattingDataset
from SimpleAICV.human_matting.common import RandomHorizontalFlip, Resize, Normalize, HumanMattingCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    input_image_size = [1024, 1024]
    network = 'dinov3_vit_base_patch16_pfan_matting'

    model = models.__dict__[network](**{})

    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['GlobalTrimapCELoss']()

    # 完整数据集必须在list中第0个位置
    val_dataset_name_list = [
        [
            'P3M-500-NP',
            'P3M-500-P',
        ],
    ]

    val_dataset_list = []
    for per_sub_dataset_list in val_dataset_name_list:
        per_sub_val_dataset = HumanMattingDataset(
            human_matting_dataset_path,
            set_name_list=per_sub_dataset_list,
            set_type='val',
            kernel_size_range=15,
            transform=transforms.Compose([
                Resize(resize=input_image_size[0]),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)
    val_collater = HumanMattingCollater(resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8

    thresh = [0.2]
    squared_beta = 0.3
