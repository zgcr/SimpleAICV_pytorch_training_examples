import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import human_matting_dataset_path

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import matting_decode
from SimpleAICV.universal_segmentation.datasets.human_matting_dataset import HumanMattingDataset
from SimpleAICV.universal_segmentation.human_matting_common import RandomHorizontalFlip, Resize, Normalize, HumanMattingTestCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'dinov3_vit_large_patch16_universal_matting'
    query_num = 100
    # num_classes has background class
    num_classes = 2
    input_image_size = [1024, 1024]

    model = models.__dict__[network](**{
        'image_size': input_image_size[0],
        'query_num': query_num,
        'num_classes': num_classes,
    })

    trained_model_path = '/root/autodl-tmp/pretrained_models/universal_matting_train_human_matting_on_human_matting_dataset/dinov3_vit_large_patch16_universal_matting_epoch_50.pth'
    load_state_dict(trained_model_path, model)

    decoder = matting_decode.__dict__['UniversalMattingDecoder'](
        **{
            'topk': 1,
            'min_score_threshold': 0.1,
        }).cuda()

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
            max_side=2048,
            kernel_size_range=[15, 15],
            transform=transforms.Compose([
                Resize(resize=input_image_size[0]),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)
    val_collater = HumanMattingTestCollater(resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 8
    # num_workers is total workers
    num_workers = 8

    thresh = [0.2]
    squared_beta = 0.3
