import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import salient_object_detection_dataset_path

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import segmentation_decode
from SimpleAICV.universal_segmentation.datasets.salient_object_detection_dataset import SalientObjectDetectionDataset
from SimpleAICV.universal_segmentation.salient_object_detection_common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, SalientObjectDetectionSegmentationTestCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'dinov3_vit_large_patch16_universal_segmentation'
    query_num = 100
    # num_classes has background class
    num_classes = 2
    input_image_size = [1024, 1024]

    model = models.__dict__[network](**{
        'image_size': input_image_size[0],
        'query_num': query_num,
        'num_classes': num_classes,
    })

    trained_model_path = '/root/autodl-tmp/pretrained_models/universal_segmentation_train_salient_object_detection_on_salient_object_detection_dataset/dinov3_vit_large_patch16_universal_segmentation_epoch_50.pth'
    load_state_dict(trained_model_path, model)

    decoder = segmentation_decode.__dict__['UniversalSegmentationDecoder'](
        **{
            'topk': 1,
            'min_score_threshold': 0.1,
            'mask_threshold': 0.5,
            'binary_mask': False,
        }).cuda()

    # 完整数据集必须在list中第0个位置
    val_dataset_name_list = [
        [
            'AM2K',
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
                YoloStyleResize(resize=input_image_size[0]),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)
    val_collater = SalientObjectDetectionSegmentationTestCollater(
        resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8

    thresh = [0.2]
    squared_beta = 0.3
