import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import segmentation_decode
from SimpleAICV.universal_segmentation.datasets.cocodataset import CocoInstanceSegmentation
from SimpleAICV.universal_segmentation.instance_segmentation_common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, InstanceSegmentationTestCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'dinov3_vit_large_patch16_universal_segmentation'
    query_num = 200
    # num_classes has background class
    num_classes = 81
    input_image_size = [1024, 1024]

    model = models.__dict__[network](**{
        'image_size': input_image_size[0],
        'query_num': query_num,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/universal_segmentation_train_on_coco/dinov3_vit_large_patch16_universal_segmentation_epoch_50.pth'
    load_state_dict(trained_model_path, model)

    decoder = segmentation_decode.__dict__['UniversalSegmentationDecoder'](
        **{
            'topk': 100,
            'min_score_threshold': 0.1,
            'mask_threshold': 0.5,
            'binary_mask': True,
        }).cuda()

    test_dataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='val2017',
        filter_no_object_image=True,
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=input_image_size[0],
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
            Normalize(),
        ]))
    test_collater = InstanceSegmentationTestCollater(
        resize=input_image_size[0], resize_type='yolo_style')

    eval_type = 'COCO'

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8
