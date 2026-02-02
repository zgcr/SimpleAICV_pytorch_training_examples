import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from tools.path import ADE20Kdataset_path

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import segmentation_decode
from SimpleAICV.universal_segmentation.datasets.ade20kdataset import ADE20KSemanticSegmentation
from SimpleAICV.universal_segmentation.semantic_segmentation_common import YoloStyleResize, RandomHorizontalFlip, Normalize, SemanticSegmentationTestCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'dinov3_vit_large_patch16_universal_segmentation'
    input_image_size = 512
    query_num = 100
    # num_classes has background class
    num_classes = 151

    model = models.__dict__[network](**{
        'image_size': input_image_size,
        'query_num': query_num,
        'num_classes': num_classes,
    })

    decoder = segmentation_decode.__dict__['UniversalSegmentationDecoder'](
        **{
            'topk': 100,
            'min_score_threshold': 0.1,
            'mask_threshold': 0.5,
            'binary_mask': True,
        }).cuda()

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/universal_segmentation_train_semantic_segmentation_on_ade20k/dinov3_vit_large_patch16_universal_segmentation_epoch_100.pth'
    load_state_dict(trained_model_path, model)

    test_dataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='validation',
        transform=transforms.Compose([
            YoloStyleResize(resize=input_image_size),
            Normalize(),
        ]))
    test_collater = SemanticSegmentationTestCollater(resize=input_image_size)

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8
