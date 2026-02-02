import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path

from SimpleAICV.semantic_segmentation import models
from SimpleAICV.semantic_segmentation import losses
from SimpleAICV.semantic_segmentation.datasets.cocosemanticsegmentationdataset import CocoSemanticSegmentation
from SimpleAICV.semantic_segmentation.common import YoloStyleResize, RandomHorizontalFlip, Normalize, SemanticSegmentationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'convformerm36_pfan_semantic_segmentation'
    input_image_size = 512
    # num_classes has background class
    num_classes = 81

    backbone_pretrained_path = ''
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    test_criterion = losses.__dict__['CELoss'](**{})

    test_dataset = CocoSemanticSegmentation(
        COCO2017_path,
        set_name='val2017',
        transform=transforms.Compose([
            YoloStyleResize(resize=input_image_size),
            Normalize(),
        ]))
    test_collater = SemanticSegmentationCollater(resize=input_image_size)

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8
