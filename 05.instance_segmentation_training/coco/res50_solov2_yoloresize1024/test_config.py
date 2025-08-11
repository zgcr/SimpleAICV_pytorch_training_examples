import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path

from simpleAICV.instance_segmentation import models
from simpleAICV.instance_segmentation import losses
from simpleAICV.instance_segmentation import decode
from simpleAICV.instance_segmentation.datasets.cocodataset import CocoInstanceSegmentation
from simpleAICV.instance_segmentation.common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, SOLOV2InstanceSegmentationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_solov2'
    num_classes = 80
    input_image_size = [1024, 1024]

    # load backbone pretrained model or not
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

    test_criterion = losses.__dict__['SOLOV2Loss'](**{
        'scale_ranges': ((1, 96), (48, 192), (96, 384), (192, 768), (384,
                                                                     2048)),
        'grid_nums': (40, 36, 24, 16, 12),
        'mask_feature_upsample_scale':
        4,
        'sigma':
        0.2,
        'alpha':
        0.25,
        'gamma':
        2.0,
        'cls_loss_weight':
        1.0,
        'dice_loss_weight':
        3.0,
    })

    decoder = decode.__dict__['SOLOV2Decoder'](
        **{
            'strides': (8, 8, 16, 32, 32),
            'grid_nums': (40, 36, 24, 16, 12),
            'mask_feature_upsample_scale': 4,
            'max_mask_num': 100,
            'topn': 500,
            'min_score_threshold': 0.1,
            'keep_score_threshold': 0.1,
            'mask_threshold': 0.5,
            'update_threshold': 0.05,
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
    test_collater = SOLOV2InstanceSegmentationCollater(
        resize=input_image_size[0], resize_type='yolo_style')

    eval_type = 'COCO'

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8
