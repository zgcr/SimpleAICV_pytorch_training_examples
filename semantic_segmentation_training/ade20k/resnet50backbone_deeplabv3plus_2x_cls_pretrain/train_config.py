import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ADE20Kdataset_path

from simpleAICV.semantic_segmentation import models
from simpleAICV.semantic_segmentation import losses
from simpleAICV.semantic_segmentation.datasets.ade20kdataset import ADE20KSemanticSegmentation
from simpleAICV.semantic_segmentation.common import Resize, RandomCrop, RandomHorizontalFlip, PhotoMetricDistortion, Normalize, SemanticSegmentationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50backbone_deeplabv3plus'
    # not include background class(class index: ignore index)
    input_image_size = 512
    num_classes = 150
    reduce_zero_label = True
    ignore_index = 255

    backbone_pretrained_path = '/root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/pretrained_models/classification_training/resnet/resnet50-acc76.264.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    loss_list = ['CELoss']
    train_criterion = {}
    for loss_name in loss_list:
        train_criterion[loss_name] = losses.__dict__[loss_name](
            **{
                'ignore_index': ignore_index,
            })
    test_criterion = losses.__dict__['CELoss'](**{
        'ignore_index': ignore_index,
    })

    train_dataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        reduce_zero_label=reduce_zero_label,
        transform=transforms.Compose([
            Resize(image_scale=(input_image_size * 4, input_image_size),
                   multi_scale=True,
                   multi_scale_range=(0.5, 2.0)),
            RandomCrop(crop_size=(input_image_size, input_image_size),
                       cat_max_ratio=0.75,
                       ignore_index=255),
            RandomHorizontalFlip(prob=0.5),
            PhotoMetricDistortion(),
            Normalize(),
        ]))

    test_dataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='validation',
        reduce_zero_label=reduce_zero_label,
        transform=transforms.Compose([
            Resize(image_scale=(input_image_size, input_image_size),
                   multi_scale=False,
                   multi_scale_range=(0.5, 2.0)),
            Normalize(),
        ]))

    train_collater = SemanticSegmentationCollater(divisor=32,
                                                  ignore_index=ignore_index)

    test_collater = SemanticSegmentationCollater(divisor=32,
                                                 ignore_index=ignore_index)

    seed = 0
    # batch_size is total size
    batch_size = 8
    # num_workers is total workers
    num_workers = 20
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-3,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [80, 120],
        },
    )

    epochs = 128
    eval_epoch = [1]
    for i in range(epochs):
        if i % 5 == 0:
            eval_epoch.append(i)
    print_interval = 100

    save_model_metric = 'mean_iou'

    sync_bn = False
    apex = True

    use_ema_model = False
    ema_model_decay = 0.9999