import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ADE20Kdataset_path

from SimpleAICV.semantic_segmentation import models
from SimpleAICV.semantic_segmentation import losses
from SimpleAICV.semantic_segmentation.datasets.ade20kdataset import ADE20KSemanticSegmentation
from SimpleAICV.semantic_segmentation.common import YoloStyleResize, RandomHorizontalFlip, Normalize, SemanticSegmentationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_pfan_semantic_segmentation'
    input_image_size = 512
    # num_classes has background class
    num_classes = 151

    backbone_pretrained_path = '/root/autodl-tmp/pretrained_models/resnet_convert_from_pytorch_official_weights/resnet50-11ad3fa6-acc1-80.858_pytorch_official_weight_convert.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    loss_list = [
        'CELoss',
    ]
    loss_ratio = {
        'CELoss': 1.0,
    }
    train_criterion = {}
    for loss_name in loss_list:
        train_criterion[loss_name] = losses.__dict__[loss_name](**{})
    test_criterion = losses.__dict__['CELoss'](**{})

    train_dataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        transform=transforms.Compose([
            YoloStyleResize(resize=input_image_size),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    test_dataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='validation',
        transform=transforms.Compose([
            YoloStyleResize(resize=input_image_size),
            Normalize(),
        ]))

    train_collater = SemanticSegmentationCollater(resize=input_image_size)

    test_collater = SemanticSegmentationCollater(resize=input_image_size)

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 32
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
        'CosineLR',
        {
            'warm_up_epochs': 1,
            'min_lr': 1e-6,
        },
    )

    epochs = 100
    eval_epoch = [1]
    for i in range(epochs):
        if i % 10 == 0:
            eval_epoch.append(i)
    print_interval = 100

    save_model_metric = 'mean_iou'

    sync_bn = False
    use_amp = True
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    use_ema_model = False
    ema_model_decay = 0.9999
