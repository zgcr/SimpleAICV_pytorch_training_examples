import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import salient_object_detection_dataset_path

from SimpleAICV.salient_object_detection import models
from SimpleAICV.salient_object_detection import losses
from SimpleAICV.salient_object_detection.datasets.salient_object_detection_dataset import SalientObjectDetectionDataset
from SimpleAICV.salient_object_detection.common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, SalientObjectDetectionSegmentationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    input_image_size = [1024, 1024]
    network = 'convformerm36_pfan_segmentation'

    backbone_pretrained_path = '/root/autodl-tmp/pretrained_models/convformer_finetune_on_imagenet1k_from_convert_official_weights/convformer_m36-acc83.980.pth'
    model = models.__dict__[network](
        **{
            'backbone_pretrained_path': backbone_pretrained_path,
        })

    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    loss_list = [
        'BCELoss',
        'BCEIouloss',
    ]
    loss_ratio = {
        'BCELoss': 1.0,
        'BCEIouloss': 1.0,
    }

    train_criterion = {}
    for loss_name in loss_list:
        train_criterion[loss_name] = losses.__dict__[loss_name]()
    test_criterion = losses.__dict__['BCELoss']()

    train_dataset = SalientObjectDetectionDataset(
        salient_object_detection_dataset_path,
        set_name_list=[
            'MAGICK',
            'AM2K',
            'DIS5K',
            'HRS10K',
            'HRSOD',
            'UHRSD',
        ],
        set_type='train',
        transform=transforms.Compose([
            YoloStyleResize(resize=input_image_size[0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

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

    train_collater = SalientObjectDetectionSegmentationCollater(
        resize=input_image_size[0])
    val_collater = SalientObjectDetectionSegmentationCollater(
        resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 64
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
    save_interval = 10

    save_model_metric = 'miou_average'
    thresh = [0.2]
    squared_beta = 0.3

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
