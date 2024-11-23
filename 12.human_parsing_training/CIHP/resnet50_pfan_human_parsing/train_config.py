import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import human_parsing_dataset_path

from simpleAICV.human_parsing import models
from simpleAICV.human_parsing import losses
from simpleAICV.human_parsing.datasets.human_parsing_dataset import HumanParsingDataset, CIHP_20_CLASSES
from simpleAICV.human_parsing.common import YoloStyleResize, RandomHorizontalFlip, Normalize, HumanParsingCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_pfan_human_parsing'
    # 包含背景类
    num_classes = 20
    input_image_size = [512, 512]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/root/code/SimpleAICV_pytorch_training_examples/pretrained_models/resnet_convert_from_pytorch_official_weights/resnet50-11ad3fa6-acc1-80.858_pytorch_official_weight_convert.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/human_parsing_train_on_LIP/resnet50_pfan_human_parsing-metric39.299.pth'
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    loss_list = [
        'CELoss',
        'IoULoss',
    ]
    loss_ratio = {
        'CELoss': 1.0,
        'IoULoss': 1.0,
    }
    train_criterion = {}
    for loss_name in loss_list:
        if loss_name == 'IoULoss':
            train_criterion[loss_name] = losses.__dict__[loss_name](
                **{
                    'logit_type': 'softmax',
                })
        else:
            train_criterion[loss_name] = losses.__dict__[loss_name](**{})
    test_criterion = losses.__dict__['CELoss'](**{})

    train_dataset = HumanParsingDataset(
        human_parsing_dataset_path,
        set_name_list=[
            'CIHP',
        ],
        set_type='train',
        cats=CIHP_20_CLASSES,
        transform=transforms.Compose([
            YoloStyleResize(resize=input_image_size[0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    val_dataset_name_list = [
        [
            'CIHP',
        ],
    ]
    val_dataset_list = []
    for per_sub_dataset_list in val_dataset_name_list:
        per_sub_val_dataset = HumanParsingDataset(
            human_parsing_dataset_path,
            set_name_list=per_sub_dataset_list,
            set_type='val',
            cats=CIHP_20_CLASSES,
            transform=transforms.Compose([
                YoloStyleResize(resize=input_image_size[0]),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)

    train_collater = HumanParsingCollater(resize=input_image_size[0])
    val_collater = HumanParsingCollater(resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 192
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
    eval_epoch = [100]
    print_interval = 50
    save_interval = 10

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
