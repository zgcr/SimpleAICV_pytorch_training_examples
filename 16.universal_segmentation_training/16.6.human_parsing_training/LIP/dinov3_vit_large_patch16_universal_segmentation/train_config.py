import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from tools.path import human_parsing_dataset_path

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import segmentation_losses
from SimpleAICV.universal_segmentation.datasets.human_parsing_dataset import HumanParsingDataset, LIP_20_CLASSES
from SimpleAICV.universal_segmentation.human_parsing_common import YoloStyleResize, RandomHorizontalFlip, Normalize, HumanParsingTrainCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'dinov3_vit_large_patch16_universal_segmentation'
    query_num = 100
    # num_classes has background class
    num_classes = 20
    input_image_size = [512, 512]

    backbone_pretrained_path = '/root/autodl-tmp/pretrained_models/dinov3_pytorch_official_weights/DINOv3_ViT_LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'image_size': input_image_size[0],
        'query_num': query_num,
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = segmentation_losses.__dict__[
        'UniversalSegmentationLoss'](**{
            'mask_cost': 5.0,
            'dice_cost': 5.0,
            'class_cost': 2.0,
            'num_classes': num_classes,
            'mask_loss_weight': 5.0,
            'dice_loss_weight': 5.0,
            'class_loss_weight': 2.0,
            'no_object_class_weight': 0.1,
        })

    train_dataset = HumanParsingDataset(
        human_parsing_dataset_path,
        set_name_list=[
            'LIP',
        ],
        set_type='train',
        cats=LIP_20_CLASSES,
        transform=transforms.Compose([
            YoloStyleResize(resize=input_image_size[0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    train_collater = HumanParsingTrainCollater(resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 32
    accumulation_steps = 1

    optimizer = (
        'Muon',
        {
            'lr': 4e-4,
            'weight_decay': 1e-3,
            'exclude_muon_layer_name_list': [],
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
    print_interval = 50
    save_interval = 20

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
