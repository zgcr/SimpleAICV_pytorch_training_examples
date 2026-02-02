import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import human_matting_dataset_path

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import matting_losses
from SimpleAICV.universal_segmentation.datasets.human_matting_dataset import HumanMattingDataset
from SimpleAICV.universal_segmentation.human_matting_common import RandomHorizontalFlip, Resize, Normalize, HumanMattingTrainCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'dinov3_vit_large_patch16_universal_matting'
    query_num = 100
    # num_classes has background class
    num_classes = 2
    input_image_size = [1024, 1024]

    backbone_pretrained_path = '/root/autodl-tmp/pretrained_models/dinov3_pytorch_official_weights/DINOv3_ViT_LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    model = models.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'image_size': input_image_size[0],
        'query_num': query_num,
        'num_classes': num_classes,
        'use_gradient_checkpoint': True,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = matting_losses.__dict__['UniversalMattingLoss'](
        **{
            'global_trimap_ce_cost': 1.0,
            'global_trimap_iou_cost': 1.0,
            'local_alpha_cost': 1.0,
            'fusion_alpha_cost': 1.0,
            'class_cost': 1.0,
            'num_classes': num_classes,
            'global_trimap_ce_loss_weight': 1.0,
            'global_trimap_iou_loss_weight': 1.0,
            'local_alpha_loss_weight': 1.0,
            'local_laplacian_loss_weight': 1.0,
            'fusion_alpha_loss_weight': 1.0,
            'fusion_laplacian_loss_weight': 1.0,
            'class_loss_weight': 1.0,
            'no_object_class_weight': 0.1,
        })

    train_dataset = HumanMattingDataset(human_matting_dataset_path,
                                        set_name_list=[
                                            'matting_human_half',
                                            'Deep_Automatic_Portrait_Matting',
                                            'RealWorldPortrait636',
                                            'P3M10K',
                                        ],
                                        set_type='train',
                                        max_side=2048,
                                        kernel_size_range=[15, 15],
                                        transform=transforms.Compose([
                                            Resize(resize=input_image_size[0]),
                                            RandomHorizontalFlip(prob=0.5),
                                            Normalize(),
                                        ]))

    train_collater = HumanMattingTrainCollater(resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 32
    accumulation_steps = 4

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

    epochs = 50
    print_interval = 100
    save_interval = 10

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
