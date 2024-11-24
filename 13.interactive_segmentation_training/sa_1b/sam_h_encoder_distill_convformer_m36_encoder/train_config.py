import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import interactive_segmentation_dataset_path

from simpleAICV.interactive_segmentation.distill_model import SAMLightImageEncoderDistillModel
from simpleAICV.interactive_segmentation.distill_losses import MSELoss
from simpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    input_image_size = 1024
    freeze_teacher = True
    teacher_encoder_trained_model_path = '/root/autodl-tmp/pretrained_models/sam_encoder_weights_from_official_pytorch_weights/sam_vit_h_4b8939_encoder_convert_from_pytorch_official_weight.pth'
    student_encoder_trained_model_path = '/root/autodl-tmp/pretrained_models/convformer_finetune_on_imagenet1k_from_convert_official_weights/convformer_m36-acc84.000.pth'

    model = SAMLightImageEncoderDistillModel(
        teacher_params={
            'image_size': input_image_size,
            'patch_size': 16,
            'inplanes': 3,
            'embedding_planes': 1280,
            'block_nums': 32,
            'head_nums': 16,
            'mlp_ratio': 4,
            'out_planes': 256,
            'window_size': 14,
            'global_attn_indexes': [7, 15, 23, 31],
            'use_gradient_checkpoint': False,
        },
        student_params={
            'backbone_type': 'convformerm36backbone',
            'planes': 256,
            'use_gradient_checkpoint': False,
        },
        teacher_pretrained_path=teacher_encoder_trained_model_path,
        student_pretrained_path='',
        freeze_teacher=freeze_teacher)
    load_state_dict(student_encoder_trained_model_path, model.student.backbone)

    train_criterion = MSELoss()

    train_dataset = SAMSegmentationDataset(
        interactive_segmentation_dataset_path,
        set_name=[
            'sa_000020',
            'sa_000021',
            'sa_000022',
            'sa_000023',
            'sa_000024',
            'sa_000025',
            'sa_000026',
            'sa_000027',
            'sa_000028',
            'sa_000029',
        ],
        set_type='train',
        per_set_image_choose_max_num={
            'sa_000020': 1000000,
            'sa_000021': 1000000,
            'sa_000022': 1000000,
            'sa_000023': 1000000,
            'sa_000024': 1000000,
            'sa_000025': 1000000,
            'sa_000026': 1000000,
            'sa_000027': 1000000,
            'sa_000028': 1000000,
            'sa_000029': 1000000,
        },
        per_image_mask_chosse_max_num=1,
        positive_points_num=9,
        negative_points_num=9,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            SamResize(resize=input_image_size),
            SamRandomHorizontalFlip(prob=0.5),
            SamNormalize(mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375]),
        ]))

    train_collater = SAMBatchCollater(resize=input_image_size,
                                      positive_point_num_range=1)

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 16
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-5,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 0,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [100],
        },
    )

    epochs = 40
    print_interval = 100
    save_interval = 2

    sync_bn = False
    use_amp = True
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    clip_max_norm = 1.
