import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import interactive_segmentation_dataset_path, video_interactive_segmentation_dataset_path, background_video_dataset_path

from SimpleAICV.video_interactive_segmentation.distill_model import DINOV3ImageEncoderDistillModel
from SimpleAICV.video_interactive_segmentation.distill_losses import MSELoss
from SimpleAICV.video_interactive_segmentation.datasets.sam2_video_segmentation_dataset import SAM2VideoSegmentationDataset
from SimpleAICV.video_interactive_segmentation.common import Sam2Resize, Sam2RandomHorizontalFlip, Sam2RandomMosaicAug, Sam2RandomRsverseFrameOrder, Sam2Normalize, SAM2VideoBatchCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    input_image_size = 1024
    freeze_teacher = True
    teacher_encoder_trained_model_path = '/root/autodl-tmp/pretrained_models/sam2.1_encoder_weights_convert_from_pytorch_official_weights/sam2.1_hiera_large_convert_from_pytorch_official_weight_encoder_convert_from_pytorch_official_weight.pth'
    student_encoder_trained_model_path = '/root/autodl-tmp/pretrained_models/dinov3_pytorch_official_weights/DINOv3_ViT_LVD-1689M/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'

    model = DINOV3ImageEncoderDistillModel(
        teacher_params={
            'inplanes': 3,
            'embedding_planes': 144,
            'head_nums': 2,
            'block_nums': [2, 6, 36, 4],
            'window_position_embedding_bkg_spatial_size': [7, 7],
            'window_specification': [8, 4, 16, 8],
            'global_attention_blocks': [23, 33, 43],
            'fpn_planes': 256,
            'use_gradient_checkpoint': False,
        },
        student_params={
            'backbone_type': 'dinov3_vit_base_patch16_backbone',
            'image_size': input_image_size,
            'fpn_planes': 256,
            'use_gradient_checkpoint': False,
        },
        teacher_pretrained_path=teacher_encoder_trained_model_path,
        student_pretrained_path='',
        freeze_teacher=freeze_teacher)

    load_state_dict(teacher_encoder_trained_model_path, model.student)
    load_state_dict(student_encoder_trained_model_path, model.student.trunk)

    train_criterion = MSELoss()

    train_dataset = SAM2VideoSegmentationDataset(
        image_root_dir=interactive_segmentation_dataset_path,
        image_set_name=[
            'SAMA-COCO',
            'lvisv1.0',
            ###########################################
            'HIM2K',
            'I-HIM50K',
            'RefMatte',
            'MAGICK',
            'AM2K',
            'DIS5K',
            'HRS10K',
            'HRSOD',
            'UHRSD',
            ###########################################
            'matting_human_half',
            'Deep_Automatic_Portrait_Matting',
            'RealWorldPortrait636',
            'P3M10K',
            ###########################################
            'sa_000000',
            'sa_000001',
            'sa_000002',
            'sa_000003',
            'sa_000004',
            'sa_000005',
            'sa_000006',
            'sa_000007',
            'sa_000008',
            'sa_000009',
            'sa_000010',
            'sa_000011',
            'sa_000012',
            'sa_000013',
            'sa_000014',
            'sa_000015',
            'sa_000016',
            'sa_000017',
            'sa_000018',
            'sa_000019',
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
            'sa_000030',
            'sa_000031',
            'sa_000032',
            'sa_000033',
            'sa_000034',
            'sa_000035',
            'sa_000036',
            'sa_000037',
            'sa_000038',
            'sa_000039',
            'sa_000040',
            'sa_000041',
            'sa_000042',
            'sa_000043',
            'sa_000044',
            'sa_000045',
            'sa_000046',
            'sa_000047',
            'sa_000048',
            'sa_000049',
        ],
        image_set_type='train',
        image_per_set_image_choose_max_num={
            'SAMA-COCO': 1000000,
            'lvisv1.0': 1000000,
            ###########################################
            'HIM2K': 1000000,
            'I-HIM50K': 1000000,
            'RefMatte': 1000000,
            'MAGICK': 1000000,
            'AM2K': 1000000,
            'DIS5K': 1000000,
            'HRS10K': 1000000,
            'HRSOD': 1000000,
            'UHRSD': 1000000,
            ###########################################
            'matting_human_half': 1000000,
            'Deep_Automatic_Portrait_Matting': 1000000,
            'RealWorldPortrait636': 1000000,
            'P3M10K': 1000000,
            ###########################################
            'sa_000000': 1000000,
            'sa_000001': 1000000,
            'sa_000002': 1000000,
            'sa_000003': 1000000,
            'sa_000004': 1000000,
            'sa_000005': 1000000,
            'sa_000006': 1000000,
            'sa_000007': 1000000,
            'sa_000008': 1000000,
            'sa_000009': 1000000,
            'sa_000010': 1000000,
            'sa_000011': 1000000,
            'sa_000012': 1000000,
            'sa_000013': 1000000,
            'sa_000014': 1000000,
            'sa_000015': 1000000,
            'sa_000016': 1000000,
            'sa_000017': 1000000,
            'sa_000018': 1000000,
            'sa_000019': 1000000,
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
            'sa_000030': 1000000,
            'sa_000031': 1000000,
            'sa_000032': 1000000,
            'sa_000033': 1000000,
            'sa_000034': 1000000,
            'sa_000035': 1000000,
            'sa_000036': 1000000,
            'sa_000037': 1000000,
            'sa_000038': 1000000,
            'sa_000039': 1000000,
            'sa_000040': 1000000,
            'sa_000041': 1000000,
            'sa_000042': 1000000,
            'sa_000043': 1000000,
            'sa_000044': 1000000,
            'sa_000045': 1000000,
            'sa_000046': 1000000,
            'sa_000047': 1000000,
            'sa_000048': 1000000,
            'sa_000049': 1000000,
        },
        per_image_mask_chosse_max_num=1,
        video_root_dir=video_interactive_segmentation_dataset_path,
        video_set_name=[
            'MOSEv2',
            'DAVIS2017',
            'YouTubeVOS2019',
            ###########################################
            'sav_000',
            'sav_001',
            'sav_002',
            'sav_003',
            'sav_004',
            'sav_005',
            'sav_006',
            'sav_007',
            'sav_008',
            'sav_009',
            'sav_010',
            'sav_011',
            'sav_012',
            'sav_013',
            'sav_014',
            'sav_015',
            'sav_016',
            'sav_017',
            'sav_018',
            'sav_019',
            'sav_020',
            'sav_021',
            'sav_022',
            'sav_023',
            'sav_024',
            'sav_025',
            'sav_026',
            'sav_027',
            'sav_028',
            'sav_029',
            'sav_030',
            'sav_031',
            'sav_032',
            'sav_033',
            'sav_034',
            'sav_035',
            'sav_036',
            'sav_037',
            'sav_038',
            'sav_039',
            'sav_040',
            'sav_041',
            'sav_042',
            'sav_043',
            'sav_044',
            'sav_045',
            'sav_046',
            'sav_047',
            'sav_048',
            'sav_049',
            'sav_050',
            'sav_051',
            'sav_052',
            'sav_053',
            'sav_054',
            'sav_055',
        ],
        video_set_type='train',
        video_matting_root_dir=video_interactive_segmentation_dataset_path,
        video_matting_set_name_list=[
            'V-HIM2K5',
            'V-HIM60_comp_easy',
            'V-HIM60_comp_medium',
            'V-HIM60_comp_hard',
            'VideoMatte240K',
        ],
        video_matting_use_background_video_prob={
            'V-HIM2K5': 0.0,
            'V-HIM60_comp_easy': 0.0,
            'V-HIM60_comp_medium': 0.0,
            'V-HIM60_comp_hard': 0.0,
            'VideoMatte240K': 1.0,
        },
        video_matting_set_type='train',
        video_matting_background_dir=background_video_dataset_path,
        video_matting_background_set_type='train',
        per_video_choose_frame_nums=8,
        per_video_choose_object_nums=1,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            Sam2Resize(resize=input_image_size),
            Sam2RandomHorizontalFlip(prob=0.5),
            Sam2RandomMosaicAug(prob=0.1),
            Sam2RandomRsverseFrameOrder(prob=0.5),
            Sam2Normalize(mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375]),
        ]))

    train_collater = SAM2VideoBatchCollater(resize=input_image_size,
                                            use_image_prob=0.)

    seed = 0
    # batch_size is total size
    batch_size = 24
    # num_workers is total workers
    num_workers = 32
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

    epochs = 20
    print_interval = 100
    save_interval = 5

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
