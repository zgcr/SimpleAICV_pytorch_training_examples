import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import interactive_segmentation_dataset_path, video_interactive_segmentation_dataset_path, background_video_dataset_path

from SimpleAICV.video_interactive_segmentation.models.segment_anything2 import sam2video_train
from SimpleAICV.video_interactive_segmentation.losses import SAM2MultiLevelLoss
from SimpleAICV.video_interactive_segmentation.datasets.sam2_video_segmentation_dataset import SAM2VideoSegmentationDataset
from SimpleAICV.video_interactive_segmentation.common import Sam2Resize, Sam2RandomHorizontalFlip, Sam2RandomMosaicAug, Sam2RandomRsverseFrameOrder, Sam2Normalize, SAM2VideoBatchCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'hiera_b_plus_sam2video'
    input_image_size = 1024
    mask_out_idxs = [0, 1, 2, 3]
    use_gradient_checkpoint = False
    frozen_image_encoder = False
    frozen_prompt_encoder = False
    frozen_mask_decoder = False
    frozen_memory_attention = False
    frozen_memory_encoder = False
    mask_threshold = 0.0

    model = sam2video_train.__dict__[network](
        **{
            'image_size': input_image_size,
            'use_gradient_checkpoint': use_gradient_checkpoint,
            'frozen_image_encoder': frozen_image_encoder,
            'frozen_prompt_encoder': frozen_prompt_encoder,
            'frozen_mask_decoder': frozen_mask_decoder,
            'frozen_memory_attention': frozen_memory_attention,
            'frozen_memory_encoder': frozen_memory_encoder,
            'mask_out_idxs': mask_out_idxs,
            'use_single_prompt': True,
            'use_point_prompt_prob': 0.25,
            'use_box_prompt_prob': 0.25,
            'use_mask_prompt_prob': 0.5,
            'max_condition_frame_num': 2,
            'random_condition_frame_num': True,
            'max_decoder_point_iters_frame_num': 2,
            'random_decoder_point_iters_frame_num': True,
            'sample_decoder_point_from_gt_mask_prob': 0.1,
            'decoder_point_iters_num': 4,
        })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/sam2_segmentation_train_on_video_interactive_segmentation_dataset/hiera_b_plus_sam2video_multilevel_stage1_epoch_2.pth'
    load_state_dict(trained_model_path, model)

    train_criterion = SAM2MultiLevelLoss(alpha=0.25,
                                         gamma=2,
                                         focal_loss_weight=20,
                                         dice_loss_weight=1,
                                         iou_predict_loss_weight=1,
                                         class_loss_weight=1,
                                         mask_threshold=mask_threshold)

    train_dataset = SAM2VideoSegmentationDataset(
        image_root_dir=interactive_segmentation_dataset_path,
        image_set_name=[
            'SAMA-COCO',
            'lvisv1.0_filter_part_object',
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
            'sa_000000_filter_part_object',
            'sa_000001_filter_part_object',
            'sa_000002_filter_part_object',
            'sa_000003_filter_part_object',
            'sa_000004_filter_part_object',
            'sa_000005_filter_part_object',
            'sa_000006_filter_part_object',
            'sa_000007_filter_part_object',
            'sa_000008_filter_part_object',
            'sa_000009_filter_part_object',
            'sa_000010_filter_part_object',
            'sa_000011_filter_part_object',
            'sa_000012_filter_part_object',
            'sa_000013_filter_part_object',
            'sa_000014_filter_part_object',
            'sa_000015_filter_part_object',
            'sa_000016_filter_part_object',
            'sa_000017_filter_part_object',
            'sa_000018_filter_part_object',
            'sa_000019_filter_part_object',
            'sa_000020_filter_part_object',
            'sa_000021_filter_part_object',
            'sa_000022_filter_part_object',
            'sa_000023_filter_part_object',
            'sa_000024_filter_part_object',
            'sa_000025_filter_part_object',
            'sa_000026_filter_part_object',
            'sa_000027_filter_part_object',
            'sa_000028_filter_part_object',
            'sa_000029_filter_part_object',
            'sa_000030_filter_part_object',
            'sa_000031_filter_part_object',
            'sa_000032_filter_part_object',
            'sa_000033_filter_part_object',
            'sa_000034_filter_part_object',
            'sa_000035_filter_part_object',
            'sa_000036_filter_part_object',
            'sa_000037_filter_part_object',
            'sa_000038_filter_part_object',
            'sa_000039_filter_part_object',
            'sa_000040_filter_part_object',
            'sa_000041_filter_part_object',
            'sa_000042_filter_part_object',
            'sa_000043_filter_part_object',
            'sa_000044_filter_part_object',
            'sa_000045_filter_part_object',
            'sa_000046_filter_part_object',
            'sa_000047_filter_part_object',
            'sa_000048_filter_part_object',
            'sa_000049_filter_part_object',
        ],
        image_set_type='train',
        image_per_set_image_choose_max_num={
            'SAMA-COCO': 1000000,
            'lvisv1.0_filter_part_object': 1000000,
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
            'sa_000000_filter_part_object': 1000000,
            'sa_000001_filter_part_object': 1000000,
            'sa_000002_filter_part_object': 1000000,
            'sa_000003_filter_part_object': 1000000,
            'sa_000004_filter_part_object': 1000000,
            'sa_000005_filter_part_object': 1000000,
            'sa_000006_filter_part_object': 1000000,
            'sa_000007_filter_part_object': 1000000,
            'sa_000008_filter_part_object': 1000000,
            'sa_000009_filter_part_object': 1000000,
            'sa_000010_filter_part_object': 1000000,
            'sa_000011_filter_part_object': 1000000,
            'sa_000012_filter_part_object': 1000000,
            'sa_000013_filter_part_object': 1000000,
            'sa_000014_filter_part_object': 1000000,
            'sa_000015_filter_part_object': 1000000,
            'sa_000016_filter_part_object': 1000000,
            'sa_000017_filter_part_object': 1000000,
            'sa_000018_filter_part_object': 1000000,
            'sa_000019_filter_part_object': 1000000,
            'sa_000020_filter_part_object': 1000000,
            'sa_000021_filter_part_object': 1000000,
            'sa_000022_filter_part_object': 1000000,
            'sa_000023_filter_part_object': 1000000,
            'sa_000024_filter_part_object': 1000000,
            'sa_000025_filter_part_object': 1000000,
            'sa_000026_filter_part_object': 1000000,
            'sa_000027_filter_part_object': 1000000,
            'sa_000028_filter_part_object': 1000000,
            'sa_000029_filter_part_object': 1000000,
            'sa_000030_filter_part_object': 1000000,
            'sa_000031_filter_part_object': 1000000,
            'sa_000032_filter_part_object': 1000000,
            'sa_000033_filter_part_object': 1000000,
            'sa_000034_filter_part_object': 1000000,
            'sa_000035_filter_part_object': 1000000,
            'sa_000036_filter_part_object': 1000000,
            'sa_000037_filter_part_object': 1000000,
            'sa_000038_filter_part_object': 1000000,
            'sa_000039_filter_part_object': 1000000,
            'sa_000040_filter_part_object': 1000000,
            'sa_000041_filter_part_object': 1000000,
            'sa_000042_filter_part_object': 1000000,
            'sa_000043_filter_part_object': 1000000,
            'sa_000044_filter_part_object': 1000000,
            'sa_000045_filter_part_object': 1000000,
            'sa_000046_filter_part_object': 1000000,
            'sa_000047_filter_part_object': 1000000,
            'sa_000048_filter_part_object': 1000000,
            'sa_000049_filter_part_object': 1000000,
        },
        per_image_mask_chosse_max_num=16,
        video_root_dir=video_interactive_segmentation_dataset_path,
        video_set_name=[
            'MOSEv2',
            'DAVIS2017',
            'YouTubeVOS2019',
            ###########################################
            'sav_000_filter_part_object',
            'sav_001_filter_part_object',
            'sav_002_filter_part_object',
            'sav_003_filter_part_object',
            'sav_004_filter_part_object',
            'sav_005_filter_part_object',
            'sav_006_filter_part_object',
            'sav_007_filter_part_object',
            'sav_008_filter_part_object',
            'sav_009_filter_part_object',
            'sav_010_filter_part_object',
            'sav_011_filter_part_object',
            'sav_012_filter_part_object',
            'sav_013_filter_part_object',
            'sav_014_filter_part_object',
            'sav_015_filter_part_object',
            'sav_016_filter_part_object',
            'sav_017_filter_part_object',
            'sav_018_filter_part_object',
            'sav_019_filter_part_object',
            'sav_020_filter_part_object',
            'sav_021_filter_part_object',
            'sav_022_filter_part_object',
            'sav_023_filter_part_object',
            'sav_024_filter_part_object',
            'sav_025_filter_part_object',
            'sav_026_filter_part_object',
            'sav_027_filter_part_object',
            'sav_028_filter_part_object',
            'sav_029_filter_part_object',
            'sav_030_filter_part_object',
            'sav_031_filter_part_object',
            'sav_032_filter_part_object',
            'sav_033_filter_part_object',
            'sav_034_filter_part_object',
            'sav_035_filter_part_object',
            'sav_036_filter_part_object',
            'sav_037_filter_part_object',
            'sav_038_filter_part_object',
            'sav_039_filter_part_object',
            'sav_040_filter_part_object',
            'sav_041_filter_part_object',
            'sav_042_filter_part_object',
            'sav_043_filter_part_object',
            'sav_044_filter_part_object',
            'sav_045_filter_part_object',
            'sav_046_filter_part_object',
            'sav_047_filter_part_object',
            'sav_048_filter_part_object',
            'sav_049_filter_part_object',
            'sav_050_filter_part_object',
            'sav_051_filter_part_object',
            'sav_052_filter_part_object',
            'sav_053_filter_part_object',
            'sav_054_filter_part_object',
            'sav_055_filter_part_object',
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
        per_video_choose_object_nums=2,
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
                                            use_image_prob=0.1)

    seed = 0
    # batch_size is total size
    batch_size = 16
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
    print_interval = 10
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

    clip_max_norm = 0.1

    find_unused_parameters = True
