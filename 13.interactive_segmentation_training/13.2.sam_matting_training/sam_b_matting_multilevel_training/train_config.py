import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import interactive_segmentation_dataset_path

from SimpleAICV.interactive_segmentation.models.segment_anything_matting import sam_matting
from SimpleAICV.interactive_segmentation import losses_matting
from SimpleAICV.interactive_segmentation.datasets.sam_matting_dataset import SAMMattingDataset
from SimpleAICV.interactive_segmentation.common_matting import SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingBatchCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'sam_b_matting'
    input_image_size = 1024
    mask_out_idxs = [0, 1, 2, 3]
    use_gradient_checkpoint = True
    frozen_image_encoder = False
    frozen_prompt_encoder = False
    frozen_mask_decoder = False
    mask_threshold = 0.5
    decoder_iters = 4

    model = sam_matting.__dict__[network](
        **{
            'image_size': input_image_size,
            'use_gradient_checkpoint': use_gradient_checkpoint,
            'frozen_image_encoder': frozen_image_encoder,
            'frozen_prompt_encoder': frozen_prompt_encoder,
            'frozen_mask_decoder': frozen_mask_decoder,
        })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/sam_segmentation_train_on_interactive_segmentation_dataset/sam_b_multilevel_epoch_2.pth'
    load_state_dict(trained_model_path, model)

    use_single_prompt = True
    # points and boxes prob must be not both 0
    prompt_probs = {
        'prompt_point': 0.5,
        'prompt_box': 0.5,
        'prompt_mask': 0.,
    }

    train_criterion = losses_matting.__dict__['SAMMattingMultiLevelLoss'](
        **{
            'global_pred_trimap_ce_loss_weight': 1,
            'global_pred_trimap_iou_loss_weight': 1,
            'local_pred_alpha_loss_weight': 1,
            'local_pred_laplacian_loss_weight': 1,
            'fusion_pred_alpha_loss_weight': 1,
            'fusion_pred_laplacian_loss_weight': 1,
            'composition_loss_weight': 1,
            'iou_predict_loss_weight': 1,
            'mask_threshold': mask_threshold,
        })

    train_dataset = SAMMattingDataset(
        interactive_segmentation_dataset_path,
        set_name=[
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
        set_type='train',
        per_set_image_choose_max_num={
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
        max_side=2048,
        kernel_size_range=[15, 15],
        per_image_mask_chosse_max_num=16,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            SAMMattingResize(resize=input_image_size),
            SAMMattingRandomHorizontalFlip(prob=0.5),
            SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375]),
        ]))

    train_collater = SAMMattingBatchCollater(resize=input_image_size)

    seed = 0
    # batch_size is total size
    batch_size = 48
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

    epochs = 2
    print_interval = 100
    save_interval = 1

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

    find_unused_parameters = True
