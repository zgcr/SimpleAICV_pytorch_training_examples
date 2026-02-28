import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import interactive_segmentation_dataset_path

from SimpleAICV.interactive_segmentation.models.segment_anything import sam
from SimpleAICV.interactive_segmentation import losses
from SimpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
from SimpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'sam_b'
    input_image_size = 1024
    mask_out_idxs = [0, 1, 2, 3]
    use_gradient_checkpoint = True
    frozen_image_encoder = False
    frozen_prompt_encoder = False
    frozen_mask_decoder = False
    mask_threshold = 0.0
    decoder_iters = 4

    model = sam.__dict__[network](**{
        'image_size': input_image_size,
        'use_gradient_checkpoint': use_gradient_checkpoint,
        'frozen_image_encoder': frozen_image_encoder,
        'frozen_prompt_encoder': frozen_prompt_encoder,
        'frozen_mask_decoder': frozen_mask_decoder,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/sam_pytorch_official_weights/sam_vit_b_01ec64.pth'
    load_state_dict(trained_model_path, model)

    use_single_prompt = True
    # points and boxes prob must be not both 0
    prompt_probs = {
        'prompt_point': 0.5,
        'prompt_box': 0.5,
        'prompt_mask': 0.,
    }

    train_criterion = losses.__dict__['SAMLoss'](
        **{
            'alpha': 0.25,
            'gamma': 2,
            'focal_loss_weight': 20,
            'dice_loss_weight': 1,
            'iou_predict_loss_weight': 1,
            'supervise_all_iou': True,
            'mask_threshold': mask_threshold,
        })

    train_dataset = SAMSegmentationDataset(
        interactive_segmentation_dataset_path,
        set_name=[
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
        set_type='train',
        per_set_image_choose_max_num={
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
        per_image_mask_chosse_max_num=16,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            SamResize(resize=input_image_size),
            SamRandomHorizontalFlip(prob=0.5),
            SamNormalize(mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375]),
        ]))

    train_collater = SAMBatchCollater(resize=input_image_size)

    seed = 0
    # batch_size is total size
    batch_size = 160
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
