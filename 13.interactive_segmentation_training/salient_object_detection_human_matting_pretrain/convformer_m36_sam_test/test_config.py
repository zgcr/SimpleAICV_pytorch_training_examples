import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import interactive_segmentation_dataset_path

from simpleAICV.interactive_segmentation.models.light_segment_anything import light_sam
from simpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'convformerm36_light_sam'
    input_image_size = 1024
    mask_out_idxs = [0, 1, 2, 3]
    sigmoid_out = False
    binary_mask_out = False
    mask_threshold = 0.0
    loop_mask_iters = 1

    model = light_sam.__dict__[network](**{
        'image_size': input_image_size,
        'sigmoid_out': sigmoid_out,
        'binary_mask_out': binary_mask_out,
        'mask_threshold': mask_threshold,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/light_sam_train_on_salient_object_detection_human_matting_dataset/convformerm36_light_sam-loss0.096.pth'
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    test_prompt_type = {
        'prompt_point': False,
        'prompt_box': True,
    }

    test_dataset_name_list = [
        [
            'DIS5K_seg',
            'HRS10K_seg',
            'HRSOD_seg',
            'UHRSD_seg',
            'Deep_Automatic_Portrait_Matting_seg',
            'P3M-500-NP_seg',
            'P3M-500-P_seg',
        ],
    ]
    test_dataset_list = []
    for per_test_dir in test_dataset_name_list:
        test_dataset = SAMSegmentationDataset(
            interactive_segmentation_dataset_path,
            set_name=per_test_dir,
            set_type='val',
            per_set_image_choose_max_num={
                'DIS5K_seg': 10000000,
                'HRS10K_seg': 10000000,
                'HRSOD_seg': 10000000,
                'UHRSD_seg': 10000000,
                'Deep_Automatic_Portrait_Matting_seg': 10000000,
                'P3M-500-NP_seg': 10000000,
                'P3M-500-P_seg': 10000000,
            },
            per_image_mask_chosse_max_num=16,
            positive_points_num=9,
            negative_points_num=9,
            area_filter_ratio=0.0001,
            box_noise_wh_ratio=0.1,
            mask_noise_area_ratio=0.04,
            transform=transforms.Compose([
                SamResize(resize=input_image_size),
                SamNormalize(mean=[123.675, 116.28, 103.53],
                             std=[58.395, 57.12, 57.375]),
            ]))
        test_dataset_list.append(test_dataset)
    test_collater = SAMBatchCollater(resize=input_image_size,
                                     positive_point_num_range=1)

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 4
