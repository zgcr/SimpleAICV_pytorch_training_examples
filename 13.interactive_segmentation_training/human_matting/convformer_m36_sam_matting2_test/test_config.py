import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import interactive_segmentation_dataset_path

from simpleAICV.interactive_segmentation.models.light_segment_anything_matting import light_sam_matting2
from simpleAICV.interactive_segmentation.datasets.sam_matting_dataset import SAMMattingDataset
from simpleAICV.interactive_segmentation.common_matting import SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingBatchCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'convformerm36_light_sam_matting2'
    input_image_size = 1024
    mask_out_idxs = [0, 1, 2, 3]
    loop_mask_iters = 1

    model = light_sam_matting2.__dict__[network](
        **{
            'image_size': input_image_size,
        })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/light_sam_matting_train_on_human_matting/convformerm36_light_sam_matting2-loss0.079.pth'
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    test_prompt_type = {
        'prompt_point': False,
        'prompt_box': True,
    }

    test_dataset_name_list = [
        [
            'Deep_Automatic_Portrait_Matting',
            'P3M-500-NP',
            'P3M-500-P',
        ],
    ]
    test_dataset_list = []
    for per_test_dir in test_dataset_name_list:
        test_dataset = SAMMattingDataset(
            interactive_segmentation_dataset_path,
            set_name=per_test_dir,
            set_type='val',
            max_side=2048,
            kernel_size_range=10,
            per_set_image_choose_max_num={
                'Deep_Automatic_Portrait_Matting': 10000000,
                'P3M-500-NP': 10000000,
                'P3M-500-P': 10000000,
            },
            per_image_mask_chosse_max_num=1,
            positive_points_num=9,
            negative_points_num=9,
            area_filter_ratio=0.0001,
            box_noise_wh_ratio=0.1,
            mask_noise_area_ratio=0.04,
            resample_num=1,
            transform=transforms.Compose([
                SAMMattingResize(resize=input_image_size),
                SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375]),
            ]))
        test_dataset_list.append(test_dataset)
    test_collater = SAMMattingBatchCollater(resize=input_image_size,
                                            positive_point_num_range=1)

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 4

    thresh = [0.2]
    # β2一般取值为0.3. 相当于增大了Precision的重要性
    squared_beta = 0.3
