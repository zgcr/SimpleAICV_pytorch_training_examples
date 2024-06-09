import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import soba_v2_dataset_path

from simpleAICV.interactive_segmentation.models import segment_anything
from simpleAICV.interactive_segmentation import losses
from simpleAICV.interactive_segmentation.datasets.sobav2dataset import SOBAV2Dataset
from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'sam_b'
    input_image_size = 1024
    mask_out_idxs = [0]
    use_gradient_checkpoint = False
    frozen_image_encoder = False
    frozen_prompt_encoder = False
    frozen_mask_decoder = False
    sigmoid_out = False
    binary_mask_out = True
    mask_threshold = 0.0

    model = segment_anything.__dict__[network](
        **{
            'image_size': input_image_size,
            'use_gradient_checkpoint': use_gradient_checkpoint,
            'frozen_image_encoder': frozen_image_encoder,
            'frozen_prompt_encoder': frozen_prompt_encoder,
            'frozen_mask_decoder': frozen_mask_decoder,
            'sigmoid_out': sigmoid_out,
            'binary_mask_out': binary_mask_out,
            'mask_threshold': mask_threshold,
        })

    # load pretrained model or not
    trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/interactive_segmentation_training/sobav2/sam_b_finetune_all_point_or_box_prompt/checkpoints/epoch_500.pth'
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    test_prompt_type = {
        'prompt_point': False,
        'prompt_box': True,
        'prompt_mask': False,
    }

    test_criterion = losses.__dict__['SAMLoss'](
        **{
            'alpha': 0.8,
            'gamma': 2,
            'smooth': 1e-4,
            'focal_loss_weight': 20,
            'dice_loss_weight': 1,
            'iou_predict_loss_weight': 1,
            'mask_threshold': mask_threshold,
        })

    test_dataset = SOBAV2Dataset(soba_v2_dataset_path,
                                 set_types=[
                                     'train',
                                     'val',
                                     'challenge',
                                 ],
                                 load_mask_type='full_mask',
                                 load_box_type='object_mask',
                                 positive_points_num=5,
                                 negative_points_num=5,
                                 area_filter_ratio=0.0001,
                                 box_noise_pixel=50,
                                 mask_noise_pixel=100,
                                 read_filtered_json=True,
                                 save_filtered_json=False,
                                 transform=transforms.Compose([
                                     SamResize(resize=input_image_size),
                                     SamRandomHorizontalFlip(prob=0.5),
                                     SamNormalize(
                                         mean=[123.675, 116.28, 103.53],
                                         std=[58.395, 57.12, 57.375]),
                                 ]))
    test_collater = SAMCollater(resize=input_image_size,
                                positive_point_num_range=1,
                                negative_point_num_range=0,
                                batch_align_random_point_num=True,
                                positive_negative_point_num_ratio=None)

    seed = 0
    # batch_size is total size
    batch_size = 4
    # num_workers is total workers
    num_workers = 4
