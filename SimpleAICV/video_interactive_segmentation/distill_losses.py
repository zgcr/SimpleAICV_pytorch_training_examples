import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MSELoss',
]


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, tea_preds, stu_preds):
        tea_features, _ = tea_preds
        stu_features, _ = stu_preds

        assert len(tea_features) == len(stu_features)

        loss = 0.
        for per_tea_feature, per_stu_feature in zip(tea_features,
                                                    stu_features):
            per_loss = self.loss(per_tea_feature, per_stu_feature)
            loss += per_loss

        loss = loss / len(tea_features)

        loss_dict = {
            'distill_mse_loss': loss,
        }

        return loss_dict


if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from tools.path import interactive_segmentation_dataset_path, video_interactive_segmentation_dataset_path, background_video_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.video_interactive_segmentation.datasets.sam2_video_segmentation_dataset import SAM2VideoSegmentationDataset
    from SimpleAICV.video_interactive_segmentation.common import Sam2Resize, Sam2RandomHorizontalFlip, Sam2RandomMosaicAug, Sam2RandomRsverseFrameOrder, Sam2Normalize, SAM2VideoBatchCollater, load_state_dict

    sam2_video_dataset = SAM2VideoSegmentationDataset(
        image_root_dir=interactive_segmentation_dataset_path,
        image_set_name=[
            ###########################################
            'sa_000000',
        ],
        image_set_type='train',
        image_per_set_image_choose_max_num={
            ###########################################
            'sa_000000': 1000000,
        },
        per_image_mask_chosse_max_num=1,
        video_root_dir=video_interactive_segmentation_dataset_path,
        video_set_name=[
            ###########################################
            'sav_000',
        ],
        video_set_type='train',
        video_matting_root_dir=video_interactive_segmentation_dataset_path,
        video_matting_set_name_list=[
            'VideoMatte240K',
        ],
        video_matting_use_background_video_prob={
            'VideoMatte240K': 1.0,
        },
        video_matting_set_type='train',
        video_matting_background_dir=background_video_dataset_path,
        video_matting_background_set_type='train',
        per_video_choose_frame_nums=2,
        per_video_choose_object_nums=1,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            Sam2Resize(resize=1024),
            Sam2Normalize(mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAM2VideoBatchCollater(resize=1024, use_image_prob=0.0)
    train_loader = DataLoader(sam2_video_dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from SimpleAICV.video_interactive_segmentation.distill_model import DINOV3ImageEncoderDistillModel
    net = DINOV3ImageEncoderDistillModel(
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
            'image_size': 1024,
            'fpn_planes': 256,
            'use_gradient_checkpoint': False,
        },
        teacher_pretrained_path='',
        student_pretrained_path='',
        freeze_teacher=True)
    loss = MSELoss()

    for data in tqdm(train_loader):
        images = data['image']
        print('1111', images.shape)

        net = net.cuda()
        images = images.cuda()
        tea_out, stu_out = net(images)

        tea_features, tea_positions = tea_out
        for per_tea_feature, per_tea_position in zip(tea_features,
                                                     tea_positions):
            print('2222', per_tea_feature.shape, per_tea_position.shape)

        stu_features, stu_positions = stu_out
        for per_stu_feature, per_stu_position in zip(stu_features,
                                                     stu_positions):
            print('3333', per_stu_feature.shape, per_stu_position.shape)

        out = loss(tea_out, stu_out)
        print('4444', out)

        break
