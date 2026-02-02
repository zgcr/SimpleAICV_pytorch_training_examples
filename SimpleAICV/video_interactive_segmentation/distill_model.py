import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from SimpleAICV.video_interactive_segmentation.models.segment_anything2.image_encoder import ImageEncoder
from SimpleAICV.video_interactive_segmentation.models.dinov3_segment_anything2.dinov3_image_encoder import DINOV3ViTImageEncoder
from SimpleAICV.classification.common import load_state_dict

__all__ = [
    'ImageEncoderDistillModel',
    'DINOV3ImageEncoderDistillModel',
]


class ImageEncoderDistillModel(nn.Module):

    def __init__(
            self,
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
                'inplanes': 3,
                'embedding_planes': 112,
                'head_nums': 2,
                'block_nums': [2, 3, 16, 3],
                'window_position_embedding_bkg_spatial_size': [14, 14],
                'window_specification': [8, 4, 14, 7],
                'global_attention_blocks': [12, 16, 20],
                'fpn_planes': 256,
                'use_gradient_checkpoint': False,
            },
            teacher_pretrained_path='',
            student_pretrained_path='',
            freeze_teacher=True):
        super(ImageEncoderDistillModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = ImageEncoder(**teacher_params)
        self.student = ImageEncoder(**student_params)

        load_state_dict(teacher_pretrained_path, self.teacher)
        load_state_dict(student_pretrained_path, self.student)

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, x):
        # [T, B, 3, H, W] -> [B, T, 3, H, W] -> [B*T, 3, H, W]
        x = x.permute(1, 0, 2, 3, 4).flatten(0, 1)

        if self.freeze_teacher:
            with torch.no_grad():
                tea_out = self.teacher(x)
        else:
            tea_out = self.teacher(x)

        stu_out = self.student(x)

        return tea_out, stu_out


class DINOV3ImageEncoderDistillModel(nn.Module):

    def __init__(
            self,
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
            freeze_teacher=True):
        super(DINOV3ImageEncoderDistillModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = ImageEncoder(**teacher_params)
        self.student = DINOV3ViTImageEncoder(**student_params)

        load_state_dict(teacher_pretrained_path, self.teacher)
        load_state_dict(student_pretrained_path, self.student)

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, x):
        # [T, B, 3, H, W] -> [B, T, 3, H, W] -> [B*T, 3, H, W]
        x = x.permute(1, 0, 2, 3, 4).flatten(0, 1)

        if self.freeze_teacher:
            with torch.no_grad():
                tea_out = self.teacher(x)
        else:
            tea_out = self.teacher(x)

        stu_out = self.student(x)

        return tea_out, stu_out


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

    net = ImageEncoderDistillModel(
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
            'inplanes': 3,
            'embedding_planes': 112,
            'head_nums': 2,
            'block_nums': [2, 3, 16, 3],
            'window_position_embedding_bkg_spatial_size': [14, 14],
            'window_specification': [8, 4, 14, 7],
            'global_attention_blocks': [12, 16, 20],
            'fpn_planes': 256,
            'use_gradient_checkpoint': False,
        },
        teacher_pretrained_path='',
        student_pretrained_path='',
        freeze_teacher=True)
    image_h, image_w = 1024, 1024
    tea_out, stu_out = net(
        torch.autograd.Variable(torch.randn(1, 1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(tea_out[0], tea_out[1]):
        print(f'1111', per_out1.shape, per_out2.shape)
    for per_out1, per_out2 in zip(stu_out[0], stu_out[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

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
    image_h, image_w = 1024, 1024
    tea_out, stu_out = net(
        torch.autograd.Variable(torch.randn(1, 1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(tea_out[0], tea_out[1]):
        print(f'1111', per_out1.shape, per_out2.shape)
    for per_out1, per_out2 in zip(stu_out[0], stu_out[1]):
        print(f'2222', per_out1.shape, per_out2.shape)
