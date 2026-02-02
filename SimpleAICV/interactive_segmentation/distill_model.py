import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from SimpleAICV.interactive_segmentation.models.segment_anything.image_encoder import ViTImageEncoder
from SimpleAICV.interactive_segmentation.models.dinov3_segment_anything.dinov3_image_encoder import DINOV3ViTImageEncoder
from SimpleAICV.classification.common import load_state_dict

__all__ = [
    'ImageEncoderDistillModel',
    'DINOV3ImageEncoderDistillModel',
]


class ImageEncoderDistillModel(nn.Module):

    def __init__(
            self,
            teacher_params={
                'image_size': 1024,
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
                'image_size': 1024,
                'patch_size': 16,
                'inplanes': 3,
                'embedding_planes': 768,
                'block_nums': 12,
                'head_nums': 12,
                'mlp_ratio': 4,
                'out_planes': 256,
                'window_size': 14,
                'global_attn_indexes': [2, 5, 8, 11],
                'use_gradient_checkpoint': False,
            },
            teacher_pretrained_path='',
            student_pretrained_path='',
            freeze_teacher=True):
        super(ImageEncoderDistillModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = ViTImageEncoder(**teacher_params)
        self.student = ViTImageEncoder(**student_params)

        load_state_dict(teacher_pretrained_path, self.teacher)
        load_state_dict(student_pretrained_path, self.student)

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, x):
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
                'image_size': 1024,
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
                'backbone_type': 'dinov3_vit_base_patch16_backbone',
                'image_size': 1024,
                'out_planes': 256,
                'use_gradient_checkpoint': False,
            },
            teacher_pretrained_path='',
            student_pretrained_path='',
            freeze_teacher=True):
        super(DINOV3ImageEncoderDistillModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = ViTImageEncoder(**teacher_params)
        self.student = DINOV3ViTImageEncoder(**student_params)

        load_state_dict(teacher_pretrained_path, self.teacher)
        load_state_dict(student_pretrained_path, self.student)

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, x):
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

    net = ImageEncoderDistillModel(teacher_params={
        'image_size': 1024,
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
                                       'image_size': 1024,
                                       'patch_size': 16,
                                       'inplanes': 3,
                                       'embedding_planes': 768,
                                       'block_nums': 12,
                                       'head_nums': 12,
                                       'mlp_ratio': 4,
                                       'out_planes': 256,
                                       'window_size': 14,
                                       'global_attn_indexes': [2, 5, 8, 11],
                                       'use_gradient_checkpoint': False,
                                   },
                                   teacher_pretrained_path='',
                                   student_pretrained_path='',
                                   freeze_teacher=True)
    image_h, image_w = 1024, 1024
    tea_out, stu_out = net(
        torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(
        f'1111, tea_out_shape: {tea_out.shape}, stu_out_shape: {stu_out.shape}'
    )

    net = DINOV3ImageEncoderDistillModel(
        teacher_params={
            'image_size': 1024,
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
            'backbone_type': 'dinov3_vit_base_patch16_backbone',
            'image_size': 1024,
            'out_planes': 256,
            'use_gradient_checkpoint': False,
        },
        teacher_pretrained_path='',
        student_pretrained_path='',
        freeze_teacher=True)
    image_h, image_w = 1024, 1024
    tea_out, stu_out = net(
        torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(
        f'2222, tea_out_shape: {tea_out.shape}, stu_out_shape: {stu_out.shape}'
    )
