import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.interactive_segmentation.models.segment_anything.image_encoder import ViTImageEncoder
from simpleAICV.interactive_segmentation.models.light_segment_anything.light_image_encoder import LightImageEncoder
from simpleAICV.interactive_segmentation.models.segment_anything import sam
from simpleAICV.interactive_segmentation.models.light_segment_anything import light_sam
from simpleAICV.classification.common import load_state_dict

__all__ = [
    'SAMImageEncoderDistillModel',
    'SAMLightImageEncoderDistillModel',
    'SAMDistillModel',
    'SAMLightDistillModel',
]


class SAMImageEncoderDistillModel(nn.Module):

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
        super(SAMImageEncoderDistillModel, self).__init__()
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


class SAMLightImageEncoderDistillModel(nn.Module):

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
                'backbone_type': 'convformerm36backbone',
                'planes': 256,
                'use_gradient_checkpoint': False,
            },
            teacher_pretrained_path='',
            student_pretrained_path='',
            freeze_teacher=True):
        super(SAMLightImageEncoderDistillModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = ViTImageEncoder(**teacher_params)
        self.student = LightImageEncoder(**student_params)

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


class SAMDistillModel(nn.Module):

    def __init__(
            self,
            teacher_type='sam_h',
            student_type='sam_b',
            teacher_params={
                'image_size': 1024,
                'use_gradient_checkpoint': False,
                'frozen_image_encoder': True,
                'frozen_prompt_encoder': True,
                'frozen_mask_decoder': True,
                'sigmoid_out': False,
                'binary_mask_out': False,
                'mask_threshold': 0.0,
            },
            student_params={
                'image_size': 1024,
                'use_gradient_checkpoint': False,
                'frozen_image_encoder': False,
                'frozen_prompt_encoder': False,
                'frozen_mask_decoder': False,
                'sigmoid_out': False,
                'binary_mask_out': False,
                'mask_threshold': 0.0,
            },
            teacher_pretrained_path='',
            student_pretrained_path='',
            freeze_teacher=True):
        super(SAMDistillModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = sam.__dict__[teacher_type](**teacher_params)
        self.student = sam.__dict__[student_type](**student_params)

        load_state_dict(teacher_pretrained_path, self.teacher)
        load_state_dict(student_pretrained_path, self.student)

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, batch_images, batch_prompts, mask_out_idxs=[0, 1, 2, 3]):
        if self.freeze_teacher:
            with torch.no_grad():
                tea_out = self.teacher(batch_images,
                                       batch_prompts,
                                       mask_out_idxs=mask_out_idxs)
        else:
            tea_out = self.teacher(batch_images,
                                   batch_prompts,
                                   mask_out_idxs=mask_out_idxs)

        stu_out = self.student(batch_images,
                               batch_prompts,
                               mask_out_idxs=mask_out_idxs)

        return tea_out, stu_out


class SAMLightDistillModel(nn.Module):

    def __init__(
            self,
            teacher_type='sam_h',
            student_type='convformerm36_light_sam',
            teacher_params={
                'image_size': 1024,
                'use_gradient_checkpoint': False,
                'frozen_image_encoder': True,
                'frozen_prompt_encoder': True,
                'frozen_mask_decoder': True,
                'sigmoid_out': False,
                'binary_mask_out': False,
                'mask_threshold': 0.0,
            },
            student_params={
                'image_size': 1024,
                'use_gradient_checkpoint': False,
                'frozen_image_encoder': False,
                'frozen_prompt_encoder': False,
                'frozen_mask_decoder': False,
                'sigmoid_out': False,
                'binary_mask_out': False,
                'mask_threshold': 0.0,
            },
            teacher_pretrained_path='',
            student_pretrained_path='',
            freeze_teacher=True):
        super(SAMLightDistillModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = sam.__dict__[teacher_type](**teacher_params)
        self.student = light_sam.__dict__[student_type](**student_params)

        load_state_dict(teacher_pretrained_path, self.teacher)
        load_state_dict(student_pretrained_path, self.student)

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, batch_images, batch_prompts, mask_out_idxs=[0, 1, 2, 3]):
        if self.freeze_teacher:
            with torch.no_grad():
                tea_out = self.teacher(batch_images,
                                       batch_prompts,
                                       mask_out_idxs=mask_out_idxs)
        else:
            tea_out = self.teacher(batch_images,
                                   batch_prompts,
                                   mask_out_idxs=mask_out_idxs)

        stu_out = self.student(batch_images,
                               batch_prompts,
                               mask_out_idxs=mask_out_idxs)

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

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from tools.path import interactive_segmentation_dataset_path

    from simpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

    # net = SAMImageEncoderDistillModel(teacher_params={
    #     'image_size': 1024,
    #     'patch_size': 16,
    #     'inplanes': 3,
    #     'embedding_planes': 1280,
    #     'block_nums': 32,
    #     'head_nums': 16,
    #     'mlp_ratio': 4,
    #     'out_planes': 256,
    #     'window_size': 14,
    #     'global_attn_indexes': [7, 15, 23, 31],
    #     'use_gradient_checkpoint': False,
    # },
    #                                   student_params={
    #                                       'image_size': 1024,
    #                                       'patch_size': 16,
    #                                       'inplanes': 3,
    #                                       'embedding_planes': 768,
    #                                       'block_nums': 12,
    #                                       'head_nums': 12,
    #                                       'mlp_ratio': 4,
    #                                       'out_planes': 256,
    #                                       'window_size': 14,
    #                                       'global_attn_indexes': [2, 5, 8, 11],
    #                                       'use_gradient_checkpoint': False,
    #                                   },
    #                                   teacher_pretrained_path='',
    #                                   student_pretrained_path='',
    #                                   freeze_teacher=True)
    # image_h, image_w = 1024, 1024
    # tea_out, stu_out = net(
    #     torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    # print(
    #     f'1111, tea_out_shape: {tea_out.shape}, stu_out_shape: {stu_out.shape}'
    # )

    # net = SAMLightImageEncoderDistillModel(teacher_params={
    #     'image_size':
    #     1024,
    #     'patch_size':
    #     16,
    #     'inplanes':
    #     3,
    #     'embedding_planes':
    #     1280,
    #     'block_nums':
    #     32,
    #     'head_nums':
    #     16,
    #     'mlp_ratio':
    #     4,
    #     'out_planes':
    #     256,
    #     'window_size':
    #     14,
    #     'global_attn_indexes': [7, 15, 23, 31],
    #     'use_gradient_checkpoint':
    #     False,
    # },
    #                                        student_params={
    #                                            'backbone_type':
    #                                            'convformerm36backbone',
    #                                            'planes': 256,
    #                                            'use_gradient_checkpoint':
    #                                            False,
    #                                        },
    #                                        teacher_pretrained_path='',
    #                                        student_pretrained_path='',
    #                                        freeze_teacher=True)
    # image_h, image_w = 1024, 1024
    # tea_out, stu_out = net(
    #     torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    # print(
    #     f'2222, tea_out_shape: {tea_out.shape}, stu_out_shape: {stu_out.shape}'
    # )

    samdataset = SAMSegmentationDataset(interactive_segmentation_dataset_path,
                                        set_name=[
                                            'sa_000020',
                                        ],
                                        set_type='train',
                                        per_set_image_choose_max_num={
                                            'sa_000020': 1000000,
                                        },
                                        per_image_mask_chosse_max_num=16,
                                        positive_points_num=9,
                                        negative_points_num=9,
                                        area_filter_ratio=0.0001,
                                        box_noise_wh_ratio=0.1,
                                        mask_noise_area_ratio=0.04,
                                        transform=transforms.Compose([
                                            SamResize(resize=1024),
                                            SamRandomHorizontalFlip(prob=0.5),
                                            SamNormalize(
                                                mean=[123.675, 116.28, 103.53],
                                                std=[58.395, 57.12, 57.375]),
                                        ]))

    from torch.utils.data import DataLoader

    collater = SAMBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    # net = SAMDistillModel(teacher_type='sam_h',
    #                       student_type='sam_b',
    #                       teacher_params={
    #                           'image_size': 1024,
    #                           'use_gradient_checkpoint': False,
    #                           'frozen_image_encoder': True,
    #                           'frozen_prompt_encoder': True,
    #                           'frozen_mask_decoder': True,
    #                           'sigmoid_out': False,
    #                           'binary_mask_out': False,
    #                           'mask_threshold': 0.0,
    #                       },
    #                       student_params={
    #                           'image_size': 1024,
    #                           'use_gradient_checkpoint': False,
    #                           'frozen_image_encoder': False,
    #                           'frozen_prompt_encoder': False,
    #                           'frozen_mask_decoder': False,
    #                           'sigmoid_out': False,
    #                           'binary_mask_out': False,
    #                           'mask_threshold': 0.0,
    #                       },
    #                       teacher_pretrained_path='',
    #                       student_pretrained_path='',
    #                       freeze_teacher=True)

    # for data in tqdm(train_loader):
    #     input_images, input_boxs, input_masks, sizes = data['image'], data[
    #         'box'], data['mask'], data['size']

    #     input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
    #         'prompt_point'], data['prompt_box'], data['prompt_mask']

    #     net = net.cuda()
    #     input_images = input_images.cuda()
    #     input_masks = input_masks.cuda()
    #     print('1111', input_images.shape, input_masks.shape)

    #     input_prompt_points = input_prompt_points.cuda()
    #     input_prompt_boxs = input_prompt_boxs.cuda()
    #     input_prompt_masks = input_prompt_masks.cuda()

    #     print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
    #           input_prompt_masks.shape)

    #     input_prompts = {
    #         'prompt_point': input_prompt_points,
    #         'prompt_box': input_prompt_boxs,
    #         'prompt_mask': input_prompt_masks,
    #     }

    #     tea_preds, stu_preds = net(input_images,
    #                                input_prompts,
    #                                mask_out_idxs=[0, 1, 2, 3])

    #     print('3333', tea_preds[0].shape, tea_preds[1].shape,
    #           tea_preds[0].dtype, tea_preds[1].dtype)

    #     print('4444', stu_preds[0].shape, stu_preds[1].shape,
    #           stu_preds[0].dtype, stu_preds[1].dtype)

    #     break

    net = SAMLightDistillModel(teacher_type='sam_h',
                               student_type='convformerm36_light_sam',
                               teacher_params={
                                   'image_size': 1024,
                                   'use_gradient_checkpoint': False,
                                   'frozen_image_encoder': True,
                                   'frozen_prompt_encoder': True,
                                   'frozen_mask_decoder': True,
                                   'sigmoid_out': False,
                                   'binary_mask_out': False,
                                   'mask_threshold': 0.0,
                               },
                               student_params={
                                   'image_size': 1024,
                                   'use_gradient_checkpoint': False,
                                   'frozen_image_encoder': False,
                                   'frozen_prompt_encoder': False,
                                   'frozen_mask_decoder': False,
                                   'sigmoid_out': False,
                                   'binary_mask_out': False,
                                   'mask_threshold': 0.0,
                               },
                               teacher_pretrained_path='',
                               student_pretrained_path='',
                               freeze_teacher=True)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        tea_preds, stu_preds = net(input_images,
                                   input_prompts,
                                   mask_out_idxs=[0, 1, 2, 3])

        print('3333', tea_preds[0].shape, tea_preds[1].shape,
              tea_preds[0].dtype, tea_preds[1].dtype)

        print('4444', stu_preds[0].shape, stu_preds[1].shape,
              stu_preds[0].dtype, stu_preds[1].dtype)

        break
