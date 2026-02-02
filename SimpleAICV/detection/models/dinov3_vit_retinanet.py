import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from SimpleAICV.detection.models import backbones
from SimpleAICV.detection.models.fpn import RetinaFPN
from SimpleAICV.detection.models.head import RetinaClsHead, RetinaRegHead
from SimpleAICV.detection.models.backbones.dinov3vit import VitPyramidNeck

__all__ = [
    'dinov3_vit_small_patch16_retinanet',
    'dinov3_vit_small_plus_patch16_retinanet',
    'dinov3_vit_base_patch16_retinanet',
    'dinov3_vit_large_patch16_retinanet',
    'dinov3_vit_large_plus_patch16_retinanet',
    'dinov3_vit_huge_plus_patch16_retinanet',
]


class RetinaNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 num_anchors=9,
                 num_classes=80,
                 use_gradient_checkpoint=False):
        super(RetinaNet, self).__init__()
        self.planes = planes
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.neck = VitPyramidNeck(inplanes=self.backbone.out_channels,
                                   planes=self.planes)
        self.fpn = RetinaFPN([
            self.planes,
            self.planes,
            self.planes,
        ],
                             self.planes,
                             use_p5=False)
        self.cls_head = RetinaClsHead(self.planes,
                                      self.num_anchors,
                                      self.num_classes,
                                      num_layers=4)
        self.reg_head = RetinaRegHead(self.planes,
                                      self.num_anchors,
                                      num_layers=4)

    def forward(self, inputs):
        features = self.backbone(inputs)
        features = self.neck(features)
        features = features[1:4]

        del inputs

        if self.use_gradient_checkpoint:
            features = checkpoint(self.fpn, features, use_reentrant=False)
        else:
            features = self.fpn(features)

        cls_heads, reg_heads = [], []
        for feature in features:
            cls_head = self.cls_head(feature)
            # [N,9*num_classes,H,W] -> [N,H,W,9*num_classes] -> [N,H,W,9,num_classes]
            cls_head = cls_head.permute(0, 2, 3, 1).contiguous()
            cls_head = cls_head.view(cls_head.shape[0], cls_head.shape[1],
                                     cls_head.shape[2], -1, self.num_classes)
            cls_heads.append(cls_head)

            reg_head = self.reg_head(feature)
            # [N, 9*4,H,W] -> [N,H,W,9*4] -> [N,H,W,9,4]
            reg_head = reg_head.permute(0, 2, 3, 1).contiguous()
            reg_head = reg_head.view(reg_head.shape[0], reg_head.shape[1],
                                     reg_head.shape[2], -1, 4)
            reg_heads.append(reg_head)

        del features

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
        # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
        return [cls_heads, reg_heads]


def _retinanet(backbone_type, backbone_pretrained_path, **kwargs):
    model = RetinaNet(backbone_type,
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)

    return model


def dinov3_vit_small_patch16_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('dinov3_vit_small_patch16_backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def dinov3_vit_small_plus_patch16_retinanet(backbone_pretrained_path='',
                                            **kwargs):
    return _retinanet('dinov3_vit_small_plus_patch16_backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def dinov3_vit_base_patch16_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('dinov3_vit_base_patch16_backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def dinov3_vit_large_patch16_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('dinov3_vit_large_patch16_backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def dinov3_vit_large_plus_patch16_retinanet(backbone_pretrained_path='',
                                            **kwargs):
    return _retinanet('dinov3_vit_large_plus_patch16_backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def dinov3_vit_huge_plus_patch16_retinanet(backbone_pretrained_path='',
                                           **kwargs):
    return _retinanet('dinov3_vit_huge_plus_patch16_backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


if __name__ == '__main__':
    import os
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

    net = dinov3_vit_small_patch16_retinanet()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = dinov3_vit_small_plus_patch16_retinanet()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = dinov3_vit_base_patch16_retinanet()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = dinov3_vit_large_patch16_retinanet()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = dinov3_vit_large_plus_patch16_retinanet()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = dinov3_vit_large_plus_patch16_retinanet(use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = dinov3_vit_huge_plus_patch16_retinanet()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)
