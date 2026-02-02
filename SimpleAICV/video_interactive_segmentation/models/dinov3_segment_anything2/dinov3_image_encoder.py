import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from SimpleAICV.detection.models import backbones
from SimpleAICV.detection.models.backbones.dinov3vit import VitPyramidNeck
from SimpleAICV.video_interactive_segmentation.models.segment_anything2.image_encoder import FpnNeck


class DINOV3ViTImageEncoder(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 image_size=1024,
                 fpn_planes=256,
                 use_gradient_checkpoint=False):
        super(DINOV3ViTImageEncoder, self).__init__()
        self.image_size = image_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.trunk = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.fpn = VitPyramidNeck(inplanes=self.trunk.out_channels,
                                  planes=self.trunk.out_channels)
        self.neck = FpnNeck(inplanes_list=[
            self.trunk.out_channels,
            self.trunk.out_channels,
            self.trunk.out_channels,
            self.trunk.out_channels,
        ],
                            planes=fpn_planes)

    def forward(self, inputs):
        features = self.trunk(inputs)

        if self.use_gradient_checkpoint:
            features = checkpoint(self.fpn, features, use_reentrant=False)
        else:
            features = self.fpn(features)

        if self.use_gradient_checkpoint:
            features, positions = checkpoint(self.neck,
                                             features,
                                             use_reentrant=False)
        else:
            features, positions = self.neck(features)

        features, positions = features[:-1], positions[:-1]

        return features, positions


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

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_small_patch16_backbone',
        fpn_planes=256,
        use_gradient_checkpoint=False)
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
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_small_patch16_backbone',
        fpn_planes=256,
        use_gradient_checkpoint=True)
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_small_plus_patch16_backbone',
        fpn_planes=256,
        use_gradient_checkpoint=False)
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
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_base_patch16_backbone',
        fpn_planes=256,
        use_gradient_checkpoint=False)
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
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_large_patch16_backbone',
        fpn_planes=256,
        use_gradient_checkpoint=False)
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
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_large_plus_patch16_backbone',
        fpn_planes=256,
        use_gradient_checkpoint=False)
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
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_huge_plus_patch16_backbone',
        fpn_planes=256,
        use_gradient_checkpoint=False)
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
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)
