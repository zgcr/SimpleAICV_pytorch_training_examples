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


class LayerNorm2d(nn.Module):

    def __init__(self, inplanes, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(inplanes))
        self.bias = nn.Parameter(torch.zeros(inplanes))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        return x


class DINOV3ViTImageEncoder(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 image_size=1024,
                 out_planes=256,
                 use_gradient_checkpoint=False):
        super(DINOV3ViTImageEncoder, self).__init__()
        self.image_size = image_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.neck = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels,
                      out_planes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), LayerNorm2d(out_planes),
            nn.Conv2d(out_planes,
                      out_planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), LayerNorm2d(out_planes))

    def forward(self, x):
        x = self.backbone(x)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.neck, x, use_reentrant=False)
        else:
            x = self.neck(x)

        return x


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
        out_planes=256,
        use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_small_patch16_backbone',
        out_planes=256,
        use_gradient_checkpoint=True)
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_small_plus_patch16_backbone',
        out_planes=256,
        use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_base_patch16_backbone',
        out_planes=256,
        use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_large_patch16_backbone',
        out_planes=256,
        use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_large_plus_patch16_backbone',
        out_planes=256,
        use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')

    net = DINOV3ViTImageEncoder(
        backbone_type='dinov3_vit_huge_plus_patch16_backbone',
        out_planes=256,
        use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')
