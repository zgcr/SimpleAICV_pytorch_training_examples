import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from SimpleAICV.detection.common import load_state_dict

__all__ = [
    'dinov3convnexttinybackbone',
    'dinov3convnextsmallbackbone',
    'dinov3convnextbasebackbone',
    'dinov3convnextlargebackbone',
]


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


class DropPathBlock(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    if drop_path_prob = 0. ,not use DropPath
    """

    def __init__(self, drop_path_prob=0., scale_by_keep=True):
        super(DropPathBlock, self).__init__()
        assert drop_path_prob >= 0.

        self.drop_path_prob = drop_path_prob
        self.keep_path_prob = 1 - drop_path_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_path_prob == 0. or not self.training:
            return x

        b = x.shape[0]
        device = x.device

        # work with diff dim tensors, not just 2D ConvNets
        shape = (b, ) + (1, ) * (len(x.shape) - 1)
        random_weight = torch.empty(shape).to(device).bernoulli_(
            self.keep_path_prob)

        if self.keep_path_prob > 0. and self.scale_by_keep:
            random_weight.div_(self.keep_path_prob)

        x = random_weight * x

        return x


class Block(nn.Module):

    def __init__(self, inplanes, drop_path_prob=0.):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(inplanes,
                                inplanes,
                                kernel_size=7,
                                padding=3,
                                groups=inplanes)
        self.norm = nn.LayerNorm(inplanes, eps=1e-6)
        self.pwconv1 = nn.Linear(inplanes, 4 * inplanes)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * inplanes, inplanes)
        self.gamma = nn.Parameter(1e-6 * torch.ones((inplanes)),
                                  requires_grad=True)

        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        input = x

        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)

        return x


class Dinov3ConvNeXtBackbone(nn.Module):

    def __init__(self,
                 inplanes=3,
                 embedding_planes=[96, 192, 384, 768],
                 block_nums=[3, 3, 9, 3],
                 drop_path_prob=0.,
                 use_gradient_checkpoint=False):
        super(Dinov3ConvNeXtBackbone, self).__init__()
        assert len(embedding_planes) == len(block_nums)

        self.block_nums = block_nums
        self.use_gradient_checkpoint = use_gradient_checkpoint

        downsample_layers = []
        stem = nn.Sequential(
            nn.Conv2d(inplanes, embedding_planes[0], kernel_size=4, stride=4),
            LayerNorm2d(embedding_planes[0], eps=1e-6))
        downsample_layers.append(stem)

        for i in range(len(block_nums) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm2d(embedding_planes[i], eps=1e-6),
                nn.Conv2d(embedding_planes[i],
                          embedding_planes[i + 1],
                          kernel_size=2,
                          stride=2))
            downsample_layers.append(downsample_layer)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        drop_path_prob_list = [
            x for x in np.linspace(0, drop_path_prob, sum(block_nums))
        ]

        stages = []
        currnet_stage_idx = 0
        for i in range(len(block_nums)):
            stage = nn.Sequential(*[
                Block(inplanes=embedding_planes[i],
                      drop_path_prob=drop_path_prob_list[currnet_stage_idx +
                                                         j])
                for j in range(block_nums[i])
            ])
            stages.append(stage)
            currnet_stage_idx += block_nums[i]
        self.stages = nn.ModuleList(stages)

        self.out_channels = [
            embedding_planes[0],
            embedding_planes[1],
            embedding_planes[2],
            embedding_planes[3],
        ]

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(len(self.block_nums)):
            if self.use_gradient_checkpoint:
                x = checkpoint(self.downsample_layers[i],
                               x,
                               use_reentrant=False)
                x = checkpoint(self.stages[i], x, use_reentrant=False)
            else:
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)

            outs.append(x)

        return outs


def _dinov3convnextbackbone(block_nums,
                            embedding_planes,
                            pretrained_path='',
                            **kwargs):
    model = Dinov3ConvNeXtBackbone(block_nums=block_nums,
                                   embedding_planes=embedding_planes,
                                   **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def dinov3convnexttinybackbone(pretrained_path='', **kwargs):
    model = _dinov3convnextbackbone(block_nums=[3, 3, 9, 3],
                                    embedding_planes=[96, 192, 384, 768],
                                    pretrained_path=pretrained_path,
                                    **kwargs)

    return model


def dinov3convnextsmallbackbone(pretrained_path='', **kwargs):
    model = _dinov3convnextbackbone(block_nums=[3, 3, 27, 3],
                                    embedding_planes=[96, 192, 384, 768],
                                    pretrained_path=pretrained_path,
                                    **kwargs)

    return model


def dinov3convnextbasebackbone(pretrained_path='', **kwargs):
    model = _dinov3convnextbackbone(block_nums=[3, 3, 27, 3],
                                    embedding_planes=[128, 256, 512, 1024],
                                    pretrained_path=pretrained_path,
                                    **kwargs)

    return model


def dinov3convnextlargebackbone(pretrained_path='', **kwargs):
    model = _dinov3convnextbackbone(block_nums=[3, 3, 27, 3],
                                    embedding_planes=[192, 384, 768, 1536],
                                    pretrained_path=pretrained_path,
                                    **kwargs)

    return model


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

    net = dinov3convnexttinybackbone()
    image_h, image_w = 640, 640
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = dinov3convnextsmallbackbone()
    image_h, image_w = 640, 640
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = dinov3convnextbasebackbone()
    image_h, image_w = 640, 640
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = dinov3convnextlargebackbone()
    image_h, image_w = 640, 640
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = dinov3convnextlargebackbone(use_gradient_checkpoint=True)
    image_h, image_w = 640, 640
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)
