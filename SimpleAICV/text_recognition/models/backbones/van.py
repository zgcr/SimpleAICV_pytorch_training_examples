import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import math
import numpy as np

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from SimpleAICV.classification.backbones.van import Block
from SimpleAICV.detection.common import load_state_dict

__all__ = [
    'vanb0backbone',
    'vanb1backbone',
    'vanb2backbone',
    'vanb3backbone',
    'vanb4backbone',
    'vanb5backbone',
    'vanb6backbone',
]


class OverlapPatchEmbed(nn.Module):

    def __init__(self, inplanes, embedding_planes, patch_size, stride,
                 padding):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(inplanes,
                              embedding_planes,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=padding)
        self.norm = nn.BatchNorm2d(embedding_planes)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        return x


class VANBackbone(nn.Module):

    def __init__(self,
                 inplanes=3,
                 embedding_planes=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 block_nums=[3, 4, 6, 3],
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 use_gradient_checkpoint=False):
        super(VANBackbone, self).__init__()
        assert len(embedding_planes) == len(mlp_ratios) == len(block_nums)

        self.block_nums = block_nums
        self.use_gradient_checkpoint = use_gradient_checkpoint

        drop_path_prob_list = [
            x for x in np.linspace(0, drop_path_prob, sum(block_nums))
        ]

        currnet_stage_idx = 0
        currnet_inplanes = inplanes
        for i in range(len(block_nums)):
            if i == 0:
                patch_embed = OverlapPatchEmbed(
                    inplanes=currnet_inplanes,
                    embedding_planes=embedding_planes[i],
                    patch_size=7,
                    stride=4,
                    padding=3)
            elif i == 1:
                patch_embed = OverlapPatchEmbed(
                    inplanes=currnet_inplanes,
                    embedding_planes=embedding_planes[i],
                    patch_size=3,
                    stride=2,
                    padding=1)
            else:
                patch_embed = OverlapPatchEmbed(
                    inplanes=currnet_inplanes,
                    embedding_planes=embedding_planes[i],
                    patch_size=(3, 1),
                    stride=(2, 1),
                    padding=(1, 0))
            currnet_inplanes = embedding_planes[i]

            block = nn.ModuleList([
                Block(inplanes=embedding_planes[i],
                      mlp_ratio=mlp_ratios[i],
                      dropout_prob=dropout_prob,
                      drop_path_prob=drop_path_prob_list[currnet_stage_idx +
                                                         j])
                for j in range(block_nums[i])
            ])
            norm = nn.BatchNorm2d(embedding_planes[i])
            currnet_stage_idx += block_nums[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.out_channels = [
            embedding_planes[0],
            embedding_planes[1],
            embedding_planes[2],
            embedding_planes[3],
        ]

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        outs = []
        for i in range(len(self.block_nums)):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            if self.use_gradient_checkpoint:
                x = checkpoint(patch_embed, x, use_reentrant=False)
            else:
                x = patch_embed(x)

            for blk in block:
                if self.use_gradient_checkpoint:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)

            if self.use_gradient_checkpoint:
                x = checkpoint(norm, x, use_reentrant=False)
            else:
                x = norm(x)

            outs.append(x)

        return outs


def _vanbackbone(embedding_planes,
                 mlp_ratios,
                 block_nums,
                 pretrained_path='',
                 **kwargs):
    model = VANBackbone(embedding_planes=embedding_planes,
                        mlp_ratios=mlp_ratios,
                        block_nums=block_nums,
                        **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def vanb0backbone(pretrained_path='', **kwargs):
    return _vanbackbone(embedding_planes=[32, 64, 160, 256],
                        mlp_ratios=[8, 8, 4, 4],
                        block_nums=[3, 3, 5, 2],
                        pretrained_path=pretrained_path,
                        **kwargs)


def vanb1backbone(pretrained_path='', **kwargs):
    return _vanbackbone(embedding_planes=[64, 128, 320, 512],
                        mlp_ratios=[8, 8, 4, 4],
                        block_nums=[2, 2, 4, 2],
                        pretrained_path=pretrained_path,
                        **kwargs)


def vanb2backbone(pretrained_path='', **kwargs):
    return _vanbackbone(embedding_planes=[64, 128, 320, 512],
                        mlp_ratios=[8, 8, 4, 4],
                        block_nums=[3, 3, 12, 3],
                        pretrained_path=pretrained_path,
                        **kwargs)


def vanb3backbone(pretrained_path='', **kwargs):
    return _vanbackbone(embedding_planes=[64, 128, 320, 512],
                        mlp_ratios=[8, 8, 4, 4],
                        block_nums=[3, 5, 27, 3],
                        pretrained_path=pretrained_path,
                        **kwargs)


def vanb4backbone(pretrained_path='', **kwargs):
    return _vanbackbone(embedding_planes=[64, 128, 320, 512],
                        mlp_ratios=[8, 8, 4, 4],
                        block_nums=[3, 6, 40, 3],
                        pretrained_path=pretrained_path,
                        **kwargs)


def vanb5backbone(pretrained_path='', **kwargs):
    return _vanbackbone(embedding_planes=[96, 192, 480, 768],
                        mlp_ratios=[8, 8, 4, 4],
                        block_nums=[3, 3, 24, 3],
                        pretrained_path=pretrained_path,
                        **kwargs)


def vanb6backbone(pretrained_path='', **kwargs):
    return _vanbackbone(embedding_planes=[96, 192, 384, 768],
                        mlp_ratios=[8, 8, 4, 4],
                        block_nums=[6, 6, 90, 6],
                        pretrained_path=pretrained_path,
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

    net = vanb0backbone()
    image_h, image_w = 32, 512
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

    net = vanb1backbone()
    image_h, image_w = 32, 512
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

    net = vanb2backbone()
    image_h, image_w = 32, 512
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

    net = vanb3backbone()
    image_h, image_w = 32, 512
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

    net = vanb4backbone()
    image_h, image_w = 32, 512
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

    net = vanb5backbone()
    image_h, image_w = 32, 512
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

    net = vanb6backbone()
    image_h, image_w = 32, 512
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

    net = vanb6backbone(use_gradient_checkpoint=True)
    image_h, image_w = 32, 512
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)
