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

from SimpleAICV.classification.backbones.convformer import MetaFormerBlock
from SimpleAICV.detection.common import load_state_dict

__all__ = [
    'convformers18backbone',
    'convformers36backbone',
    'convformerm36backbone',
    'convformerb36backbone',
]


class Downsampling(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 pre_norm=False,
                 post_norm=False):
        super(Downsampling, self).__init__()
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True)

        self.pre_norm = nn.BatchNorm2d(inplanes) if pre_norm else nn.Identity()
        self.post_norm = nn.BatchNorm2d(planes) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)

        x = self.conv(x)

        x = self.post_norm(x)

        return x


class MetaFormerBackbone(nn.Module):

    def __init__(self,
                 inplanes=3,
                 embedding_planes=[64, 128, 320, 512],
                 block_nums=[2, 2, 6, 2],
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 use_gradient_checkpoint=False):
        super(MetaFormerBackbone, self).__init__()
        assert len(embedding_planes) == len(block_nums)

        self.block_nums = block_nums
        self.use_gradient_checkpoint = use_gradient_checkpoint

        downsample_layers = []
        down_embedding_planes = [inplanes] + embedding_planes
        for i in range(len(block_nums)):
            if i == 0:
                per_downsample_layer = Downsampling(down_embedding_planes[i],
                                                    down_embedding_planes[i +
                                                                          1],
                                                    kernel_size=7,
                                                    stride=4,
                                                    padding=2,
                                                    pre_norm=False,
                                                    post_norm=True)
            elif i == 1:
                per_downsample_layer = Downsampling(down_embedding_planes[i],
                                                    down_embedding_planes[i +
                                                                          1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    pre_norm=True,
                                                    post_norm=False)
            else:
                per_downsample_layer = Downsampling(down_embedding_planes[i],
                                                    down_embedding_planes[i +
                                                                          1],
                                                    kernel_size=(3, 1),
                                                    stride=(2, 1),
                                                    padding=(1, 0),
                                                    pre_norm=True,
                                                    post_norm=False)
            downsample_layers.append(per_downsample_layer)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        drop_path_prob_list = [
            x for x in np.linspace(0, drop_path_prob, sum(block_nums))
        ]

        stages = []
        currnet_stage_idx = 0
        for i in range(len(block_nums)):
            stage = nn.Sequential(*[
                MetaFormerBlock(inplanes=embedding_planes[i],
                                dropout_prob=dropout_prob,
                                drop_path_prob=drop_path_prob_list[
                                    currnet_stage_idx + j])
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
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
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


def _metaformerbackbone(block_nums,
                        embedding_planes,
                        pretrained_path='',
                        **kwargs):
    model = MetaFormerBackbone(block_nums=block_nums,
                               embedding_planes=embedding_planes,
                               **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def convformers18backbone(pretrained_path='', **kwargs):
    model = _metaformerbackbone(block_nums=[3, 3, 9, 3],
                                embedding_planes=[64, 128, 320, 512],
                                pretrained_path=pretrained_path,
                                **kwargs)

    return model


def convformers36backbone(pretrained_path='', **kwargs):
    model = _metaformerbackbone(block_nums=[3, 12, 18, 3],
                                embedding_planes=[64, 128, 320, 512],
                                pretrained_path=pretrained_path,
                                **kwargs)

    return model


def convformerm36backbone(pretrained_path='', **kwargs):
    model = _metaformerbackbone(block_nums=[3, 12, 18, 3],
                                embedding_planes=[96, 192, 384, 576],
                                pretrained_path=pretrained_path,
                                **kwargs)

    return model


def convformerb36backbone(pretrained_path='', **kwargs):
    model = _metaformerbackbone(block_nums=[3, 12, 18, 3],
                                embedding_planes=[128, 256, 512, 768],
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

    net = convformers18backbone()
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

    net = convformers36backbone()
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

    net = convformerm36backbone()
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

    net = convformerb36backbone()
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

    net = convformerb36backbone(use_gradient_checkpoint=True)
    image_h, image_w = 32, 512
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)
