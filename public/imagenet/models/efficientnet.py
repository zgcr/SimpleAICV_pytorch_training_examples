"""
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
Unofficial implementation, not completely same as the official implementation
"""
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
    'efficientnet_b8',
    'efficientnet_l2',
]

model_urls = {
    'efficientnet_b0':
    '{}/efficientnet/efficientnet_b0-epoch100-acc75.508.pth'.format(
        pretrained_models_path),
    'efficientnet_b1':
    '{}/efficientnet/efficientnet_b1-epoch100-acc76.908.pth'.format(
        pretrained_models_path),
    'efficientnet_b2':
    '{}/efficientnet/efficientnet_b2-epoch100-acc77.776.pth'.format(
        pretrained_models_path),
    'efficientnet_b3':
    '{}/efficientnet/efficientnet_b3-epoch100-acc78.116.pth'.format(
        pretrained_models_path),
    'efficientnet_b4':
    'empty',
    'efficientnet_b5':
    'empty',
    'efficientnet_b6':
    'empty',
    'efficientnet_b7':
    'empty',
    'efficientnet_b8':
    'empty',
    'efficientnet_l2':
    'empty',
}


def round_filters(filters, efficientnet_super_parameters):
    width_coefficient = efficientnet_super_parameters.width_coefficient
    depth_divisor = efficientnet_super_parameters.depth_divisor
    filters *= width_coefficient
    min_depth = depth_divisor
    new_filters = max(
        min_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, efficientnet_super_parameters):
    depth_coefficient = efficientnet_super_parameters.depth_coefficient
    return int(math.ceil(depth_coefficient * repeats))


GlobalParams = collections.namedtuple('GlobalParams', [
    "depth_divisor",
    "width_coefficient",
    "depth_coefficient",
    "dropout",
    "image_size",
    "blocks_args",
])
GlobalParams.__new__.__defaults__ = (None, ) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size',
    'num_repeat',
    'input_filter',
    'output_filter',
    'expand_ratio',
    'stride',
    'padding',
    'se_ratio',
])
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)

origin_globalparams = GlobalParams(
    depth_divisor=8,
    width_coefficient=None,
    depth_coefficient=None,
    dropout=None,
    image_size=None,
    blocks_args=None,
)

origin_blocks_args = [
    BlockArgs(kernel_size=3,
              num_repeat=1,
              input_filter=32,
              output_filter=16,
              expand_ratio=1,
              stride=1,
              padding=1,
              se_ratio=0.25),
    BlockArgs(kernel_size=3,
              num_repeat=2,
              input_filter=16,
              output_filter=24,
              expand_ratio=6,
              stride=2,
              padding=1,
              se_ratio=0.25),
    BlockArgs(kernel_size=5,
              num_repeat=2,
              input_filter=24,
              output_filter=40,
              expand_ratio=6,
              stride=2,
              padding=2,
              se_ratio=0.25),
    BlockArgs(kernel_size=3,
              num_repeat=3,
              input_filter=40,
              output_filter=80,
              expand_ratio=6,
              stride=2,
              padding=1,
              se_ratio=0.25),
    BlockArgs(kernel_size=5,
              num_repeat=3,
              input_filter=80,
              output_filter=112,
              expand_ratio=6,
              stride=1,
              padding=2,
              se_ratio=0.25),
    BlockArgs(kernel_size=5,
              num_repeat=4,
              input_filter=112,
              output_filter=192,
              expand_ratio=6,
              stride=2,
              padding=2,
              se_ratio=0.25),
    BlockArgs(kernel_size=3,
              num_repeat=1,
              input_filter=192,
              output_filter=320,
              expand_ratio=6,
              stride=1,
              padding=1,
              se_ratio=0.25)
]

efficientnet_types_config = {
    'efficientnet_b0': {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "dropout": 0.2,
        "resolution": 224,
    },
    'efficientnet_b1': {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "dropout": 0.2,
        "resolution": 240,
    },
    'efficientnet_b2': {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "dropout": 0.3,
        "resolution": 260,
    },
    'efficientnet_b3': {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "dropout": 0.3,
        "resolution": 300,
    },
    'efficientnet_b4': {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "dropout": 0.4,
        "resolution": 380,
    },
    'efficientnet_b5': {
        "width_coefficient": 1.6,
        "depth_coefficient": 2.2,
        "dropout": 0.4,
        "resolution": 456,
    },
    'efficientnet_b6': {
        "width_coefficient": 1.8,
        "depth_coefficient": 2.6,
        "dropout": 0.5,
        "resolution": 528,
    },
    'efficientnet_b7': {
        "width_coefficient": 2.0,
        "depth_coefficient": 3.1,
        "dropout": 0.5,
        "resolution": 600,
    },
    'efficientnet_b8': {
        "width_coefficient": 2.2,
        "depth_coefficient": 3.6,
        "dropout": 0.5,
        "resolution": 672,
    },
    'efficientnet_l2': {
        "width_coefficient": 4.3,
        "depth_coefficient": 5.3,
        "dropout": 0.5,
        "resolution": 800,
    },
}


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def hard_swish(self, x, inplace):
        inner = F.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if inplace else x.mul(inner)

    def forward(self, x):
        return self.hard_swish(x, self.inplace)


class SeBlock(nn.Module):
    def __init__(self, inplanes, se_ratio=0.25):
        super(SeBlock, self).__init__()
        squeezed_planes = max(1, int(inplanes * se_ratio))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(inplanes,
                               squeezed_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeezed_planes,
                               inplanes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = inputs * x

        return x


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(planes)
        if self.has_act:
            self.act = HardSwish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.act(x)

        return x


class MBConvBlock(nn.Module):
    def __init__(self, blockArgs):
        super(MBConvBlock, self).__init__()
        self.inplanes = blockArgs.input_filter
        self.planes = blockArgs.output_filter
        self.expand_ratio = blockArgs.expand_ratio
        self.expand_planes = int(self.inplanes * self.expand_ratio)
        self.kernel_size = blockArgs.kernel_size
        self.stride = blockArgs.stride
        self.padding = blockArgs.padding
        self.se_ratio = blockArgs.se_ratio

        if self.expand_ratio != 1:
            self.expand = ConvBnActBlock(self.inplanes,
                                         self.expand_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True)
        self.depthwise_conv = ConvBnActBlock(self.expand_planes,
                                             self.expand_planes,
                                             kernel_size=self.kernel_size,
                                             stride=self.stride,
                                             padding=self.padding,
                                             groups=self.expand_planes,
                                             has_bn=True,
                                             has_act=True)
        self.se = SeBlock(self.expand_planes, se_ratio=self.se_ratio)
        self.pointwise_conv = ConvBnActBlock(self.expand_planes,
                                             self.planes,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             has_bn=True,
                                             has_act=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        if self.expand_ratio != 1:
            x = self.expand(inputs)
        else:
            x = inputs

        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)

        if self.stride == 1 and self.inplanes == self.planes:
            x += inputs

        return x


class EfficientNet(nn.Module):
    def __init__(self,
                 efficientnet_type,
                 origin_globalparams=origin_globalparams,
                 origin_blocks_args=origin_blocks_args,
                 num_classes=1000):
        super(EfficientNet, self).__init__()
        self.efficientnet_type = efficientnet_types_config[efficientnet_type]
        self.efficientnet_superparams = self.get_efficientnet_superparams(
            self.efficientnet_type,
            origin_globalparams=origin_globalparams,
            origin_blocks_args=origin_blocks_args)

        self.dropout_rate = self.efficientnet_superparams.dropout
        self.blocks_args = self.efficientnet_superparams.blocks_args
        self.stem = ConvBnActBlock(3,
                                   round_filters(
                                       32, self.efficientnet_superparams),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True)

        self.blocks = nn.ModuleList([])
        for block_args in self.blocks_args:
            for _ in range(block_args.num_repeat):
                self.blocks.append(MBConvBlock(block_args))
                if block_args.num_repeat > 0:
                    block_args = block_args._replace(
                        input_filter=block_args.output_filter, stride=1)

        self.conv_head = ConvBnActBlock(self.blocks_args[6].output_filter,
                                        round_filters(
                                            1280,
                                            self.efficientnet_superparams),
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        has_bn=True,
                                        has_act=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(round_filters(1280, self.efficientnet_superparams),
                            num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def get_efficientnet_superparams(self,
                                     efficientnet_type,
                                     origin_globalparams=origin_globalparams,
                                     origin_blocks_args=origin_blocks_args):
        origin_globalparams = origin_globalparams._replace(
            width_coefficient=efficientnet_type["width_coefficient"],
            depth_coefficient=efficientnet_type["depth_coefficient"],
            dropout=self.efficientnet_type["dropout"],
            image_size=efficientnet_type["resolution"])

        efficientnet_blocks_args = []
        for per_block_args in origin_blocks_args:
            per_block_args = per_block_args._replace(
                input_filter=round_filters(per_block_args.input_filter,
                                           origin_globalparams),
                output_filter=round_filters(per_block_args.output_filter,
                                            origin_globalparams),
                num_repeat=round_repeats(per_block_args.num_repeat,
                                         origin_globalparams))
            efficientnet_blocks_args.append(per_block_args)

        efficientnet_superparams = origin_globalparams._replace(
            blocks_args=efficientnet_blocks_args)

        return efficientnet_superparams

    def forward(self, x):
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def _efficientnet(arch, origin_globalparams, origin_blocks_args, pretrained,
                  progress, **kwargs):
    model = EfficientNet(arch,
                         origin_globalparams=origin_globalparams,
                         origin_blocks_args=origin_blocks_args,
                         **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')))

    return model


def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b0', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b1', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b2', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b3', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b4', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b5', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b6', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b7', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_b8(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_b8', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)


def efficientnet_l2(pretrained=False, progress=True, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet('efficientnet_l2', origin_globalparams,
                         origin_blocks_args, pretrained, progress, **kwargs)