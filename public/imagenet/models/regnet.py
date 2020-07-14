"""
Designing Network Design Spaces
https://arxiv.org/pdf/2003.13678.pdf
"""
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

import numpy as np

import torch
import torch.nn as nn

__all__ = [
    'RegNetY_200MF',
    'RegNetY_400MF',
    'RegNetY_600MF',
    'RegNetY_800MF',
    'RegNetY_1_6GF',
    'RegNetY_3_2GF',
    'RegNetY_4_0GF',
    'RegNetY_6_4GF',
    'RegNetY_8_0GF',
    'RegNetY_12GF',
    'RegNetY_16GF',
    'RegNetY_32GF',
]

model_urls = {
    'RegNetY_200MF':
    '{}/regnet/RegNetY_200MF-epoch100-acc70.096.pth'.format(
        pretrained_models_path),
    'RegNetY_400MF':
    '{}/regnet/RegNetY_400MF-epoch100-acc73.79.pth'.format(
        pretrained_models_path),
    'RegNetY_600MF':
    '{}/regnet/RegNetY_600MF-epoch100-acc74.724.pth'.format(
        pretrained_models_path),
    'RegNetY_800MF':
    '{}/regnet/RegNetY_800MF-epoch100-acc75.994.pth'.format(
        pretrained_models_path),
    'RegNetY_1_6GF':
    '{}/regnet/RegNetY_1_6GF-epoch100-acc77.308.pth'.format(
        pretrained_models_path),
    'RegNetY_3_2GF':
    '{}/regnet/RegNetY_3_2GF-epoch100-acc78.908.pth'.format(
        pretrained_models_path),
    'RegNetY_4_0GF':
    '{}/regnet/RegNetY_4_0GF-epoch100-acc78.136.pth'.format(
        pretrained_models_path),
    'RegNetY_6_4GF':
    '{}/regnet/RegNetY_6_4GF-epoch100-acc78.77.pth'.format(
        pretrained_models_path),
    'RegNetY_8_0GF':
    'empty',
    'RegNetY_12GF':
    'empty',
    'RegNetY_16GF':
    'empty',
    'RegNetY_32GF':
    'empty',
}

RegnetY_types_config = {
    'RegNetY_200MF': {
        "stem_width": 32,
        "w_a": 36.44,
        "w_0": 24,
        "w_m": 2.49,
        "groups": 8,
        "depth": 13,
        "has_se": True,
    },
    'RegNetY_400MF': {
        "stem_width": 32,
        "w_a": 27.89,
        "w_0": 48,
        "w_m": 2.09,
        "groups": 8,
        "depth": 16,
        "has_se": True,
    },
    'RegNetY_600MF': {
        "stem_width": 32,
        "w_a": 32.54,
        "w_0": 48,
        "w_m": 2.32,
        "groups": 16,
        "depth": 15,
        "has_se": True,
    },
    'RegNetY_800MF': {
        "stem_width": 32,
        "w_a": 38.84,
        "w_0": 56,
        "w_m": 2.4,
        "groups": 16,
        "depth": 14,
        "has_se": True,
    },
    'RegNetY_1_6GF': {
        "stem_width": 32,
        "w_a": 20.71,
        "w_0": 48,
        "w_m": 2.65,
        "groups": 24,
        "depth": 27,
        "has_se": True,
    },
    'RegNetY_3_2GF': {
        "stem_width": 32,
        "w_a": 42.63,
        "w_0": 80,
        "w_m": 2.66,
        "groups": 24,
        "depth": 21,
        "has_se": True,
    },
    'RegNetY_4_0GF': {
        "stem_width": 32,
        "w_a": 31.41,
        "w_0": 96,
        "w_m": 2.24,
        "groups": 64,
        "depth": 22,
        "has_se": True,
    },
    'RegNetY_6_4GF': {
        "stem_width": 32,
        "w_a": 33.22,
        "w_0": 112,
        "w_m": 2.27,
        "groups": 72,
        "depth": 25,
        "has_se": True,
    },
    'RegNetY_8_0GF': {
        "stem_width": 32,
        "w_a": 76.82,
        "w_0": 192,
        "w_m": 2.19,
        "groups": 56,
        "depth": 17,
        "has_se": True,
    },
    'RegNetY_12GF': {
        "stem_width": 32,
        "w_a": 73.36,
        "w_0": 168,
        "w_m": 2.37,
        "groups": 112,
        "depth": 19,
        "has_se": True,
    },
    'RegNetY_16GF': {
        "stem_width": 32,
        "w_a": 106.23,
        "w_0": 200,
        "w_m": 2.48,
        "groups": 112,
        "depth": 18,
        "has_se": True,
    },
    'RegNetY_32GF': {
        "stem_width": 32,
        "w_a": 115.89,
        "w_0": 232,
        "w_m": 2.53,
        "groups": 232,
        "depth": 20,
        "has_se": True,
    },
}


def get_regnet_config(regnet_type, q=8):
    stem_width, has_se = regnet_type["stem_width"], regnet_type["has_se"]

    w_a, w_0, w_m, depth, groups = regnet_type["w_a"], regnet_type[
        "w_0"], regnet_type["w_m"], regnet_type["depth"], regnet_type["groups"]

    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ks = np.round(np.log((np.arange(depth) * w_a + w_0) / w_0) / np.log(w_m))
    per_stage_width = w_0 * np.power(w_m, ks)
    per_stage_width = (np.round(np.divide(per_stage_width, q)) *
                       q).astype(int).tolist()

    ts_temp = zip(per_stage_width + [0], [0] + per_stage_width,
                  per_stage_width + [0], [0] + per_stage_width)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    per_stage_depth = np.diff([d for d, t in zip(range(len(ts)), ts)
                               if t]).tolist()

    per_stage_width = np.unique(per_stage_width).tolist()

    per_stage_groups = [groups for _ in range(len(per_stage_width))]
    per_stage_groups = [
        min(per_g, per_w)
        for per_g, per_w in zip(per_stage_groups, per_stage_width)
    ]
    per_stage_width = [
        int(round(per_w / per_g) * per_g)
        for per_w, per_g in zip(per_stage_width, per_stage_groups)
    ]

    return stem_width, has_se, per_stage_width, per_stage_depth, per_stage_groups


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
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.act(x)

        return x


class SeBlock(nn.Module):
    def __init__(self, inplanes, reduction=4):
        super(SeBlock, self).__init__()
        squeezed_planes = max(1, int(inplanes // reduction))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(inplanes,
                               squeezed_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeezed_planes,
                               inplanes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = inputs * x

        return x


class XBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 groups,
                 neck_ratio=1,
                 has_se=True,
                 downsample=False):
        super(XBlock, self).__init__()
        self.has_se = has_se
        self.downsample = downsample

        if self.downsample:
            self.downsample_layer = ConvBnActBlock(inplanes,
                                                   planes,
                                                   kernel_size=1,
                                                   stride=stride,
                                                   padding=0,
                                                   groups=1,
                                                   has_bn=True,
                                                   has_act=False)
        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes // neck_ratio,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=groups,
                                    has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(planes // neck_ratio,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.has_se:
            self.se_block = SeBlock(planes)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.has_se:
            x = self.se_block(x)

        if self.downsample:
            inputs = self.downsample_layer(inputs)

        x += inputs
        x = self.relu(x)
        return x


class RegNet(nn.Module):
    def __init__(self, regnet_type, num_classes=1000):
        super(RegNet, self).__init__()
        stem_width, has_se, per_stage_width, per_stage_depth, per_stage_groups = get_regnet_config(
            RegnetY_types_config[regnet_type], q=8)

        self.stem_width = stem_width
        self.has_se = has_se
        self.per_stage_width = per_stage_width
        self.per_stage_depth = per_stage_depth
        self.per_stage_groups = per_stage_groups

        assert len(self.per_stage_width) == len(self.per_stage_depth)
        assert len(self.per_stage_depth) == len(self.per_stage_groups)

        self.conv1 = ConvBnActBlock(3,
                                    self.stem_width,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)

        self.layer1 = self.make_layer(self.stem_width,
                                      self.per_stage_width[0],
                                      stride=2,
                                      block_num=self.per_stage_depth[0],
                                      group_num=self.per_stage_groups[0],
                                      has_se=self.has_se)
        self.layer2 = self.make_layer(self.per_stage_width[0],
                                      self.per_stage_width[1],
                                      stride=2,
                                      block_num=self.per_stage_depth[1],
                                      group_num=self.per_stage_groups[1],
                                      has_se=self.has_se)
        self.layer3 = self.make_layer(self.per_stage_width[1],
                                      self.per_stage_width[2],
                                      stride=2,
                                      block_num=self.per_stage_depth[2],
                                      group_num=self.per_stage_groups[2],
                                      has_se=self.has_se)
        self.layer4 = self.make_layer(self.per_stage_width[2],
                                      self.per_stage_width[3],
                                      stride=2,
                                      block_num=self.per_stage_depth[3],
                                      group_num=self.per_stage_groups[3],
                                      has_se=self.has_se)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.per_stage_width[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, inplanes, planes, stride, block_num, group_num,
                   has_se):
        layers = []
        for block_index in range(block_num):
            downsample = True if block_index == 0 and (
                stride != 1 or inplanes != planes) else False
            inplanes = planes if block_index > 0 else inplanes
            stride = 1 if block_index > 0 else stride
            layers.append(
                XBlock(inplanes,
                       planes,
                       stride=stride,
                       groups=group_num,
                       has_se=has_se,
                       downsample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _regnet(arch, pretrained, progress, **kwargs):
    model = RegNet(arch, **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')))

    return model


def RegNetY_200MF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_200MF', pretrained, progress, **kwargs)


def RegNetY_400MF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_400MF', pretrained, progress, **kwargs)


def RegNetY_600MF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_600MF', pretrained, progress, **kwargs)


def RegNetY_800MF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_800MF', pretrained, progress, **kwargs)


def RegNetY_1_6GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_1_6GF', pretrained, progress, **kwargs)


def RegNetY_3_2GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_3_2GF', pretrained, progress, **kwargs)


def RegNetY_4_0GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_4_0GF', pretrained, progress, **kwargs)


def RegNetY_6_4GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_6_4GF', pretrained, progress, **kwargs)


def RegNetY_8_0GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_8_0GF', pretrained, progress, **kwargs)


def RegNetY_12GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_12GF', pretrained, progress, **kwargs)


def RegNetY_16GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_16GF', pretrained, progress, **kwargs)


def RegNetY_32GF(pretrained=False, progress=True, **kwargs):
    return _regnet('RegNetY_32GF', pretrained, progress, **kwargs)