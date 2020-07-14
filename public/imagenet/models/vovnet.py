"""
CenterMask : Real-Time Anchor-Free Instance Segmentation
https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py
"""
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'VoVNet19_slim_depthwise_se',
    'VoVNet19_depthwise_se',
    'VoVNet19_slim_se',
    'VoVNet19_se',
    'VoVNet39_se',
    'VoVNet57_se',
    'VoVNet99_se',
]

model_urls = {
    'VoVNet19_slim_depthwise_se':
    '{}/vovnet/VoVNet19_slim_depthwise_se-epoch100-acc66.724.pth'.format(
        pretrained_models_path),
    'VoVNet19_depthwise_se':
    '{}/vovnet/VoVNet19_depthwise_se-epoch100-acc73.042.pth'.format(
        pretrained_models_path),
    'VoVNet19_slim_se':
    '{}/vovnet/VoVNet19_slim_se-epoch100-acc69.354.pth'.format(
        pretrained_models_path),
    'VoVNet19_se':
    '{}/vovnet/VoVNet19_se-epoch100-acc74.636.pth'.format(
        pretrained_models_path),
    'VoVNet39_se':
    '{}/vovnet/VoVNet39_se-epoch100-acc77.338.pth'.format(
        pretrained_models_path),
    'VoVNet57_se':
    '{}/vovnet/VoVNet57_se-epoch100-acc77.986.pth'.format(
        pretrained_models_path),
    'VoVNet99_se':
    '{}/vovnet/VoVNet99_se-epoch100-acc78.392.pth'.format(
        pretrained_models_path),
}

vovnet_configs = {
    'VoVNet19_slim_depthwise_se': {
        'stem_channel': [64, 64, 64],
        'per_stage_inter_channels': [64, 80, 96, 112],
        'per_stage_inout_channels': [112, 256, 384, 512],
        'per_block_layer_nums': 3,
        'per_stage_block_nums': [1, 1, 1, 1],
        "has_se": True,
        'has_depthwise': True,
    },
    'VoVNet19_depthwise_se': {
        'stem_channel': [64, 64, 64],
        'per_stage_inter_channels': [128, 160, 192, 224],
        'per_stage_inout_channels': [256, 512, 768, 1024],
        'per_block_layer_nums': 3,
        'per_stage_block_nums': [1, 1, 1, 1],
        "has_se": True,
        'has_depthwise': True,
    },
    'VoVNet19_slim_se': {
        'stem_channel': [64, 64, 128],
        'per_stage_inter_channels': [64, 80, 96, 112],
        'per_stage_inout_channels': [112, 256, 384, 512],
        'per_block_layer_nums': 3,
        'per_stage_block_nums': [1, 1, 1, 1],
        "has_se": True,
        'has_depthwise': False,
    },
    'VoVNet19_se': {
        'stem_channel': [64, 64, 128],
        'per_stage_inter_channels': [128, 160, 192, 224],
        'per_stage_inout_channels': [256, 512, 768, 1024],
        'per_block_layer_nums': 3,
        'per_stage_block_nums': [1, 1, 1, 1],
        "has_se": True,
        'has_depthwise': False,
    },
    'VoVNet39_se': {
        'stem_channel': [64, 64, 128],
        'per_stage_inter_channels': [128, 160, 192, 224],
        'per_stage_inout_channels': [256, 512, 768, 1024],
        'per_block_layer_nums': 5,
        'per_stage_block_nums': [1, 1, 2, 2],
        "has_se": True,
        'has_depthwise': False,
    },
    'VoVNet57_se': {
        'stem_channel': [64, 64, 128],
        'per_stage_inter_channels': [128, 160, 192, 224],
        'per_stage_inout_channels': [256, 512, 768, 1024],
        'per_block_layer_nums': 5,
        'per_stage_block_nums': [1, 1, 4, 3],
        "has_se": True,
        'has_depthwise': False,
    },
    'VoVNet99_se': {
        'stem_channel': [64, 64, 128],
        'per_stage_inter_channels': [128, 160, 192, 224],
        'per_stage_inout_channels': [256, 512, 768, 1024],
        'per_block_layer_nums': 5,
        'per_stage_block_nums': [1, 3, 9, 3],
        "has_se": True,
        'has_depthwise': False,
    },
}


class Conv3x3Block(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 has_bn=True,
                 has_act=True,
                 has_depthwise=False):
        super(Conv3x3Block, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act
        if has_depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size,
                          stride=stride,
                          padding=padding,
                          groups=planes,
                          bias=False),
                nn.Conv2d(planes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=False))
        else:
            self.conv = nn.Conv2d(inplanes,
                                  planes,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  groups=1,
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


class Conv1x1Block(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 has_bn=True,
                 has_act=True):
        super(Conv1x1Block, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=1,
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


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(eSEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=True)
        self.hardsigmoid = HardSigmoid()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv(x)
        x = self.hardsigmoid(x)
        x = inputs * x

        return x


class OSABlock(nn.Module):
    def __init__(self,
                 inplanes,
                 interplanes,
                 planes,
                 per_block_layer_nums,
                 has_se=False,
                 has_depthwise=False,
                 has_identity=False):
        super(OSABlock, self).__init__()
        self.inplanes = inplanes
        self.interplanes = interplanes
        self.has_se = has_se
        self.has_depthwise = has_depthwise
        self.has_identity = has_identity

        if self.has_depthwise and self.inplanes != self.interplanes:
            self.reduce_conv = Conv1x1Block(inplanes,
                                            interplanes,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            has_bn=True,
                                            has_act=True)
        self.OSABlocklayers = nn.ModuleList()

        input_planes = inplanes
        for _ in range(per_block_layer_nums):
            if self.has_depthwise:
                input_planes = interplanes
            self.OSABlocklayers.append(
                Conv3x3Block(input_planes,
                             interplanes,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             has_bn=True,
                             has_act=True,
                             has_depthwise=self.has_depthwise))
            input_planes = interplanes

        concatplanes = inplanes + per_block_layer_nums * interplanes

        self.concat_conv = Conv1x1Block(concatplanes,
                                        planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        has_bn=True,
                                        has_act=True)
        if self.has_se:
            self.ese = eSEBlock(planes, planes)

    def forward(self, inputs):
        outputs = []
        outputs.append(inputs)
        if self.has_depthwise and self.inplanes != self.interplanes:
            x = self.reduce_conv(inputs)
        else:
            x = inputs
        for layer in self.OSABlocklayers:
            x = layer(x)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        x = self.concat_conv(x)

        if self.has_se:
            x = self.ese(x)

        if self.has_identity:
            x = x + inputs

        return x


class OSAStage(nn.Module):
    def __init__(self,
                 inplanes,
                 interplanes,
                 planes,
                 per_stage_block_nums,
                 per_block_layer_nums,
                 has_se=False,
                 has_depthwise=False,
                 first_stage=False):
        super(OSAStage, self).__init__()
        self.first_stage = first_stage
        if not self.first_stage:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.has_depthwise = has_depthwise
        self.has_se = has_se
        if per_stage_block_nums > 1:
            self.has_se = False

        identity = False
        input_planes = inplanes
        layers = []
        for i in range(per_stage_block_nums):
            if i > 0:
                input_planes = planes
                identity = True
            if i != per_stage_block_nums - 1:
                self.has_se = False
            layers.append(
                OSABlock(input_planes,
                         interplanes,
                         planes,
                         per_block_layer_nums,
                         has_se=self.has_se,
                         has_depthwise=self.has_depthwise,
                         has_identity=identity))

        self.OSAStageblocks = nn.Sequential(*layers)

    def forward(self, x):
        if not self.first_stage:
            x = self.pool(x)

        x = self.OSAStageblocks(x)

        return x


class VoVNet(nn.Module):
    def __init__(self, vovnet_type, num_classes=1000):
        super(VoVNet, self).__init__()
        vovnet_type_config = vovnet_configs[vovnet_type]
        self.stem_channel = vovnet_type_config['stem_channel']
        self.per_stage_inter_channels = vovnet_type_config[
            'per_stage_inter_channels']
        self.per_stage_inout_channels = vovnet_type_config[
            'per_stage_inout_channels']
        self.per_block_layer_nums = vovnet_type_config['per_block_layer_nums']
        self.per_stage_block_nums = vovnet_type_config['per_stage_block_nums']
        self.has_se = vovnet_type_config['has_se']
        self.has_depthwise = vovnet_type_config['has_depthwise']

        self.stem = nn.Sequential(
            Conv3x3Block(3,
                         self.stem_channel[0],
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         has_bn=True,
                         has_act=True,
                         has_depthwise=False),
            Conv3x3Block(self.stem_channel[0],
                         self.stem_channel[1],
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         has_bn=True,
                         has_act=True,
                         has_depthwise=self.has_depthwise),
            Conv3x3Block(self.stem_channel[1],
                         self.stem_channel[2],
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         has_bn=True,
                         has_act=True,
                         has_depthwise=self.has_depthwise))

        input_planes = self.stem_channel[2]
        first_stage = True
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                input_planes = self.per_stage_inout_channels[i - 1]
                first_stage = False
            self.stages.append(
                OSAStage(input_planes,
                         self.per_stage_inter_channels[i],
                         self.per_stage_inout_channels[i],
                         self.per_stage_block_nums[i],
                         self.per_block_layer_nums,
                         has_se=self.has_se,
                         has_depthwise=self.has_depthwise,
                         first_stage=first_stage))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.per_stage_inout_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _vovnet(arch, pretrained, progress, **kwargs):
    model = VoVNet(arch, **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')))

    return model


def VoVNet19_slim_depthwise_se(pretrained=False, progress=True, **kwargs):
    return _vovnet('VoVNet19_slim_depthwise_se', pretrained, progress,
                   **kwargs)


def VoVNet19_depthwise_se(pretrained=False, progress=True, **kwargs):
    return _vovnet('VoVNet19_depthwise_se', pretrained, progress, **kwargs)


def VoVNet19_slim_se(pretrained=False, progress=True, **kwargs):
    return _vovnet('VoVNet19_slim_se', pretrained, progress, **kwargs)


def VoVNet19_se(pretrained=False, progress=True, **kwargs):
    return _vovnet('VoVNet19_se', pretrained, progress, **kwargs)


def VoVNet39_se(pretrained=False, progress=True, **kwargs):
    return _vovnet('VoVNet39_se', pretrained, progress, **kwargs)


def VoVNet57_se(pretrained=False, progress=True, **kwargs):
    return _vovnet('VoVNet57_se', pretrained, progress, **kwargs)


def VoVNet99_se(pretrained=False, progress=True, **kwargs):
    return _vovnet('VoVNet99_se', pretrained, progress, **kwargs)