'''
Designing Network Design Spaces
https://arxiv.org/pdf/2003.13678.pdf
'''
import numpy as np

import torch
import torch.nn as nn

__all__ = [
    'RegNetX_200MF',
    'RegNetX_400MF',
    'RegNetX_600MF',
    'RegNetX_800MF',
    'RegNetX_1_6GF',
    'RegNetX_3_2GF',
    'RegNetX_4_0GF',
    'RegNetX_6_4GF',
    'RegNetX_8_0GF',
    'RegNetX_12GF',
    'RegNetX_16GF',
    'RegNetX_32GF',
]

types_config = {
    'RegNetX_200MF': {
        'stem_width': 32,
        'depth': 13,
        'w_0': 24,
        'w_a': 36.44,
        'w_m': 2.49,
        'group_width': 8,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_400MF': {
        'stem_width': 32,
        'depth': 22,
        'w_0': 24,
        'w_a': 24.48,
        'w_m': 2.54,
        'group_width': 16,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_600MF': {
        'stem_width': 32,
        'depth': 16,
        'w_0': 48,
        'w_a': 36.97,
        'w_m': 2.24,
        'group_width': 24,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_800MF': {
        'stem_width': 32,
        'depth': 16,
        'w_0': 56,
        'w_a': 35.73,
        'w_m': 2.28,
        'group_width': 16,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_1_6GF': {
        'stem_width': 32,
        'depth': 18,
        'w_0': 80,
        'w_a': 34.01,
        'w_m': 2.25,
        'group_width': 24,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_3_2GF': {
        'stem_width': 32,
        'depth': 25,
        'w_0': 88,
        'w_a': 26.31,
        'w_m': 2.25,
        'group_width': 48,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_4_0GF': {
        'stem_width': 32,
        'depth': 23,
        'w_0': 96,
        'w_a': 38.65,
        'w_m': 2.43,
        'group_width': 40,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_6_4GF': {
        'stem_width': 32,
        'depth': 17,
        'w_0': 184,
        'w_a': 60.83,
        'w_m': 2.07,
        'group_width': 56,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_8_0GF': {
        'stem_width': 32,
        'depth': 23,
        'w_0': 80,
        'w_a': 49.56,
        'w_m': 2.88,
        'group_width': 120,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_12GF': {
        'stem_width': 32,
        'depth': 19,
        'w_0': 168,
        'w_a': 73.36,
        'w_m': 2.37,
        'group_width': 112,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_16GF': {
        'stem_width': 32,
        'depth': 22,
        'w_0': 216,
        'w_a': 55.59,
        'w_m': 2.1,
        'group_width': 128,
        'bottleneck_ratio': 1.0,
    },
    'RegNetX_32GF': {
        'stem_width': 32,
        'depth': 23,
        'w_0': 320,
        'w_a': 69.86,
        'w_m': 2.0,
        'group_width': 168,
        'bottleneck_ratio': 1.0,
    },
}


def get_regnet_config(regnet_type, q=8):
    stem_width, depth, w_0, w_a, w_m, group_width, bottleneck_ratio = regnet_type[
        'stem_width'], regnet_type['depth'], regnet_type['w_0'], regnet_type[
            'w_a'], regnet_type['w_m'], regnet_type[
                'group_width'], regnet_type['bottleneck_ratio']
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_0 >= 0 and w_a > 0 and w_m > 1 and w_0 % q == 0

    # Generate quantized per-block ws
    ks = np.round(np.log((np.arange(depth) * w_a + w_0) / w_0) / np.log(w_m))
    width_all = (np.round(np.divide(w_0 * np.power(w_m, ks), q)) *
                 q).astype(int)
    # Generate per stage width and depth (assumes width_all are sorted)
    all_stage_width, all_stage_depth = np.unique(width_all, return_counts=True)

    all_stage_group_width = [
        int(
            min(group_width, per_stage_width // bottleneck_ratio) *
            bottleneck_ratio) for per_stage_width in all_stage_width
    ]
    all_stage_width = [
        int(
            round(per_stage_width // bottleneck_ratio / group_width) *
            group_width) for per_stage_width in all_stage_width
    ]
    all_stage_bottleneck_ratio = [bottleneck_ratio for _ in all_stage_width]

    return stem_width, all_stage_width, all_stage_depth, all_stage_bottleneck_ratio, all_stage_group_width


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
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class XBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 bottleneck_ratio,
                 group_width,
                 downsample=False):
        super(XBlock, self).__init__()
        self.downsample = downsample
        inter_planes = int(planes // bottleneck_ratio)
        groups = inter_planes // group_width

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
                                    inter_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(inter_planes,
                                    inter_planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=groups,
                                    has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(inter_planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample:
            inputs = self.downsample_layer(inputs)

        x += inputs
        x = self.relu(x)

        return x


class RegNet(nn.Module):

    def __init__(self, regnet_type, num_classes=1000):
        super(RegNet, self).__init__()
        stem_width, all_stage_width, all_stage_depth, all_stage_bottleneck_ratio, all_stage_group_width = get_regnet_config(
            types_config[regnet_type], q=8)

        self.stem_width = stem_width
        self.all_stage_width = all_stage_width
        self.all_stage_depth = all_stage_depth
        self.all_stage_bottleneck_ratio = all_stage_bottleneck_ratio
        self.all_stage_group_width = all_stage_group_width
        self.num_classes = num_classes

        assert len(self.all_stage_width) == len(self.all_stage_depth) == len(
            self.all_stage_bottleneck_ratio) == len(self.all_stage_group_width)

        self.conv1 = ConvBnActBlock(3,
                                    self.stem_width,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)

        self.layer1 = self.make_layer(
            inplanes=self.stem_width,
            planes=self.all_stage_width[0],
            stride=2,
            block_num=self.all_stage_depth[0],
            bottleneck_ratio=self.all_stage_bottleneck_ratio[0],
            group_width=self.all_stage_group_width[0])
        self.layer2 = self.make_layer(
            inplanes=self.all_stage_width[0],
            planes=self.all_stage_width[1],
            stride=2,
            block_num=self.all_stage_depth[1],
            bottleneck_ratio=self.all_stage_bottleneck_ratio[1],
            group_width=self.all_stage_group_width[1])
        self.layer3 = self.make_layer(
            inplanes=self.all_stage_width[1],
            planes=self.all_stage_width[2],
            stride=2,
            block_num=self.all_stage_depth[2],
            bottleneck_ratio=self.all_stage_bottleneck_ratio[2],
            group_width=self.all_stage_group_width[2])
        self.layer4 = self.make_layer(
            inplanes=self.all_stage_width[2],
            planes=self.all_stage_width[3],
            stride=2,
            block_num=self.all_stage_depth[3],
            bottleneck_ratio=self.all_stage_bottleneck_ratio[3],
            group_width=self.all_stage_group_width[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.all_stage_width[3], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, inplanes, planes, stride, block_num, bottleneck_ratio,
                   group_width):
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
                       bottleneck_ratio=bottleneck_ratio,
                       group_width=group_width,
                       downsample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _regnet(arch, **kwargs):
    model = RegNet(arch, **kwargs)

    return model


def RegNetX_200MF(**kwargs):
    return _regnet('RegNetX_200MF', **kwargs)


def RegNetX_400MF(**kwargs):
    return _regnet('RegNetX_400MF', **kwargs)


def RegNetX_600MF(**kwargs):
    return _regnet('RegNetX_600MF', **kwargs)


def RegNetX_800MF(**kwargs):
    return _regnet('RegNetX_800MF', **kwargs)


def RegNetX_1_6GF(**kwargs):
    return _regnet('RegNetX_1_6GF', **kwargs)


def RegNetX_3_2GF(**kwargs):
    return _regnet('RegNetX_3_2GF', **kwargs)


def RegNetX_4_0GF(**kwargs):
    return _regnet('RegNetX_4_0GF', **kwargs)


def RegNetX_6_4GF(**kwargs):
    return _regnet('RegNetX_6_4GF', **kwargs)


def RegNetX_8_0GF(**kwargs):
    return _regnet('RegNetX_8_0GF', **kwargs)


def RegNetX_12GF(**kwargs):
    return _regnet('RegNetX_12GF', **kwargs)


def RegNetX_16GF(**kwargs):
    return _regnet('RegNetX_16GF', **kwargs)


def RegNetX_32GF(**kwargs):
    return _regnet('RegNetX_32GF', **kwargs)


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

    net = RegNetX_200MF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_400MF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_600MF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_800MF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_1_6GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_3_2GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'6666, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_4_0GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'7777, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_6_4GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'8888, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_8_0GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'9999, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_12GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'9191, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_16GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'9292, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = RegNetX_32GF(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'9393, macs: {macs}, params: {params},out_shape: {out.shape}')