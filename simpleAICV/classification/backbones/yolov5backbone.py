import math

import torch
import torch.nn as nn

__all__ = [
    'yolov5nbackbone',
    'yolov5sbackbone',
    'yolov5mbackbone',
    'yolov5lbackbone',
    'yolov5xbackbone',
]

types_config = {
    'yolov5n': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.25,
    },
    'yolov5s': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.50,
    },
    'yolov5m': {
        'depth_coefficient': 0.67,
        'width_coefficient': 0.75,
    },
    'yolov5l': {
        'depth_coefficient': 1.0,
        'width_coefficient': 1.0,
    },
    'yolov5x': {
        'depth_coefficient': 1.33,
        'width_coefficient': 1.25,
    },
}


class ActivationBlock(nn.Module):

    def __init__(self, act_type='silu', inplace=True):
        super(ActivationBlock, self).__init__()
        assert act_type in ['silu', 'relu',
                            'leakyrelu'], 'Unsupport activation function!'
        if act_type == 'silu':
            self.act = nn.SiLU(inplace=inplace)
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1, inplace=inplace)

    def forward(self, x):
        x = self.act(x)

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
                 has_act=True,
                 act_type='silu'):
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
            ActivationBlock(act_type=act_type, inplace=True)
            if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(Bottleneck, self).__init__()
        squeezed_planes = max(1, int(planes * reduction))
        self.conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           squeezed_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(squeezed_planes,
                           planes,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))

        self.shortcut = True if shortcut and inplanes == planes else False

    def forward(self, x):
        out = self.conv(x)

        if self.shortcut:
            out = out + x

        del x

        return out


class CSPBottleneck(nn.Module):
    '''
    CSP Bottleneck with 3 convolution layers
    CSPBottleneck:https://github.com/WongKinYiu/CrossStagePartialNetworks
    '''

    def __init__(self,
                 inplanes,
                 planes,
                 bottleneck_nums=1,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(CSPBottleneck, self).__init__()
        squeezed_planes = max(1, int(planes * reduction))
        self.conv1 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv3 = ConvBnActBlock(2 * squeezed_planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)

        self.bottlenecks = nn.Sequential(*[
            Bottleneck(squeezed_planes,
                       squeezed_planes,
                       reduction=1.0,
                       shortcut=shortcut,
                       act_type=act_type) for _ in range(bottleneck_nums)
        ])

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bottlenecks(y1)
        y2 = self.conv2(x)

        del x

        out = torch.cat([y1, y2], axis=1)
        out = self.conv3(out)

        del y1, y2

        return out


class SPPF(nn.Module):

    def __init__(self, inplanes, planes, kernel=5, act_type='silu'):
        super(SPPF, self).__init__()
        squeezed_planes = max(1, int(inplanes // 2))
        self.conv1 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(squeezed_planes * 4,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel,
                                    stride=1,
                                    padding=kernel // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)

        out = torch.cat([x, y1, y2, y3], dim=1)

        del x, y1, y2, y3

        out = self.conv2(out)

        return out


class Yolov5Backbone(nn.Module):

    def __init__(self,
                 yolo_backbone_type,
                 planes=[64, 128, 256, 512, 1024],
                 csp_nums=[3, 6, 9, 3],
                 csp_shortcut=[True, True, True, True],
                 act_type='silu',
                 num_classes=1000):
        super(Yolov5Backbone, self).__init__()
        depth_scale = types_config[yolo_backbone_type]['depth_coefficient']
        width_scale = types_config[yolo_backbone_type]['width_coefficient']

        self.planes = [self.compute_width(num, width_scale) for num in planes]
        self.csp_nums = [
            self.compute_depth(num, depth_scale) for num in csp_nums
        ]
        self.num_classes = num_classes

        self.conv = ConvBnActBlock(3,
                                   self.planes[0],
                                   kernel_size=6,
                                   stride=2,
                                   padding=2,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)

        middle_layers = []
        self.middle_planes = self.planes[0]
        for i in range(7):
            idx = (i // 2) + 1
            middle_layers.append(
                ConvBnActBlock(self.middle_planes,
                               self.planes[idx],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               has_bn=True,
                               has_act=True,
                               act_type=act_type) if i %
                2 == 0 else CSPBottleneck(self.middle_planes,
                                          self.planes[idx],
                                          bottleneck_nums=self.csp_nums[idx],
                                          reduction=0.5,
                                          shortcut=csp_shortcut[idx],
                                          act_type=act_type))
            self.middle_planes = self.planes[idx]

        self.middle_layers = nn.Sequential(*middle_layers)

        self.sppf = SPPF(self.planes[-1],
                         self.planes[-1],
                         kernel=5,
                         act_type=act_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes[-1], self.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.middle_layers(x)
        x = self.sppf(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def compute_depth(self, depth, scale):
        return max(round(depth * scale), 1) if depth > 1 else depth

    def compute_width(self, width, scale, divisor=8):
        return math.ceil((width * scale) / divisor) * divisor


def _yolov5backbone(yolo_backbone_type, **kwargs):
    model = Yolov5Backbone(yolo_backbone_type, **kwargs)

    return model


def yolov5nbackbone(**kwargs):
    return _yolov5backbone('yolov5n', **kwargs)


def yolov5sbackbone(**kwargs):
    return _yolov5backbone('yolov5s', **kwargs)


def yolov5mbackbone(**kwargs):
    return _yolov5backbone('yolov5m', **kwargs)


def yolov5lbackbone(**kwargs):
    return _yolov5backbone('yolov5l', **kwargs)


def yolov5xbackbone(**kwargs):
    return _yolov5backbone('yolov5x', **kwargs)


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

    net = yolov5nbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yolov5sbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yolov5mbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yolov5lbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yolov5xbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params},out_shape: {out.shape}')