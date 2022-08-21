import math

import torch
import torch.nn as nn

__all__ = [
    'yoloxnbackbone',
    'yoloxtbackbone',
    'yoloxsbackbone',
    'yoloxmbackbone',
    'yoloxlbackbone',
    'yoloxxbackbone',
]

types_config = {
    'yoloxn': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.25,
    },
    'yoloxt': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.375,
    },
    'yoloxs': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.50,
    },
    'yoloxm': {
        'depth_coefficient': 0.67,
        'width_coefficient': 0.75,
    },
    'yoloxl': {
        'depth_coefficient': 1.0,
        'width_coefficient': 1.0,
    },
    'yoloxx': {
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


class DWConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 has_bn=True,
                 has_act=True,
                 act_type='silu'):
        super(DWConvBnActBlock, self).__init__()

        self.depthwise_conv = ConvBnActBlock(inplanes,
                                             inplanes,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             groups=inplanes,
                                             has_bn=has_bn,
                                             has_act=has_act,
                                             act_type=act_type)
        self.pointwise_conv = ConvBnActBlock(inplanes,
                                             planes,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=groups,
                                             has_bn=has_bn,
                                             has_act=has_act,
                                             act_type=act_type)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class YOLOXBottleneck(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 block=ConvBnActBlock,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(YOLOXBottleneck, self).__init__()
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
            block(squeezed_planes,
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


class YOLOXCSPBottleneck(nn.Module):
    '''
    CSP Bottleneck with 3 convolution layers
    CSPBottleneck:https://github.com/WongKinYiu/CrossStagePartialNetworks
    '''

    def __init__(self,
                 inplanes,
                 planes,
                 bottleneck_nums=1,
                 bottleneck_block_type=ConvBnActBlock,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(YOLOXCSPBottleneck, self).__init__()
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
            YOLOXBottleneck(squeezed_planes,
                            squeezed_planes,
                            block=bottleneck_block_type,
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


class SPP(nn.Module):
    '''
    Spatial pyramid pooling layer used in YOLOv3-SPP
    '''

    def __init__(self, inplanes, planes, kernels=[5, 9, 13], act_type='silu'):
        super(SPP, self).__init__()
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
        self.conv2 = ConvBnActBlock(squeezed_planes * (len(kernels) + 1),
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
            for kernel in kernels
        ])

    def forward(self, x):
        x = self.conv1(x)

        out = torch.cat([x] + [layer(x) for layer in self.maxpool_layers],
                        dim=1)
        out = self.conv2(out)

        return out


class YoloxBackbone(nn.Module):

    def __init__(self,
                 yolo_backbone_type,
                 planes=[64, 128, 256, 512, 1024],
                 csp_nums=[3, 9, 9, 3],
                 csp_shortcut=[True, True, True, False],
                 block=ConvBnActBlock,
                 act_type='silu',
                 num_classes=1000):
        super(YoloxBackbone, self).__init__()
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

        self.layer1 = nn.Sequential(
            block(self.planes[0],
                  self.planes[1],
                  kernel_size=3,
                  stride=2,
                  padding=1,
                  groups=1,
                  has_bn=True,
                  has_act=True,
                  act_type=act_type),
            YOLOXCSPBottleneck(self.planes[1],
                               self.planes[1],
                               bottleneck_nums=self.csp_nums[0],
                               bottleneck_block_type=block,
                               reduction=0.5,
                               shortcut=csp_shortcut[0],
                               act_type=act_type))

        self.layer2 = nn.Sequential(
            block(self.planes[1],
                  self.planes[2],
                  kernel_size=3,
                  stride=2,
                  padding=1,
                  groups=1,
                  has_bn=True,
                  has_act=True,
                  act_type=act_type),
            YOLOXCSPBottleneck(self.planes[2],
                               self.planes[2],
                               bottleneck_nums=self.csp_nums[1],
                               bottleneck_block_type=block,
                               reduction=0.5,
                               shortcut=csp_shortcut[1],
                               act_type=act_type))

        self.layer3 = nn.Sequential(
            block(self.planes[2],
                  self.planes[3],
                  kernel_size=3,
                  stride=2,
                  padding=1,
                  groups=1,
                  has_bn=True,
                  has_act=True,
                  act_type=act_type),
            YOLOXCSPBottleneck(self.planes[3],
                               self.planes[3],
                               bottleneck_nums=self.csp_nums[2],
                               bottleneck_block_type=block,
                               reduction=0.5,
                               shortcut=csp_shortcut[2],
                               act_type=act_type))

        self.layer4 = nn.Sequential(
            block(self.planes[3],
                  self.planes[4],
                  kernel_size=3,
                  stride=2,
                  padding=1,
                  groups=1,
                  has_bn=True,
                  has_act=True,
                  act_type=act_type),
            SPP(self.planes[4],
                self.planes[4],
                kernels=[5, 9, 13],
                act_type=act_type),
            YOLOXCSPBottleneck(self.planes[4],
                               self.planes[4],
                               bottleneck_nums=self.csp_nums[3],
                               bottleneck_block_type=block,
                               reduction=0.5,
                               shortcut=csp_shortcut[3],
                               act_type=act_type))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes[-1], self.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def compute_depth(self, depth, scale):
        return max(round(depth * scale), 1) if depth > 1 else depth

    def compute_width(self, width, scale, divisor=8):
        return math.ceil((width * scale) / divisor) * divisor


def _yoloxbackbone(yolo_backbone_type, **kwargs):
    model = YoloxBackbone(yolo_backbone_type, **kwargs)

    return model


def yoloxnbackbone(**kwargs):
    return _yoloxbackbone('yoloxn', block=DWConvBnActBlock, **kwargs)


def yoloxtbackbone(**kwargs):
    return _yoloxbackbone('yoloxt', **kwargs)


def yoloxsbackbone(**kwargs):
    return _yoloxbackbone('yoloxs', **kwargs)


def yoloxmbackbone(**kwargs):
    return _yoloxbackbone('yoloxm', **kwargs)


def yoloxlbackbone(**kwargs):
    return _yoloxbackbone('yoloxl', **kwargs)


def yoloxxbackbone(**kwargs):
    return _yoloxbackbone('yoloxx', **kwargs)


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

    net = yoloxnbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yoloxtbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yoloxsbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yoloxmbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yoloxlbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yoloxxbackbone(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'6666, macs: {macs}, params: {params},out_shape: {out.shape}')