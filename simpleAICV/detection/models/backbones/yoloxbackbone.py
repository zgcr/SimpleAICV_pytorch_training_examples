import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import math

import torch
import torch.nn as nn

from simpleAICV.detection.common import load_state_dict

__all__ = [
    'yoloxnbackbone',
    'yoloxtbackbone',
    'yoloxsbackbone',
    'yoloxmbackbone',
    'yoloxlbackbone',
    'yoloxxbackbone',
]

types_config = {
    'yoloxnbackbone': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.25,
    },
    'yoloxtbackbone': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.375,
    },
    'yoloxsbackbone': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.50,
    },
    'yoloxmbackbone': {
        'depth_coefficient': 0.67,
        'width_coefficient': 0.75,
    },
    'yoloxlbackbone': {
        'depth_coefficient': 1.0,
        'width_coefficient': 1.0,
    },
    'yoloxxbackbone': {
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


class FocusBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=1,
                 stride=1,
                 act_type="silu"):
        super(FocusBlock, self).__init__()
        self.conv = ConvBnActBlock(int(inplanes * 4),
                                   planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=kernel_size // 2,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)

    def forward(self, x):
        patch_top_left, patch_bottom_left = x[..., ::2, ::2], x[..., 1::2, ::2]
        patch_top_right, patch_bottom_right = x[..., ::2, 1::2], x[..., 1::2,
                                                                   1::2]
        x = torch.cat(
            [
                patch_top_left,
                patch_bottom_left,
                patch_top_right,
                patch_bottom_right,
            ],
            dim=1,
        )
        x = self.conv(x)

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

        x = torch.cat([x] + [layer(x) for layer in self.maxpool_layers], dim=1)
        x = self.conv2(x)

        return x


class YoloxBackbone(nn.Module):

    def __init__(self,
                 yolo_backbone_type,
                 planes=[64, 128, 256, 512, 1024],
                 csp_nums=[3, 9, 9, 3],
                 csp_shortcut=[True, True, True, False],
                 block=ConvBnActBlock,
                 act_type='silu'):
        super(YoloxBackbone, self).__init__()
        depth_scale = types_config[yolo_backbone_type]['depth_coefficient']
        width_scale = types_config[yolo_backbone_type]['width_coefficient']

        self.planes = [self.compute_width(num, width_scale) for num in planes]
        self.csp_nums = [
            self.compute_depth(num, depth_scale) for num in csp_nums
        ]

        self.focus = FocusBlock(3,
                                self.planes[0],
                                kernel_size=3,
                                stride=1,
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

        self.out_channels = [self.planes[2], self.planes[3], self.planes[4]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.focus(x)

        x = self.layer1(x)
        C3 = self.layer2(x)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        return [C3, C4, C5]

    def compute_depth(self, depth, scale):
        return max(round(depth * scale), 1) if depth > 1 else depth

    def compute_width(self, width, scale, divisor=8):
        return math.ceil((width * scale) / divisor) * divisor


def _yoloxbackbone(yolo_backbone_type, pretrained_path='', **kwargs):
    model = YoloxBackbone(yolo_backbone_type, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def yoloxnbackbone(pretrained_path='', **kwargs):
    model = _yoloxbackbone('yoloxnbackbone',
                           pretrained_path=pretrained_path,
                           block=DWConvBnActBlock,
                           **kwargs)

    return model


def yoloxtbackbone(pretrained_path='', **kwargs):
    model = _yoloxbackbone('yoloxtbackbone',
                           pretrained_path=pretrained_path,
                           **kwargs)

    return model


def yoloxsbackbone(pretrained_path='', **kwargs):
    model = _yoloxbackbone('yoloxsbackbone',
                           pretrained_path=pretrained_path,
                           **kwargs)

    return model


def yoloxmbackbone(pretrained_path='', **kwargs):
    model = _yoloxbackbone('yoloxmbackbone',
                           pretrained_path=pretrained_path,
                           **kwargs)

    return model


def yoloxlbackbone(pretrained_path='', **kwargs):
    model = _yoloxbackbone('yoloxlbackbone',
                           pretrained_path=pretrained_path,
                           **kwargs)

    return model


def yoloxxbackbone(pretrained_path='', **kwargs):
    model = _yoloxbackbone('yoloxxbackbone',
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

    net = yoloxnbackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = yoloxtbackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = yoloxsbackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = yoloxmbackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = yoloxlbackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = yoloxxbackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)