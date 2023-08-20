import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.semantic_segmentation.models import backbones

__all__ = [
    'resnet18backbone_deeplabv3plus',
    'resnet34backbone_deeplabv3plus',
    'resnet50backbone_deeplabv3plus',
    'resnet101backbone_deeplabv3plus',
    'resnet152backbone_deeplabv3plus',
]


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 dilation=1,
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
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class LightConvBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(LightConvBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=dilation,
                           groups=inplanes,
                           dilation=dilation,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(inplanes,
                           planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
        )

    def forward(self, x):
        x = self.layers(x)

        return x


class ASPPBlock(nn.Module):

    def __init__(self, inplanes=2048, planes=256, output_stride=8):
        super(ASPPBlock, self).__init__()
        assert output_stride in [8, 16]
        if output_stride == 8:
            self.dilation = [12, 24, 36]
        else:
            self.dilation = [6, 12, 18]

        self.aspp0 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    dilation=1,
                                    has_bn=True,
                                    has_act=True)
        self.aspp1 = LightConvBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    dilation=self.dilation[0])
        self.aspp2 = LightConvBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    dilation=self.dilation[1])
        self.aspp3 = LightConvBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    dilation=self.dilation[2])

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBnActBlock(inplanes,
                           planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
        )

        self.fuse_conv = ConvBnActBlock(int(planes * 5),
                                        planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

    def forward(self, x):
        pool = self.pooling(x)
        pool = F.interpolate(pool,
                             size=(x.shape[2], x.shape[3]),
                             mode='bilinear',
                             align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)
        x = self.fuse_conv(x)

        return x


class DeepLabV3PlusHead(nn.Module):

    def __init__(self,
                 c1_inplanes,
                 c4_inplanes,
                 c1_planes=64,
                 planes=256,
                 output_stride=8,
                 num_classes=150):
        super(DeepLabV3PlusHead, self).__init__()
        self.aspp = ASPPBlock(inplanes=c4_inplanes,
                              planes=planes,
                              output_stride=output_stride)
        self.c1_conv = ConvBnActBlock(c1_inplanes,
                                      c1_planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      dilation=1,
                                      has_bn=True,
                                      has_act=True)
        self.fuse_conv = nn.Sequential(
            LightConvBlock(int(planes + c1_planes),
                           planes,
                           kernel_size=3,
                           stride=1,
                           dilation=1),
            LightConvBlock(planes, planes, kernel_size=3, stride=1,
                           dilation=1),
        )
        self.predict_conv = nn.Conv2d(planes,
                                      num_classes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

    def forward(self, x):
        C1, _, _, C4 = x

        C4 = self.aspp(C4)
        C4 = F.interpolate(C4,
                           size=(C1.shape[2], C1.shape[3]),
                           mode='bilinear',
                           align_corners=True)
        C1 = self.c1_conv(C1)
        x = torch.cat([C4, C1], dim=1)
        x = self.fuse_conv(x)
        x = self.predict_conv(x)

        return x


class DeepLabV3Plus(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 c1_planes=64,
                 planes=256,
                 output_stride=8,
                 num_classes=150):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })
        self.head = DeepLabV3PlusHead(
            c1_inplanes=self.backbone.out_channels[0],
            c4_inplanes=self.backbone.out_channels[-1],
            c1_planes=c1_planes,
            planes=planes,
            output_stride=output_stride,
            num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        features = self.backbone(x)
        x = self.head(features)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x


def _deeplabv3plus(backbone_type, backbone_pretrained_path, c1_planes, planes,
                   output_stride, **kwargs):
    model = DeepLabV3Plus(backbone_type,
                          backbone_pretrained_path=backbone_pretrained_path,
                          c1_planes=c1_planes,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)

    return model


def resnet18backbone_deeplabv3plus(c1_planes=64,
                                   planes=256,
                                   output_stride=8,
                                   backbone_pretrained_path='',
                                   **kwargs):
    return _deeplabv3plus('resnet18backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          c1_planes=c1_planes,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet34backbone_deeplabv3plus(c1_planes=64,
                                   planes=256,
                                   output_stride=8,
                                   backbone_pretrained_path='',
                                   **kwargs):
    return _deeplabv3plus('resnet34backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          c1_planes=c1_planes,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet50backbone_deeplabv3plus(c1_planes=64,
                                   planes=256,
                                   output_stride=8,
                                   backbone_pretrained_path='',
                                   **kwargs):
    return _deeplabv3plus('resnet50backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          c1_planes=c1_planes,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet101backbone_deeplabv3plus(c1_planes=64,
                                    planes=256,
                                    output_stride=8,
                                    backbone_pretrained_path='',
                                    **kwargs):
    return _deeplabv3plus('resnet101backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          c1_planes=c1_planes,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet152backbone_deeplabv3plus(c1_planes=64,
                                    planes=256,
                                    output_stride=8,
                                    backbone_pretrained_path='',
                                    **kwargs):
    return _deeplabv3plus('resnet152backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          c1_planes=c1_planes,
                          planes=planes,
                          output_stride=output_stride,
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

    net = resnet50backbone_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')