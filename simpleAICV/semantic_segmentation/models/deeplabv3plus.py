import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from simpleAICV.detection.models import backbones

__all__ = [
    'resnet18_deeplabv3plus',
    'resnet34_deeplabv3plus',
    'resnet50_deeplabv3plus',
    'resnet101_deeplabv3plus',
    'resnet152_deeplabv3plus',
    'vanb0_deeplabv3plus',
    'vanb1_deeplabv3plus',
    'vanb2_deeplabv3plus',
    'vanb3_deeplabv3plus',
    'convformers18_deeplabv3plus',
    'convformers36_deeplabv3plus',
    'convformerm36_deeplabv3plus',
    'convformerb36_deeplabv3plus',
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

    def __init__(self, inplanes, planes=256, output_stride=8, num_classes=150):
        super(DeepLabV3PlusHead, self).__init__()
        self.aspp1 = ASPPBlock(inplanes=inplanes[0],
                               planes=planes,
                               output_stride=output_stride)
        self.aspp2 = ASPPBlock(inplanes=inplanes[1],
                               planes=planes,
                               output_stride=output_stride)
        self.aspp3 = ASPPBlock(inplanes=inplanes[2],
                               planes=planes,
                               output_stride=output_stride)
        self.aspp4 = ASPPBlock(inplanes=inplanes[3],
                               planes=planes,
                               output_stride=output_stride)

        self.fuse_conv = nn.Sequential(
            LightConvBlock(planes * 4,
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
        C1, C2, C3, C4 = x

        C1 = self.aspp1(C1)

        C2 = self.aspp2(C2)
        C2 = F.interpolate(C2,
                           size=(C1.shape[2], C1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        C3 = self.aspp3(C3)
        C3 = F.interpolate(C3,
                           size=(C1.shape[2], C1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        C4 = self.aspp4(C4)
        C4 = F.interpolate(C4,
                           size=(C1.shape[2], C1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        x = torch.cat([C1, C2, C3, C4], dim=1)

        x = self.fuse_conv(x)
        x = self.predict_conv(x)

        return x


class DeepLabV3Plus(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 output_stride=8,
                 num_classes=150,
                 use_gradient_checkpoint=False):
        super(DeepLabV3Plus, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.head = DeepLabV3PlusHead(inplanes=self.backbone.out_channels,
                                      planes=planes,
                                      output_stride=output_stride,
                                      num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        features = self.backbone(x)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.head, features, use_reentrant=False)
        else:
            x = self.head(features)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x


def _deeplabv3plus(backbone_type, backbone_pretrained_path, planes,
                   output_stride, **kwargs):
    model = DeepLabV3Plus(backbone_type,
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)

    return model


def resnet18_deeplabv3plus(planes=256,
                           output_stride=8,
                           backbone_pretrained_path='',
                           **kwargs):
    return _deeplabv3plus('resnet18backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet34_deeplabv3plus(planes=256,
                           output_stride=8,
                           backbone_pretrained_path='',
                           **kwargs):
    return _deeplabv3plus('resnet34backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet50_deeplabv3plus(planes=256,
                           output_stride=8,
                           backbone_pretrained_path='',
                           **kwargs):
    return _deeplabv3plus('resnet50backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet101_deeplabv3plus(planes=256,
                            output_stride=8,
                            backbone_pretrained_path='',
                            **kwargs):
    return _deeplabv3plus('resnet101backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def resnet152_deeplabv3plus(planes=256,
                            output_stride=8,
                            backbone_pretrained_path='',
                            **kwargs):
    return _deeplabv3plus('resnet152backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def vanb0_deeplabv3plus(planes=256,
                        output_stride=8,
                        backbone_pretrained_path='',
                        **kwargs):
    return _deeplabv3plus('vanb0backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def vanb1_deeplabv3plus(planes=256,
                        output_stride=8,
                        backbone_pretrained_path='',
                        **kwargs):
    return _deeplabv3plus('vanb1backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def vanb2_deeplabv3plus(planes=256,
                        output_stride=8,
                        backbone_pretrained_path='',
                        **kwargs):
    return _deeplabv3plus('vanb2backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def vanb3_deeplabv3plus(planes=256,
                        output_stride=8,
                        backbone_pretrained_path='',
                        **kwargs):
    return _deeplabv3plus('vanb3backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def convformers18_deeplabv3plus(planes=256,
                                output_stride=8,
                                backbone_pretrained_path='',
                                **kwargs):
    return _deeplabv3plus('convformers18backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def convformers36_deeplabv3plus(planes=256,
                                output_stride=8,
                                backbone_pretrained_path='',
                                **kwargs):
    return _deeplabv3plus('convformers36backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def convformerm36_deeplabv3plus(planes=256,
                                output_stride=8,
                                backbone_pretrained_path='',
                                **kwargs):
    return _deeplabv3plus('convformerm36backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
                          planes=planes,
                          output_stride=output_stride,
                          **kwargs)


def convformerb36_deeplabv3plus(planes=256,
                                output_stride=8,
                                backbone_pretrained_path='',
                                **kwargs):
    return _deeplabv3plus('convformerb36backbone',
                          backbone_pretrained_path=backbone_pretrained_path,
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

    net = resnet18_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = resnet34_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = resnet50_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = resnet101_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = resnet152_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = resnet152_deeplabv3plus(use_gradient_checkpoint=True)
    image_h, image_w = 512, 512
    outs = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', outs.shape)

    net = vanb0_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vanb1_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vanb2_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vanb3_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformers18_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformers36_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformerm36_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformerb36_deeplabv3plus()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')
