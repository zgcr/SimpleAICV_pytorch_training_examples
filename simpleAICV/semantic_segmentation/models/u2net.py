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
from simpleAICV.semantic_segmentation.models.backbones.u2netbackbone import RSU4FBlock, RSU4Block, RSU5Block, RSU6Block, RSU7Block

__all__ = [
    'u2net',
    'u2net_small',
]


class U2Net(nn.Module):

    def __init__(self, backbone_pretrained_path='', num_classes=150):
        super(U2Net, self).__init__()

        self.backbone = backbones.__dict__['u2netbackbone'](
            **{
                'pretrained_path': backbone_pretrained_path,
            })

        # decoder
        self.stage5d = RSU4FBlock(inplanes=1024, middle_planes=256, planes=512)
        self.stage4d = RSU4Block(inplanes=1024, middle_planes=128, planes=256)
        self.stage3d = RSU5Block(inplanes=512, middle_planes=64, planes=128)
        self.stage2d = RSU6Block(inplanes=256, middle_planes=32, planes=64)
        self.stage1d = RSU7Block(inplanes=128, middle_planes=16, planes=64)

        self.side1 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side2 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side3 = nn.Conv2d(128,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side4 = nn.Conv2d(256,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side5 = nn.Conv2d(512,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side6 = nn.Conv2d(512,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.outconv = nn.Conv2d(6 * num_classes,
                                 num_classes,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.backbone(x)

        x6up = F.interpolate(x6,
                             size=(x5.shape[2], x5.shape[3]),
                             mode='bilinear',
                             align_corners=True)

        #-------------------- decoder --------------------
        x5d = self.stage5d(torch.cat((x6up, x5), dim=1))
        x5dup = F.interpolate(x5d,
                              size=(x4.shape[2], x4.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x4d = self.stage4d(torch.cat((x5dup, x4), dim=1))
        x4dup = F.interpolate(x4d,
                              size=(x3.shape[2], x3.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x3d = self.stage3d(torch.cat((x4dup, x3), dim=1))
        x3dup = F.interpolate(x3d,
                              size=(x2.shape[2], x2.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x2d = self.stage2d(torch.cat((x3dup, x2), dim=1))
        x2dup = F.interpolate(x2d,
                              size=(x1.shape[2], x1.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x1d = self.stage1d(torch.cat((x2dup, x1), dim=1))

        #side output
        d1 = self.side1(x1d)

        d2 = self.side2(x2d)
        d2 = F.interpolate(d2,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d3 = self.side3(x3d)
        d3 = F.interpolate(d3,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d4 = self.side4(x4d)
        d4 = F.interpolate(d4,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d5 = self.side5(x5d)
        d5 = F.interpolate(d5,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d6 = self.side6(x6)
        d6 = F.interpolate(d6,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

        return d0


class U2NetSmall(nn.Module):

    def __init__(self, backbone_pretrained_path='', num_classes=150):
        super(U2NetSmall, self).__init__()

        self.backbone = backbones.__dict__['u2netsmallbackbone'](
            **{
                'pretrained_path': backbone_pretrained_path,
            })

        # decoder
        self.stage5d = RSU4FBlock(inplanes=128, middle_planes=16, planes=64)
        self.stage4d = RSU4Block(inplanes=128, middle_planes=16, planes=64)
        self.stage3d = RSU5Block(inplanes=128, middle_planes=16, planes=64)
        self.stage2d = RSU6Block(inplanes=128, middle_planes=16, planes=64)
        self.stage1d = RSU7Block(inplanes=128, middle_planes=16, planes=64)

        self.side1 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side2 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side3 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side4 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side5 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.side6 = nn.Conv2d(64,
                               num_classes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.outconv = nn.Conv2d(6 * num_classes,
                                 num_classes,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.backbone(x)

        x6up = F.interpolate(x6,
                             size=(x5.shape[2], x5.shape[3]),
                             mode='bilinear',
                             align_corners=True)

        #decoder
        x5d = self.stage5d(torch.cat((x6up, x5), dim=1))
        x5dup = F.interpolate(x5d,
                              size=(x4.shape[2], x4.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x4d = self.stage4d(torch.cat((x5dup, x4), dim=1))
        x4dup = F.interpolate(x4d,
                              size=(x3.shape[2], x3.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x3d = self.stage3d(torch.cat((x4dup, x3), dim=1))
        x3dup = F.interpolate(x3d,
                              size=(x2.shape[2], x2.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x2d = self.stage2d(torch.cat((x3dup, x2), dim=1))
        x2dup = F.interpolate(x2d,
                              size=(x1.shape[2], x1.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x1d = self.stage1d(torch.cat((x2dup, x1), dim=1))

        #side output
        d1 = self.side1(x1d)

        d2 = self.side2(x2d)
        d2 = F.interpolate(d2,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d3 = self.side3(x3d)
        d3 = F.interpolate(d3,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d4 = self.side4(x4d)
        d4 = F.interpolate(d4,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d5 = self.side5(x5d)
        d5 = F.interpolate(d5,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d6 = self.side6(x6)
        d6 = F.interpolate(d6,
                           size=(d1.shape[2], d1.shape[3]),
                           mode='bilinear',
                           align_corners=True)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

        return d0


def u2net(**kwargs):
    model = U2Net(**kwargs)

    return model


def u2net_small(**kwargs):
    model = U2NetSmall(**kwargs)

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

    net = u2net()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = u2net_small()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')