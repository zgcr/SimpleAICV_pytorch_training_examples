import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.text_detection.models import backbones
from simpleAICV.text_detection.models.fpn import DBNetFPN
from simpleAICV.text_detection.models.head import DBNetHead

__all__ = [
    'RepVGGDBNet',
    'RepVGGEnhanceDBNet',
    'ResnetDBNet',
    'VANDBNet',
]


class RepVGGDBNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=[16, 16, 32, 48, 64, 80],
                 deploy=False,
                 inter_planes=64,
                 k=50):
        super(RepVGGDBNet, self).__init__()
        assert backbone_type in [
            'RepVGGNetBackbone',
        ]

        self.backbone = backbones.__dict__[backbone_type](**{
            'pretrained_path': backbone_pretrained_path,
            'planes': planes,
            'deploy': deploy,
        })

        self.fpn = DBNetFPN(inplanes_list=self.backbone.out_channels,
                            inter_planes=inter_planes)
        self.head = DBNetHead(inplanes=inter_planes, k=k)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)

        return x


class RepVGGEnhanceDBNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=[16, 16, 32, 48, 64, 80],
                 repvgg_k=4,
                 deploy=False,
                 inter_planes=64,
                 k=50):
        super(RepVGGEnhanceDBNet, self).__init__()
        assert backbone_type in [
            'RepVGGEnhanceNetBackbone',
            'RepVGGEnhanceDilationNetBackbone',
        ]

        self.backbone = backbones.__dict__[backbone_type](**{
            'pretrained_path': backbone_pretrained_path,
            'planes': planes,
            'deploy': deploy,
            'k': repvgg_k,
        })

        self.fpn = DBNetFPN(inplanes_list=self.backbone.out_channels,
                            inter_planes=inter_planes)
        self.head = DBNetHead(inplanes=inter_planes, k=k)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)

        return x


class ResnetDBNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 inter_planes=256,
                 k=50):
        super(ResnetDBNet, self).__init__()
        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })

        self.fpn = DBNetFPN(inplanes_list=self.backbone.out_channels,
                            inter_planes=inter_planes)
        self.head = DBNetHead(inplanes=inter_planes, k=k)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)

        return x


class VANDBNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 inter_planes=256,
                 k=50):
        super(VANDBNet, self).__init__()
        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })

        self.fpn = DBNetFPN(inplanes_list=self.backbone.out_channels,
                            inter_planes=inter_planes)
        self.head = DBNetHead(inplanes=inter_planes, k=k)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)

        return x


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

    net = RepVGGDBNet(backbone_type='RepVGGNetBackbone',
                      planes=[16, 16, 32, 48, 64, 80],
                      deploy=True,
                      inter_planes=96,
                      k=50)
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params}, out: {out.shape}')

    net = RepVGGEnhanceDBNet(backbone_type='RepVGGEnhanceNetBackbone',
                             planes=[16, 16, 32, 48, 64, 80],
                             repvgg_k=4,
                             deploy=True,
                             inter_planes=96,
                             k=50)
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params}, out: {out.shape}')

    net = ResnetDBNet(backbone_type='resnet50backbone', inter_planes=256, k=50)
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params}, out: {out.shape}')

    net = VANDBNet(backbone_type='van_b1_backbone', inter_planes=256, k=50)
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params}, out: {out.shape}')
