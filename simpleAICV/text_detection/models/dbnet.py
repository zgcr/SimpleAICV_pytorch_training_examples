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
from simpleAICV.text_detection.models.fpn import DBNetFPN
from simpleAICV.text_detection.models.head import DBNetHead

__all__ = [
    'resnet18_dbnet',
    'resnet34_dbnet',
    'resnet50_dbnet',
    'resnet101_dbnet',
    'resnet152_dbnet',
    'vanb0_dbnet',
    'vanb1_dbnet',
    'vanb2_dbnet',
    'vanb3_dbnet',
    'convformers18_dbnet',
    'convformers36_dbnet',
    'convformerm36_dbnet',
    'convformerb36_dbnet',
]


class DBNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 inter_planes=256,
                 k=50,
                 use_gradient_checkpoint=False):
        super(DBNet, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        self.fpn = DBNetFPN(inplanes_list=self.backbone.out_channels,
                            inter_planes=inter_planes)
        self.head = DBNetHead(inplanes=inter_planes, k=k)

    def forward(self, x):
        x = self.backbone(x)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.fpn, x, use_reentrant=False)
        else:
            x = self.fpn(x)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.head, x, use_reentrant=False)
        else:
            x = self.head(x)

        return x


def _dbnet(backbone_type, backbone_pretrained_path, **kwargs):
    model = DBNet(backbone_type=backbone_type,
                  backbone_pretrained_path=backbone_pretrained_path,
                  **kwargs)

    return model


def resnet18_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('resnet18backbone', backbone_pretrained_path, **kwargs)


def resnet34_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('resnet34backbone', backbone_pretrained_path, **kwargs)


def resnet50_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('resnet50backbone', backbone_pretrained_path, **kwargs)


def resnet101_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('resnet101backbone', backbone_pretrained_path, **kwargs)


def resnet152_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('resnet152backbone', backbone_pretrained_path, **kwargs)


def vanb0_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('vanb0backbone', backbone_pretrained_path, **kwargs)


def vanb1_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('vanb1backbone', backbone_pretrained_path, **kwargs)


def vanb2_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('vanb2backbone', backbone_pretrained_path, **kwargs)


def vanb3_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('vanb3backbone', backbone_pretrained_path, **kwargs)


def convformers18_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('convformers18backbone', backbone_pretrained_path, **kwargs)


def convformers36_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('convformers36backbone', backbone_pretrained_path, **kwargs)


def convformerm36_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('convformerm36backbone', backbone_pretrained_path, **kwargs)


def convformerb36_dbnet(backbone_pretrained_path='', **kwargs):
    return _dbnet('convformerb36backbone', backbone_pretrained_path, **kwargs)


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

    net = resnet18_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = resnet34_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = resnet50_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = resnet101_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = resnet152_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = vanb0_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = vanb1_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = vanb2_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = vanb3_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = vanb3_dbnet(use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    outs = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', outs.shape)

    net = convformers18_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = convformers36_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = convformerm36_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)

    net = convformerb36_dbnet()
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    print('2222', outs.shape)
