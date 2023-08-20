import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.head import TTFHetWhHead

__all__ = [
    'resnet18_ttfnet',
    'resnet34_ttfnet',
    'resnet50_ttfnet',
    'resnet101_ttfnet',
    'resnet152_ttfnet',
]


class TTFNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=[256, 128, 64],
                 num_classes=80):
        super(TTFNet, self).__init__()
        assert len(planes) == 3

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })
        self.ttf_head = TTFHetWhHead(self.backbone.out_channels,
                                     num_classes,
                                     planes=planes,
                                     short_cut_layers_num=[1, 2],
                                     num_layers=3)

    def forward(self, inputs):
        [C3, C4, C5] = self.backbone(inputs)

        del inputs

        heatmap_output, wh_output = self.ttf_head([C3, C4, C5])

        del C5

        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # wh_output shape:[3, 4, 160, 160]
        return [heatmap_output, wh_output]


def _ttfnet(backbone_type, backbone_pretrained_path, **kwargs):
    model = TTFNet(backbone_type,
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)

    return model


def resnet18_ttfnet(backbone_pretrained_path='', **kwargs):
    return _ttfnet('resnet18backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet34_ttfnet(backbone_pretrained_path='', **kwargs):
    return _ttfnet('resnet34backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet50_ttfnet(backbone_pretrained_path='', **kwargs):
    return _ttfnet('resnet50backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet101_ttfnet(backbone_pretrained_path='', **kwargs):
    return _ttfnet('resnet101backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet152_ttfnet(backbone_pretrained_path='', **kwargs):
    return _ttfnet('resnet152backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
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

    net = resnet18_ttfnet()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet50_ttfnet()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)