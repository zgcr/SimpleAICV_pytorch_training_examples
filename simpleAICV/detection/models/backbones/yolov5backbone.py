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

from simpleAICV.classification.backbones.yolov5backbone import ConvBnActBlock, CSPBottleneck, SPPF
from simpleAICV.detection.common import load_state_dict

__all__ = [
    'yolov5nbackbone',
    'yolov5sbackbone',
    'yolov5mbackbone',
    'yolov5lbackbone',
    'yolov5xbackbone',
]

types_config = {
    'yolov5nbackbone': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.25,
    },
    'yolov5sbackbone': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.50,
    },
    'yolov5mbackbone': {
        'depth_coefficient': 0.67,
        'width_coefficient': 0.75,
    },
    'yolov5lbackbone': {
        'depth_coefficient': 1.0,
        'width_coefficient': 1.0,
    },
    'yolov5xbackbone': {
        'depth_coefficient': 1.33,
        'width_coefficient': 1.25,
    },
}


class Yolov5Backbone(nn.Module):

    def __init__(self,
                 yolo_backbone_type,
                 planes=[64, 128, 256, 512, 1024],
                 csp_nums=[3, 6, 9, 3],
                 csp_shortcut=[True, True, True, True],
                 act_type='silu'):
        super(Yolov5Backbone, self).__init__()
        depth_scale = types_config[yolo_backbone_type]['depth_coefficient']
        width_scale = types_config[yolo_backbone_type]['width_coefficient']

        self.planes = [self.compute_width(num, width_scale) for num in planes]
        self.csp_nums = [
            self.compute_depth(num, depth_scale) for num in csp_nums
        ]

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
        x = self.conv(x)

        out = []
        for i, layer in enumerate(self.middle_layers):
            x = layer(x)
            if i % 2 == 1 and i > 2 and i < 6:
                out.append(x)

        x = self.sppf(x)
        out.append(x)

        return out

    def compute_depth(self, depth, scale):
        return max(round(depth * scale), 1) if depth > 1 else depth

    def compute_width(self, width, scale, divisor=8):
        return math.ceil((width * scale) / divisor) * divisor


def _yolov5backbone(yolo_backbone_type, pretrained_path='', **kwargs):
    model = Yolov5Backbone(yolo_backbone_type, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def yolov5nbackbone(pretrained_path='', **kwargs):
    model = _yolov5backbone('yolov5nbackbone',
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def yolov5sbackbone(pretrained_path='', **kwargs):
    model = _yolov5backbone('yolov5sbackbone',
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def yolov5mbackbone(pretrained_path='', **kwargs):
    model = _yolov5backbone('yolov5mbackbone',
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def yolov5lbackbone(pretrained_path='', **kwargs):
    model = _yolov5backbone('yolov5lbackbone',
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def yolov5xbackbone(pretrained_path='', **kwargs):
    model = _yolov5backbone('yolov5xbackbone',
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

    net = yolov5nbackbone()
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

    net = yolov5sbackbone()
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

    net = yolov5mbackbone()
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

    net = yolov5lbackbone()
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

    net = yolov5xbackbone()
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