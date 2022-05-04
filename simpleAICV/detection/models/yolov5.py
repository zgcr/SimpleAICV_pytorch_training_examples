'''
yolov5 official code
https://github.com/ultralytics/yolov5
'''
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.fpn import YOLOV5FPNHead

__all__ = [
    'yolov5n',
    'yolov5s',
    'yolov5m',
    'yolov5l',
    'yolov5x',
]


class YOLOV5(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 act_type='silu',
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YOLOV5, self).__init__()
        assert backbone_type in [
            'yolov5nbackbone',
            'yolov5sbackbone',
            'yolov5mbackbone',
            'yolov5lbackbone',
            'yolov5xbackbone',
        ]
        self.per_level_num_anchors = per_level_num_anchors
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type](**{
            'pretrained_path': backbone_pretrained_path,
            'act_type': act_type,
        })
        self.fpn = YOLOV5FPNHead(
            self.backbone.out_channels,
            csp_nums=self.backbone.csp_nums[-1],
            csp_shortcut=False,
            per_level_num_anchors=self.per_level_num_anchors,
            num_classes=self.num_classes,
            act_type=act_type)

    def forward(self, x):
        features = self.backbone(x)
        features = self.fpn(features)

        obj_reg_cls_heads = []
        for feature in features:
            # feature shape:[B,H,W,3,85]

            # obj_head:feature[:, :, :, :, 0:1], shape:[B,H,W,3,1]
            # reg_head:feature[:, :, :, :, 1:5], shape:[B,H,W,3,4]
            # cls_head:feature[:, :, :, :, 5:],  shape:[B,H,W,3,80]
            obj_reg_cls_heads.append(feature)

        del features

        # if input size:[B,3,416,416]
        # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
        # obj_reg_cls_heads shape:[[B, 52, 52, 3, 85],[B, 26, 26, 3, 85],[B, 13, 13, 3, 85]]
        return [obj_reg_cls_heads]


def _yolov5(backbone_type, backbone_pretrained_path, **kwargs):
    model = YOLOV5(backbone_type,
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)

    return model


def yolov5n(backbone_pretrained_path='', **kwargs):
    return _yolov5('yolov5nbackbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def yolov5s(backbone_pretrained_path='', **kwargs):
    return _yolov5('yolov5sbackbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def yolov5m(backbone_pretrained_path='', **kwargs):
    return _yolov5('yolov5mbackbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def yolov5l(backbone_pretrained_path='', **kwargs):
    return _yolov5('yolov5lbackbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def yolov5x(backbone_pretrained_path='', **kwargs):
    return _yolov5('yolov5xbackbone',
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

    net = yolov5n()
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
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = yolov5s()
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
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = yolov5m()
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
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = yolov5l()
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
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = yolov5x()
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
        for per_level_out in out:
            print('2222', per_level_out.shape)
