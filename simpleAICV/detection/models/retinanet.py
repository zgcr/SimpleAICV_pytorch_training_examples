import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.fpn import RetinaFPN
from simpleAICV.detection.models.head import RetinaClsHead, RetinaRegHead

__all__ = [
    'resnet18_retinanet',
    'resnet34_retinanet',
    'resnet50_retinanet',
    'resnet101_retinanet',
    'resnet152_retinanet',
]


class RetinaNet(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 num_anchors=9,
                 num_classes=80):
        super(RetinaNet, self).__init__()
        self.planes = planes
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })
        self.fpn = RetinaFPN(self.backbone.out_channels,
                             self.planes,
                             use_p5=False)
        self.cls_head = RetinaClsHead(self.planes,
                                      self.num_anchors,
                                      self.num_classes,
                                      num_layers=4)
        self.reg_head = RetinaRegHead(self.planes,
                                      self.num_anchors,
                                      num_layers=4)

    def forward(self, inputs):
        features = self.backbone(inputs)

        del inputs

        features = self.fpn(features)

        cls_heads, reg_heads = [], []
        for feature in features:
            cls_head = self.cls_head(feature)
            # [N,9*num_classes,H,W] -> [N,H,W,9*num_classes] -> [N,H,W,9,num_classes]
            cls_head = cls_head.permute(0, 2, 3, 1).contiguous()
            cls_head = cls_head.view(cls_head.shape[0], cls_head.shape[1],
                                     cls_head.shape[2], -1, self.num_classes)
            cls_heads.append(cls_head)

            reg_head = self.reg_head(feature)
            # [N, 9*4,H,W] -> [N,H,W,9*4] -> [N,H,W,9,4]
            reg_head = reg_head.permute(0, 2, 3, 1).contiguous()
            reg_head = reg_head.view(reg_head.shape[0], reg_head.shape[1],
                                     reg_head.shape[2], -1, 4)
            reg_heads.append(reg_head)

        del features

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
        # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
        return [cls_heads, reg_heads]


def _retinanet(backbone_type, backbone_pretrained_path, **kwargs):
    model = RetinaNet(backbone_type,
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)

    return model


def resnet18_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet18backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet34_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet34backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet50_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet50backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet101_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet101backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet152_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet152backbone',
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

    net = resnet50_retinanet()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = resnet50_retinanet()
    image_h, image_w = 800, 800
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = resnet50_retinanet()
    image_h, image_w = 800, 1333
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)