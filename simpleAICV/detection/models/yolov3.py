import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.fpn import Yolov3TinyFPNHead, Yolov3FPNHead

__all__ = [
    'darknettiny_yolov3',
    'darknet19_yolov3',
    'darknet53_yolov3',
]


class YOLOV3(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 act_type='leakyrelu',
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YOLOV3, self).__init__()
        assert backbone_type in [
            'darknettinybackbone', 'darknet19backbone', 'darknet53backbone'
        ]
        self.per_level_num_anchors = per_level_num_anchors
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type](**{
            'pretrained_path': backbone_pretrained_path,
            'act_type': act_type,
        })

        if backbone_type == 'darknettinybackbone':
            self.fpn = Yolov3TinyFPNHead(
                self.backbone.out_channels,
                per_level_num_anchors=self.per_level_num_anchors,
                num_classes=self.num_classes,
                act_type=act_type)
        elif backbone_type in ['darknet19backbone', 'darknet53backbone']:
            self.fpn = Yolov3FPNHead(
                self.backbone.out_channels,
                per_level_num_anchors=self.per_level_num_anchors,
                num_classes=self.num_classes,
                act_type=act_type)

    def forward(self, inputs):
        features = self.backbone(inputs)

        del inputs

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


def _yolov3(backbone_type, backbone_pretrained_path, **kwargs):
    model = YOLOV3(backbone_type,
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)

    return model


def darknettiny_yolov3(backbone_pretrained_path='', **kwargs):
    return _yolov3('darknettinybackbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def darknet19_yolov3(backbone_pretrained_path='', **kwargs):
    return _yolov3('darknet19backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def darknet53_yolov3(backbone_pretrained_path='', **kwargs):
    return _yolov3('darknet53backbone',
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

    net = darknettiny_yolov3()
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

    net = darknet53_yolov3()
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