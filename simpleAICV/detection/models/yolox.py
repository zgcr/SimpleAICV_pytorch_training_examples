'''
yolox official code
https://github.com/Megvii-BaseDetection/YOLOX
'''
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.detection.models.backbones.yoloxbackbone import ConvBnActBlock, DWConvBnActBlock
from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.fpn import YOLOXFPN
from simpleAICV.detection.models.head import YOLOXHead

__all__ = [
    'yoloxn',
    'yoloxt',
    'yoloxs',
    'yoloxm',
    'yoloxl',
    'yoloxx',
]


class YOLOX(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 act_type='silu',
                 planes=256,
                 num_classes=80):
        super(YOLOX, self).__init__()
        assert backbone_type in [
            'yoloxnbackbone',
            'yoloxtbackbone',
            'yoloxsbackbone',
            'yoloxmbackbone',
            'yoloxlbackbone',
            'yoloxxbackbone',
        ]
        self.backbone = backbones.__dict__[backbone_type](**{
            'pretrained_path': backbone_pretrained_path,
            'act_type': act_type,
        })
        self.block_type = DWConvBnActBlock if backbone_type == 'yoloxnbackbone' else ConvBnActBlock
        self.fpn = YOLOXFPN(self.backbone.out_channels,
                            csp_nums=self.backbone.csp_nums[-1],
                            csp_shortcut=False,
                            block=self.block_type,
                            act_type=act_type)
        self.head = YOLOXHead(self.backbone.out_channels,
                              planes=planes,
                              num_classes=num_classes,
                              block=self.block_type,
                              act_type=act_type)

    def forward(self, x):
        features = self.backbone(x)
        features = self.fpn(features)

        cls_outputs, reg_outputs, obj_outputs = self.head(features)

        del features

        obj_preds, cls_preds, reg_preds = [], [], []
        for cls_output, reg_output, obj_output in zip(cls_outputs, reg_outputs,
                                                      obj_outputs):
            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_output = cls_output.permute(0, 2, 3, 1).contiguous()
            # [N,4,H,W] -> [N,H,W,4]
            reg_output = reg_output.permute(0, 2, 3, 1).contiguous()
            # [N,1,H,W] -> [N,H,W,1]
            obj_output = obj_output.permute(0, 2, 3, 1).contiguous()

            cls_preds.append(cls_output)
            reg_preds.append(reg_output)
            obj_preds.append(obj_output)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # obj_preds shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        # cls_preds shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_preds shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        return [cls_preds, reg_preds, obj_preds]


def _yolox(backbone_type, backbone_pretrained_path, **kwargs):
    model = YOLOX(backbone_type,
                  backbone_pretrained_path=backbone_pretrained_path,
                  **kwargs)

    return model


def yoloxn(backbone_pretrained_path='', **kwargs):
    return _yolox('yoloxnbackbone',
                  backbone_pretrained_path=backbone_pretrained_path,
                  **kwargs)


def yoloxt(backbone_pretrained_path='', **kwargs):
    return _yolox('yoloxtbackbone',
                  backbone_pretrained_path=backbone_pretrained_path,
                  **kwargs)


def yoloxs(backbone_pretrained_path='', **kwargs):
    return _yolox('yoloxsbackbone',
                  backbone_pretrained_path=backbone_pretrained_path,
                  **kwargs)


def yoloxm(backbone_pretrained_path='', **kwargs):
    return _yolox('yoloxmbackbone',
                  backbone_pretrained_path=backbone_pretrained_path,
                  **kwargs)


def yoloxl(backbone_pretrained_path='', **kwargs):
    return _yolox('yoloxlbackbone',
                  backbone_pretrained_path=backbone_pretrained_path,
                  **kwargs)


def yoloxx(backbone_pretrained_path='', **kwargs):
    return _yolox('yoloxxbackbone',
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

    net = yoloxn()
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

    net = yoloxt()
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

    net = yoloxs()
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

    net = yoloxm()
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

    net = yoloxl()
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

    net = yoloxx()
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