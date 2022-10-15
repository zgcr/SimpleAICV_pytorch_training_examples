import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.fpn import RetinaFPN, VitDetFPN
from simpleAICV.detection.models.head import FCOSClsRegCntHead

__all__ = [
    'resnet18_fcos',
    'resnet34_fcos',
    'resnet50_fcos',
    'resnet101_fcos',
    'resnet152_fcos',
    'vit_base_backbone_patch16_fcos',
]


class FCOS(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 num_classes=80):
        super(FCOS, self).__init__()
        self.planes = planes
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })
        self.fpn = RetinaFPN(self.backbone.out_channels,
                             self.planes,
                             use_p5=True)
        self.clsregcnt_head = FCOSClsRegCntHead(self.planes,
                                                self.num_classes,
                                                num_layers=4,
                                                use_gn=True,
                                                cnt_on_reg=True)
        self.scales = nn.Parameter(
            torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32))

    def forward(self, inputs):
        features = self.backbone(inputs)

        del inputs

        features = self.fpn(features)

        cls_heads, reg_heads, center_heads = [], [], []
        for feature, scale in zip(features, self.scales):
            cls_outs, reg_outs, center_outs = self.clsregcnt_head(feature)

            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_outs = cls_outs.permute(0, 2, 3, 1).contiguous()
            cls_heads.append(cls_outs)
            # [N,4,H,W] -> [N,H,W,4]
            reg_outs = reg_outs.permute(0, 2, 3, 1).contiguous()
            reg_outs = reg_outs * torch.exp(scale)
            reg_heads.append(reg_outs)
            # [N,1,H,W] -> [N,H,W,1]
            center_outs = center_outs.permute(0, 2, 3, 1).contiguous()
            center_heads.append(center_outs)

        del features

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        return [cls_heads, reg_heads, center_heads]


def _fcos(backbone_type, backbone_pretrained_path, **kwargs):
    model = FCOS(backbone_type,
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)

    return model


def resnet18_fcos(backbone_pretrained_path='', **kwargs):
    return _fcos('resnet18backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet34_fcos(backbone_pretrained_path='', **kwargs):
    return _fcos('resnet34backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet50_fcos(backbone_pretrained_path='', **kwargs):
    return _fcos('resnet50backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet101_fcos(backbone_pretrained_path='', **kwargs):
    return _fcos('resnet101backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet152_fcos(backbone_pretrained_path='', **kwargs):
    return _fcos('resnet152backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


class VitDetFCOS(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 image_size=512,
                 planes=256,
                 num_classes=80):
        super(VitDetFCOS, self).__init__()
        self.planes = planes
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'image_size': image_size,
                'pretrained_path': backbone_pretrained_path,
            })
        self.fpn = VitDetFPN(self.backbone.out_channels, planes)
        self.clsregcnt_head = FCOSClsRegCntHead(self.planes,
                                                self.num_classes,
                                                num_layers=4,
                                                use_gn=True,
                                                cnt_on_reg=True)
        self.scales = nn.Parameter(
            torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32))

    def forward(self, inputs):
        features = self.backbone(inputs)

        del inputs

        features = self.fpn(features)

        cls_heads, reg_heads, center_heads = [], [], []
        for feature, scale in zip(features, self.scales):
            cls_outs, reg_outs, center_outs = self.clsregcnt_head(feature)

            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_outs = cls_outs.permute(0, 2, 3, 1).contiguous()
            cls_heads.append(cls_outs)
            # [N,4,H,W] -> [N,H,W,4]
            reg_outs = reg_outs.permute(0, 2, 3, 1).contiguous()
            reg_outs = reg_outs * torch.exp(scale)
            reg_heads.append(reg_outs)
            # [N,1,H,W] -> [N,H,W,1]
            center_outs = center_outs.permute(0, 2, 3, 1).contiguous()
            center_heads.append(center_outs)

        del features

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        return [cls_heads, reg_heads, center_heads]


def _vitdetfcos(backbone_type, backbone_pretrained_path, **kwargs):
    model = VitDetFCOS(backbone_type,
                       backbone_pretrained_path=backbone_pretrained_path,
                       **kwargs)

    return model


def vit_base_backbone_patch16_fcos(backbone_pretrained_path='', **kwargs):
    return _vitdetfcos('vit_base_backbone_patch16',
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

    net = resnet18_fcos()
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

    net = resnet50_fcos()
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

    net = vit_base_backbone_patch16_fcos(image_size=512)
    image_h, image_w = 512, 512
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