import os
import sys
import math
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.backbone import ResNetBackbone
from public.detection.models.fpn import RetinaFPN
from public.detection.models.head import FCOSClsHead, FCOSRegCenterHead

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_fcos',
    'resnet34_fcos',
    'resnet50_fcos',
    'resnet101_fcos',
    'resnet152_fcos',
]

model_urls = {
    'resnet18_fcos': 'empty',
    'resnet34_fcos': 'empty',
    'resnet50_fcos': 'empty',
    'resnet101_fcos': 'empty',
    'resnet152_fcos': 'empty',
}


# assert input annotations are[x_min,y_min,x_max,y_max]
class FCOS(nn.Module):
    def __init__(self, resnet_type, num_classes=80, planes=256):
        super(FCOS, self).__init__()
        self.backbone = ResNetBackbone(resnet_type=resnet_type)
        expand_ratio = {
            "resnet18": 1,
            "resnet34": 1,
            "resnet50": 4,
            "resnet101": 4,
            "resnet152": 4
        }
        C3_inplanes, C4_inplanes, C5_inplanes = int(
            128 * expand_ratio[resnet_type]), int(
                256 * expand_ratio[resnet_type]), int(
                    512 * expand_ratio[resnet_type])
        self.fpn = RetinaFPN(C3_inplanes, C4_inplanes, C5_inplanes, planes)

        self.num_classes = num_classes
        self.planes = planes

        self.cls_head = FCOSClsHead(self.planes,
                                    self.num_classes,
                                    num_layers=4,
                                    prior=0.01)
        self.regcenter_head = FCOSRegCenterHead(self.planes, num_layers=4)

    def forward(self, inputs):
        [C3, C4, C5] = self.backbone(inputs)

        del inputs

        features = self.fpn([C3, C4, C5])

        del C3, C4, C5

        cls_heads, reg_heads, center_heads = [], [], []
        for feature in features:
            cls_outs = self.cls_head(feature)
            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_outs = cls_outs.permute(0, 2, 3, 1).contiguous()
            cls_heads.append(cls_outs)

            reg_outs, center_outs = self.regcenter_head(feature)
            # [N,4,H,W] -> [N,H,W,4]
            reg_outs = reg_outs.permute(0, 2, 3, 1).contiguous()
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

        return cls_heads, reg_heads, center_heads


def _fcos(arch, pretrained, progress, **kwargs):
    model = FCOS(arch, **kwargs)
    # only load state_dict()
    if pretrained:
        pretrained_models = torch.load(model_urls[arch + "_fcos"],
                                       map_location=torch.device('cpu'))
        # del pretrained_models['cls_head.cls_head.8.weight']
        # del pretrained_models['cls_head.cls_head.8.bias']
        # del pretrained_models['reg_head.reg_head.8.weight']
        # del pretrained_models['reg_head.reg_head.8.bias']

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def resnet18_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet18', pretrained, progress, **kwargs)


def resnet34_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet34', pretrained, progress, **kwargs)


def resnet50_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet50', pretrained, progress, **kwargs)


def resnet101_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet101', pretrained, progress, **kwargs)


def resnet152_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet152', pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = FCOS(resnet_type="resnet50")
    image_h, image_w = 600, 600
    cls_heads, reg_heads, center_heads = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])