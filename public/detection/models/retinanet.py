import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.backbone import ResNetBackbone
from public.detection.models.fpn import RetinaFPN
from public.detection.models.head import RetinaClsHead, RetinaRegHead
from public.detection.models.anchor import RetinaAnchors

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_retinanet',
    'resnet34_retinanet',
    'resnet50_retinanet',
    'resnet101_retinanet',
    'resnet152_retinanet',
]

model_urls = {
    'resnet18_retinanet':
    'empty',
    'resnet34_retinanet':
    'empty',
    'resnet50_retinanet':
    '{}/detection_models/resnet50_retinanet-epoch30-coco-mAP0.279.pth'.format(
        pretrained_models_path),
    'resnet101_retinanet':
    'empty',
    'resnet152_retinanet':
    'empty',
}


# assert input annotations are[x_min,y_min,x_max,y_max]
class RetinaNet(nn.Module):
    def __init__(self, resnet_type, num_anchors=9, num_classes=80, planes=256):
        super(RetinaNet, self).__init__()
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

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.planes = planes

        self.cls_head = RetinaClsHead(self.planes,
                                      self.num_anchors,
                                      self.num_classes,
                                      num_layers=4,
                                      prior=0.01)

        self.reg_head = RetinaRegHead(self.planes,
                                      self.num_anchors,
                                      num_layers=4)

        self.areas = torch.tensor([[32, 32], [64, 64], [128, 128], [256, 256],
                                   [512, 512]])
        self.ratios = torch.tensor([0.5, 1, 2])
        self.scales = torch.tensor([2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)])
        self.strides = torch.tensor([8, 16, 32, 64, 128], dtype=torch.float)

        self.anchors = RetinaAnchors(self.areas, self.ratios, self.scales,
                                     self.strides)

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device

        [C3, C4, C5] = self.backbone(inputs)

        del inputs

        features = self.fpn([C3, C4, C5])

        del C3, C4, C5

        self.fpn_feature_sizes = []
        cls_heads, reg_heads = [], []
        for feature in features:
            self.fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])
            cls_head = self.cls_head(feature)
            # [N,9*num_classes,H,W] -> [N,H*W*9,num_classes]
            cls_head = cls_head.permute(0, 2, 3, 1).contiguous().view(
                self.batch_size, -1, self.num_classes)
            cls_heads.append(cls_head)

            reg_head = self.reg_head(feature)
            # [N, 9*4,H,W] -> [N,H*W*9, 4]
            reg_head = reg_head.permute(0, 2, 3, 1).contiguous().view(
                self.batch_size, -1, 4)
            reg_heads.append(reg_head)

        del features

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 57600, 80],[B, 14400, 80],[B, 3600, 80],[B, 900, 80],[B, 225, 80]]
        # reg_heads shape:[[B, 57600, 4],[B, 14400, 4],[B, 3600, 4],[B, 900, 4],[B, 225, 4]]
        # batch_anchors shape:[[B, 57600, 4],[B, 14400, 4],[B, 3600, 4],[B, 900, 4],[B, 225, 4]]

        batch_anchors = self.anchors(self.batch_size, self.fpn_feature_sizes)

        return cls_heads, reg_heads, batch_anchors


def _retinanet(arch, pretrained, progress, **kwargs):
    model = RetinaNet(arch, **kwargs)

    if pretrained:
        pretrained_models = torch.load(model_urls[arch + "_retinanet"],
                                       map_location=torch.device('cpu'))

        # del pretrained_models['cls_head.cls_head.8.weight']
        # del pretrained_models['cls_head.cls_head.8.bias']
        # del pretrained_models['reg_head.reg_head.8.weight']
        # del pretrained_models['reg_head.reg_head.8.bias']

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def resnet18_retinanet(pretrained=False, progress=True, **kwargs):
    return _retinanet('resnet18', pretrained, progress, **kwargs)


def resnet34_retinanet(pretrained=False, progress=True, **kwargs):
    return _retinanet('resnet34', pretrained, progress, **kwargs)


def resnet50_retinanet(pretrained=False, progress=True, **kwargs):
    return _retinanet('resnet50', pretrained, progress, **kwargs)


def resnet101_retinanet(pretrained=False, progress=True, **kwargs):
    return _retinanet('resnet101', pretrained, progress, **kwargs)


def resnet152_retinanet(pretrained=False, progress=True, **kwargs):
    return _retinanet('resnet152', pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = RetinaNet(resnet_type="resnet50")
    image_h, image_w = 600, 600
    cls_heads, reg_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])