import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.backbone import EfficientNetBackbone
from public.detection.models.fpn import EfficientDetBiFPN
from public.detection.models.head import EfficientDetClsHead, EfficientDetRegHead
from public.detection.models.anchor import RetinaAnchors

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'efficientdet_d0',
    'efficientdet_d1',
    'efficientdet_d2',
    'efficientdet_d3',
    'efficientdet_d4',
    'efficientdet_d5',
    'efficientdet_d6',
    'efficientdet_d7',
]

model_urls = {
    'efficientdet_d0': 'empty',
    'efficientdet_d1': 'empty',
    'efficientdet_d2': 'empty',
    'efficientdet_d3': 'empty',
    'efficientdet_d4': 'empty',
    'efficientdet_d5': 'empty',
    'efficientdet_d6': 'empty',
    'efficientdet_d7': 'empty',
}

efficientdet_types_config = {
    'efficientdet_d0': {
        'efficientnet_type': 'efficientnet_b0',
        'resolution': 512,
        'fpn_channel_nums': 64,
        'fpn_cell_repeats': 3,
        'head_layer_nums': 3,
        'fpn_inplanes': [40, 112, 320],
    },
    'efficientdet_d1': {
        'efficientnet_type': 'efficientnet_b1',
        'resolution': 640,
        'fpn_channel_nums': 88,
        'fpn_cell_repeats': 4,
        'head_layer_nums': 3,
        'fpn_inplanes': [40, 112, 320],
    },
    'efficientdet_d2': {
        'efficientnet_type': 'efficientnet_b2',
        'resolution': 768,
        'fpn_channel_nums': 112,
        'fpn_cell_repeats': 5,
        'head_layer_nums': 3,
        'fpn_inplanes': [48, 120, 352],
    },
    'efficientdet_d3': {
        'efficientnet_type': 'efficientnet_b3',
        'resolution': 896,
        'fpn_channel_nums': 160,
        'fpn_cell_repeats': 6,
        'head_layer_nums': 4,
        'fpn_inplanes': [48, 136, 384],
    },
    'efficientdet_d4': {
        'efficientnet_type': 'efficientnet_b4',
        'resolution': 1024,
        'fpn_channel_nums': 224,
        'fpn_cell_repeats': 7,
        'head_layer_nums': 4,
        'fpn_inplanes': [56, 160, 448],
    },
    'efficientdet_d5': {
        'efficientnet_type': 'efficientnet_b5',
        'resolution': 1280,
        'fpn_channel_nums': 288,
        'fpn_cell_repeats': 7,
        'head_layer_nums': 4,
        'fpn_inplanes': [64, 176, 512],
    },
    'efficientdet_d6': {
        'efficientnet_type': 'efficientnet_b6',
        'resolution': 1280,
        'fpn_channel_nums': 384,
        'fpn_cell_repeats': 8,
        'head_layer_nums': 5,
        'fpn_inplanes': [72, 200, 576],
    },
    'efficientdet_d7': {
        'efficientnet_type': 'efficientnet_b7',
        'resolution': 1536,
        'fpn_channel_nums': 384,
        'fpn_cell_repeats': 8,
        'head_layer_nums': 5,
        'fpn_inplanes': [72, 200, 576],
    },
}


class EfficientDet(nn.Module):
    def __init__(self, efficientdet_type, num_anchors=9, num_classes=80):
        super(EfficientDet, self).__init__()
        self.efficientdet_superparams = efficientdet_types_config[
            efficientdet_type]
        self.backbone = EfficientNetBackbone(
            efficientnet_type=self.
            efficientdet_superparams['efficientnet_type'])
        self.fpn_channel_nums = self.efficientdet_superparams[
            'fpn_channel_nums']
        self.fpn_cell_repeats = self.efficientdet_superparams[
            'fpn_cell_repeats']
        self.head_layer_nums = self.efficientdet_superparams['head_layer_nums']
        self.fpn_inplanes = self.efficientdet_superparams['fpn_inplanes']

        self.fpn = nn.Sequential(*[
            EfficientDetBiFPN(self.fpn_inplanes[0], self.fpn_inplanes[1],
                              self.fpn_inplanes[2], self.fpn_channel_nums,
                              True if _ == 0 else False)
            for _ in range(self.fpn_cell_repeats)
        ])

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.cls_head = EfficientDetClsHead(inplanes=self.fpn_channel_nums,
                                            num_anchors=self.num_anchors,
                                            num_classes=self.num_classes,
                                            num_layers=self.head_layer_nums)
        self.reg_head = EfficientDetRegHead(inplanes=self.fpn_channel_nums,
                                            num_anchors=self.num_anchors,
                                            num_layers=self.head_layer_nums)

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


def _efficientdet(arch, pretrained, **kwargs):
    model = EfficientDet(arch, **kwargs)

    if pretrained:
        pretrained_models = torch.load(
            model_urls[efficientdet_types_config[arch]['efficientnet_type']],
            map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def efficientdet_d0(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d0', pretrained, **kwargs)


def efficientdet_d1(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d1', pretrained, **kwargs)


def efficientdet_d2(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d2', pretrained, **kwargs)


def efficientdet_d3(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d3', pretrained, **kwargs)


def efficientdet_d4(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d4', pretrained, **kwargs)


def efficientdet_d5(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d5', pretrained, **kwargs)


def efficientdet_d6(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d6', pretrained, **kwargs)


def efficientdet_d7(pretrained=False, **kwargs):
    return _efficientdet('efficientdet_d7', pretrained, **kwargs)


if __name__ == '__main__':
    net = EfficientDet(efficientdet_type='efficientdet_d0')
    image_h, image_w = 640, 640
    cls_heads, reg_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])