import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.backbone import ResNetBackbone
from public.detection.models.head import CenterNetHetRegWhHead

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_centernet',
    'resnet34_centernet',
    'resnet50_centernet',
    'resnet101_centernet',
    'resnet152_centernet',
]

model_urls = {
    'resnet18_centernet':
    '{}/detection_models/resnet18dcn_centernet_coco_multi_scale_resize512_mAP0.266.pth'
    .format(pretrained_models_path),
    'resnet34_centernet':
    'empty',
    'resnet50_centernet':
    'empty',
    'resnet101_centernet':
    'empty',
    'resnet152_centernet':
    'empty',
}


# assert input annotations are[x_min,y_min,x_max,y_max]
class CenterNet(nn.Module):
    def __init__(self, resnet_type, num_classes=80):
        super(CenterNet, self).__init__()
        self.backbone = ResNetBackbone(resnet_type=resnet_type)
        expand_ratio = {
            "resnet18": 1,
            "resnet34": 1,
            "resnet50": 4,
            "resnet101": 4,
            "resnet152": 4
        }
        C5_inplanes = int(512 * expand_ratio[resnet_type])

        self.centernet_head = CenterNetHetRegWhHead(
            C5_inplanes,
            num_classes,
            num_layers=3,
            out_channels=[256, 128, 64])

    def forward(self, inputs):
        [C3, C4, C5] = self.backbone(inputs)

        del inputs, C3, C4

        heatmap_output, offset_output, wh_output = self.centernet_head(C5)

        del C5

        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # offset_output shape:[3, 2, 160, 160]
        # wh_output shape:[3, 2, 160, 160]

        return heatmap_output, offset_output, wh_output


def _centernet(arch, pretrained, **kwargs):
    model = CenterNet(arch, **kwargs)

    if pretrained:
        pretrained_models = torch.load(model_urls[arch + "_centernet"],
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def resnet18_centernet(pretrained=False, **kwargs):
    return _centernet('resnet18', pretrained, **kwargs)


def resnet34_centernet(pretrained=False, **kwargs):
    return _centernet('resnet34', pretrained, **kwargs)


def resnet50_centernet(pretrained=False, **kwargs):
    return _centernet('resnet50', pretrained, **kwargs)


def resnet101_centernet(pretrained=False, **kwargs):
    return _centernet('resnet101', pretrained, **kwargs)


def resnet152_centernet(pretrained=False, **kwargs):
    return _centernet('resnet152', pretrained, **kwargs)


if __name__ == '__main__':
    net = CenterNet(resnet_type="resnet18")
    image_h, image_w = 600, 600
    heatmap_output, offset_output, wh_output = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    print("1111", heatmap_output.shape, offset_output.shape, wh_output.shape)
