import os
import sys
import math
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import pretrained_models_path

from simpleAICV.detection.common import load_state_dict
from simpleAICV.detection.models.backbone import ResNetBackbone
from simpleAICV.detection.models.fpn import RetinaFPN
from simpleAICV.detection.models.head import FCOSClsRegCntHead
from simpleAICV.detection.models.anchor import FCOSPositions

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
        self.backbone = ResNetBackbone(resnet_type=resnet_type,
                                       pretrained=True)
        expand_ratio = {
            'resnet18': 1,
            'resnet34': 1,
            'resnet50': 4,
            'resnet101': 4,
            'resnet152': 4,
        }
        C3_inplanes, C4_inplanes, C5_inplanes = int(
            128 * expand_ratio[resnet_type]), int(
                256 * expand_ratio[resnet_type]), int(
                    512 * expand_ratio[resnet_type])
        self.fpn = RetinaFPN(C3_inplanes,
                             C4_inplanes,
                             C5_inplanes,
                             planes,
                             use_p5=True)

        self.num_classes = num_classes
        self.planes = planes

        self.clsregcnt_head = FCOSClsRegCntHead(self.planes,
                                                self.num_classes,
                                                num_layers=4,
                                                prior=0.01,
                                                use_gn=True,
                                                cnt_on_reg=True)

        self.strides = torch.tensor([8, 16, 32, 64, 128], dtype=torch.float)
        self.positions = FCOSPositions(self.strides)

        self.scales = nn.Parameter(
            torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32))

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device
        [C3, C4, C5] = self.backbone(inputs)

        del inputs

        features = self.fpn([C3, C4, C5])

        del C3, C4, C5

        self.fpn_feature_sizes = []
        cls_heads, reg_heads, center_heads = [], [], []
        for feature, scale in zip(features, self.scales):
            self.fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])

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

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        batch_positions = self.positions(self.batch_size,
                                         self.fpn_feature_sizes)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]

        return cls_heads, reg_heads, center_heads, batch_positions


def _fcos(arch, pretrained, **kwargs):
    model = FCOS(arch, **kwargs)

    if pretrained:
        load_state_dict(
            torch.load(model_urls[arch + '_fcos'],
                       map_location=torch.device('cpu')), model)

    return model


def resnet18_fcos(pretrained=False, **kwargs):
    return _fcos('resnet18', pretrained, **kwargs)


def resnet34_fcos(pretrained=False, **kwargs):
    return _fcos('resnet34', pretrained, **kwargs)


def resnet50_fcos(pretrained=False, **kwargs):
    return _fcos('resnet50', pretrained, **kwargs)


def resnet101_fcos(pretrained=False, **kwargs):
    return _fcos('resnet101', pretrained, **kwargs)


def resnet152_fcos(pretrained=False, **kwargs):
    return _fcos('resnet152', pretrained, **kwargs)


if __name__ == '__main__':
    net = FCOS(resnet_type='resnet50')
    image_h, image_w = 640, 640
    cls_heads, reg_heads, center_heads, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    for x, y, z, m in zip(cls_heads, reg_heads, center_heads, batch_positions):
        print('1111', x.shape, y.shape, z.shape, m.shape)

    from thop import profile
    from thop import clever_format
    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    print(f"2222, flops: {flops}, params: {params}")