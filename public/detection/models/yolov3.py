import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.backbone import Darknet53Backbone
from public.detection.models.fpn import YOLOV3FPNHead
from public.detection.models.anchor import YOLOV3Anchors

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'darknet53_yolov3',
]

model_urls = {
    'darknet53_yolov3': 'empty',
}

# [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
# [[10,13], [16,30], [33,23], [30,61],
#  [62,45], [59,119], [116,90], [156,198],
#  [373,326]]

# anchor for voc dataset
# [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],[78.4, 201.92], [178.24, 178.56], [129.6, 294.72],[331.84, 194.56], [227.84, 325.76], [365.44, 358.72]]

# anchor for coco dataset
# [[12.48, 19.2], [31.36, 46.4], [46.4, 113.92], [97.28, 55.04],
#  [133.12, 127.36], [79.04, 224.], [301.12, 150.4], [172.16, 285.76],
#  [348.16, 341.12]]


# assert input annotations are[x_min,y_min,x_max,y_max]
class YOLOV3(nn.Module):
    def __init__(self,
                 backbone_type="darknet53",
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YOLOV3, self).__init__()
        if backbone_type == "darknet53":
            self.backbone = Darknet53Backbone()
            C3_inplanes, C4_inplanes, C5_inplanes = 256, 512, 1024

        self.fpn = YOLOV3FPNHead(C3_inplanes,
                                 C4_inplanes,
                                 C5_inplanes,
                                 num_anchors=per_level_num_anchors,
                                 num_classes=num_classes)

        self.anchor_sizes = torch.tensor(
            [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
             [116, 90], [156, 198], [373, 326]],
            dtype=torch.float)
        self.per_level_num_anchors = per_level_num_anchors
        self.strides = torch.tensor([8, 16, 32], dtype=torch.float)
        self.anchors = YOLOV3Anchors(
            anchor_sizes=self.anchor_sizes,
            per_level_num_anchors=self.per_level_num_anchors,
            strides=self.strides)

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device

        [C3, C4, C5] = self.backbone(inputs)

        del inputs

        features = self.fpn([C3, C4, C5])

        del C3, C4, C5

        self.fpn_feature_sizes = []
        obj_heads, reg_heads, cls_heads = [], [], []
        for feature in features:
            # feature shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
            self.fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])

            feature = feature.permute(0, 2, 3, 1).contiguous()
            feature = feature.view(feature.shape[0], feature.shape[1],
                                   feature.shape[2],
                                   self.per_level_num_anchors,
                                   -1).contiguous()

            # obj_head shape:[B,H,W,3,1]
            # reg_head shape:[B,H,W,3,4]
            # cls_head shape:[B,H,W,3,80]
            obj_head = feature[:, :, :, :, 0:1]
            reg_head = feature[:, :, :, :, 1:5]
            cls_head = feature[:, :, :, :, 5:]

            obj_heads.append(obj_head)
            reg_heads.append(reg_head)
            cls_heads.append(cls_head)

        del features

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        # if input size:[B,3,416,416]
        # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
        # obj_heads shape:[[B, 52, 52, 3, 1],[B, 26, 26, 3, 1],[B, 13, 13, 3, 1]]
        # reg_heads shape:[[B, 52, 52, 3, 4],[B, 26, 26, 3, 4],[B, 13, 13, 3, 4]]
        # cls_heads shape:[[B, 52, 52, 3, 80],[B, 26, 26, 3, 80],[B, 13, 13, 3, 80]]
        # batch_anchors shape:[[B, 52, 52, 3, 5],[B, 26, 26, 3, 5],[B, 13, 13, 3, 5]]

        batch_anchors = self.anchors(self.batch_size, self.fpn_feature_sizes)

        return obj_heads, reg_heads, cls_heads, batch_anchors


def _yolov3(arch, pretrained, **kwargs):
    model = YOLOV3(arch, **kwargs)

    if pretrained:
        pretrained_models = torch.load(model_urls[arch + "_yolov3"],
                                       map_location=torch.device('cpu'))

        # del pretrained_models['cls_head.cls_head.8.weight']
        # del pretrained_models['cls_head.cls_head.8.bias']
        # del pretrained_models['reg_head.reg_head.8.weight']
        # del pretrained_models['reg_head.reg_head.8.bias']

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def darknet53_yolov3(pretrained=False, **kwargs):
    return _yolov3('darknet53', pretrained, **kwargs)


if __name__ == '__main__':
    net = YOLOV3(backbone_type="darknet53")
    image_h, image_w = 416, 416
    obj_heads, reg_heads, cls_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    print("1111", obj_heads[0].shape, reg_heads[0].shape, cls_heads[0].shape,
          batch_anchors[0].shape)
