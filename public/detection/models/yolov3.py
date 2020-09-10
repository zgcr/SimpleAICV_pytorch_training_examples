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
from public.detection.models.anchor import RetinaAnchors

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'darknet53_yolov3',
]

model_urls = {
    'darknet53_yolov3': 'empty',
}

# anchor for voc dataset
# [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],[78.4, 201.92], [178.24, 178.56], [129.6, 294.72],[331.84, 194.56], [227.84, 325.76], [365.44, 358.72]]

# anchor for coco dataset
# [[12.48, 19.2], [31.36, 46.4], [46.4, 113.92], [97.28, 55.04],
#  [133.12, 127.36], [79.04, 224.], [301.12, 150.4], [172.16, 285.76],
#  [348.16, 341.12]]


# assert input annotations are[x_min,y_min,x_max,y_max]
class YOLOV3(nn.Module):
    def __init__(self,
                 backbone_type,
                 anchor_sizes=[[12.48, 19.2], [31.36, 46.4], [46.4, 113.92],
                               [97.28, 55.04], [133.12, 127.36], [79.04, 224.],
                               [301.12, 150.4], [172.16, 285.76],
                               [348.16, 341.12]],
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

        # self.anchors = RetinaAnchors(self.areas, self.ratios, self.scales,
        #                              self.strides)

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
            # cls_head = self.cls_head(feature)
            # # [N,9*num_classes,H,W] -> [N,H*W*9,num_classes]
            # cls_head = cls_head.permute(0, 2, 3, 1).contiguous().view(
            #     self.batch_size, -1, self.num_classes)
            # cls_heads.append(cls_head)

            # reg_head = self.reg_head(feature)
            # # [N, 9*4,H,W] -> [N,H*W*9, 4]
            # reg_head = reg_head.permute(0, 2, 3, 1).contiguous().view(
            #     self.batch_size, -1, 4)
            # reg_heads.append(reg_head)

        del features

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        print(self.fpn_feature_sizes)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 57600, 80],[B, 14400, 80],[B, 3600, 80],[B, 900, 80],[B, 225, 80]]
        # reg_heads shape:[[B, 57600, 4],[B, 14400, 4],[B, 3600, 4],[B, 900, 4],[B, 225, 4]]
        # batch_anchors shape:[[B, 57600, 4],[B, 14400, 4],[B, 3600, 4],[B, 900, 4],[B, 225, 4]]

        # batch_anchors = self.anchors(self.batch_size, self.fpn_feature_sizes)

        # return cls_heads, reg_heads, batch_anchors


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
    net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])