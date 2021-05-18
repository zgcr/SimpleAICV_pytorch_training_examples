import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import pretrained_models_path

from simpleAICV.detection.common import load_state_dict
from simpleAICV.detection.models.backbone import DarknetTinyBackbone, Darknet53Backbone
from simpleAICV.detection.models.fpn import Yolov3TinyFPNHead, Yolov3FPNHead
from simpleAICV.detection.models.anchor import Yolov3Anchors

import torch
import torch.nn as nn

__all__ = [
    'yolov3_tiny',
    'yolov3',
    'yolov3_spp',
]

model_urls = {
    'yolov3_tiny': 'empty',
    'yolov3': 'empty',
    'yolov3_spp': 'empty',
}

# yolov3_tiny anchor
#[[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
# strides=[16,32]

# yolov3/yolov3_spp anchor
# [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
# strides=[8,16,32]


# assert input annotations are[x_min,y_min,x_max,y_max]
class YOLOV3(nn.Module):
    def __init__(self,
                 yolo_type='yolov3',
                 anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]],
                 strides=[8, 16, 32],
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YOLOV3, self).__init__()
        assert yolo_type in ['yolov3_tiny', 'yolov3',
                             'yolov3_spp'], 'Not supported type!'

        if yolo_type == 'yolov3_tiny':
            self.backbone = DarknetTinyBackbone(pretrained=False)
            C4_inplanes, C5_inplanes = 256, 512
            self.fpn = Yolov3TinyFPNHead(C4_inplanes,
                                         C5_inplanes,
                                         num_anchors=per_level_num_anchors,
                                         num_classes=num_classes)
        elif yolo_type in ['yolov3', 'yolov3_spp']:
            self.backbone = Darknet53Backbone(pretrained=False)
            C3_inplanes, C4_inplanes, C5_inplanes = 256, 512, 1024
            if yolo_type == 'yolov3_spp':
                self.fpn = Yolov3FPNHead(C3_inplanes,
                                         C4_inplanes,
                                         C5_inplanes,
                                         num_anchors=per_level_num_anchors,
                                         num_classes=num_classes,
                                         use_spp=True)
            else:
                self.fpn = Yolov3FPNHead(C3_inplanes,
                                         C4_inplanes,
                                         C5_inplanes,
                                         num_anchors=per_level_num_anchors,
                                         num_classes=num_classes,
                                         use_spp=False)

        self.anchor_sizes = torch.tensor(anchor_sizes)
        self.per_level_num_anchors = per_level_num_anchors
        self.strides = torch.tensor(strides)

        self.anchors = Yolov3Anchors(
            anchor_sizes=self.anchor_sizes,
            per_level_num_anchors=self.per_level_num_anchors,
            strides=self.strides)

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device

        outs = self.backbone(inputs)

        del inputs

        features = self.fpn(outs)

        del outs

        self.fpn_feature_sizes, obj_reg_cls_heads = [], []
        for feature in features:
            # feature shape:[B,H,W,3,85]
            self.fpn_feature_sizes.append([feature.shape[2], feature.shape[1]])

            # obj_head:feature[:, :, :, :, 0:1], shape:[B,H,W,3,1]
            # reg_head:feature[:, :, :, :, 1:5], shape:[B,H,W,3,4]
            # cls_head:feature[:, :, :, :, 5:],  shape:[B,H,W,3,80]
            obj_reg_cls_heads.append(feature)

        del features

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        # if input size:[B,3,416,416]
        # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
        # obj_reg_cls_heads shape:[[B, 52, 52, 3, 85],[B, 26, 26, 3, 85],[B, 13, 13, 3, 85]]
        # batch_anchors shape:[[B, 52, 52, 3, 5],[B, 26, 26, 3, 5],[B, 13, 13, 3, 5]]
        batch_anchors = self.anchors(self.batch_size, self.fpn_feature_sizes)

        return obj_reg_cls_heads, batch_anchors


def _yolov3(arch, anchor_sizes, strides, pretrained, **kwargs):
    model = YOLOV3(arch, anchor_sizes, strides, **kwargs)

    if pretrained:
        load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')),
            model)

    return model


def yolov3_tiny(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov3('yolov3_tiny',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


def yolov3(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov3('yolov3',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


def yolov3_spp(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov3('yolov3_spp',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


if __name__ == '__main__':
    net = YOLOV3(
        yolo_type='yolov3_spp',
        anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                      [59, 119], [116, 90], [156, 198], [373, 326]],
        strides=[8, 16, 32],
    )
    # net = YOLOV3(
    #     yolo_type='yolov3_tiny',
    #     anchor_sizes=[[10, 14], [23, 27], [37, 58], [81, 82], [135, 169],
    #                   [344, 319]],
    #     strides=[16, 32],
    # )
    image_h, image_w = 640, 640
    obj_reg_cls_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    for x, y in zip(obj_reg_cls_heads, batch_anchors):
        print("1111", x.shape, y.shape)

    from thop import profile
    from thop import clever_format
    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    print(f"2222, flops: {flops}, params: {params}")