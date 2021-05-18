import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import math

from tools.path import pretrained_models_path

from simpleAICV.detection.common import load_state_dict
from simpleAICV.detection.models.fpn import Yolov4TinyFPNHead, Yolov4FPNHead
from simpleAICV.detection.models.anchor import Yolov3Anchors

import torch
import torch.nn as nn

__all__ = [
    'yolov4_tiny',
    'yolov4',
]

model_urls = {
    'yolov4_tiny': 'empty',
    'yolov4': 'empty',
}


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True) if has_act else nn.Sequential())

    def forward(self, x):
        x = self.layer(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, squeeze=False):
        super(ResBlock, self).__init__()
        squeezed_planes = max(1, int(inplanes // 2)) if squeeze else inplanes
        self.conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           squeezed_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0),
            ConvBnActBlock(squeezed_planes,
                           planes,
                           kernel_size=3,
                           stride=1,
                           padding=1))

    def forward(self, x):
        x = x + self.conv(x)

        return x


class CSPDarkNetTinyBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(CSPDarkNetTinyBlock, self).__init__()
        self.planes = planes
        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.conv2 = ConvBnActBlock(planes // 2,
                                    planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        self.conv3 = ConvBnActBlock(planes // 2,
                                    planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.conv4 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)

        _, x = torch.split(x1, self.planes // 2, dim=1)

        x2 = self.conv2(x)
        x = self.conv3(x2)

        x = torch.cat([x, x2], dim=1)

        x3 = self.conv4(x)

        x = torch.cat([x1, x3], dim=1)

        x = self.maxpool(x)

        return x, x3


class CSPDarkNetBlock(nn.Module):
    def __init__(self, inplanes, planes, num_blocks, reduction=True):
        super(CSPDarkNetBlock, self).__init__()
        self.front_conv = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
        blocks = nn.Sequential(*[
            ResBlock(planes // 2 if reduction else planes,
                     planes // 2 if reduction else planes,
                     squeeze=not reduction) for _ in range(num_blocks)
        ])
        self.left_conv = nn.Sequential(
            ConvBnActBlock(planes,
                           planes // 2 if reduction else planes,
                           kernel_size=1,
                           stride=1,
                           padding=0), blocks,
            ConvBnActBlock(planes // 2 if reduction else planes,
                           planes // 2 if reduction else planes,
                           kernel_size=1,
                           stride=1,
                           padding=0))
        self.right_conv = ConvBnActBlock(planes,
                                         planes // 2 if reduction else planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.out_conv = ConvBnActBlock(planes if reduction else planes * 2,
                                       planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        x = self.front_conv(x)
        left = self.left_conv(x)
        right = self.right_conv(x)

        del x

        out = torch.cat([left, right], dim=1)
        out = self.out_conv(out)

        return out


class CSPDarknetTiny(nn.Module):
    def __init__(self, planes=[64, 128, 256, 512]):
        super(CSPDarknetTiny, self).__init__()
        self.conv1 = ConvBnActBlock(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBnActBlock(32,
                                    planes[0],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.block1 = CSPDarkNetTinyBlock(planes[0], planes[0])
        self.block2 = CSPDarkNetTinyBlock(planes[1], planes[1])
        self.block3 = CSPDarkNetTinyBlock(planes[2], planes[2])
        self.conv3 = ConvBnActBlock(planes[3],
                                    planes[3],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, C4 = self.block3(x)
        C5 = self.conv3(x)

        return C4, C5


class CSPDarknet53(nn.Module):
    def __init__(self, inplanes=32, planes=[64, 128, 256, 512, 1024]):
        super(CSPDarknet53, self).__init__()
        self.conv1 = ConvBnActBlock(3,
                                    inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.block1 = CSPDarkNetBlock(inplanes,
                                      planes[0],
                                      num_blocks=1,
                                      reduction=False)
        self.block2 = CSPDarkNetBlock(planes[0],
                                      planes[1],
                                      num_blocks=2,
                                      reduction=True)
        self.block3 = CSPDarkNetBlock(planes[1],
                                      planes[2],
                                      num_blocks=8,
                                      reduction=True)
        self.block4 = CSPDarkNetBlock(planes[2],
                                      planes[3],
                                      num_blocks=8,
                                      reduction=True)
        self.block5 = CSPDarkNetBlock(planes[3],
                                      planes[4],
                                      num_blocks=4,
                                      reduction=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        C3 = self.block3(x)
        C4 = self.block4(C3)
        C5 = self.block5(C4)

        del x

        return [C3, C4, C5]


# yolov4_tiny anchor
# [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
# yolov4 anchor
# [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110],
#     [192, 243], [459, 401]]


# assert input annotations are[x_min,y_min,x_max,y_max]
class YOLOV4(nn.Module):
    def __init__(self,
                 yolo_type='yolov4',
                 anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]],
                 strides=[8, 16, 32],
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YOLOV4, self).__init__()
        assert yolo_type in ['yolov4_tiny', 'yolov4']
        if yolo_type == 'yolov4_tiny':
            self.backbone = CSPDarknetTiny(planes=[64, 128, 256, 512])
            C4_inplanes, C5_inplanes = 256, 512
            self.fpn = Yolov4TinyFPNHead(C4_inplanes,
                                         C5_inplanes,
                                         num_anchors=per_level_num_anchors,
                                         num_classes=num_classes)
        elif yolo_type == 'yolov4':
            self.backbone = CSPDarknet53(inplanes=32,
                                         planes=[64, 128, 256, 512, 1024])
            C3_inplanes, C4_inplanes, C5_inplanes = 256, 512, 1024
            self.fpn = Yolov4FPNHead(C3_inplanes,
                                     C4_inplanes,
                                     C5_inplanes,
                                     num_anchors=per_level_num_anchors,
                                     num_classes=num_classes)

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


def _yolov4(arch, anchor_sizes, strides, pretrained, **kwargs):
    model = YOLOV4(arch, anchor_sizes, strides, **kwargs)

    if pretrained:
        load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')),
            model)

    return model


def yolov4_tiny(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov4('yolov4_tiny',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


def yolov4(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov4('yolov4',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


if __name__ == '__main__':
    net = YOLOV4(
        'yolov4',
        anchor_sizes=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                      [72, 146], [142, 110], [192, 243], [459, 401]],
        strides=[8, 16, 32],
    )
    # net = YOLOV4(
    #     'yolov4_tiny',
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