"""
yolov5 official code
https://github.com/ultralytics/yolov5
"""
import os
import sys
import math

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.anchor import YOLOV3Anchors

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'yolov5s',
    'yolov5m',
    'yolov5l',
    'yolov5x',
]

model_urls = {
    'yolov5s': 'empty',
    'yolov5m': 'empty',
    'yolov5l': 'empty',
    'yolov5x': 'empty',
}

yolov5_types_config = {
    'yolov5s': {
        "depth_coefficient": 0.33,
        "width_coefficient": 0.5,
    },
    'yolov5m': {
        "depth_coefficient": 0.67,
        "width_coefficient": 0.75,
    },
    'yolov5l': {
        "depth_coefficient": 1.0,
        "width_coefficient": 1.0,
    },
    'yolov5x': {
        "depth_coefficient": 1.33,
        "width_coefficient": 1.25,
    },
}


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=kernel_size // 2,
                              groups=groups,
                              bias=False)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(planes)
        if self.has_act:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.act(x)

        return x


class Focus(nn.Module):
    """
    Focus wh dim information into channel dim
    """
    def __init__(self, inplanes, planes, kernel_size, stride=1):
        super(Focus, self).__init__()
        self.conv = ConvBnActBlock(inplanes * 4,
                                   planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   has_bn=True,
                                   has_act=True)

    def forward(self, x):
        # x:[B,C,H,W] -> [B,4C,H/2,W/2]
        x = torch.cat([
            x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ],
                      axis=1)
        x = self.conv(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, reduction=0.5, shortcut=True):
        super(Bottleneck, self).__init__()
        squeezed_planes = max(1, int(planes * reduction))
        self.conv1 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(squeezed_planes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)

        self.shortcut = True if shortcut and inplanes == planes else False

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.shortcut:
            out = out + x

        return out


class CSPBottleneck(nn.Module):
    """
    CSPBottleneck:https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    def __init__(self,
                 inplanes,
                 planes,
                 Bottleneck_nums=1,
                 reduction=0.5,
                 shortcut=True):
        super(CSPBottleneck, self).__init__()
        squeezed_planes = max(1, int(planes * reduction))
        self.conv1 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = nn.Conv2d(inplanes,
                               squeezed_planes,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.conv3 = nn.Conv2d(squeezed_planes,
                               squeezed_planes,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.conv4 = ConvBnActBlock(2 * squeezed_planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)

        self.bn = nn.BatchNorm2d(2 * squeezed_planes)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        layers = []
        for _ in range(Bottleneck_nums):
            layers.append(
                Bottleneck(squeezed_planes,
                           squeezed_planes,
                           reduction=1.0,
                           shortcut=shortcut))
        self.bottlenecks = nn.Sequential(*layers)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bottlenecks(y1)
        y1 = self.conv3(y1)
        y2 = self.conv2(x)

        out = torch.cat([y1, y2], axis=1)
        out = self.bn(out)
        out = self.act(out)
        out = self.conv4(out)

        return out


class SPP(nn.Module):
    """
    Spatial pyramid pooling layer used in YOLOv3-SPP
    """
    def __init__(self, inplanes, planes, kernels=[5, 9, 13], reduction=0.5):
        super(SPP, self).__init__()
        squeezed_planes = max(1, int(inplanes * reduction))
        self.conv1 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(squeezed_planes * (len(kernels) + 1),
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)
        layers = []
        for kernel in kernels:
            layers.append(
                nn.MaxPool2d(kernel_size=kernel, stride=1,
                             padding=kernel // 2))

        self.maxpools = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)

        features = [x]
        for maxpool in self.maxpools:
            features.append(maxpool(x))

        out = torch.cat(features, axis=1)
        out = self.conv2(out)

        return out


class YOLOV5FPNHead(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 depth_scale,
                 num_anchors=3,
                 num_classes=80,
                 CSP_nums=3,
                 CSP_shortcut=False):
        super(YOLOV5FPNHead, self).__init__()
        CSP_nums = max(round(CSP_nums *
                             depth_scale), 1) if CSP_nums > 1 else CSP_nums
        self.P5_1 = CSPBottleneck(C5_inplanes,
                                  C5_inplanes,
                                  Bottleneck_nums=CSP_nums,
                                  reduction=0.5,
                                  shortcut=CSP_shortcut)
        self.P5_2 = ConvBnActBlock(C5_inplanes,
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)

        self.P4_1 = CSPBottleneck(C4_inplanes * 2,
                                  C4_inplanes,
                                  Bottleneck_nums=CSP_nums,
                                  reduction=0.5,
                                  shortcut=CSP_shortcut)
        self.P4_2 = ConvBnActBlock(C4_inplanes,
                                   C3_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)

        self.P3_1 = CSPBottleneck(C3_inplanes * 2,
                                  C3_inplanes,
                                  Bottleneck_nums=CSP_nums,
                                  reduction=0.5,
                                  shortcut=CSP_shortcut)
        self.P3_to_P4 = ConvBnActBlock(C3_inplanes,
                                       C3_inplanes,
                                       kernel_size=3,
                                       stride=2,
                                       has_bn=True,
                                       has_act=True)
        self.P4_3 = CSPBottleneck(C4_inplanes,
                                  C4_inplanes,
                                  Bottleneck_nums=CSP_nums,
                                  reduction=0.5,
                                  shortcut=CSP_shortcut)
        self.P4_to_P5 = ConvBnActBlock(C4_inplanes,
                                       C4_inplanes,
                                       kernel_size=3,
                                       stride=2,
                                       has_bn=True,
                                       has_act=True)
        self.P5_3 = CSPBottleneck(C5_inplanes,
                                  C5_inplanes,
                                  Bottleneck_nums=CSP_nums,
                                  reduction=0.5,
                                  shortcut=CSP_shortcut)

        self.P5_pred_conv = nn.Conv2d(C5_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P4_pred_conv = nn.Conv2d(C4_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P3_pred_conv = nn.Conv2d(C3_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        del C5
        P5 = self.P5_2(P5)

        P5_upsample = F.interpolate(P5,
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')

        P4 = torch.cat([C4, P5_upsample], axis=1)
        del C4, P5_upsample
        P4 = self.P4_1(P4)
        P4 = self.P4_2(P4)

        P4_upsample = F.interpolate(P4,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='nearest')

        P3 = torch.cat([C3, P4_upsample], axis=1)
        del C3, P4_upsample
        P3 = self.P3_1(P3)
        P3_out = self.P3_pred_conv(P3)

        P3 = self.P3_to_P4(P3)
        P4 = torch.cat([P3, P4], axis=1)
        P4 = self.P4_3(P4)
        P4_out = self.P4_pred_conv(P4)

        P4 = self.P4_to_P5(P4)
        P5 = torch.cat([P4, P5], axis=1)
        P5 = self.P5_3(P5)
        P5_out = self.P5_pred_conv(P5)

        del P3, P4, P5

        return [P3_out, P4_out, P5_out]


class YOLOV5(nn.Module):
    def __init__(self,
                 yolov5_type='yolov5s',
                 per_level_num_anchors=3,
                 num_classes=80,
                 CSP_nums=[3, 9, 9, 3],
                 CSP_shortcut=[True, True, True, False],
                 planes=[128, 256, 512, 1024]):
        super(YOLOV5, self).__init__()
        depth_scale = yolov5_types_config[yolov5_type]["depth_coefficient"]
        width_scale = yolov5_types_config[yolov5_type]["width_coefficient"]
        assert len(CSP_nums) == len(CSP_shortcut)

        scaled_csp_num = []
        for num in CSP_nums:
            scaled_csp_num.append(self.compute_depth(num, depth_scale))

        scaled_planes = []
        for num in planes:
            scaled_planes.append(self.compute_width(num, width_scale))

        self.focus = Focus(3,
                           self.compute_width(64, width_scale),
                           kernel_size=3,
                           stride=1)

        inplanes = self.compute_width(64, width_scale)

        layers = []
        for i in range(7):
            if i % 2 == 0:
                layers.append(
                    ConvBnActBlock(inplanes,
                                   scaled_planes[i // 2],
                                   kernel_size=3,
                                   stride=2,
                                   has_bn=True,
                                   has_act=True))
            else:
                layers.append(
                    CSPBottleneck(inplanes,
                                  scaled_planes[i // 2],
                                  Bottleneck_nums=scaled_csp_num[i // 2],
                                  reduction=0.5,
                                  shortcut=CSP_shortcut[i // 2]))

            inplanes = scaled_planes[i // 2]

        self.middle_layers = nn.Sequential(*layers)

        self.spp = SPP(inplanes,
                       scaled_planes[-1],
                       kernels=[5, 9, 13],
                       reduction=0.5)

        C3_inplanes, C4_inplanes, C5_inplanes = scaled_planes[1:]

        self.fpn = YOLOV5FPNHead(C3_inplanes,
                                 C4_inplanes,
                                 C5_inplanes,
                                 depth_scale,
                                 num_anchors=per_level_num_anchors,
                                 num_classes=num_classes,
                                 CSP_nums=CSP_nums[-1],
                                 CSP_shortcut=CSP_shortcut[-1])

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

    def compute_depth(self, depth, scale):
        return max(round(depth * scale), 1) if depth > 1 else depth

    def compute_width(self, width, scale, divisor=8):
        return math.ceil((width * scale) / divisor) * divisor

    def forward(self, x):
        self.batch_size, _, _, _ = x.shape
        device = x.device
        x = self.focus(x)
        features = []
        for i, layer in enumerate(self.middle_layers):
            x = layer(x)
            if i % 2 == 1 and i != 1:
                features.append(x)
        x = self.spp(x)

        features.append(x)

        del x

        features = self.fpn(features)

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


def _yolov5(arch, pretrained, **kwargs):
    model = YOLOV5(arch, **kwargs)

    if pretrained:
        pretrained_models = torch.load(model_urls[arch],
                                       map_location=torch.device('cpu'))

        # del pretrained_models['cls_head.cls_head.8.weight']
        # del pretrained_models['cls_head.cls_head.8.bias']
        # del pretrained_models['reg_head.reg_head.8.weight']
        # del pretrained_models['reg_head.reg_head.8.bias']

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def yolov5s(pretrained=False, **kwargs):
    return _yolov5('yolov5s', pretrained, **kwargs)


def yolov5m(pretrained=False, **kwargs):
    return _yolov5('yolov5m', pretrained, **kwargs)


def yolov5l(pretrained=False, **kwargs):
    return _yolov5('yolov5l', pretrained, **kwargs)


def yolov5x(pretrained=False, **kwargs):
    return _yolov5('yolov5x', pretrained, **kwargs)


if __name__ == '__main__':
    net = YOLOV5(yolov5_type='yolov5s')
    image_h, image_w = 640, 640
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