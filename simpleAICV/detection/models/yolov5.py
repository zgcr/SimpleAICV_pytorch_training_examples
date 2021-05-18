'''
yolov5 official code
https://github.com/ultralytics/yolov5
'''
import os
import sys
import math

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import pretrained_models_path

from simpleAICV.detection.common import load_state_dict
from simpleAICV.detection.models.anchor import Yolov3Anchors

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

types_config = {
    'yolov5s': {
        'depth_coefficient': 0.33,
        'width_coefficient': 0.50,
    },
    'yolov5m': {
        'depth_coefficient': 0.67,
        'width_coefficient': 0.75,
    },
    'yolov5l': {
        'depth_coefficient': 1.0,
        'width_coefficient': 1.0,
    },
    'yolov5x': {
        'depth_coefficient': 1.33,
        'width_coefficient': 1.25,
    },
}


class SiLU(nn.Module):
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        inner = torch.sigmoid(x)

        return x.mul_(inner) if self.inplace else x.mul(inner)


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
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            SiLU(inplace=True) if has_act else nn.Sequential())

    def forward(self, x):
        x = self.layer(x)

        return x


class Focus(nn.Module):
    '''
    Focus wh dim information into channel dim
    '''
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
        self.conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           squeezed_planes,
                           kernel_size=1,
                           stride=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(squeezed_planes,
                           planes,
                           kernel_size=3,
                           stride=1,
                           has_bn=True,
                           has_act=True))

        self.shortcut = True if shortcut and inplanes == planes else False

    def forward(self, x):
        out = self.conv(x)

        if self.shortcut:
            out = out + x

        return out


class CSPBottleneck(nn.Module):
    '''
    CSP Bottleneck with 3 convolution layers
    CSPBottleneck:https://github.com/WongKinYiu/CrossStagePartialNetworks
    '''
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
        self.conv2 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(2 * squeezed_planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    has_bn=True,
                                    has_act=True)

        self.bottlenecks = nn.Sequential(*[
            Bottleneck(squeezed_planes,
                       squeezed_planes,
                       reduction=1.0,
                       shortcut=shortcut) for _ in range(Bottleneck_nums)
        ])

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bottlenecks(y1)
        y2 = self.conv2(x)

        out = torch.cat([y1, y2], axis=1)
        out = self.conv3(out)

        return out


class Yolov5SPP(nn.Module):
    '''
    Spatial pyramid pooling layer used in YOLOv3-SPP
    '''
    def __init__(self, inplanes, planes, kernels=[5, 9, 13]):
        super(Yolov5SPP, self).__init__()
        squeezed_planes = max(1, int(inplanes // 2))
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
        self.maxpool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
            for kernel in kernels
        ])

    def forward(self, x):
        x = self.conv1(x)

        out = torch.cat([x] + [layer(x) for layer in self.maxpool_layers],
                        dim=1)
        out = self.conv2(out)

        return out


class Yolov5Backbone(nn.Module):
    def __init__(self, inplanes, scaled_csp_nums, CSP_shortcut, scaled_planes):
        super(Yolov5Backbone, self).__init__()
        self.inplanes = inplanes

        self.focus = Focus(3, self.inplanes, kernel_size=3, stride=1)

        middle_layers = []
        for i in range(7):
            idx = i // 2
            middle_layers.append(
                ConvBnActBlock(self.inplanes,
                               scaled_planes[idx],
                               kernel_size=3,
                               stride=2,
                               has_bn=True,
                               has_act=True) if i %
                2 == 0 else CSPBottleneck(self.inplanes,
                                          scaled_planes[idx],
                                          Bottleneck_nums=scaled_csp_nums[idx],
                                          reduction=0.5,
                                          shortcut=CSP_shortcut[idx]))
            self.inplanes = scaled_planes[idx]

        self.middle_layers = nn.Sequential(*middle_layers)

        self.spp = Yolov5SPP(self.inplanes,
                             scaled_planes[-1],
                             kernels=[5, 9, 13])

    def forward(self, x):
        x = self.focus(x)

        features = []

        for i, layer in enumerate(self.middle_layers):
            x = layer(x)
            if i % 2 == 1 and i > 2 and i < 6:
                features.append(x)

        x = self.spp(x)
        features.append(x)

        return features


class YOLOV5FPNHead(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 depth_scale,
                 num_anchors=3,
                 num_classes=80,
                 scaled_CSP_nums=3,
                 CSP_shortcut=False):
        super(YOLOV5FPNHead, self).__init__()
        self.P5_fpn_1 = CSPBottleneck(C5_inplanes,
                                      C5_inplanes,
                                      Bottleneck_nums=scaled_CSP_nums,
                                      reduction=0.5,
                                      shortcut=CSP_shortcut)
        self.P5_fpn_2 = ConvBnActBlock(C5_inplanes,
                                       C4_inplanes,
                                       kernel_size=1,
                                       stride=1,
                                       has_bn=True,
                                       has_act=True)

        self.P4_fpn_1 = CSPBottleneck(int(C4_inplanes * 2),
                                      C4_inplanes,
                                      Bottleneck_nums=scaled_CSP_nums,
                                      reduction=0.5,
                                      shortcut=CSP_shortcut)
        self.P4_fpn_2 = ConvBnActBlock(C4_inplanes,
                                       C3_inplanes,
                                       kernel_size=1,
                                       stride=1,
                                       has_bn=True,
                                       has_act=True)

        self.P3_out = CSPBottleneck(int(C3_inplanes * 2),
                                    C3_inplanes,
                                    Bottleneck_nums=scaled_CSP_nums,
                                    reduction=0.5,
                                    shortcut=CSP_shortcut)
        self.P3_pred_conv = nn.Conv2d(C3_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P3_pan_1 = ConvBnActBlock(C3_inplanes,
                                       C3_inplanes,
                                       kernel_size=3,
                                       stride=2,
                                       has_bn=True,
                                       has_act=True)

        self.P4_out = CSPBottleneck(C4_inplanes,
                                    C4_inplanes,
                                    Bottleneck_nums=scaled_CSP_nums,
                                    reduction=0.5,
                                    shortcut=CSP_shortcut)
        self.P4_pred_conv = nn.Conv2d(C4_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P4_pan_1 = ConvBnActBlock(C4_inplanes,
                                       C4_inplanes,
                                       kernel_size=3,
                                       stride=2,
                                       has_bn=True,
                                       has_act=True)

        self.P5_out = CSPBottleneck(C5_inplanes,
                                    C5_inplanes,
                                    Bottleneck_nums=scaled_CSP_nums,
                                    reduction=0.5,
                                    shortcut=CSP_shortcut)
        self.P5_pred_conv = nn.Conv2d(C5_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        # https://arxiv.org/abs/1708.02002 section 3.3
        p5_bias = self.P5_pred_conv.bias.view(num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=32
        p5_bias.data[:, 0] += math.log(8 / (640 / 32)**2)
        # init cls pred value
        p5_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P5_pred_conv.bias = torch.nn.Parameter(p5_bias.view(-1),
                                                    requires_grad=True)

        p4_bias = self.P4_pred_conv.bias.view(num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=16
        p4_bias.data[:, 0] += math.log(8 / (640 / 16)**2)
        # init cls pred value
        p4_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P4_pred_conv.bias = torch.nn.Parameter(p4_bias.view(-1),
                                                    requires_grad=True)

        p3_bias = self.P3_pred_conv.bias.view(num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=8
        p3_bias.data[:, 0] += math.log(8 / (640 / 8)**2)
        # init cls pred value
        p3_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P3_pred_conv.bias = torch.nn.Parameter(p3_bias.view(-1),
                                                    requires_grad=True)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_fpn_1(C5)
        P5 = self.P5_fpn_2(P5)

        del C5

        P5_upsample = F.interpolate(P5,
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')
        P4 = torch.cat([C4, P5_upsample], axis=1)

        del C4, P5_upsample

        P4 = self.P4_fpn_1(P4)
        P4 = self.P4_fpn_2(P4)

        P4_upsample = F.interpolate(P4,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='nearest')
        P3 = torch.cat([C3, P4_upsample], axis=1)

        del C3, P4_upsample

        P3 = self.P3_out(P3)
        P3_out = self.P3_pred_conv(P3)

        P3 = self.P3_pan_1(P3)
        P4 = torch.cat([P3, P4], axis=1)

        del P3

        P4 = self.P4_out(P4)
        P4_out = self.P4_pred_conv(P4)

        P4 = self.P4_pan_1(P4)
        P5 = torch.cat([P4, P5], axis=1)

        del P4

        P5 = self.P5_out(P5)
        P5_out = self.P5_pred_conv(P5)

        del P5

        P3_out = torch.sigmoid(P3_out)
        P4_out = torch.sigmoid(P4_out)
        P5_out = torch.sigmoid(P5_out)

        return [P3_out, P4_out, P5_out]


class YOLOV5(nn.Module):
    def __init__(self,
                 yolo_type='yolov5s',
                 anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]],
                 strides=[8, 16, 32],
                 per_level_num_anchors=3,
                 num_classes=80,
                 CSP_nums=[3, 9, 9, 3],
                 CSP_shortcut=[True, True, True, False],
                 planes=[128, 256, 512, 1024]):
        super(YOLOV5, self).__init__()
        depth_scale = types_config[yolo_type]['depth_coefficient']
        width_scale = types_config[yolo_type]['width_coefficient']
        assert len(CSP_nums) == len(CSP_shortcut) == len(
            planes), 'wrong CSP_nums/CSP_shortcut/planes!'

        scaled_csp_nums = [
            self.compute_depth(num, depth_scale) for num in CSP_nums
        ]
        scaled_planes = [
            self.compute_width(num, width_scale) for num in planes
        ]
        inplanes = self.compute_width(64, width_scale)
        self.backbone = Yolov5Backbone(inplanes, scaled_csp_nums, CSP_shortcut,
                                       scaled_planes)

        C3_inplanes, C4_inplanes, C5_inplanes = scaled_planes[1:]

        self.fpn = YOLOV5FPNHead(C3_inplanes,
                                 C4_inplanes,
                                 C5_inplanes,
                                 depth_scale,
                                 num_anchors=per_level_num_anchors,
                                 num_classes=num_classes,
                                 scaled_CSP_nums=scaled_csp_nums[-1],
                                 CSP_shortcut=CSP_shortcut[-1])

        self.anchor_sizes = torch.tensor(anchor_sizes)
        self.per_level_num_anchors = per_level_num_anchors
        self.strides = torch.tensor(strides)

        self.anchors = Yolov3Anchors(
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

        features = self.backbone(x)
        features = self.fpn(features)

        self.fpn_feature_sizes, obj_reg_cls_heads = [], []
        for feature in features:
            self.fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])
            # feature shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
            _, _, H, W = feature.shape
            feature = feature.permute(0, 2, 3, 1).contiguous().view(
                self.batch_size, H, W, self.per_level_num_anchors, -1)

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


def _yolov5(arch, anchor_sizes, strides, pretrained, **kwargs):
    model = YOLOV5(arch, anchor_sizes, strides, **kwargs)

    if pretrained:
        load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')),
            model)

    return model


def yolov5s(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov5('yolov5s',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


def yolov5m(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov5('yolov5m',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


def yolov5l(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov5('yolov5l',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


def yolov5x(anchor_sizes, strides, pretrained=False, **kwargs):
    return _yolov5('yolov5x',
                   anchor_sizes=anchor_sizes,
                   strides=strides,
                   pretrained=pretrained,
                   **kwargs)


if __name__ == '__main__':
    net = YOLOV5(yolo_type='yolov5l')
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
