import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.classification.backbones.darknet import ConvBnActBlock
from simpleAICV.classification.backbones.yolov5backbone import CSPBottleneck
from simpleAICV.classification.backbones.yoloxbackbone import YOLOXCSPBottleneck


class RetinaFPN(nn.Module):

    def __init__(self, inplanes, planes, use_p5=False):
        super(RetinaFPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.use_p5 = use_p5
        self.P3_1 = nn.Conv2d(inplanes[0],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P4_1 = nn.Conv2d(inplanes[1],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P5_1 = nn.Conv2d(inplanes[2],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P5_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        self.P6 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=2,
            padding=1) if self.use_p5 else nn.Conv2d(
                inplanes[2], planes, kernel_size=3, stride=2, padding=1)

        self.P7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5,
                           size=(P4.shape[2], P4.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4,
                           size=(P3.shape[2], P3.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P3

        del C3, C4

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        P6 = self.P6(P5) if self.use_p5 else self.P6(C5)

        del C5

        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]


class Yolov3TinyFPNHead(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov3TinyFPNHead, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.conv1 = ConvBnActBlock(inplanes[1],
                                    1024,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(1024,
                                    256,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.P5_conv = ConvBnActBlock(256,
                                      512,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True,
                                      act_type=act_type)
        self.P5_pred_conv = nn.Conv2d(512,
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)
        self.conv3 = ConvBnActBlock(256,
                                    128,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.P4_conv = ConvBnActBlock(int(128 + inplanes[0]),
                                      256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True,
                                      act_type=act_type)
        self.P4_pred_conv = nn.Conv2d(256,
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        [C4, C5] = inputs

        C5 = self.conv1(C5)
        C5 = self.conv2(C5)

        P5 = self.P5_conv(C5)
        P5 = self.P5_pred_conv(P5)

        C5_upsample = F.interpolate(self.conv3(C5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        del C5
        C4 = torch.cat([C4, C5_upsample], dim=1)

        P4 = self.P4_conv(C4)
        P4 = self.P4_pred_conv(P4)
        del C4

        # P4 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4 = P4.permute(0, 2, 3, 1).contiguous()
        P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2],
                     self.per_level_num_anchors, -1)
        # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5 = P5.permute(0, 2, 3, 1).contiguous()
        P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2],
                     self.per_level_num_anchors, -1)

        P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
        P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
        P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
        P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

        return [P4, P5]


class Yolov3FPNHead(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov3FPNHead, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        P5_1_layers = []
        for i in range(5):
            P5_1_layers.append(
                ConvBnActBlock(inplanes[2],
                               inplanes[2] // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               has_bn=True,
                               has_act=True,
                               act_type=act_type) if i %
                2 == 0 else ConvBnActBlock(inplanes[2] // 2,
                                           inplanes[2],
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type))
        self.P5_1 = nn.Sequential(*P5_1_layers)
        self.P5_2 = ConvBnActBlock(inplanes[2] // 2,
                                   inplanes[2],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        self.P5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)

        self.P5_up_conv = ConvBnActBlock(inplanes[2] // 2,
                                         inplanes[1] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)

        P4_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P4_1_layers.append(
                    ConvBnActBlock((inplanes[1] // 2) + inplanes[1],
                                   inplanes[1] // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type) if i ==
                    0 else ConvBnActBlock(inplanes[1],
                                          inplanes[1] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type))
            else:
                P4_1_layers.append(
                    ConvBnActBlock(inplanes[1] // 2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type))
        self.P4_1 = nn.Sequential(*P4_1_layers)
        self.P4_2 = ConvBnActBlock(inplanes[1] // 2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        self.P4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)

        self.P4_up_conv = ConvBnActBlock(inplanes[1] // 2,
                                         inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)

        P3_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P3_1_layers.append(
                    ConvBnActBlock((inplanes[0] // 2) + inplanes[0],
                                   inplanes[0] // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type) if i ==
                    0 else ConvBnActBlock(inplanes[0],
                                          inplanes[0] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type))
            else:
                P3_1_layers.append(
                    ConvBnActBlock(inplanes[0] // 2,
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type))
        self.P3_1 = nn.Sequential(*P3_1_layers)
        self.P3_2 = ConvBnActBlock(inplanes[0] // 2,
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        self.P3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        del C5

        C5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C4 = torch.cat([C4, C5_upsample], axis=1)
        del C5_upsample

        P4 = self.P4_1(C4)
        del C4

        C4_upsample = F.interpolate(self.P4_up_conv(P4),
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C3 = torch.cat([C3, C4_upsample], axis=1)
        del C4_upsample

        P3 = self.P3_1(C3)
        del C3

        P5 = self.P5_2(P5)
        P5 = self.P5_pred_conv(P5)

        P4 = self.P4_2(P4)
        P4 = self.P4_pred_conv(P4)

        P3 = self.P3_2(P3)
        P3 = self.P3_pred_conv(P3)

        # P3 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P3 = P3.permute(0, 2, 3, 1).contiguous()
        P3 = P3.view(P3.shape[0], P3.shape[1], P3.shape[2],
                     self.per_level_num_anchors, -1)
        # P4 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4 = P4.permute(0, 2, 3, 1).contiguous()
        P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2],
                     self.per_level_num_anchors, -1)
        # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5 = P5.permute(0, 2, 3, 1).contiguous()
        P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2],
                     self.per_level_num_anchors, -1)

        P3[:, :, :, :, 0:3] = torch.sigmoid(P3[:, :, :, :, 0:3])
        P3[:, :, :, :, 5:] = torch.sigmoid(P3[..., 5:])
        P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
        P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
        P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
        P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

        return [P3, P4, P5]


class Yolov4TinyFPNHead(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov4TinyFPNHead, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_1 = ConvBnActBlock(inplanes[1],
                                   inplanes[1] // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        self.P5_up_conv = ConvBnActBlock(inplanes[1] // 2,
                                         inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)

        self.P5_2 = ConvBnActBlock(inplanes[1] // 2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        self.P5_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        self.P4_1 = ConvBnActBlock(int(inplanes[0] + inplanes[0] // 2),
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        self.P4_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        [C4, C5] = inputs

        P5 = self.P5_1(C5)

        del C5

        P5_out = self.P5_2(P5)
        P5_out = self.P5_pred_conv(P5_out)

        P5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P4 = torch.cat([C4, P5_upsample], dim=1)

        del C4, P5, P5_upsample

        P4 = self.P4_1(P4)
        P4_out = self.P4_pred_conv(P4)

        del P4

        # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.per_level_num_anchors, -1)
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.per_level_num_anchors, -1)

        P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
        P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
        P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
        P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

        return P4_out, P5_out


class SPP(nn.Module):
    '''
    Spatial pyramid pooling layer used in YOLOv3-SPP
    '''

    def __init__(self, kernels=[5, 9, 13]):
        super(SPP, self).__init__()
        self.maxpool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
            for kernel in kernels
        ])

    def forward(self, x):
        out = torch.cat([x] + [layer(x) for layer in self.maxpool_layers],
                        dim=1)

        return out


class Yolov4FPNHead(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov4FPNHead, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        p5_block1 = nn.Sequential(*[
            ConvBnActBlock(inplanes[2],
                           inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[2] // 2,
                                       inplanes[2],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(3)
        ])
        p5_spp_block = SPP(kernels=(5, 9, 13))
        p5_block2 = nn.Sequential(
            ConvBnActBlock(inplanes[2] * 2,
                           inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(inplanes[2] // 2,
                           inplanes[2],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(inplanes[2],
                           inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))
        self.P5_1 = nn.Sequential(p5_block1, p5_spp_block, p5_block2)
        self.P5_up_conv = ConvBnActBlock(inplanes[2] // 2,
                                         inplanes[1] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.P4_cat_conv = ConvBnActBlock(inplanes[1],
                                          inplanes[1] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P4_1 = nn.Sequential(*[
            ConvBnActBlock(inplanes[1],
                           inplanes[1] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[1] // 2,
                                       inplanes[1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(5)
        ])
        self.P4_up_conv = ConvBnActBlock(inplanes[1] // 2,
                                         inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.P3_cat_conv = ConvBnActBlock(inplanes[0],
                                          inplanes[0] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P3_1 = nn.Sequential(*[
            ConvBnActBlock(inplanes[0],
                           inplanes[0] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[0] // 2,
                                       inplanes[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(5)
        ])
        self.P3_out_conv = ConvBnActBlock(inplanes[0] // 2,
                                          inplanes[0],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P3_down_conv = ConvBnActBlock(inplanes[0] // 2,
                                           inplanes[1] // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type)
        self.P4_2 = nn.Sequential(*[
            ConvBnActBlock(inplanes[1],
                           inplanes[1] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[1] // 2,
                                       inplanes[1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(5)
        ])
        self.P4_out_conv = ConvBnActBlock(inplanes[1] // 2,
                                          inplanes[1],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P4_down_conv = ConvBnActBlock(inplanes[1] // 2,
                                           inplanes[2] // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type)
        self.P5_2 = nn.Sequential(*[
            ConvBnActBlock(inplanes[2],
                           inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[2] // 2,
                                       inplanes[2],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(5)
        ])
        self.P5_out_conv = ConvBnActBlock(inplanes[2] // 2,
                                          inplanes[2],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        del C5

        P5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C4 = torch.cat([self.P4_cat_conv(C4), P5_upsample], dim=1)
        del P5_upsample

        P4 = self.P4_1(C4)
        del C4

        P4_upsample = F.interpolate(self.P4_up_conv(P4),
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C3 = torch.cat([self.P3_cat_conv(C3), P4_upsample], dim=1)
        del P4_upsample

        P3 = self.P3_1(C3)
        del C3

        P3_out = self.P3_out_conv(P3)
        P3_out = self.P3_pred_conv(P3_out)

        P4 = torch.cat([P4, self.P3_down_conv(P3)], dim=1)
        del P3
        P4 = self.P4_2(P4)

        P4_out = self.P4_out_conv(P4)
        P4_out = self.P4_pred_conv(P4_out)

        P5 = torch.cat([P5, self.P4_down_conv(P4)], dim=1)
        del P4
        P5 = self.P5_2(P5)

        P5_out = self.P5_out_conv(P5)
        P5_out = self.P5_pred_conv(P5_out)
        del P5

        # P3_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P3_out = P3_out.permute(0, 2, 3, 1).contiguous()
        P3_out = P3_out.view(P3_out.shape[0], P3_out.shape[1], P3_out.shape[2],
                             self.per_level_num_anchors, -1)
        # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.per_level_num_anchors, -1)
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.per_level_num_anchors, -1)

        P3_out[:, :, :, :, 0:3] = torch.sigmoid(P3_out[:, :, :, :, 0:3])
        P3_out[:, :, :, :, 5:] = torch.sigmoid(P3_out[..., 5:])
        P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
        P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
        P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
        P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

        return [P3_out, P4_out, P5_out]


class YOLOV5FPNHead(nn.Module):

    def __init__(self,
                 inplanes,
                 csp_nums=3,
                 csp_shortcut=False,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='silu'):
        super(YOLOV5FPNHead, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_fpn_1 = CSPBottleneck(inplanes[2],
                                      inplanes[2],
                                      bottleneck_nums=csp_nums,
                                      reduction=0.5,
                                      shortcut=csp_shortcut,
                                      act_type=act_type)
        self.P5_fpn_2 = ConvBnActBlock(inplanes[2],
                                       inplanes[1],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P4_fpn_1 = CSPBottleneck(int(inplanes[1] * 2),
                                      inplanes[1],
                                      bottleneck_nums=csp_nums,
                                      reduction=0.5,
                                      shortcut=csp_shortcut,
                                      act_type=act_type)
        self.P4_fpn_2 = ConvBnActBlock(inplanes[1],
                                       inplanes[0],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P3_out = CSPBottleneck(int(inplanes[0] * 2),
                                    inplanes[0],
                                    bottleneck_nums=csp_nums,
                                    reduction=0.5,
                                    shortcut=csp_shortcut,
                                    act_type=act_type)
        self.P3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P3_pan_1 = ConvBnActBlock(inplanes[0],
                                       inplanes[0],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P4_out = CSPBottleneck(inplanes[1],
                                    inplanes[1],
                                    bottleneck_nums=csp_nums,
                                    reduction=0.5,
                                    shortcut=csp_shortcut,
                                    act_type=act_type)
        self.P4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P4_pan_1 = ConvBnActBlock(inplanes[1],
                                       inplanes[1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P5_out = CSPBottleneck(inplanes[2],
                                    inplanes[2],
                                    bottleneck_nums=csp_nums,
                                    reduction=0.5,
                                    shortcut=csp_shortcut,
                                    act_type=act_type)
        self.P5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

        # https://arxiv.org/abs/1708.02002 section 3.3
        p5_bias = self.P5_pred_conv.bias.view(per_level_num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=32
        p5_bias.data[:, 0] += math.log(8 / (640 / 32)**2)
        # init cls pred value
        p5_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P5_pred_conv.bias = torch.nn.Parameter(p5_bias.view(-1),
                                                    requires_grad=True)

        p4_bias = self.P4_pred_conv.bias.view(per_level_num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=16
        p4_bias.data[:, 0] += math.log(8 / (640 / 16)**2)
        # init cls pred value
        p4_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P4_pred_conv.bias = torch.nn.Parameter(p4_bias.view(-1),
                                                    requires_grad=True)

        p3_bias = self.P3_pred_conv.bias.view(per_level_num_anchors, -1)
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
                                    mode='bilinear',
                                    align_corners=True)
        P4 = torch.cat([C4, P5_upsample], axis=1)

        del C4, P5_upsample

        P4 = self.P4_fpn_1(P4)
        P4 = self.P4_fpn_2(P4)

        P4_upsample = F.interpolate(P4,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
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

        # P3_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P3_out = P3_out.permute(0, 2, 3, 1).contiguous()
        P3_out = P3_out.view(P3_out.shape[0], P3_out.shape[1], P3_out.shape[2],
                             self.per_level_num_anchors, -1).contiguous()
        # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.per_level_num_anchors, -1).contiguous()
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.per_level_num_anchors, -1).contiguous()

        P3_out = self.sigmoid(P3_out)
        P4_out = self.sigmoid(P4_out)
        P5_out = self.sigmoid(P5_out)

        return [P3_out, P4_out, P5_out]


class YOLOXFPN(nn.Module):

    def __init__(self,
                 inplanes,
                 csp_nums=3,
                 csp_shortcut=False,
                 block=ConvBnActBlock,
                 act_type='silu'):
        super(YOLOXFPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]

        self.p5_reduce_conv = ConvBnActBlock(inplanes[2],
                                             inplanes[1],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             has_bn=True,
                                             has_act=True,
                                             act_type=act_type)
        self.p4_conv1 = YOLOXCSPBottleneck(int(inplanes[1] * 2),
                                           inplanes[1],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)
        self.p4_reduce_conv = ConvBnActBlock(inplanes[1],
                                             inplanes[0],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             has_bn=True,
                                             has_act=True,
                                             act_type=act_type)
        self.p3_conv1 = YOLOXCSPBottleneck(int(inplanes[0] * 2),
                                           inplanes[0],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)
        self.p3_up_conv = ConvBnActBlock(inplanes[0],
                                         inplanes[0],
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.p4_conv2 = YOLOXCSPBottleneck(int(inplanes[0] * 2),
                                           inplanes[1],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)
        self.p4_up_conv = ConvBnActBlock(inplanes[1],
                                         inplanes[1],
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.p5_conv1 = YOLOXCSPBottleneck(int(inplanes[1] * 2),
                                           inplanes[2],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.p5_reduce_conv(C5)

        del C5

        P5_upsample = F.interpolate(P5,
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P4 = torch.cat([C4, P5_upsample], axis=1)

        del C4, P5_upsample

        P4 = self.p4_conv1(P4)
        P4 = self.p4_reduce_conv(P4)

        P4_upsample = F.interpolate(P4,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P3 = torch.cat([C3, P4_upsample], axis=1)

        del C3, P4_upsample

        P3_out = self.p3_conv1(P3)

        P3_up = self.p3_up_conv(P3_out)
        P4 = torch.cat([P3_up, P4], axis=1)
        P4_out = self.p4_conv2(P4)

        del P4

        P4_up = self.p4_up_conv(P4_out)
        P5 = torch.cat([P4_up, P5], axis=1)
        P5_out = self.p5_conv1(P5)

        del P5

        return [P3_out, P4_out, P5_out]


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    net = RetinaFPN([512, 1024, 2048], 256, use_p5=False)
    C3, C4, C5 = torch.randn(3, 512, 80, 80), torch.randn(3, 1024, 40,
                                                          40), torch.randn(
                                                              3, 2048, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C3, C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C3, C4, C5])
    for out in outs:
        print('2222', out.shape)

    net = Yolov3TinyFPNHead([256, 512],
                            per_level_num_anchors=3,
                            num_classes=80,
                            act_type='leakyrelu')
    C4, C5 = torch.randn(3, 256, 40, 40), torch.randn(3, 512, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C4, C5])
    for out in outs:
        print('2222', out.shape)

    net = Yolov3FPNHead([256, 512, 1024],
                        per_level_num_anchors=3,
                        num_classes=80,
                        act_type='leakyrelu')
    C3, C4, C5 = torch.randn(3, 256, 80, 80), torch.randn(3, 512, 40,
                                                          40), torch.randn(
                                                              3, 1024, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C3, C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C3, C4, C5])
    for out in outs:
        print('2222', out.shape)

    net = Yolov4TinyFPNHead([256, 512],
                            per_level_num_anchors=3,
                            num_classes=80,
                            act_type='leakyrelu')
    C4, C5 = torch.randn(3, 256, 40, 40), torch.randn(3, 512, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C4, C5])
    for out in outs:
        print('2222', out.shape)

    net = Yolov4FPNHead([256, 512, 1024],
                        per_level_num_anchors=3,
                        num_classes=80,
                        act_type='leakyrelu')
    C3, C4, C5 = torch.randn(3, 256, 80, 80), torch.randn(3, 512, 40,
                                                          40), torch.randn(
                                                              3, 1024, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C3, C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C3, C4, C5])
    for out in outs:
        print('2222', out.shape)

    net = YOLOV5FPNHead([256, 512, 1024],
                        csp_nums=3,
                        csp_shortcut=False,
                        per_level_num_anchors=3,
                        num_classes=80,
                        act_type='silu')
    C3, C4, C5 = torch.randn(3, 256, 80, 80), torch.randn(3, 512, 40,
                                                          40), torch.randn(
                                                              3, 1024, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C3, C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C3, C4, C5])
    for out in outs:
        print('2222', out.shape)

    net = YOLOXFPN([256, 512, 1024],
                   csp_nums=3,
                   csp_shortcut=False,
                   block=ConvBnActBlock,
                   per_level_num_anchors=3,
                   act_type='silu')
    C3, C4, C5 = torch.randn(3, 256, 80, 80), torch.randn(3, 512, 40,
                                                          40), torch.randn(
                                                              3, 1024, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C3, C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C3, C4, C5])
    for out in outs:
        print('2222', out.shape)