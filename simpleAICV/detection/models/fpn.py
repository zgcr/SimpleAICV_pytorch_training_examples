import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaFPN(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 planes,
                 use_p5=False):
        super(RetinaFPN, self).__init__()
        self.use_p5 = use_p5
        self.P3_1 = nn.Conv2d(C3_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P4_1 = nn.Conv2d(C4_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P5_1 = nn.Conv2d(C5_inplanes,
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
                C5_inplanes, planes, kernel_size=3, stride=2, padding=1)

        self.P7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5, size=(P4.shape[2], P4.shape[3]),
                           mode='nearest') + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        P6 = self.P6(P5) if self.use_p5 else self.P6(C5)

        del C3, C4, C5

        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding=1,
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
            nn.LeakyReLU(0.1, inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


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


class Yolov3TinyFPNHead(nn.Module):
    def __init__(self,
                 C4_inplanes,
                 C5_inplanes,
                 num_anchors=3,
                 num_classes=80):
        super(Yolov3TinyFPNHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1 = ConvBnActBlock(C5_inplanes,
                                    1024,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(1024,
                                    256,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    has_bn=True,
                                    has_act=True)
        self.P5_conv = ConvBnActBlock(256,
                                      512,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      has_bn=True,
                                      has_act=True)
        self.P5_pred_conv = nn.Conv2d(512,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.conv3 = ConvBnActBlock(256,
                                    128,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    has_bn=True,
                                    has_act=True)
        self.P4_conv = ConvBnActBlock(int(128 + C4_inplanes),
                                      256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      has_bn=True,
                                      has_act=True)
        self.P4_pred_conv = nn.Conv2d(256,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

    def forward(self, inputs):
        [C4, C5] = inputs

        C5 = self.conv1(C5)
        C5 = self.conv2(C5)

        P5 = self.P5_conv(C5)
        P5 = self.P5_pred_conv(P5)

        C5_upsample = F.interpolate(self.conv3(C5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')
        del C5
        C4 = torch.cat([C4, C5_upsample], dim=1)

        P4 = self.P4_conv(C4)
        P4 = self.P4_pred_conv(P4)
        del C4

        # P4 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4 = P4.permute(0, 2, 3, 1).contiguous()
        P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2], self.num_anchors,
                     -1)
        # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5 = P5.permute(0, 2, 3, 1).contiguous()
        P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2], self.num_anchors,
                     -1)

        P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
        P4[:, :, :, :, 3:5] = torch.exp(P4[:, :, :, :, 3:5])
        P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
        P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
        P5[:, :, :, :, 3:5] = torch.exp(P5[:, :, :, :, 3:5])
        P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

        return [P4, P5]


class Yolov3FPNHead(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 num_anchors=3,
                 num_classes=80,
                 use_spp=False):
        super(Yolov3FPNHead, self).__init__()
        self.num_anchors = num_anchors
        P5_1_layers = []
        if use_spp:
            for i in range(3):
                P5_1_layers.append(
                    ConvBnActBlock(C5_inplanes,
                                   C5_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   has_bn=True,
                                   has_act=True) if i %
                    2 == 0 else ConvBnActBlock(C5_inplanes // 2,
                                               C5_inplanes,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               has_bn=True,
                                               has_act=True))
            P5_1_layers.append(SPP(kernels=(5, 9, 13)))
            P5_1_layers.append(
                ConvBnActBlock(C5_inplanes * 2,
                               C5_inplanes // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0))
            P5_1_layers.append(
                ConvBnActBlock(C5_inplanes // 2,
                               C5_inplanes,
                               kernel_size=3,
                               stride=1,
                               padding=1))
            P5_1_layers.append(
                ConvBnActBlock(C5_inplanes,
                               C5_inplanes // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0))
        else:
            for i in range(5):
                P5_1_layers.append(
                    ConvBnActBlock(C5_inplanes,
                                   C5_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   has_bn=True,
                                   has_act=True) if i %
                    2 == 0 else ConvBnActBlock(C5_inplanes // 2,
                                               C5_inplanes,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               has_bn=True,
                                               has_act=True))
        self.P5_1 = nn.Sequential(*P5_1_layers)
        self.P5_2 = ConvBnActBlock(C5_inplanes // 2,
                                   C5_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   has_bn=True,
                                   has_act=True)
        self.P5_pred_conv = nn.Conv2d(C5_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        self.P5_up_conv = ConvBnActBlock(C5_inplanes // 2,
                                         C4_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         has_bn=True,
                                         has_act=True)

        P4_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P4_1_layers.append(
                    ConvBnActBlock((C4_inplanes // 2) + C4_inplanes,
                                   C4_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   has_bn=True,
                                   has_act=True) if i ==
                    0 else ConvBnActBlock(C4_inplanes,
                                          C4_inplanes // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          has_bn=True,
                                          has_act=True))
            else:
                P4_1_layers.append(
                    ConvBnActBlock(C4_inplanes // 2,
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   has_bn=True,
                                   has_act=True))
        self.P4_1 = nn.Sequential(*P4_1_layers)
        self.P4_2 = ConvBnActBlock(C4_inplanes // 2,
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   has_bn=True,
                                   has_act=True)
        self.P4_pred_conv = nn.Conv2d(C4_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        self.P4_up_conv = ConvBnActBlock(C4_inplanes // 2,
                                         C3_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         has_bn=True,
                                         has_act=True)

        P3_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P3_1_layers.append(
                    ConvBnActBlock((C3_inplanes // 2) + C3_inplanes,
                                   C3_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   has_bn=True,
                                   has_act=True) if i ==
                    0 else ConvBnActBlock(C3_inplanes,
                                          C3_inplanes // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          has_bn=True,
                                          has_act=True))
            else:
                P3_1_layers.append(
                    ConvBnActBlock(C3_inplanes // 2,
                                   C3_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   has_bn=True,
                                   has_act=True))
        self.P3_1 = nn.Sequential(*P3_1_layers)
        self.P3_2 = ConvBnActBlock(C3_inplanes // 2,
                                   C3_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   has_bn=True,
                                   has_act=True)
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

        C5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')
        C4 = torch.cat([C4, C5_upsample], axis=1)
        del C5_upsample

        P4 = self.P4_1(C4)
        del C4

        C4_upsample = F.interpolate(self.P4_up_conv(P4),
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='nearest')
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
        P3 = P3.view(P3.shape[0], P3.shape[1], P3.shape[2], self.num_anchors,
                     -1)
        # P4 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4 = P4.permute(0, 2, 3, 1).contiguous()
        P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2], self.num_anchors,
                     -1)
        # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5 = P5.permute(0, 2, 3, 1).contiguous()
        P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2], self.num_anchors,
                     -1)

        P3[:, :, :, :, 0:3] = torch.sigmoid(P3[:, :, :, :, 0:3])
        P3[:, :, :, :, 3:5] = torch.exp(P3[:, :, :, :, 3:5])
        P3[:, :, :, :, 5:] = torch.sigmoid(P3[..., 5:])
        P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
        P4[:, :, :, :, 3:5] = torch.exp(P4[:, :, :, :, 3:5])
        P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
        P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
        P5[:, :, :, :, 3:5] = torch.exp(P5[:, :, :, :, 3:5])
        P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

        return [P3, P4, P5]


class Yolov4TinyFPNHead(nn.Module):
    def __init__(self,
                 C4_inplanes,
                 C5_inplanes,
                 num_anchors=3,
                 num_classes=80):
        super(Yolov4TinyFPNHead, self).__init__()
        self.num_anchors = num_anchors
        self.P5_1 = ConvBnActBlock(C5_inplanes,
                                   C5_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.P5_up_conv = ConvBnActBlock(C5_inplanes // 2,
                                         C4_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        self.P5_2 = ConvBnActBlock(C5_inplanes // 2,
                                   C5_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.P5_pred_conv = nn.Conv2d(C5_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        self.P4_1 = ConvBnActBlock(int(C4_inplanes + C4_inplanes // 2),
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.P4_pred_conv = nn.Conv2d(C4_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

    def forward(self, inputs):
        [C4, C5] = inputs

        P5 = self.P5_1(C5)

        del C5

        P5_out = self.P5_2(P5)
        P5_out = self.P5_pred_conv(P5_out)

        P5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')
        P4 = torch.cat([C4, P5_upsample], dim=1)

        del C4, P5, P5_upsample

        P4 = self.P4_1(P4)
        P4_out = self.P4_pred_conv(P4)

        del P4

        # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.num_anchors, -1)
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.num_anchors, -1)

        P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
        P4_out[:, :, :, :, 3:5] = torch.exp(P4_out[:, :, :, :, 3:5])
        P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
        P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
        P5_out[:, :, :, :, 3:5] = torch.exp(P5_out[:, :, :, :, 3:5])
        P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

        return P4_out, P5_out


class Yolov4FPNHead(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 num_anchors=3,
                 num_classes=80):
        super(Yolov4FPNHead, self).__init__()
        self.num_anchors = num_anchors
        p5_block1 = nn.Sequential(*[
            ConvBnActBlock(C5_inplanes,
                           C5_inplanes // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0) if i %
            2 == 0 else ConvBnActBlock(C5_inplanes // 2,
                                       C5_inplanes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1) for i in range(3)
        ])
        p5_spp_block = SPP(kernels=(5, 9, 13))
        p5_block2 = nn.Sequential(
            ConvBnActBlock(C5_inplanes * 2,
                           C5_inplanes // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0),
            ConvBnActBlock(C5_inplanes // 2,
                           C5_inplanes,
                           kernel_size=3,
                           stride=1,
                           padding=1),
            ConvBnActBlock(C5_inplanes,
                           C5_inplanes // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0))
        self.P5_1 = nn.Sequential(p5_block1, p5_spp_block, p5_block2)
        self.P5_up_conv = ConvBnActBlock(C5_inplanes // 2,
                                         C4_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.P4_cat_conv = ConvBnActBlock(C4_inplanes,
                                          C4_inplanes // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)
        self.P4_1 = nn.Sequential(*[
            ConvBnActBlock(C4_inplanes,
                           C4_inplanes // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0) if i %
            2 == 0 else ConvBnActBlock(C4_inplanes // 2,
                                       C4_inplanes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1) for i in range(5)
        ])
        self.P4_up_conv = ConvBnActBlock(C4_inplanes // 2,
                                         C3_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.P3_cat_conv = ConvBnActBlock(C3_inplanes,
                                          C3_inplanes // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)
        self.P3_1 = nn.Sequential(*[
            ConvBnActBlock(C3_inplanes,
                           C3_inplanes // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0) if i %
            2 == 0 else ConvBnActBlock(C3_inplanes // 2,
                                       C3_inplanes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1) for i in range(5)
        ])
        self.P3_out_conv = ConvBnActBlock(C3_inplanes // 2,
                                          C3_inplanes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        self.P3_down_conv = ConvBnActBlock(C3_inplanes // 2,
                                           C4_inplanes // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1)
        self.P4_2 = nn.Sequential(*[
            ConvBnActBlock(C4_inplanes,
                           C4_inplanes // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0) if i %
            2 == 0 else ConvBnActBlock(C4_inplanes // 2,
                                       C4_inplanes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1) for i in range(5)
        ])
        self.P4_out_conv = ConvBnActBlock(C4_inplanes // 2,
                                          C4_inplanes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        self.P4_down_conv = ConvBnActBlock(C4_inplanes // 2,
                                           C5_inplanes // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1)
        self.P5_2 = nn.Sequential(*[
            ConvBnActBlock(C5_inplanes,
                           C5_inplanes // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0) if i %
            2 == 0 else ConvBnActBlock(C5_inplanes // 2,
                                       C5_inplanes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1) for i in range(5)
        ])
        self.P5_out_conv = ConvBnActBlock(C5_inplanes // 2,
                                          C5_inplanes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
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

        P5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')
        C4 = torch.cat([self.P4_cat_conv(C4), P5_upsample], dim=1)
        del P5_upsample

        P4 = self.P4_1(C4)
        del C4

        P4_upsample = F.interpolate(self.P4_up_conv(P4),
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='nearest')
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
                             self.num_anchors, -1)
        # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.num_anchors, -1)
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.num_anchors, -1)

        P3_out[:, :, :, :, 0:3] = torch.sigmoid(P3_out[:, :, :, :, 0:3])
        P3_out[:, :, :, :, 3:5] = torch.exp(P3_out[:, :, :, :, 3:5])
        P3_out[:, :, :, :, 5:] = torch.sigmoid(P3_out[..., 5:])
        P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
        P4_out[:, :, :, :, 3:5] = torch.exp(P4_out[:, :, :, :, 3:5])
        P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
        P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
        P5_out[:, :, :, :, 3:5] = torch.exp(P5_out[:, :, :, :, 3:5])
        P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

        return [P3_out, P4_out, P5_out]


if __name__ == '__main__':
    image_h, image_w = 640, 640
    fpn = RetinaFPN(512, 1024, 2048, 256)
    C3, C4, C5 = torch.randn(3, 512, 80, 80), torch.randn(3, 1024, 40,
                                                          40), torch.randn(
                                                              3, 2048, 20, 20)
    features = fpn([C3, C4, C5])

    for feature in features:
        print('1111', feature.shape)

    image_h, image_w = 416, 416
    fpn = Yolov3TinyFPNHead(256, 512, num_anchors=3, num_classes=80)
    C4, C5 = torch.randn(3, 256, 26, 26), torch.randn(3, 512, 13, 13)
    features = fpn([C4, C5])

    for feature in features:
        print('2222', feature.shape)

    image_h, image_w = 416, 416
    fpn = Yolov3FPNHead(256,
                        512,
                        1024,
                        num_anchors=3,
                        num_classes=80,
                        use_spp=True)
    C3, C4, C5 = torch.randn(3, 256, 52, 52), torch.randn(3, 512, 26,
                                                          26), torch.randn(
                                                              3, 1024, 13, 13)
    features = fpn([C3, C4, C5])

    for feature in features:
        print('3333', feature.shape)

    image_h, image_w = 608, 608
    fpn = Yolov4TinyFPNHead(256, 512, num_anchors=3, num_classes=80)
    C4, C5 = torch.randn(3, 256, 26, 26), torch.randn(3, 512, 13, 13)
    features = fpn([C4, C5])

    for feature in features:
        print('4444', feature.shape)

    image_h, image_w = 608, 608
    fpn = Yolov4FPNHead(256, 512, 1024, num_anchors=3, num_classes=80)
    C3, C4, C5 = torch.randn(3, 256, 52, 52), torch.randn(3, 512, 26,
                                                          26), torch.randn(
                                                              3, 1024, 13, 13)
    features = fpn([C3, C4, C5])

    for feature in features:
        print('5555', feature.shape)