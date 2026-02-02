import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class DBNetFPN(nn.Module):

    def __init__(self, inplanes_list, inter_planes=256):
        super(DBNetFPN, self).__init__()
        self.c2_conv = ConvBnActBlock(inplanes_list[0],
                                      inter_planes // 4,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True)
        self.c3_conv = ConvBnActBlock(inplanes_list[1],
                                      inter_planes // 4,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True)
        self.c4_conv = ConvBnActBlock(inplanes_list[2],
                                      inter_planes // 4,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True)
        self.c5_conv = ConvBnActBlock(inplanes_list[3],
                                      inter_planes // 4,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True)

        self.p2_conv = ConvBnActBlock(inter_planes // 4,
                                      inter_planes // 4,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True)
        self.p3_conv = ConvBnActBlock(inter_planes // 4,
                                      inter_planes // 4,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True)
        self.p4_conv = ConvBnActBlock(inter_planes // 4,
                                      inter_planes // 4,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True)
        self.last_conv = ConvBnActBlock(inter_planes,
                                        inter_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        has_bn=True,
                                        has_act=True)
        self.out_channels = inter_planes

    def forward(self, x):
        C2, C3, C4, C5 = x

        del x

        P5 = self.c5_conv(C5)

        del C5

        P4 = self.c4_conv(C4)

        del C4

        P4 = F.interpolate(
            P5, size=(P4.shape[2], P4.shape[3]), mode='bilinear') + P4

        P4 = self.p4_conv(P4)

        P3 = self.c3_conv(C3)

        del C3

        P3 = F.interpolate(
            P4, size=(P3.shape[2], P3.shape[3]), mode='bilinear') + P3
        P3 = self.p3_conv(P3)

        P2 = self.c2_conv(C2)

        del C2

        P2 = F.interpolate(
            P3, size=(P2.shape[2], P2.shape[3]), mode='bilinear') + P2
        P2 = self.p2_conv(P2)

        p2_h, p2_w = P2.shape[2], P2.shape[3]
        P3 = F.interpolate(P3, size=(p2_h, p2_w), mode='bilinear')
        P4 = F.interpolate(P4, size=(p2_h, p2_w), mode='bilinear')
        P5 = F.interpolate(P5, size=(p2_h, p2_w), mode='bilinear')

        x = torch.cat([P2, P3, P4, P5], dim=1)

        del P2, P3, P4, P5

        x = self.last_conv(x)

        return x


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

    net = DBNetFPN(inplanes_list=[256, 512, 1024, 2048], inter_planes=256)
    C2, C3, C4, C5 = torch.randn(1, 256, 240, 240), torch.randn(
        1, 512, 120, 120), torch.randn(1, 1024, 60,
                                       60), torch.randn(1, 2048, 30, 30)
    out = net([C2, C3, C4, C5])
    print(f'1111, out: {out.shape}')
    print(net.out_channels)
