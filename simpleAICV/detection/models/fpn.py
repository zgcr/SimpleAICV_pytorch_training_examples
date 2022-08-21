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

    net = YOLOXFPN([256, 512, 1024],
                   csp_nums=3,
                   csp_shortcut=False,
                   block=ConvBnActBlock,
                   act_type='silu')
    C3, C4, C5 = torch.randn(3, 256, 80, 80), torch.randn(3, 512, 40,
                                                          40), torch.randn(
                                                              3, 1024, 20, 20)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C3, C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'3333, macs: {macs}, params: {params}')
    outs = net([C3, C4, C5])
    for out in outs:
        print('4444', out.shape)