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


class VitDetFPN(nn.Module):

    def __init__(self, inplanes, planes):
        super(VitDetFPN, self).__init__()
        self.P3 = nn.ConvTranspose2d(inplanes,
                                     planes,
                                     kernel_size=2,
                                     stride=2,
                                     padding=0,
                                     output_padding=0,
                                     bias=True)
        self.P4 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.GELU(),
        )
        self.P5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.P6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.P7 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        P3 = self.P3(x)
        P4 = self.P4(x)
        P5 = self.P5(P4)
        P6 = self.P6(P5)
        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]


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

    net = VitDetFPN(768, 256)
    x = torch.randn(3, 768, 32, 32)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(x, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'3333, macs: {macs}, params: {params}')
    outs = net(x)
    for out in outs:
        print('4444', out.shape)