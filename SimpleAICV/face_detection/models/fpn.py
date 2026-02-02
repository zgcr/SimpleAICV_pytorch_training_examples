import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

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
                 dilation=1,
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
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class RetinaFaceFPN(nn.Module):

    def __init__(self, inplanes, planes):
        super(RetinaFaceFPN, self).__init__()
        self.output1 = ConvBnActBlock(inplanes[0],
                                      planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      dilation=1,
                                      has_bn=True,
                                      has_act=True)

        self.output2 = ConvBnActBlock(inplanes[1],
                                      planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      dilation=1,
                                      has_bn=True,
                                      has_act=True)

        self.output3 = ConvBnActBlock(inplanes[2],
                                      planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      dilation=1,
                                      has_bn=True,
                                      has_act=True)

        self.merge1 = ConvBnActBlock(planes,
                                     planes,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1,
                                     dilation=1,
                                     has_bn=True,
                                     has_act=True)

        self.merge2 = ConvBnActBlock(planes,
                                     planes,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1,
                                     dilation=1,
                                     has_bn=True,
                                     has_act=True)

    def forward(self, inputs):
        [x2, x3, x4] = inputs

        out1 = self.output1(x2)
        out2 = self.output2(x3)
        out3 = self.output3(x4)

        up3 = F.interpolate(out3,
                            size=(out2.shape[2], out2.shape[3]),
                            mode='bilinear')
        out2 = out2 + up3
        out2 = self.merge2(out2)

        up2 = F.interpolate(out2,
                            size=(out1.shape[2], out1.shape[3]),
                            mode='bilinear')
        out1 = out1 + up2
        out1 = self.merge1(out1)

        return [out1, out2, out3]


class RetinaFaceSSH(nn.Module):

    def __init__(self, inplanes, planes):
        super(RetinaFaceSSH, self).__init__()
        self.conv3X3 = ConvBnActBlock(inplanes,
                                      planes // 2,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      dilation=1,
                                      has_bn=True,
                                      has_act=False)

        self.conv5X5_1 = ConvBnActBlock(inplanes,
                                        planes // 4,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

        self.conv5X5_2 = ConvBnActBlock(planes // 4,
                                        planes // 4,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=False)

        self.conv7X7_2 = ConvBnActBlock(planes // 4,
                                        planes // 4,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

        self.conv7x7_3 = ConvBnActBlock(planes // 4,
                                        planes // 4,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        conv3X3 = self.conv3X3(inputs)

        conv5X5_1 = self.conv5X5_1(inputs)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)

        out = self.relu(out)

        return out


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

    net = RetinaFaceFPN(inplanes=[512, 1024, 2048], planes=256)
    C3, C4, C5 = torch.randn(3, 512, 120,
                             120), torch.randn(3, 1024, 60, 60), torch.randn(
                                 3, 2048, 30, 30)
    outs = net([C3, C4, C5])
    for out in outs:
        print('1111', out.shape)

    net = RetinaFaceSSH(inplanes=256, planes=256)
    x = torch.randn(3, 256, 120, 120)
    outs = net(x)
    for out in outs:
        print('1111', out.shape)
