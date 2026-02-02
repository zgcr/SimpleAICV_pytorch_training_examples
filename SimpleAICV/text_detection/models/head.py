import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn


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


class ConvTransposeBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvTransposeBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(inplanes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               groups=groups,
                               bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class DBNetHead(nn.Module):

    def __init__(self, inplanes, k=50):
        super(DBNetHead, self).__init__()
        self.k = k
        self.binary_conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes // 4,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),
            ConvTransposeBnActBlock(inplanes // 4,
                                    inplanes // 4,
                                    kernel_size=2,
                                    stride=2,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True),
            nn.ConvTranspose2d(inplanes // 4,
                               1,
                               kernel_size=2,
                               stride=2,
                               groups=1,
                               bias=True))
        self.thresh_conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes // 4,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),
            ConvTransposeBnActBlock(inplanes // 4,
                                    inplanes // 4,
                                    kernel_size=2,
                                    stride=2,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True),
            nn.ConvTranspose2d(inplanes // 4,
                               1,
                               kernel_size=2,
                               stride=2,
                               groups=1,
                               bias=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        probability_map = self.binary_conv(x)
        probability_map = probability_map.float()
        probability_map = self.sigmoid(probability_map)

        threshold_map = self.thresh_conv(x)
        threshold_map = threshold_map.float()
        threshold_map = self.sigmoid(threshold_map)

        preds = torch.cat([probability_map, threshold_map], dim=1)

        return preds


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

    net = DBNetHead(inplanes=256, k=50)
    x = torch.randn(1, 256, 240, 240)
    out = net(x)
    print(f'1111, out: {out.shape}')
