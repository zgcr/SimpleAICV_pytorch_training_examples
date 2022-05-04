import math

import torch
import torch.nn as nn
import torchvision.ops


class DeformableConv2d(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.Tensor(planes, inplanes // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(planes))
        else:
            self.bias = None

        self.offset_conv = nn.Conv2d(inplanes,
                                     2 * groups * kernel_size * kernel_size,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(inplanes,
                                   1 * groups * kernel_size * kernel_size,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   bias=True)

        n = inplanes * kernel_size * kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias.data.zero_()

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          mask=mask)

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

    feature = torch.randn(3, 256, 80, 80)
    model = DeformableConv2d(256,
                             128,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             dilation=1,
                             groups=1,
                             bias=True)
    out = model(feature)
    print('1111', out.shape)