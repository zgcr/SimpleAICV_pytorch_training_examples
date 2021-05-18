import math

import torch
import torch.nn as nn
import torchvision.ops


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 groups=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        self.padding = padding

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size[0],
                         kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * groups * kernel_size[0] *
                                     kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(in_channels,
                                   1 * groups * kernel_size[0] *
                                   kernel_size[1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=self.padding,
                                   bias=True)

        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
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
                                          padding=self.padding,
                                          mask=mask)

        return x