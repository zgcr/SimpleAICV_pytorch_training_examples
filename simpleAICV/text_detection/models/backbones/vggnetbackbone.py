import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.classification.common import load_state_dict

__all__ = [
    'VGGNetBackbone',
    'RepVGGNetBackbone',
    'RepVGGEnhanceNetBackbone',
    'RepVGGEnhanceDilationNetBackbone',
]


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


class VGGNet(nn.Module):

    def __init__(self, planes=[16, 16, 32, 48, 64, 80]):
        super(VGGNet, self).__init__()
        self.layers = nn.Sequential(
            ConvBnActBlock(3,
                           planes[0],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),  # [B,16,h,w]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/2,w/2]
            ConvBnActBlock(planes[0],
                           planes[1],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),  # [B,16,h/2,w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/4,w/4]
            ConvBnActBlock(planes[1],
                           planes[2],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(planes[2],
                           planes[2],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),  # [B,32,h/4,w/4]        #5
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,32,h/8,w/8]
            ConvBnActBlock(planes[2],
                           planes[3],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(planes[3],
                           planes[3],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),  # [B,48,h/8,w/8]         #8
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,48,h/16,w/16]
            ConvBnActBlock(planes[3],
                           planes[4],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(planes[4],
                           planes[4],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),  # [B,64,h/16,w/16]      #11
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,64,h/32,w/32]
            ConvBnActBlock(planes[4],
                           planes[5],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(planes[5],
                           planes[5],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True))  # [B,80,h/32,w/32]      #14

        self.last_layer = ConvBnActBlock(planes[5],
                                         planes[5] * 6,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True)

        self.out_idx = [5, 8, 11, 14]
        self.out_channels = [planes[2], planes[3], planes[4], planes[5] * 6]

    def forward(self, x):
        out = []
        for idx, per_layer in enumerate(self.layers):
            x = per_layer(x)
            if idx in self.out_idx:
                out.append(x)

        out[-1] = self.last_layer(out[-1])

        return out


def conv_bn_layer(inplanes, planes, kernel_size, stride, padding=1, groups=1):
    layer = nn.Sequential(
        OrderedDict([
            ('conv',
             nn.Conv2d(inplanes,
                       planes,
                       kernel_size,
                       stride=stride,
                       padding=padding,
                       groups=groups,
                       bias=False)),
            ('bn', nn.BatchNorm2d(planes)),
        ]))

    return layer


class RepVGGBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 deploy=False):
        super(RepVGGBlock, self).__init__()
        self.inplanes = inplanes
        self.groups = groups
        self.deploy = deploy

        assert kernel_size == 3 and padding == 1

        if self.deploy:
            self.fuse_equivalent_conv = nn.Conv2d(inplanes,
                                                  planes,
                                                  kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=groups,
                                                  bias=True,
                                                  padding_mode='zeros')

        else:
            self.identity = nn.BatchNorm2d(
                inplanes) if inplanes == planes and stride == 1 else None
            self.conv3x3 = conv_bn_layer(inplanes,
                                         planes,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=groups)
            self.conv1x1 = conv_bn_layer(inplanes,
                                         planes,
                                         kernel_size=1,
                                         stride=stride,
                                         padding=padding - kernel_size // 2,
                                         groups=groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            x = self.relu(self.fuse_equivalent_conv(x))

            return x

        if self.identity:
            identity_out = self.identity(x)
        else:
            identity_out = 0

        x = self.relu(self.conv3x3(x) + self.conv1x1(x) + identity_out)

        return x

    def _fuse_bn_layer(self, branch):
        '''
        fuse conv and bn layers to get equivalent conv layer kernel and bias
        '''
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            # make sure conv layer doesn't have bias
            kernel = branch.conv.weight
            running_mean, running_var = branch.bn.running_mean, branch.bn.running_var
            gamma, beta, eps = branch.bn.weight, branch.bn.bias, branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            # convert identity branch to get a equivalent 1x1 conv layer kernel and bias
            input_dim = self.inplanes // self.groups
            kernel_value = np.zeros((self.inplanes, input_dim, 3, 3),
                                    dtype=np.float32)
            for i in range(self.inplanes):
                kernel_value[i, i % input_dim, 1, 1] = 1

            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma, beta, eps = branch.weight, branch.bias, branch.eps

        # fuse conv and bn layer
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        equivalent_kernel, equivalent_bias = kernel * t, beta - running_mean * gamma / std

        return equivalent_kernel, equivalent_bias

    def get_equivalent_conv_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_layer(self.conv3x3)
        kernel1x1, bias1x1 = self._fuse_bn_layer(self.conv1x1)
        kernelidentity, biasidentity = self._fuse_bn_layer(self.identity)

        # 1x1kernel must be pad to 3x3kernel before add
        kernel, bias = kernel3x3 + F.pad(
            kernel1x1,
            [1, 1, 1, 1]) + kernelidentity, bias3x3 + bias1x1 + biasidentity
        kernel, bias = kernel.detach().cpu(), bias.detach().cpu()

        return kernel, bias


def conv_bn_dilation_layer(inplanes,
                           planes,
                           kernel_size,
                           stride,
                           padding=1,
                           dilation=1,
                           groups=1):
    layer = nn.Sequential(
        OrderedDict([
            ('conv',
             nn.Conv2d(inplanes,
                       planes,
                       kernel_size,
                       stride=stride,
                       padding=padding,
                       dilation=dilation,
                       groups=groups,
                       bias=False)),
            ('bn', nn.BatchNorm2d(planes)),
        ]))

    return layer


class RepVGGEnhanceDilationBlock(nn.Module):
    '''
    RepVGGEnhanceDilationBlock use k to control 1x1 branch nums and 3x3 branch nums
    RepVGGEnhanceDilationBlock use dilation to control 3x3 conv dilation setting
    '''

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 k=4,
                 deploy=False):
        super(RepVGGEnhanceDilationBlock, self).__init__()
        self.inplanes = inplanes
        self.groups = groups
        self.k = k
        self.deploy = deploy

        assert kernel_size == 3 and padding == 1

        if self.deploy:
            self.fuse_equivalent_conv = nn.Conv2d(inplanes,
                                                  planes,
                                                  kernel_size,
                                                  stride=stride,
                                                  padding=padding * dilation,
                                                  dilation=dilation,
                                                  groups=groups,
                                                  bias=True,
                                                  padding_mode='zeros')

        else:
            self.identity = nn.BatchNorm2d(
                inplanes) if inplanes == planes and stride == 1 else None
            self.conv3x3_list = []
            for _ in range(self.k):
                self.conv3x3_list.append(
                    conv_bn_dilation_layer(inplanes,
                                           planes,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding * dilation,
                                           dilation=dilation,
                                           groups=groups))
            self.conv3x3_list = nn.ModuleList(self.conv3x3_list)

            self.conv1x1_list = []
            for _ in range(self.k):
                self.conv1x1_list.append(
                    conv_bn_layer(inplanes,
                                  planes,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=padding - kernel_size // 2,
                                  groups=groups))
            self.conv1x1_list = nn.ModuleList(self.conv1x1_list)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            x = self.relu(self.fuse_equivalent_conv(x))

            return x

        if self.identity:
            identity_out = self.identity(x)
        else:
            identity_out = 0

        out = identity_out
        for idx in range(self.k):
            out += self.conv1x1_list[idx](x)

        for idx in range(self.k):
            out += self.conv3x3_list[idx](x)

        out = self.relu(out)

        return out

    def _fuse_bn_layer(self, branch):
        '''
        fuse conv and bn layers to get equivalent conv layer kernel and bias
        '''
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            # make sure conv layer doesn't have bias
            kernel = branch.conv.weight
            running_mean, running_var = branch.bn.running_mean, branch.bn.running_var
            gamma, beta, eps = branch.bn.weight, branch.bn.bias, branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            # convert identity branch to get a equivalent 1x1 conv layer kernel and bias
            input_dim = self.inplanes // self.groups
            kernel_value = np.zeros((self.inplanes, input_dim, 3, 3),
                                    dtype=np.float32)
            for i in range(self.inplanes):
                kernel_value[i, i % input_dim, 1, 1] = 1

            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma, beta, eps = branch.weight, branch.bias, branch.eps

        # fuse conv and bn layer
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        equivalent_kernel, equivalent_bias = kernel * t, beta - running_mean * gamma / std

        return equivalent_kernel, equivalent_bias

    def get_equivalent_conv_kernel_bias(self):
        fuse_kernel3x3, fuse_bias3x3 = None, None
        for idx in range(self.k):
            kernel3x3, bias3x3 = self._fuse_bn_layer(self.conv3x3_list[idx])
            if idx == 0:
                fuse_kernel3x3, fuse_bias3x3 = kernel3x3, bias3x3
            else:
                fuse_kernel3x3 += kernel3x3
                fuse_bias3x3 += bias3x3

        fuse_kernel1x1, fuse_bias1x1 = None, None
        for idx in range(self.k):
            kernel1x1, bias1x1 = self._fuse_bn_layer(self.conv1x1_list[idx])
            if idx == 0:
                fuse_kernel1x1, fuse_bias1x1 = kernel1x1, bias1x1
            else:
                fuse_kernel1x1 += kernel1x1
                fuse_bias1x1 += bias1x1

        kernelidentity, biasidentity = self._fuse_bn_layer(self.identity)

        # 1x1kernel must be pad to 3x3kernel before add
        kernel, bias = fuse_kernel3x3 + F.pad(fuse_kernel1x1, [
            1, 1, 1, 1
        ]) + kernelidentity, fuse_bias3x3 + fuse_bias1x1 + biasidentity
        kernel, bias = kernel.detach().cpu(), bias.detach().cpu()

        return kernel, bias


class RepVGGNet(nn.Module):

    def __init__(self, planes=[16, 16, 32, 48, 64, 80], deploy=False):
        super(RepVGGNet, self).__init__()
        self.deploy = deploy

        self.layers = nn.Sequential(
            RepVGGBlock(3,
                        planes[0],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),  # [B,16,h,w]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/2,w/2]
            RepVGGBlock(planes[0],
                        planes[1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),  # [B,16,h/2,w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/4,w/4]
            RepVGGBlock(planes[1],
                        planes[2],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),
            RepVGGBlock(planes[2],
                        planes[2],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),  # [B,32,h/4,w/4]    #5
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,32,h/8,w/8]
            RepVGGBlock(planes[2],
                        planes[3],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),
            RepVGGBlock(planes[3],
                        planes[3],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),  # [B,48,h/8,w/8]     #8
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,48,h/16,w/16]
            RepVGGBlock(planes[3],
                        planes[4],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),
            RepVGGBlock(planes[4],
                        planes[4],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),  # [B,64,h/16,w/16]   #11
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,64,h/32,w/32]
            RepVGGBlock(planes[4],
                        planes[5],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy),
            RepVGGBlock(planes[5],
                        planes[5],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        deploy=self.deploy))  # [B,80,h/32,w/32]   #14

        self.last_layer = RepVGGBlock(planes[5],
                                      planes[5] * 6,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      deploy=self.deploy)

        self.out_idx = [5, 8, 11, 14]
        self.out_channels = [planes[2], planes[3], planes[4], planes[5] * 6]

    def forward(self, x):
        out = []
        for idx, per_layer in enumerate(self.layers):
            x = per_layer(x)
            if idx in self.out_idx:
                out.append(x)

        out[-1] = self.last_layer(out[-1])

        return out


class RepVGGEnhanceNet(nn.Module):

    def __init__(self, planes=[16, 16, 32, 48, 64, 80], k=4, deploy=False):
        super(RepVGGEnhanceNet, self).__init__()
        self.k = k
        self.deploy = deploy

        self.layers = nn.Sequential(
            RepVGGEnhanceDilationBlock(3,
                                       planes[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),  # [B,16,h,w]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/2,w/2]
            RepVGGEnhanceDilationBlock(planes[0],
                                       planes[1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),  # [B,16,h/2,w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/4,w/4]
            RepVGGEnhanceDilationBlock(planes[1],
                                       planes[2],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[2],
                planes[2],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy),  # [B,32,h/4,w/4]    #5
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,32,h/8,w/8]
            RepVGGEnhanceDilationBlock(planes[2],
                                       planes[3],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[3],
                planes[3],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy),  # [B,48,h/8,w/8]     #8
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,48,h/16,w/16]
            RepVGGEnhanceDilationBlock(planes[3],
                                       planes[4],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[4],
                planes[4],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy),  # [B,64,h/16,w/16]   #11
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,64,h/32,w/32]
            RepVGGEnhanceDilationBlock(planes[4],
                                       planes[5],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[5],
                planes[5],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy))  # [B,80,h/32,w/32]   #14

        self.last_layer = RepVGGEnhanceDilationBlock(planes[5],
                                                     planes[5] * 6,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=1,
                                                     groups=1,
                                                     k=self.k,
                                                     deploy=self.deploy)

        self.out_idx = [5, 8, 11, 14]
        self.out_channels = [planes[2], planes[3], planes[4], planes[5] * 6]

    def forward(self, x):
        out = []
        for idx, per_layer in enumerate(self.layers):
            x = per_layer(x)
            if idx in self.out_idx:
                out.append(x)

        out[-1] = self.last_layer(out[-1])

        return out


class RepVGGEnhanceDilationNet(nn.Module):

    def __init__(self, planes=[16, 16, 32, 48, 64, 80], k=4, deploy=False):
        super(RepVGGEnhanceDilationNet, self).__init__()
        self.k = k
        self.deploy = deploy

        self.layers = nn.Sequential(
            RepVGGEnhanceDilationBlock(3,
                                       planes[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),  # [B,16,h,w]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/2,w/2]
            RepVGGEnhanceDilationBlock(planes[0],
                                       planes[1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),  # [B,16,h/2,w/2]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,16,h/4,w/4]
            RepVGGEnhanceDilationBlock(planes[1],
                                       planes[2],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[2],
                planes[2],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy),  # [B,32,h/4,w/4]    #5
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,32,h/8,w/8]
            RepVGGEnhanceDilationBlock(planes[2],
                                       planes[3],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=3,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[3],
                planes[3],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy),  # [B,48,h/8,w/8]     #8
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,48,h/16,w/16]
            RepVGGEnhanceDilationBlock(planes[3],
                                       planes[4],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=5,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[4],
                planes[4],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy),  # [B,64,h/16,w/16]   #11
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B,64,h/32,w/32]
            RepVGGEnhanceDilationBlock(planes[4],
                                       planes[5],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=5,
                                       groups=1,
                                       k=self.k,
                                       deploy=self.deploy),
            RepVGGEnhanceDilationBlock(
                planes[5],
                planes[5],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                k=self.k,
                deploy=self.deploy))  # [B,80,h/32,w/32]   #14

        self.last_layer = RepVGGEnhanceDilationBlock(planes[5],
                                                     planes[5] * 6,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=1,
                                                     groups=1,
                                                     k=self.k,
                                                     deploy=self.deploy)

        self.out_idx = [5, 8, 11, 14]
        self.out_channels = [planes[2], planes[3], planes[4], planes[5] * 6]

    def forward(self, x):
        out = []
        for idx, per_layer in enumerate(self.layers):
            x = per_layer(x)
            if idx in self.out_idx:
                out.append(x)

        out[-1] = self.last_layer(out[-1])

        return out


def VGGNetBackbone(planes, pretrained_path='', **kwargs):
    model = VGGNet(planes, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def RepVGGNetBackbone(planes, deploy=True, pretrained_path='', **kwargs):
    model = RepVGGNet(planes, deploy=deploy, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def RepVGGEnhanceNetBackbone(planes,
                             k=4,
                             deploy=True,
                             pretrained_path='',
                             **kwargs):
    model = RepVGGEnhanceNet(planes, k=k, deploy=deploy, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def RepVGGEnhanceDilationNetBackbone(planes,
                                     k=4,
                                     deploy=True,
                                     pretrained_path='',
                                     **kwargs):
    model = RepVGGEnhanceDilationNet(planes, k=k, deploy=deploy, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def deploy_model(trained_model, deployed_model):
    deploy_model_weights = {}
    for name, module in trained_model.named_modules():
        if hasattr(module, 'get_equivalent_conv_kernel_bias'):
            kernel, bias = module.get_equivalent_conv_kernel_bias()
            deploy_model_weights[name +
                                 '.fuse_equivalent_conv.weight'] = kernel
            deploy_model_weights[name + '.fuse_equivalent_conv.bias'] = bias
        elif isinstance(module, nn.Linear):
            deploy_model_weights[name +
                                 '.weight'] = module.weight.detach().cpu()
            deploy_model_weights[name + '.bias'] = module.bias.detach().cpu()
        else:
            # named_parameters return all layers that need to be backpropagated,such as conv layer or linear layer
            for layer_name, layer_weights in module.named_parameters():
                full_name = name + '.' + layer_name
                if full_name not in deploy_model_weights.keys():
                    deploy_model_weights[full_name] = layer_weights.detach(
                    ).cpu()
            # named_buffers return all layers that don't need to be backpropagated,such as bn layer
            for layer_name, layer_weights in module.named_buffers():
                full_name = name + '.' + layer_name
                if full_name not in deploy_model_weights.keys():
                    deploy_model_weights[full_name] = layer_weights.cpu()

    # load all equivalent weights,and the other weights will be abandoned(self.conv3x3,self.conv1x1,self.identity in RepVGGBlock).
    deployed_model.load_state_dict(deploy_model_weights, strict=False)

    return deployed_model


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

    net = VGGNetBackbone(planes=[16, 16, 32, 48, 64, 80])
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params}')
    print(net.out_channels)

    for per_feature in out:
        print('1111', per_feature.shape)

    net = RepVGGNetBackbone(planes=[16, 16, 32, 48, 64, 80], deploy=True)
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params}')
    print(net.out_channels)

    for per_feature in out:
        print('2222', per_feature.shape)

    net = RepVGGEnhanceNetBackbone(planes=[16, 16, 32, 48, 64, 80],
                                   k=4,
                                   deploy=True)
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params}')
    print(net.out_channels)

    for per_feature in out:
        print('3333', per_feature.shape)

    net = RepVGGEnhanceDilationNetBackbone(planes=[16, 16, 32, 48, 64, 80],
                                           k=4,
                                           deploy=True)
    image_h, image_w = 960, 960
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params}')
    print(net.out_channels)

    for per_feature in out:
        print('4444', per_feature.shape)

    ########################################################
    image_h, image_w = 960, 960
    images = torch.randn(1, 3, image_h, image_w)

    net1 = RepVGGEnhanceNetBackbone(planes=[16, 16, 32, 48, 64, 80],
                                    k=4,
                                    deploy=False)
    torch.save(net1.state_dict(), '1111_trained.pth')
    net1.load_state_dict(
        torch.load('1111_trained.pth', map_location=torch.device('cpu')))
    net1.eval()

    net2 = RepVGGEnhanceNetBackbone(planes=[16, 16, 32, 48, 64, 80],
                                    k=4,
                                    deploy=True)
    net2 = deploy_model(net1, net2)
    net2.eval()

    out1 = net1(images)
    out2 = net2(images)
    print(((out1[0] - out2[0])**2).sum(), ((out1[1] - out2[1])**2).sum())

    image_h, image_w = 960, 960
    images = torch.randn(1, 3, image_h, image_w)

    net1 = RepVGGEnhanceDilationNetBackbone(planes=[16, 16, 32, 48, 64, 80],
                                            k=4,
                                            deploy=False)
    torch.save(net1.state_dict(), '1111_trained.pth')
    net1.load_state_dict(
        torch.load('1111_trained.pth', map_location=torch.device('cpu')))
    net1.eval()

    net2 = RepVGGEnhanceDilationNetBackbone(planes=[16, 16, 32, 48, 64, 80],
                                            k=4,
                                            deploy=True)
    net2 = deploy_model(net1, net2)
    net2.eval()

    out1 = net1(images)
    out2 = net2(images)
    print(((out1[0] - out2[0])**2).sum(), ((out1[1] - out2[1])**2).sum())
