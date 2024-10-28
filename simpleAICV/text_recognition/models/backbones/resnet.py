import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from simpleAICV.classification.backbones.resnet import ConvBnActBlock
from simpleAICV.detection.common import load_state_dict

__all__ = [
    'resnet18backbone',
    'resnet34backbone',
    'resnet50backbone',
    'resnet101backbone',
    'resnet152backbone',
]


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        if isinstance(stride, tuple):
            self.downsample = True if max(
                stride) != 1 or inplanes != planes * 1 else False
        else:
            self.downsample = True if stride != 1 or inplanes != planes * 1 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(Bottleneck, self).__init__()
        if isinstance(stride, tuple):
            self.downsample = True if max(
                stride) != 1 or inplanes != planes * 4 else False
        else:
            self.downsample = True if stride != 1 or inplanes != planes * 4 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(planes,
                                    planes * 4,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes * 4,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class ResNetBackbone(nn.Module):

    def __init__(self,
                 block,
                 layer_nums,
                 inplanes=64,
                 use_gradient_checkpoint=False):
        super(ResNetBackbone, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.conv1 = ConvBnActBlock(3,
                                    self.inplanes,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block,
                                      self.planes[0],
                                      self.layer_nums[0],
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.layer2 = self.make_layer(self.block,
                                      self.planes[1],
                                      self.layer_nums[1],
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)
        self.layer3 = self.make_layer(self.block,
                                      self.planes[2],
                                      self.layer_nums[2],
                                      kernel_size=(3, 1),
                                      stride=(2, 1),
                                      padding=(1, 0))
        self.layer4 = self.make_layer(self.block,
                                      self.planes[3],
                                      self.layer_nums[3],
                                      kernel_size=(3, 1),
                                      stride=(2, 1),
                                      padding=(1, 0))

        self.out_channels = [
            self.planes[0] * self.expansion,
            self.planes[1] * self.expansion,
            self.planes[2] * self.expansion,
            self.planes[3] * self.expansion,
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, planes, layer_nums, kernel_size, stride,
                   padding):
        layers = []
        for i in range(0, layer_nums):
            if i == 0:
                layers.append(
                    block(self.inplanes, planes, kernel_size, stride, padding))
            else:
                layers.append(
                    block(self.inplanes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.inplanes = planes * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        if self.use_gradient_checkpoint:
            C2 = checkpoint(self.layer1, x, use_reentrant=False)
            C3 = checkpoint(self.layer2, C2, use_reentrant=False)
            C4 = checkpoint(self.layer3, C3, use_reentrant=False)
            C5 = checkpoint(self.layer4, C4, use_reentrant=False)
        else:
            C2 = self.layer1(x)
            C3 = self.layer2(C2)
            C4 = self.layer3(C3)
            C5 = self.layer4(C4)

        del x

        return [C2, C3, C4, C5]


def _resnetbackbone(block, layers, inplanes, pretrained_path='', **kwargs):
    model = ResNetBackbone(block, layers, inplanes, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def resnet18backbone(pretrained_path='', **kwargs):
    model = _resnetbackbone(BasicBlock, [2, 2, 2, 2],
                            64,
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def resnet34backbone(pretrained_path='', **kwargs):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            64,
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def resnet50backbone(pretrained_path='', **kwargs):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            64,
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def resnet101backbone(pretrained_path='', **kwargs):
    model = _resnetbackbone(Bottleneck, [3, 4, 23, 3],
                            64,
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


def resnet152backbone(pretrained_path='', **kwargs):
    model = _resnetbackbone(Bottleneck, [3, 8, 36, 3],
                            64,
                            pretrained_path=pretrained_path,
                            **kwargs)

    return model


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

    net = resnet18backbone()
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet34backbone()
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet50backbone()
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet101backbone()
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet152backbone()
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet152backbone(use_gradient_checkpoint=True)
    image_h, image_w = 32, 512
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)
