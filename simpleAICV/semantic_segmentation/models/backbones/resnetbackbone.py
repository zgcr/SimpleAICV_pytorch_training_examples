import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.semantic_segmentation.common import load_state_dict

__all__ = [
    'resnet18backbone',
    'resnet34halfbackbone',
    'resnet34backbone',
    'resnet50halfbackbone',
    'resnet50backbone',
    'resnet101backbone',
    'resnet152backbone',
]


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


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, padding=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 1 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=padding,
                                    groups=1,
                                    dilation=dilation,
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

    def __init__(self, inplanes, planes, stride=1, padding=1, dilation=1):
        super(Bottleneck, self).__init__()
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
                                    kernel_size=3,
                                    stride=stride,
                                    padding=padding,
                                    groups=1,
                                    dilation=dilation,
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

    def __init__(self, block, layer_nums, inplanes=64, dilation=[1, 1, 2, 4]):
        super(ResNetBackbone, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4
        self.dilation = dilation

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
                                      stride=1,
                                      last_dilation=1,
                                      dilation=self.dilation[0])
        self.layer2 = self.make_layer(self.block,
                                      self.planes[1],
                                      self.layer_nums[1],
                                      stride=2,
                                      last_dilation=self.dilation[0],
                                      dilation=self.dilation[1])
        self.layer3 = self.make_layer(self.block,
                                      self.planes[2],
                                      self.layer_nums[2],
                                      stride=2,
                                      last_dilation=self.dilation[1],
                                      dilation=self.dilation[2])
        self.layer4 = self.make_layer(self.block,
                                      self.planes[3],
                                      self.layer_nums[3],
                                      stride=2,
                                      last_dilation=self.dilation[2],
                                      dilation=self.dilation[3])

        self.out_channels = [
            inplanes * self.expansion,
            inplanes * 2 * self.expansion,
            inplanes * 4 * self.expansion,
            inplanes * 8 * self.expansion,
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self,
                   block,
                   planes,
                   layer_nums,
                   stride,
                   last_dilation=1,
                   dilation=1):
        layers = []
        for i in range(0, layer_nums):
            if i == 0:
                layers.append(
                    block(self.inplanes,
                          planes,
                          stride,
                          padding=last_dilation,
                          dilation=last_dilation))
            else:
                layers.append(
                    block(self.inplanes,
                          planes,
                          padding=dilation,
                          dilation=dilation))
            self.inplanes = planes * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        C1 = self.layer1(x)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3)

        return C1, C2, C3, C4


def _resnetbackbone(block,
                    layers,
                    inplanes,
                    dilation,
                    pretrained_path='',
                    **kwargs):
    model = ResNetBackbone(block, layers, inplanes, dilation, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path,
                        model,
                        loading_new_input_size_position_encoding_weight=False)
    else:
        print('no backbone pretrained model!')

    return model

    return model


def resnet18backbone(pretrained_path='', **kwargs):
    return _resnetbackbone(BasicBlock, [2, 2, 2, 2],
                           64, [1, 1, 1, 1],
                           pretrained_path=pretrained_path,
                           **kwargs)


def resnet34halfbackbone(pretrained_path='', **kwargs):
    return _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                           32, [1, 1, 1, 1],
                           pretrained_path=pretrained_path,
                           **kwargs)


def resnet34backbone(pretrained_path='', **kwargs):
    return _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                           64, [1, 1, 1, 1],
                           pretrained_path=pretrained_path,
                           **kwargs)


def resnet50halfbackbone(pretrained_path='', **kwargs):
    return _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                           32, [1, 1, 2, 4],
                           pretrained_path=pretrained_path,
                           **kwargs)


def resnet50backbone(pretrained_path='', **kwargs):
    return _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                           64, [1, 1, 2, 4],
                           pretrained_path=pretrained_path,
                           **kwargs)


def resnet101backbone(pretrained_path='', **kwargs):
    return _resnetbackbone(Bottleneck, [3, 4, 23, 3],
                           64, [1, 1, 2, 4],
                           pretrained_path=pretrained_path,
                           **kwargs)


def resnet152backbone(pretrained_path='', **kwargs):
    return _resnetbackbone(Bottleneck, [3, 8, 36, 3],
                           64, [1, 1, 2, 4],
                           pretrained_path=pretrained_path,
                           **kwargs)


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
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    outs = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params}')
    for per_out in outs:
        print(f'1111', per_out.shape)

    net = resnet34halfbackbone()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    outs = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params}')
    for per_out in outs:
        print(f'2222', per_out.shape)

    net = resnet34backbone()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    outs = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params}')
    for per_out in outs:
        print(f'3333', per_out.shape)

    net = resnet50halfbackbone()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    outs = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params}')
    for per_out in outs:
        print(f'4444', per_out.shape)

    net = resnet50backbone()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    outs = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params}')
    for per_out in outs:
        print(f'5555', per_out.shape)

    net = resnet101backbone()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    outs = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'6666, macs: {macs}, params: {params}')
    for per_out in outs:
        print(f'6666', per_out.shape)

    net = resnet152backbone()
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    outs = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'7777, macs: {macs}, params: {params}')
    for per_out in outs:
        print(f'7777', per_out.shape)