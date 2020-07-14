"""
YOLOv3: An Incremental Improvement
https://arxiv.org/pdf/1804.02767.pdf
"""
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

import torch
import torch.nn as nn

__all__ = [
    'darknet19',
    'darknet53',
]

model_urls = {
    'darknet19':
    '{}/darknet/darknet19-input256-epoch100-acc73.868.pth'.format(
        pretrained_models_path),
    'darknet53':
    '{}/darknet/darknet53-input256-epoch100-acc77.008.pth'.format(
        pretrained_models_path),
}


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding=1,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(planes)
        if self.has_act:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.act(x)

        return x


class Darknet19Block(nn.Module):
    def __init__(self, inplanes, planes, layer_num, use_maxpool=False):
        super(Darknet19Block, self).__init__()
        self.use_maxpool = use_maxpool
        layers = []
        for i in range(0, layer_num):
            if i % 2 == 0:
                layers.append(
                    ConvBnActBlock(inplanes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True))
            else:
                layers.append(
                    ConvBnActBlock(planes,
                                   inplanes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True))

        self.Darknet19Block = nn.Sequential(*layers)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.Darknet19Block(x)

        if self.use_maxpool:
            x = self.MaxPool(x)

        return x


class Darknet53Block(nn.Module):
    def __init__(self, inplanes):
        super(Darknet53Block, self).__init__()
        reduced_planes = int(inplanes * 0.5)
        self.conv1 = ConvBnActBlock(inplanes,
                                    reduced_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(reduced_planes,
                                    inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = x + inputs

        return x


class Darknet19(nn.Module):
    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.layer1 = ConvBnActBlock(3,
                                     32,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1,
                                     has_bn=True,
                                     has_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Darknet19Block(32, 64, 1, use_maxpool=True)
        self.layer3 = Darknet19Block(64, 128, 3, use_maxpool=True)
        self.layer4 = Darknet19Block(128, 256, 3, use_maxpool=True)
        self.layer5 = Darknet19Block(256, 512, 5, use_maxpool=True)
        self.layer6 = Darknet19Block(512, 1024, 5, use_maxpool=False)
        self.layer7 = ConvBnActBlock(1024,
                                     1000,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=1,
                                     has_bn=False,
                                     has_act=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class Darknet53(nn.Module):
    def __init__(self, num_classes=1000):
        super(Darknet53, self).__init__()
        self.conv1 = ConvBnActBlock(3,
                                    32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(32,
                                    64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.block1 = self.make_layer(inplanes=64, num_blocks=1)
        self.conv3 = ConvBnActBlock(64,
                                    128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.block2 = self.make_layer(inplanes=128, num_blocks=2)
        self.conv4 = ConvBnActBlock(128,
                                    256,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.block3 = self.make_layer(inplanes=256, num_blocks=8)
        self.conv5 = ConvBnActBlock(256,
                                    512,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.block4 = self.make_layer(inplanes=512, num_blocks=8)
        self.conv6 = ConvBnActBlock(512,
                                    1024,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.block5 = self.make_layer(inplanes=1024, num_blocks=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.block2(x)
        x = self.conv4(x)
        x = self.block3(x)
        x = self.conv5(x)
        x = self.block4(x)
        x = self.conv6(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x

    def make_layer(self, inplanes, num_blocks):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(Darknet53Block(inplanes))
        return nn.Sequential(*layers)


def darknet19(pretrained=False, progress=True, **kwargs):
    model = Darknet19(**kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls['darknet19'],
                       map_location=torch.device('cpu')))
    return model


def darknet53(pretrained=False, progress=True, **kwargs):
    model = Darknet53(**kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls['darknet53'],
                       map_location=torch.device('cpu')))
    return model