import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.classification.backbones.resnet import ConvBnActBlock, BasicBlock, Bottleneck
from simpleAICV.detection.common import load_state_dict

__all__ = [
    'resnet18backbone',
    'resnet34halfbackbone',
    'resnet34backbone',
    'resnet50halfbackbone',
    'resnet50backbone',
    'resnet101backbone',
    'resnet152backbone',
]


class ResNetBackbone(nn.Module):

    def __init__(self, block, layer_nums, inplanes=64):
        super(ResNetBackbone, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4

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
                                      stride=1)
        self.layer2 = self.make_layer(self.block,
                                      self.planes[1],
                                      self.layer_nums[1],
                                      stride=2)
        self.layer3 = self.make_layer(self.block,
                                      self.planes[2],
                                      self.layer_nums[2],
                                      stride=2)
        self.layer4 = self.make_layer(self.block,
                                      self.planes[3],
                                      self.layer_nums[3],
                                      stride=2)

        self.out_channels = [
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

    def make_layer(self, block, planes, layer_nums, stride):
        layers = []
        for i in range(0, layer_nums):
            if i == 0:
                layers.append(block(self.inplanes, planes, stride))
            else:
                layers.append(block(self.inplanes, planes))
            self.inplanes = planes * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        C3 = self.layer2(x)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        del x

        return [C3, C4, C5]


def _resnetbackbone(block, layers, inplanes, pretrained_path=''):
    model = ResNetBackbone(block, layers, inplanes)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def resnet18backbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [2, 2, 2, 2],
                            64,
                            pretrained_path=pretrained_path)

    return model


def resnet34halfbackbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            32,
                            pretrained_path=pretrained_path)

    return model


def resnet34backbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def resnet50halfbackbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            32,
                            pretrained_path=pretrained_path)

    return model


def resnet50backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def resnet101backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 23, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def resnet152backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 8, 36, 3],
                            64,
                            pretrained_path=pretrained_path)

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
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet34halfbackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = resnet50backbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)