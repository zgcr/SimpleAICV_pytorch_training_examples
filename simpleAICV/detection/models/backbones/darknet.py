import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.classification.backbones.darknet import ConvBnActBlock, Darknet19Block, Darknet53Block
from simpleAICV.detection.common import load_state_dict

__all__ = [
    'darknettinybackbone',
    'darknet19backbone',
    'darknet53backbone',
]


class DarknetTinyBackbone(nn.Module):

    def __init__(self, act_type='leakyrelu'):
        super(DarknetTinyBackbone, self).__init__()

        self.conv1 = ConvBnActBlock(3,
                                    16,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBnActBlock(16,
                                    32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBnActBlock(32,
                                    64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = ConvBnActBlock(64,
                                    128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBnActBlock(128,
                                    256,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = ConvBnActBlock(256,
                                    512,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.out_channels = [256, 512]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)

        C4 = self.conv5(x)
        C5 = self.maxpool5(C4)
        C5 = self.conv6(C5)
        C5 = self.zeropad(C5)
        C5 = self.maxpool6(C5)

        del x

        return [C4, C5]


class Darknet19Backbone(nn.Module):

    def __init__(self, act_type='leakyrelu'):
        super(Darknet19Backbone, self).__init__()
        self.layer1 = ConvBnActBlock(3,
                                     32,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1,
                                     has_bn=True,
                                     has_act=True,
                                     act_type=act_type)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Darknet19Block(32,
                                     64,
                                     layer_num=1,
                                     use_maxpool=True,
                                     act_type=act_type)
        self.layer3 = Darknet19Block(64,
                                     128,
                                     layer_num=3,
                                     use_maxpool=True,
                                     act_type=act_type)
        self.layer4 = Darknet19Block(128,
                                     256,
                                     layer_num=3,
                                     use_maxpool=True,
                                     act_type=act_type)
        self.layer5 = Darknet19Block(256,
                                     512,
                                     layer_num=5,
                                     use_maxpool=True,
                                     act_type=act_type)
        self.layer6 = Darknet19Block(512,
                                     1024,
                                     layer_num=5,
                                     use_maxpool=False,
                                     act_type=act_type)

        self.out_channels = [128, 256, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)

        C3 = self.layer3(x)
        C4 = self.layer4(C3)
        C5 = self.layer5(C4)
        C5 = self.layer6(C5)

        del x

        return [C3, C4, C5]


class Darknet53Backbone(nn.Module):

    def __init__(self, act_type='leakyrelu'):
        super(Darknet53Backbone, self).__init__()
        self.conv1 = ConvBnActBlock(3,
                                    32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(32,
                                    64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = self.make_layer(inplanes=64,
                                      num_blocks=1,
                                      act_type=act_type)
        self.conv3 = ConvBnActBlock(64,
                                    128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block2 = self.make_layer(inplanes=128,
                                      num_blocks=2,
                                      act_type=act_type)
        self.conv4 = ConvBnActBlock(128,
                                    256,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block3 = self.make_layer(inplanes=256,
                                      num_blocks=8,
                                      act_type=act_type)
        self.conv5 = ConvBnActBlock(256,
                                    512,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block4 = self.make_layer(inplanes=512,
                                      num_blocks=8,
                                      act_type=act_type)
        self.conv6 = ConvBnActBlock(512,
                                    1024,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block5 = self.make_layer(inplanes=1024,
                                      num_blocks=4,
                                      act_type=act_type)

        self.out_channels = [256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.block2(x)
        x = self.conv4(x)

        C3 = self.block3(x)
        C4 = self.conv5(C3)
        C4 = self.block4(C4)
        C5 = self.conv6(C4)
        C5 = self.block5(C5)

        del x

        return [C3, C4, C5]

    def make_layer(self, inplanes, num_blocks, act_type):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(Darknet53Block(inplanes, act_type=act_type))
        return nn.Sequential(*layers)


def darknettinybackbone(pretrained_path='', **kwargs):
    model = DarknetTinyBackbone(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def darknet19backbone(pretrained_path='', **kwargs):
    model = Darknet19Backbone(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def darknet53backbone(pretrained_path='', **kwargs):
    model = Darknet53Backbone(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

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

    net = darknettinybackbone()
    image_h, image_w = 416, 416
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

    net = darknet19backbone()
    image_h, image_w = 640, 640
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

    net = darknet53backbone()
    image_h, image_w = 640, 640
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