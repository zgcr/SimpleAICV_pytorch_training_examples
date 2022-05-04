'''
YOLOv3: An Incremental Improvement
https://arxiv.org/pdf/1804.02767.pdf
'''
import torch
import torch.nn as nn

__all__ = [
    'darknettiny',
    'darknet19',
    'darknet53',
]


class ActivationBlock(nn.Module):

    def __init__(self, act_type='leakyrelu', inplace=True):
        super(ActivationBlock, self).__init__()
        assert act_type in ['silu', 'relu',
                            'leakyrelu'], 'Unsupport activation function!'
        if act_type == 'silu':
            self.act = nn.SiLU(inplace=inplace)
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1, inplace=inplace)

    def forward(self, x):
        x = self.act(x)

        return x


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True,
                 act_type='leakyrelu'):
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
            ActivationBlock(act_type=act_type, inplace=True)
            if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class Darknet19Block(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 layer_num,
                 use_maxpool=False,
                 act_type='leakyrelu'):
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
                                   has_act=True,
                                   act_type=act_type))
            else:
                layers.append(
                    ConvBnActBlock(planes,
                                   inplanes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type))

        self.Darknet19Block = nn.Sequential(*layers)
        if self.use_maxpool:
            self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.Darknet19Block(x)

        if self.use_maxpool:
            x = self.MaxPool(x)

        return x


class Darknet53Block(nn.Module):

    def __init__(self, inplanes, act_type='leakyrelu'):
        super(Darknet53Block, self).__init__()
        squeezed_planes = int(inplanes // 2)
        self.conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           squeezed_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(squeezed_planes,
                           inplanes,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))

    def forward(self, x):
        x = self.conv(x) + x

        return x


class DarknetTiny(nn.Module):

    def __init__(self, act_type='leakyrelu', num_classes=1000):
        super(DarknetTiny, self).__init__()
        self.num_classes = num_classes

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = self.zeropad(x)
        x = self.maxpool6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Darknet19(nn.Module):

    def __init__(self, act_type='leakyrelu', num_classes=1000):
        super(Darknet19, self).__init__()
        self.num_classes = num_classes

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
        self.layer7 = ConvBnActBlock(1024,
                                     self.num_classes,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=1,
                                     has_bn=False,
                                     has_act=False,
                                     act_type=act_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
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

    def __init__(self, act_type='leakyrelu', num_classes=1000):
        super(Darknet53, self).__init__()
        self.num_classes = num_classes

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def make_layer(self, inplanes, num_blocks, act_type):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(Darknet53Block(inplanes, act_type=act_type))
        return nn.Sequential(*layers)


def darknettiny(**kwargs):
    model = DarknetTiny(**kwargs)

    return model


def darknet19(**kwargs):
    model = Darknet19(**kwargs)

    return model


def darknet53(**kwargs):
    model = Darknet53(**kwargs)

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

    net = darknettiny(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = darknet19(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = darknet53(num_classes=1000)
    image_h, image_w = 256, 256
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')
