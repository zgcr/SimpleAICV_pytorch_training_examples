import torch
import torch.nn as nn

__all__ = [
    'yolov4cspdarknettiny',
    'yolov4cspdarknet53',
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


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, squeeze=False, act_type='leakyrelu'):
        super(ResBlock, self).__init__()
        squeezed_planes = max(1, int(inplanes // 2)) if squeeze else inplanes
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
                           planes,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))

    def forward(self, x):
        x = x + self.conv(x)

        return x


class CSPDarkNetTinyBlock(nn.Module):

    def __init__(self, inplanes, planes, act_type='leakyrelu'):
        super(CSPDarkNetTinyBlock, self).__init__()
        self.planes = planes

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(planes // 2,
                                    planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv3 = ConvBnActBlock(planes // 2,
                                    planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv4 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)

        _, x = torch.split(x1, self.planes // 2, dim=1)

        x2 = self.conv2(x)
        x = self.conv3(x2)

        x = torch.cat([x, x2], dim=1)

        x3 = self.conv4(x)

        x = torch.cat([x1, x3], dim=1)

        x = self.maxpool(x)

        return x, x3


class CSPDarkNetBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 num_blocks,
                 reduction=True,
                 act_type='leakyrelu'):
        super(CSPDarkNetBlock, self).__init__()
        self.front_conv = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        blocks = nn.Sequential(*[
            ResBlock(planes // 2 if reduction else planes,
                     planes // 2 if reduction else planes,
                     squeeze=not reduction) for _ in range(num_blocks)
        ])
        self.left_conv = nn.Sequential(
            ConvBnActBlock(planes,
                           planes // 2 if reduction else planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type), blocks,
            ConvBnActBlock(planes // 2 if reduction else planes,
                           planes // 2 if reduction else planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))
        self.right_conv = ConvBnActBlock(planes,
                                         planes // 2 if reduction else planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.out_conv = ConvBnActBlock(planes if reduction else planes * 2,
                                       planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

    def forward(self, x):
        x = self.front_conv(x)
        left = self.left_conv(x)
        right = self.right_conv(x)

        x = torch.cat([left, right], dim=1)

        del left, right

        x = self.out_conv(x)

        return x


class CSPDarknetTiny(nn.Module):

    def __init__(self,
                 planes=[64, 128, 256, 512],
                 act_type='leakyrelu',
                 num_classes=1000):
        super(CSPDarknetTiny, self).__init__()
        self.num_classes = num_classes

        self.conv1 = ConvBnActBlock(3,
                                    32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(32,
                                    planes[0],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetTinyBlock(planes[0],
                                          planes[0],
                                          act_type=act_type)
        self.block2 = CSPDarkNetTinyBlock(planes[1],
                                          planes[1],
                                          act_type=act_type)
        self.block3 = CSPDarkNetTinyBlock(planes[2],
                                          planes[2],
                                          act_type=act_type)
        self.conv3 = ConvBnActBlock(planes[3],
                                    planes[3],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3], self.num_classes)

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
        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, _ = self.block3(x)
        x = self.conv3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CSPDarknet53(nn.Module):

    def __init__(self,
                 inplanes=32,
                 planes=[64, 128, 256, 512, 1024],
                 act_type='leakyrelu',
                 num_classes=1000):
        super(CSPDarknet53, self).__init__()
        self.num_classes = num_classes

        self.conv1 = ConvBnActBlock(3,
                                    inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetBlock(inplanes,
                                      planes[0],
                                      num_blocks=1,
                                      reduction=False,
                                      act_type=act_type)
        self.block2 = CSPDarkNetBlock(planes[0],
                                      planes[1],
                                      num_blocks=2,
                                      reduction=True,
                                      act_type=act_type)
        self.block3 = CSPDarkNetBlock(planes[1],
                                      planes[2],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block4 = CSPDarkNetBlock(planes[2],
                                      planes[3],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block5 = CSPDarkNetBlock(planes[3],
                                      planes[4],
                                      num_blocks=4,
                                      reduction=True,
                                      act_type=act_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[4], self.num_classes)

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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def yolov4cspdarknettiny(**kwargs):
    model = CSPDarknetTiny(**kwargs)

    return model


def yolov4cspdarknet53(**kwargs):
    model = CSPDarknet53(**kwargs)

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

    net = yolov4cspdarknettiny(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = yolov4cspdarknet53(num_classes=1000)
    image_h, image_w = 256, 256
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')