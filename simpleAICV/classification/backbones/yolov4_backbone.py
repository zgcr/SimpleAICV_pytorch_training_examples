import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from simpleAICV.classification.common import load_state_dict
from tools.path import pretrained_models_path

import torch
import torch.nn as nn

__all__ = [
    'yolov4_cspdarknettiny',
    'yolov4_cspdarknet53',
]

model_urls = {
    'yolov4_cspdarknettiny': 'empty',
    'yolov4_cspdarknet53': 'empty',
}


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True) if has_act else nn.Sequential())

    def forward(self, x):
        x = self.layer(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, squeeze=False):
        super(ResBlock, self).__init__()
        squeezed_planes = max(1, int(inplanes // 2)) if squeeze else inplanes
        self.conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           squeezed_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0),
            ConvBnActBlock(squeezed_planes,
                           planes,
                           kernel_size=3,
                           stride=1,
                           padding=1))

    def forward(self, x):
        x = x + self.conv(x)

        return x


class CSPDarkNetTinyBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(CSPDarkNetTinyBlock, self).__init__()
        self.planes = planes
        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.conv2 = ConvBnActBlock(planes // 2,
                                    planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        self.conv3 = ConvBnActBlock(planes // 2,
                                    planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.conv4 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

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
    def __init__(self, inplanes, planes, num_blocks, reduction=True):
        super(CSPDarkNetBlock, self).__init__()
        self.front_conv = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
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
                           padding=0), blocks,
            ConvBnActBlock(planes // 2 if reduction else planes,
                           planes // 2 if reduction else planes,
                           kernel_size=1,
                           stride=1,
                           padding=0))
        self.right_conv = ConvBnActBlock(planes,
                                         planes // 2 if reduction else planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.out_conv = ConvBnActBlock(planes if reduction else planes * 2,
                                       planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        x = self.front_conv(x)
        left = self.left_conv(x)
        right = self.right_conv(x)

        del x

        out = torch.cat([left, right], dim=1)
        out = self.out_conv(out)

        return out


class CSPDarknetTiny(nn.Module):
    def __init__(self, planes=[64, 128, 256, 512], num_classes=1000):
        super(CSPDarknetTiny, self).__init__()
        self.conv1 = ConvBnActBlock(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBnActBlock(32,
                                    planes[0],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.block1 = CSPDarkNetTinyBlock(planes[0], planes[0])
        self.block2 = CSPDarkNetTinyBlock(planes[1], planes[1])
        self.block3 = CSPDarkNetTinyBlock(planes[2], planes[2])
        self.conv3 = ConvBnActBlock(planes[3],
                                    planes[3],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3], num_classes)

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
        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, _ = self.block3(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x


class CSPDarknet53(nn.Module):
    def __init__(self,
                 inplanes=32,
                 planes=[64, 128, 256, 512, 1024],
                 num_classes=1000):
        super(CSPDarknet53, self).__init__()
        self.conv1 = ConvBnActBlock(3,
                                    inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.block1 = CSPDarkNetBlock(inplanes,
                                      planes[0],
                                      num_blocks=1,
                                      reduction=False)
        self.block2 = CSPDarkNetBlock(planes[0],
                                      planes[1],
                                      num_blocks=2,
                                      reduction=True)
        self.block3 = CSPDarkNetBlock(planes[1],
                                      planes[2],
                                      num_blocks=8,
                                      reduction=True)
        self.block4 = CSPDarkNetBlock(planes[2],
                                      planes[3],
                                      num_blocks=8,
                                      reduction=True)
        self.block5 = CSPDarkNetBlock(planes[3],
                                      planes[4],
                                      num_blocks=4,
                                      reduction=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x


def cspdarknettiny(pretrained=False, **kwargs):
    model = CSPDarknetTiny(**kwargs)
    # only load state_dict()
    if pretrained:
        load_state_dict(
            torch.load(model_urls['yolov4_cspdarknettiny'],
                       map_location=torch.device('cpu')), model)
    return model


def cspdarknet53(pretrained=False, **kwargs):
    model = CSPDarknet53(**kwargs)
    # only load state_dict()
    if pretrained:
        load_state_dict(
            torch.load(model_urls['yolov4_cspdarknet53'],
                       map_location=torch.device('cpu')), model)
    return model


if __name__ == '__main__':
    net = CSPDarknetTiny(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, flops: {flops}, params: {params},out_shape: {out.shape}')
    net = CSPDarknet53(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'2222, flops: {flops}, params: {params},out_shape: {out.shape}')