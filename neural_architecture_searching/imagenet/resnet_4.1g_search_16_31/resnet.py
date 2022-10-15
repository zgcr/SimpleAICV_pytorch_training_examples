'''
Deep Residual Learning for Image Recognition
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import numpy as np

import torch
import torch.nn as nn


def get_resnet_config(resnet_type, q=8):
    stem_width, depth, w_0, w_a, w_m = resnet_type['stem_width'], resnet_type[
        'depth'], resnet_type['w_0'], resnet_type['w_a'], resnet_type['w_m']
    """Generates per stage widths and depths from ResNet parameters."""
    assert w_0 >= 0 and w_a > 0 and w_m > 1 and w_0 % q == 0

    # Generate quantized per-block ws
    ks = np.round(np.log((np.arange(depth) * w_a + w_0) / w_0) / np.log(w_m))
    width_all = (np.round(np.divide(w_0 * np.power(w_m, ks), q)) *
                 q).astype(int)
    # Generate per stage width and depth (assumes width_all are sorted)
    all_stage_width, all_stage_depth = np.unique(width_all, return_counts=True)

    return stem_width, all_stage_width, all_stage_depth


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


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 1 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
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

    def __init__(self, inplanes, planes, stride=1):
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
                                    padding=1,
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


class ResNet(nn.Module):

    def __init__(self, resnet_config, block=Bottleneck, num_classes=1000):
        super(ResNet, self).__init__()
        stem_width, all_stage_width, all_stage_depth = get_resnet_config(
            resnet_config, q=8)

        self.inplanes = stem_width
        self.all_stage_width = all_stage_width
        self.all_stage_depth = all_stage_depth
        self.block = block
        self.num_classes = num_classes
        self.expansion = 1 if block is BasicBlock else 4

        assert len(self.all_stage_width) == len(self.all_stage_depth)

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
                                      self.all_stage_width[0],
                                      self.all_stage_depth[0],
                                      stride=1)
        self.layer2 = self.make_layer(self.block,
                                      self.all_stage_width[1],
                                      self.all_stage_depth[1],
                                      stride=2)
        self.layer3 = self.make_layer(self.block,
                                      self.all_stage_width[2],
                                      self.all_stage_depth[2],
                                      stride=2)
        self.layer4 = self.make_layer(self.block,
                                      self.all_stage_width[3],
                                      self.all_stage_depth[3],
                                      stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.all_stage_width[3] * self.expansion,
                            self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


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

    # 'ResNet50'
    resnet_config = {
        'stem_width': 64,
        'depth': 14,
        'w_0': 32,
        'w_a': 18.639926481306944,
        'w_m': 2.065381050297783,
    }

    net = ResNet(resnet_config=resnet_config,
                 block=Bottleneck,
                 num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')