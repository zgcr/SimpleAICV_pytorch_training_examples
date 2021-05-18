import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from simpleAICV.classification import backbones


class DarknetTinyBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(DarknetTinyBackbone, self).__init__()
        self.model = backbones.__dict__['darknettiny'](**{
            'pretrained': pretrained
        })
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool1(x)
        x = self.model.conv2(x)
        x = self.model.maxpool2(x)
        x = self.model.conv3(x)
        x = self.model.maxpool3(x)
        x = self.model.conv4(x)
        x = self.model.maxpool4(x)
        C4 = self.model.conv5(x)
        C5 = self.model.maxpool5(C4)
        C5 = self.model.conv6(C5)
        C5 = self.model.zeropad(C5)
        C5 = self.model.maxpool6(C5)

        del x

        return [C4, C5]


class Darknet19Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Darknet19Backbone, self).__init__()
        self.model = backbones.__dict__['darknet19'](**{
            'pretrained': pretrained
        })
        del self.model.avgpool
        del self.model.layer7

    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.maxpool1(x)
        x = self.model.layer2(x)
        C3 = self.model.layer3(x)
        C4 = self.model.layer4(C3)
        C5 = self.model.layer5(C4)
        C5 = self.model.layer6(C5)

        del x

        return [C3, C4, C5]


class Darknet53Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Darknet53Backbone, self).__init__()
        self.model = backbones.__dict__['darknet53'](**{
            'pretrained': pretrained
        })
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.block1(x)
        x = self.model.conv3(x)
        x = self.model.block2(x)
        x = self.model.conv4(x)
        C3 = self.model.block3(x)
        C4 = self.model.conv5(C3)
        C4 = self.model.block4(C4)
        C5 = self.model.conv6(C4)
        C5 = self.model.block5(C5)

        del x

        return [C3, C4, C5]


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.model = backbones.__dict__[resnet_type](**{
            'pretrained': pretrained
        })
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        C3 = self.model.layer2(x)
        C4 = self.model.layer3(C3)
        C5 = self.model.layer4(C4)

        del x

        return [C3, C4, C5]


class VovNetBackbone(nn.Module):
    def __init__(self, vovnet_type='VoVNet39_se', pretrained=True):
        super(VovNetBackbone, self).__init__()
        self.model = backbones.__dict__[vovnet_type](**{
            'pretrained': pretrained
        })
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.stem(x)

        features = []
        for stage in self.model.stages:
            x = stage(x)
            features.append(x)

        del x

        return features[1:]


if __name__ == '__main__':
    net = ResNetBackbone(resnet_type='resnet50', pretrained=True)
    images = torch.randn(8, 3, 640, 640)
    [C3, C4, C5] = net(images)
    print('1111', C3.shape, C4.shape, C5.shape)
    net = VovNetBackbone(vovnet_type='VoVNet39_se', pretrained=False)
    images = torch.randn(8, 3, 640, 640)
    [C3, C4, C5] = net(images)
    print('1111', C3.shape, C4.shape, C5.shape)
    net1 = Darknet53Backbone(pretrained=False)
    images = torch.randn(8, 3, 416, 416)
    [C3, C4, C5] = net1(images)
    print('1111', C3.shape, C4.shape, C5.shape)
    net2 = Darknet19Backbone(pretrained=False)
    images = torch.randn(8, 3, 416, 416)
    [C3, C4, C5] = net2(images)
    print('1111', C3.shape, C4.shape, C5.shape)
    net3 = DarknetTinyBackbone(pretrained=False)
    images = torch.randn(8, 3, 416, 416)
    [C4, C5] = net3(images)
    print('1111', C4.shape, C5.shape)
