import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from public.imagenet import models


class Darknet19Backbone(nn.Module):
    def __init__(self):
        super(Darknet19Backbone, self).__init__()
        self.model = models.__dict__['darknet19'](**{"pretrained": True})
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
    def __init__(self):
        super(Darknet53Backbone, self).__init__()
        self.model = models.__dict__['darknet53'](**{"pretrained": True})
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


class EfficientNetBackbone(nn.Module):
    def __init__(self, efficientnet_type="efficientnet_b0"):
        super(EfficientNetBackbone, self).__init__()
        self.model = models.__dict__[efficientnet_type](**{"pretrained": True})
        del self.model.dropout
        del self.model.fc
        del self.model.avgpool
        del self.model.conv_head

    def forward(self, x):
        x = self.model.stem(x)

        feature_maps = []
        last_x = None
        for index, block in enumerate(self.model.blocks):
            x = block(x)
            if block.stride == 2:
                feature_maps.append(last_x)
            elif index == len(self.model.blocks) - 1:
                feature_maps.append(x)
            last_x = x

        del last_x

        return feature_maps[2:]


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type="resnet50"):
        super(ResNetBackbone, self).__init__()
        self.model = models.__dict__[resnet_type](**{"pretrained": True})
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
    def __init__(self, vovnet_type='VoVNet39_se'):
        super(VovNetBackbone, self).__init__()
        self.model = models.__dict__[vovnet_type](**{"pretrained": True})
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
    # net = ResNetBackbone(resnet_type="resnet50")
    # images = torch.randn(8, 3, 640, 640)
    # [C3, C4, C5] = net(images)
    # print("1111", C3.shape, C4.shape, C5.shape)
    # net = EfficientNetBackbone(efficientnet_type="efficientnet_b0")
    # images = torch.randn(8, 3, 640, 640)
    # [C3, C4, C5] = net(images)
    # print("1111", C3.shape, C4.shape, C5.shape)
    net1 = Darknet53Backbone()
    images = torch.randn(8, 3, 416, 416)
    [C3, C4, C5] = net1(images)
    print("1111", C3.shape, C4.shape, C5.shape)
    net2 = Darknet19Backbone()
    images = torch.randn(8, 3, 416, 416)
    [C3, C4, C5] = net2(images)
    print("1111", C3.shape, C4.shape, C5.shape)