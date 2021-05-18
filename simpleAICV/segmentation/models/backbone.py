import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from simpleAICV.classification import backbones


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

        C2 = self.model.layer1(x)
        C3 = self.model.layer2(C2)
        C4 = self.model.layer3(C3)
        C5 = self.model.layer4(C4)

        del x

        return [C2, C3, C4, C5]


if __name__ == '__main__':
    net = ResNetBackbone(resnet_type='resnet50', pretrained=True)
    images = torch.randn(8, 3, 640, 640)
    [C2, C3, C4, C5] = net(images)
    print('1111', C2.shape, C3.shape, C4.shape, C5.shape)
