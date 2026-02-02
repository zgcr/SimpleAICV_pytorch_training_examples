import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import math

import torch
import torch.nn as nn


class RetinaClsHead(nn.Module):

    def __init__(self, inplanes, num_anchors, num_classes, num_layers=4):
        super(RetinaClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*layers)
        self.cls_out = nn.Conv2d(inplanes,
                                 num_anchors * num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.cls_out.bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)
        x = self.cls_out(x)
        x = x.float()
        x = self.sigmoid(x)

        return x


class RetinaRegHead(nn.Module):

    def __init__(self, inplanes, num_anchors, num_layers=4):
        super(RetinaRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*layers)
        self.reg_out = nn.Conv2d(inplanes,
                                 num_anchors * 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.reg_head(x)
        x = self.reg_out(x)

        return x


class FCOSClsRegCntHead(nn.Module):

    def __init__(self,
                 inplanes,
                 num_classes,
                 num_layers=4,
                 use_gn=True,
                 cnt_on_reg=True):
        super(FCOSClsRegCntHead, self).__init__()
        self.cnt_on_reg = cnt_on_reg

        cls_layers = []
        for _ in range(num_layers):
            cls_layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          bias=use_gn is False))
            if use_gn:
                cls_layers.append(nn.GroupNorm(32, inplanes))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*cls_layers)

        reg_layers = []
        for _ in range(num_layers):
            reg_layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          bias=use_gn is False))
            if use_gn:
                reg_layers.append(nn.GroupNorm(32, inplanes))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*reg_layers)

        self.cls_out = nn.Conv2d(inplanes,
                                 num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=1,
                                 bias=True)
        self.reg_out = nn.Conv2d(inplanes,
                                 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=1,
                                 bias=True)
        self.center_out = nn.Conv2d(inplanes,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    bias=True)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.cls_out.bias.data.fill_(b)

    def forward(self, x):
        cls_x = self.cls_head(x)
        reg_x = self.reg_head(x)

        del x

        cls_output = self.cls_out(cls_x)
        reg_output = self.reg_out(reg_x)

        if self.cnt_on_reg:
            center_output = self.center_out(reg_x)
        else:
            center_output = self.center_out(cls_x)

        cls_output = cls_output.float()
        center_output = center_output.float()
        cls_output = self.sigmoid(cls_output)
        center_output = self.sigmoid(center_output)

        return cls_output, reg_output, center_output


class DETRClsRegHead(nn.Module):

    def __init__(self, hidden_inplanes, num_classes, num_layers=3):
        super(DETRClsRegHead, self).__init__()

        self.cls_head = nn.Linear(hidden_inplanes, num_classes)

        reg_layers = []
        for _ in range(num_layers - 1):
            reg_layers.append(nn.Linear(hidden_inplanes, hidden_inplanes))
            reg_layers.append(nn.ReLU(inplace=True))
        reg_layers.append(nn.Linear(hidden_inplanes, 4))
        self.reg_head = nn.Sequential(*reg_layers)

        self.sigmoid = nn.Sigmoid()

        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def forward(self, x):
        cls_output = self.cls_head(x)
        reg_output = self.reg_head(x)

        del x

        reg_output = reg_output.float()
        reg_output = self.sigmoid(reg_output)

        return cls_output, reg_output


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

    inputs = torch.randn(3, 256, 80, 80)
    net1 = RetinaClsHead(256, 9, 80, num_layers=4)
    outs1 = net1(inputs)
    print('1111', outs1.shape)

    inputs = torch.randn(3, 256, 80, 80)
    net2 = RetinaRegHead(256, 9, num_layers=4)
    outs2 = net2(inputs)
    print('2222', outs2.shape)

    inputs = torch.randn(3, 256, 80, 80)
    net = FCOSClsRegCntHead(256,
                            80,
                            num_layers=4,
                            use_gn=True,
                            cnt_on_reg=True)
    outs = net(inputs)
    for out in outs:
        print('1111', out.shape)

    inputs = torch.randn(6, 3, 100, 256)
    net = DETRClsRegHead(hidden_inplanes=256, num_classes=80, num_layers=3)
    outs = net(inputs)
    for out in outs:
        print('1111', out.shape)
