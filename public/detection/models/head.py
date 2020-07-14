import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaClsHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors,
                 num_classes,
                 num_layers=4,
                 prior=0.01):
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
        layers.append(
            nn.Conv2d(inplanes,
                      num_anchors * num_classes,
                      kernel_size=3,
                      stride=1,
                      padding=1))
        self.cls_head = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_head[-1].bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)
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
        layers.append(
            nn.Conv2d(inplanes,
                      num_anchors * 4,
                      kernel_size=3,
                      stride=1,
                      padding=1))

        self.reg_head = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.reg_head(x)

        return x


class FCOSClsHead(nn.Module):
    def __init__(self, inplanes, num_classes, num_layers=4, prior=0.01):
        super(FCOSClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(inplanes,
                      num_classes,
                      kernel_size=3,
                      stride=1,
                      padding=1))
        self.cls_head = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_head[-1].bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)

        return x


class FCOSRegCenterHead(nn.Module):
    def __init__(self, inplanes, num_layers=4):
        super(FCOSRegCenterHead, self).__init__()
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
                                 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.center_out = nn.Conv2d(inplanes,
                                    1,
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
        reg_output = self.reg_out(x)
        center_output = self.center_out(x)

        return reg_output, center_output


if __name__ == '__main__':
    image_h, image_w = 640, 640
    from fpn import RetinaFPN
    fpn_model = RetinaFPN(512, 1024, 2048, 256)
    C3, C4, C5 = torch.randn(3, 512, 80, 80), torch.randn(3, 1024, 40,
                                                          40), torch.randn(
                                                              3, 2048, 20, 20)
    features = fpn_model([C3, C4, C5])

    print("1111", features[0].shape)

    cls_model = RetinaClsHead(256, 9, 80)
    reg_model = RetinaRegHead(256, 9)

    cls_output = cls_model(features[0])
    reg_output = reg_model(features[0])

    print("2222", cls_output.shape, reg_output.shape)

    cls_model2 = FCOSClsHead(256, 80)
    reg_model2 = FCOSRegCenterHead(256)

    cls_output2 = cls_model2(features[0])
    reg_output2, center_output2 = reg_model2(features[0])
    print("3333", cls_output2.shape, reg_output2.shape, center_output2.shape)
