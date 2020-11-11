import os
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.detection.models.DCNv2 import DCN


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


class FCOSClsRegCntHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_classes,
                 num_layers=4,
                 prior=0.01,
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
                          padding=1))
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
                          padding=1))
            if use_gn:
                reg_layers.append(nn.GroupNorm(32, inplanes))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*reg_layers)

        self.cls_out = nn.Conv2d(inplanes,
                                 num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
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

        prior = prior
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

        return cls_output, reg_output, center_output


class CenterNetHetRegWhHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_classes,
                 num_layers=3,
                 out_channels=[256, 128, 64]):
        super(CenterNetHetRegWhHead, self).__init__()
        self.inplanes = inplanes
        layers = []
        for i in range(num_layers):
            layers.append(
                DCN(in_channels=self.inplanes,
                    out_channels=out_channels[i],
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                    dilation=1,
                    deformable_groups=1))
            layers.append(nn.BatchNorm2d(out_channels[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.ConvTranspose2d(in_channels=out_channels[i],
                                   out_channels=out_channels[i],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   output_padding=0,
                                   bias=False))
            layers.append(nn.BatchNorm2d(out_channels[i]))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = out_channels[i]

        self.public_deconv_head = nn.Sequential(*layers)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64,
                      out_channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[-1],
                      num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(64,
                      out_channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(64,
                      out_channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )

        for m in self.public_deconv_head.modules():
            if isinstance(m, nn.ConvTranspose2d):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                            1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]

        self.heatmap_head[-1].bias.data.fill_(-2.19)

        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.public_deconv_head(x)

        heatmap_output = self.heatmap_head(x)
        offset_output = self.offset_head(x)
        wh_output = self.wh_head(x)

        return heatmap_output, offset_output, wh_output


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def hard_swish(self, x, inplace):
        inner = F.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if inplace else x.mul(inner)

    def forward(self, x):
        return self.hard_swish(x, self.inplace)


class SeparableConvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(inplanes,
                                        inplanes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=inplanes,
                                        bias=False)
        self.pointwise_conv = nn.Conv2d(inplanes,
                                        planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class EfficientDetClsHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors,
                 num_classes,
                 num_layers,
                 prior=0.01):
        super(EfficientDetClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(SeparableConvBlock(inplanes, inplanes))
            layers.append(nn.BatchNorm2d(inplanes))
            layers.append(HardSwish(inplace=True))

        layers.append(SeparableConvBlock(
            inplanes,
            num_anchors * num_classes,
        ))
        self.cls_head = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_head[-1].pointwise_conv.bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)
        x = self.sigmoid(x)

        return x


class EfficientDetRegHead(nn.Module):
    def __init__(self, inplanes, num_anchors, num_layers):
        super(EfficientDetRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(SeparableConvBlock(inplanes, inplanes))
            layers.append(nn.BatchNorm2d(inplanes))
            layers.append(HardSwish(inplace=True))

        layers.append(SeparableConvBlock(
            inplanes,
            num_anchors * 4,
        ))
        self.reg_head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.reg_head(x)

        return x


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

    model2 = FCOSClsRegCntHead(256, 80)

    cls_output2, reg_output2, center_output2 = model2(features[0])
    print("3333", cls_output2.shape, reg_output2.shape, center_output2.shape)

    cls_model3 = EfficientDetClsHead(256, 9, 80, 4)
    reg_model3 = EfficientDetRegHead(256, 9, 4)

    cls_output3 = cls_model3(features[0])
    reg_output3 = reg_model3(features[0])
    print("4444", cls_output3.shape, reg_output3.shape)

    head_model4 = CenterNetHetRegWhHead(2048,
                                        80,
                                        num_layers=3,
                                        out_channels=[256, 128, 64])

    heatmap_output4, offset_output4, wh_output4 = head_model4(C5)

    print("5555", heatmap_output4.shape, offset_output4.shape,
          wh_output4.shape)
