import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import math

import torch
import torch.nn as nn

from simpleAICV.detection.models.dcnv2 import DeformableConv2d
from simpleAICV.detection.models.backbones.yoloxbackbone import ConvBnActBlock, DWConvBnActBlock


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

        cls_output = self.sigmoid(cls_output)
        center_output = self.sigmoid(center_output)

        return cls_output, reg_output, center_output


class CenterNetHetRegWhHead(nn.Module):

    def __init__(self,
                 inplanes,
                 num_classes,
                 planes=[256, 128, 64],
                 num_layers=3):
        super(CenterNetHetRegWhHead, self).__init__()
        self.inplanes = inplanes
        layers = []
        for i in range(num_layers):
            layers.append(
                DeformableConv2d(self.inplanes,
                                 planes[i],
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 dilation=1,
                                 groups=1,
                                 bias=False))
            layers.append(nn.BatchNorm2d(planes[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.ConvTranspose2d(planes[i],
                                   planes[i],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   output_padding=0,
                                   bias=False))
            layers.append(nn.BatchNorm2d(planes[i]))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes[i]

        self.public_deconv_head = nn.Sequential(*layers)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.sigmoid = nn.Sigmoid()

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

        heatmap_output = self.sigmoid(heatmap_output)

        return heatmap_output, offset_output, wh_output


class TTFHetWhHead(nn.Module):

    def __init__(self,
                 inplanes,
                 num_classes,
                 planes=[256, 128, 64],
                 short_cut_layers_num=[1, 2],
                 num_layers=3):
        super(TTFHetWhHead, self).__init__()

        self.deconv_layers = nn.ModuleList()
        inter_planes = inplanes[-1]
        for i in range(num_layers):
            self.deconv_layers.append(
                nn.Sequential(
                    DeformableConv2d(inter_planes,
                                     planes[i],
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     dilation=1,
                                     groups=1,
                                     bias=False), nn.BatchNorm2d(planes[i]),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(planes[i],
                                       planes[i],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       output_padding=0,
                                       bias=False), nn.BatchNorm2d(planes[i]),
                    nn.ReLU(inplace=True)))
            inter_planes = planes[i]

        self.shortcut_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            single_shortcut_layer = []
            inter_planes = inplanes[len(inplanes) - i - 2]
            for j in range(short_cut_layers_num[i]):
                single_shortcut_layer.append(
                    nn.Conv2d(inter_planes,
                              planes[i],
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True))
                inter_planes = planes[i]
                if j < short_cut_layers_num[i] - 1:
                    single_shortcut_layer.append(nn.ReLU(inplace=True))
            single_shortcut_layer = nn.Sequential(*single_shortcut_layer)
            self.shortcut_layers.append(single_shortcut_layer)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      4,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
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
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.heatmap_head[-1].bias.data.fill_(b)

    def forward(self, inputs):
        x = inputs[-1]
        for i, upsample_layer in enumerate(self.deconv_layers):
            x = upsample_layer(x)
            if i < len(self.shortcut_layers):
                shortcut = self.shortcut_layers[i](inputs[-i - 2])
                x = x + shortcut

        heatmap_output = self.heatmap_head(x)
        wh_output = self.relu(self.wh_head(x))

        heatmap_output = self.sigmoid(heatmap_output)

        return heatmap_output, wh_output


class YOLOXHead(nn.Module):

    def __init__(self,
                 inplanes_list,
                 planes,
                 num_classes,
                 block=ConvBnActBlock,
                 act_type='silu'):
        super(YOLOXHead, self).__init__()

        self.stem_conv_list = nn.ModuleList()
        self.cls_conv_list = nn.ModuleList()
        self.reg_conv_list = nn.ModuleList()
        self.cls_pred_list = nn.ModuleList()
        self.reg_pred_list = nn.ModuleList()
        self.obj_pred_list = nn.ModuleList()

        for i in range(len(inplanes_list)):
            self.stem_conv_list.append(
                ConvBnActBlock(inplanes_list[i],
                               planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               has_bn=True,
                               has_act=True,
                               act_type=act_type))
            self.cls_conv_list.append(
                nn.Sequential(
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type),
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type)))
            self.reg_conv_list.append(
                nn.Sequential(
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type),
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type)))
            self.cls_pred_list.append(
                nn.Conv2d(planes,
                          num_classes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=True))
            self.reg_pred_list.append(
                nn.Conv2d(planes,
                          4,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=True))
            self.obj_pred_list.append(
                nn.Conv2d(planes,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        obj_outputs, cls_outputs, reg_outputs = [], [], []
        for i, x in enumerate(inputs):
            x = self.stem_conv_list[i](x)

            cls_out = self.cls_conv_list[i](x)
            reg_out = self.reg_conv_list[i](x)

            cls_out = self.cls_pred_list[i](cls_out)
            obj_out = self.obj_pred_list[i](reg_out)
            reg_out = self.reg_pred_list[i](reg_out)

            cls_out = self.sigmoid(cls_out)
            obj_out = self.sigmoid(obj_out)

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            obj_outputs.append(obj_out)

        return cls_outputs, reg_outputs, obj_outputs


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

    inputs = torch.randn(3, 256, 80,
                         80), torch.randn(3, 256, 40, 40), torch.randn(
                             3, 256, 20,
                             20), torch.randn(3, 256, 10,
                                              10), torch.randn(3, 256, 5, 5)
    net1 = RetinaClsHead(256, 9, 80, num_layers=4)
    net2 = RetinaRegHead(256, 9, num_layers=4)
    from thop import profile
    from thop import clever_format
    for input in inputs:
        macs, params = profile(net1, inputs=(input, ), verbose=False)
        macs, params = clever_format([macs, params], '%.3f')
        print(f'1111, macs: {macs}, params: {params}')
        macs, params = profile(net2, inputs=(input, ), verbose=False)
        macs, params = clever_format([macs, params], '%.3f')
        print(f'2222, macs: {macs}, params: {params}')

    inputs = torch.randn(3, 256, 80,
                         80), torch.randn(3, 256, 40, 40), torch.randn(
                             3, 256, 20,
                             20), torch.randn(3, 256, 10,
                                              10), torch.randn(3, 256, 5, 5)
    net = FCOSClsRegCntHead(256,
                            80,
                            num_layers=4,
                            use_gn=True,
                            cnt_on_reg=True)
    from thop import profile
    from thop import clever_format
    for input in inputs:
        macs, params = profile(net, inputs=(input, ), verbose=False)
        macs, params = clever_format([macs, params], '%.3f')
        print(f'3333, macs: {macs}, params: {params}')

    inputs = torch.randn(3, 2048, 20, 20)
    net = CenterNetHetRegWhHead(2048, 80, planes=[256, 128, 64], num_layers=3)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(inputs, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'4444, macs: {macs}, params: {params}')

    inputs = [
        torch.randn(3, 256, 80, 80),
        torch.randn(3, 512, 40, 40),
        torch.randn(3, 1024, 20, 20)
    ]
    net = TTFHetWhHead([256, 512, 1024],
                       80,
                       planes=[256, 128, 64],
                       num_layers=3)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(inputs, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'5555, macs: {macs}, params: {params}')

    P3, P4, P5 = torch.randn(3, 256, 80, 80), torch.randn(3, 512, 40,
                                                          40), torch.randn(
                                                              3, 1024, 20, 20)
    net = YOLOXHead(inplanes_list=[256, 512, 1024],
                    planes=256,
                    num_classes=80,
                    block=ConvBnActBlock,
                    act_type='silu')
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([P3, P4, P5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'6666, macs: {macs}, params: {params}')