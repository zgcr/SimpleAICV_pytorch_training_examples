import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from SimpleAICV.detection.models import backbones
from SimpleAICV.detection.models.backbones.dinov3vit import VitPyramidNeck

__all__ = [
    'dinov3_vit_small_patch16_solov2',
    'dinov3_vit_small_plus_patch16_solov2',
    'dinov3_vit_base_patch16_solov2',
    'dinov3_vit_large_patch16_solov2',
    'dinov3_vit_large_plus_patch16_solov2',
    'dinov3_vit_huge_plus_patch16_solov2',
]


class SOLOV2FPN(nn.Module):

    def __init__(self, inplanes, planes=256):
        super(SOLOV2FPN, self).__init__()
        assert isinstance(inplanes, list)
        self.inplanes = inplanes
        self.planes = planes

        lateral_conv_layers = []
        fpn_conv_layers = []
        for i in range(len(inplanes)):
            lateral_conv_layers.append(
                nn.Conv2d(inplanes[i],
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True))
            fpn_conv_layers.append(
                nn.Conv2d(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
        self.lateral_conv_layers = nn.ModuleList(lateral_conv_layers)
        self.fpn_conv_layers = nn.ModuleList(fpn_conv_layers)

        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, inputs):
        assert len(inputs) == len(self.inplanes)

        laterals_features = []
        for i, per_lateral_conv_layer in enumerate(self.lateral_conv_layers):
            laterals_features.append(per_lateral_conv_layer(inputs[i]))

        del inputs

        for i in range(len(self.inplanes) - 1, 0, -1):
            laterals_features[i - 1] = F.interpolate(
                laterals_features[i],
                size=(laterals_features[i - 1].shape[2],
                      laterals_features[i - 1].shape[3]),
                mode='bilinear') + laterals_features[i - 1]

        outs_features = []
        for i in range(len(self.inplanes)):
            outs_features.append(self.fpn_conv_layers[i](laterals_features[i]))
        outs_features.append(self.maxpool(outs_features[-1]))

        del laterals_features

        return outs_features


class ConvGnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 dilation=1,
                 has_gn=True,
                 has_act=True):
        super(ConvGnActBlock, self).__init__()
        bias = False if has_gn else True
        self.has_gn = has_gn
        self.has_act = has_act

        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              dilation=dilation,
                              bias=bias)

        if self.has_gn:
            self.gn = nn.GroupNorm(num_groups=32, num_channels=planes)
        if self.has_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)

        if self.has_gn:
            x = self.gn(x)

        if self.has_act:
            x = self.act(x)

        return x


class SOLOV2MaskFeatHead(nn.Module):

    def __init__(self, inplanes=256, planes=128, num_classes=256):
        super(SOLOV2MaskFeatHead, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.num_classes = num_classes

        self.level_0_conv1 = ConvGnActBlock(inplanes,
                                            planes,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            groups=1,
                                            dilation=1,
                                            has_gn=True,
                                            has_act=True)

        self.level_1_conv1 = ConvGnActBlock(inplanes,
                                            planes,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            groups=1,
                                            dilation=1,
                                            has_gn=True,
                                            has_act=True)

        self.level_2_conv1 = ConvGnActBlock(inplanes,
                                            planes,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            groups=1,
                                            dilation=1,
                                            has_gn=True,
                                            has_act=True)
        self.level_2_conv2 = ConvGnActBlock(planes,
                                            planes,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            groups=1,
                                            dilation=1,
                                            has_gn=True,
                                            has_act=True)

        self.level_3_conv1 = ConvGnActBlock(inplanes + 2,
                                            planes,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            groups=1,
                                            dilation=1,
                                            has_gn=True,
                                            has_act=True)
        self.level_3_conv2 = ConvGnActBlock(planes,
                                            planes,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            groups=1,
                                            dilation=1,
                                            has_gn=True,
                                            has_act=True)
        self.level_3_conv3 = ConvGnActBlock(planes,
                                            planes,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            groups=1,
                                            dilation=1,
                                            has_gn=True,
                                            has_act=True)

        self.pred_conv = ConvGnActBlock(planes,
                                        num_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        dilation=1,
                                        has_gn=True,
                                        has_act=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, inputs):
        fused_features = self.level_0_conv1(inputs[0])

        for i in range(1, len(inputs)):
            per_level_inputs = inputs[i]
            if i == 3:
                x_range = torch.linspace(-1,
                                         1,
                                         per_level_inputs.shape[-1],
                                         device=per_level_inputs.device)
                y_range = torch.linspace(-1,
                                         1,
                                         per_level_inputs.shape[-2],
                                         device=per_level_inputs.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([per_level_inputs.shape[0], 1, -1, -1])
                x = x.expand([per_level_inputs.shape[0], 1, -1, -1])
                coord_features = torch.cat([x, y], dim=1)
                per_level_inputs = torch.cat(
                    [per_level_inputs, coord_features], dim=1)

            if i == 1:
                per_level_inputs = self.level_1_conv1(per_level_inputs)
                per_level_inputs = F.interpolate(per_level_inputs,
                                                 size=(inputs[i - 1].shape[2],
                                                       inputs[i - 1].shape[3]),
                                                 mode='bilinear')
            elif i == 2:
                per_level_inputs = self.level_2_conv1(per_level_inputs)
                per_level_inputs = F.interpolate(per_level_inputs,
                                                 size=(inputs[i - 1].shape[2],
                                                       inputs[i - 1].shape[3]),
                                                 mode='bilinear')
                per_level_inputs = self.level_2_conv2(per_level_inputs)
                per_level_inputs = F.interpolate(per_level_inputs,
                                                 size=(inputs[i - 2].shape[2],
                                                       inputs[i - 2].shape[3]),
                                                 mode='bilinear')

            elif i == 3:
                per_level_inputs = self.level_3_conv1(per_level_inputs)
                per_level_inputs = F.interpolate(per_level_inputs,
                                                 size=(inputs[i - 1].shape[2],
                                                       inputs[i - 1].shape[3]),
                                                 mode='bilinear')
                per_level_inputs = self.level_3_conv2(per_level_inputs)
                per_level_inputs = F.interpolate(per_level_inputs,
                                                 size=(inputs[i - 2].shape[2],
                                                       inputs[i - 2].shape[3]),
                                                 mode='bilinear')
                per_level_inputs = self.level_3_conv3(per_level_inputs)
                per_level_inputs = F.interpolate(per_level_inputs,
                                                 size=(inputs[i - 3].shape[2],
                                                       inputs[i - 3].shape[3]),
                                                 mode='bilinear')

            fused_features = fused_features + per_level_inputs

        del inputs, per_level_inputs

        pred_feature = self.pred_conv(fused_features)

        return pred_feature


class SOLOV2BboxHead(nn.Module):

    def __init__(self,
                 inplanes=256,
                 inter_planes=512,
                 instance_planes=256,
                 stacked_conv_nums=4,
                 grid_nums=(40, 36, 24, 16, 12),
                 num_classes=80):
        super(SOLOV2BboxHead, self).__init__()
        kernel_planes = instance_planes * 1 * 1
        self.grid_nums = grid_nums

        cate_conv_layers = []
        kernel_conv_layers = []
        for i in range(stacked_conv_nums):
            current_inplanes = inplanes if i == 0 else inter_planes
            cate_conv_layers.append(
                ConvGnActBlock(current_inplanes,
                               inter_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               dilation=1,
                               has_gn=True,
                               has_act=True))

            current_inplanes = inplanes + 2 if i == 0 else inter_planes
            kernel_conv_layers.append(
                ConvGnActBlock(current_inplanes,
                               inter_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               dilation=1,
                               has_gn=True,
                               has_act=True))
        self.cate_conv_layers = nn.ModuleList(cate_conv_layers)
        self.kernel_conv_layers = nn.ModuleList(kernel_conv_layers)

        self.cate_pred_conv = nn.Conv2d(inter_planes,
                                        num_classes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.kernel_pred_conv = nn.Conv2d(inter_planes,
                                          kernel_planes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.cate_pred_conv.bias.data.fill_(b)

    def forward(self, inputs):
        P2, P3, P4, P5, P6 = inputs

        del inputs

        P2 = F.interpolate(P2,
                           size=(P3.shape[2], P3.shape[3]),
                           mode='bilinear')
        P6 = F.interpolate(P6,
                           size=(P5.shape[2], P5.shape[3]),
                           mode='bilinear')

        features = [P2, P3, P4, P5, P6]
        kernel_preds, cate_preds = [], []
        for i in range(len(features)):
            kernel_features = features[i]

            x_range = torch.linspace(-1,
                                     1,
                                     kernel_features.shape[-1],
                                     device=kernel_features.device)
            y_range = torch.linspace(-1,
                                     1,
                                     kernel_features.shape[-2],
                                     device=kernel_features.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([kernel_features.shape[0], 1, -1, -1])
            x = x.expand([kernel_features.shape[0], 1, -1, -1])
            coord_features = torch.cat([x, y], 1)
            kernel_features = torch.cat([kernel_features, coord_features], 1)

            kernel_features = F.interpolate(kernel_features,
                                            size=self.grid_nums[i],
                                            mode='bilinear')
            cate_features = kernel_features[:, :-2, :, :]

            for per_kernel_layer in self.kernel_conv_layers:
                kernel_features = per_kernel_layer(kernel_features)
            per_kernel_pred = self.kernel_pred_conv(kernel_features)

            for per_cate_layer in self.cate_conv_layers:
                cate_features = per_cate_layer(cate_features)
            per_cate_pred = self.cate_pred_conv(cate_features)

            kernel_preds.append(per_kernel_pred)
            cate_preds.append(per_cate_pred)

        return kernel_preds, cate_preds


class SOLOV2(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 fpn_planes=256,
                 mask_feature_planes=128,
                 mask_feature_num_classes=256,
                 bbox_inter_planes=512,
                 instance_planes=256,
                 grid_nums=(40, 36, 24, 16, 12),
                 num_classes=80,
                 use_gradient_checkpoint=False):
        super(SOLOV2, self).__init__()
        self.fpn_planes = fpn_planes
        self.mask_feature_planes = mask_feature_planes
        self.mask_feature_num_classes = mask_feature_num_classes
        self.bbox_inter_planes = bbox_inter_planes
        self.instance_planes = instance_planes
        self.grid_nums = grid_nums
        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        self.neck = VitPyramidNeck(inplanes=self.backbone.out_channels,
                                   planes=self.fpn_planes)

        self.fpn = SOLOV2FPN([
            self.fpn_planes,
            self.fpn_planes,
            self.fpn_planes,
            self.fpn_planes,
        ], fpn_planes)

        self.mask_feature_head = SOLOV2MaskFeatHead(
            inplanes=self.fpn_planes,
            planes=self.mask_feature_planes,
            num_classes=self.mask_feature_num_classes)

        self.bbox_head = SOLOV2BboxHead(inplanes=self.fpn_planes,
                                        inter_planes=self.bbox_inter_planes,
                                        instance_planes=self.instance_planes,
                                        stacked_conv_nums=4,
                                        grid_nums=self.grid_nums,
                                        num_classes=self.num_classes)

    def forward(self, inputs):
        x = self.backbone(inputs)
        # torch.Size([1, 256, 160, 160])
        # torch.Size([1, 256, 80, 80])
        # torch.Size([1, 256, 40, 40])
        # torch.Size([1, 256, 20, 20])
        x = self.neck(x)

        # torch.Size([1, 256, 160, 160])
        # torch.Size([1, 256, 80, 80])
        # torch.Size([1, 256, 40, 40])
        # torch.Size([1, 256, 20, 20])
        # torch.Size([1, 256, 10, 10])
        if self.use_gradient_checkpoint:
            x = checkpoint(self.fpn, x, use_reentrant=False)
        else:
            x = self.fpn(x)

        # torch.Size([16, 256, 160, 160])
        mask_feat_pred = self.mask_feature_head(x[0:4])

        # kernel_preds
        # torch.Size([1, 256, 40, 40])
        # torch.Size([1, 256, 36, 36])
        # torch.Size([1, 256, 24, 24])
        # torch.Size([1, 256, 16, 16])
        # torch.Size([1, 256, 12, 12])

        # cate_preds
        # torch.Size([1, 80, 40, 40])
        # torch.Size([1, 80, 36, 36])
        # torch.Size([1, 80, 24, 24])
        # torch.Size([1, 80, 16, 16])
        # torch.Size([1, 80, 12, 12])
        kernel_preds, cate_preds = self.bbox_head(x)

        return mask_feat_pred, kernel_preds, cate_preds


def _solov2(backbone_type, backbone_pretrained_path, **kwargs):
    model = SOLOV2(backbone_type,
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)

    return model


def dinov3_vit_small_patch16_solov2(backbone_pretrained_path='', **kwargs):
    return _solov2('dinov3_vit_small_patch16_backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def dinov3_vit_small_plus_patch16_solov2(backbone_pretrained_path='',
                                         **kwargs):
    return _solov2('dinov3_vit_small_plus_patch16_backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def dinov3_vit_base_patch16_solov2(backbone_pretrained_path='', **kwargs):
    return _solov2('dinov3_vit_base_patch16_backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def dinov3_vit_large_patch16_solov2(backbone_pretrained_path='', **kwargs):
    return _solov2('dinov3_vit_large_patch16_backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def dinov3_vit_large_plus_patch16_solov2(backbone_pretrained_path='',
                                         **kwargs):
    return _solov2('dinov3_vit_large_plus_patch16_backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def dinov3_vit_huge_plus_patch16_solov2(backbone_pretrained_path='', **kwargs):
    return _solov2('dinov3_vit_huge_plus_patch16_backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


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

    net = dinov3_vit_small_patch16_solov2()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out1, out2, out3 = net(
        torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', out1.shape)
    for per_out in out2:
        print('3333', per_out.shape)
    for per_out in out3:
        print('4444', per_out.shape)

    net = dinov3_vit_small_patch16_solov2(use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    out1, out2, out3 = net(
        torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', out1.shape)
    for per_out in out2:
        print('3333', per_out.shape)
    for per_out in out3:
        print('4444', per_out.shape)

    net = dinov3_vit_small_plus_patch16_solov2()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out1, out2, out3 = net(
        torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', out1.shape)
    for per_out in out2:
        print('3333', per_out.shape)
    for per_out in out3:
        print('4444', per_out.shape)

    net = dinov3_vit_base_patch16_solov2()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out1, out2, out3 = net(
        torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', out1.shape)
    for per_out in out2:
        print('3333', per_out.shape)
    for per_out in out3:
        print('4444', per_out.shape)

    net = dinov3_vit_large_patch16_solov2()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out1, out2, out3 = net(
        torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', out1.shape)
    for per_out in out2:
        print('3333', per_out.shape)
    for per_out in out3:
        print('4444', per_out.shape)

    net = dinov3_vit_large_plus_patch16_solov2()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out1, out2, out3 = net(
        torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', out1.shape)
    for per_out in out2:
        print('3333', per_out.shape)
    for per_out in out3:
        print('4444', per_out.shape)

    net = dinov3_vit_huge_plus_patch16_solov2()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'inputs':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    out1, out2, out3 = net(
        torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', out1.shape)
    for per_out in out2:
        print('3333', per_out.shape)
    for per_out in out3:
        print('4444', per_out.shape)
