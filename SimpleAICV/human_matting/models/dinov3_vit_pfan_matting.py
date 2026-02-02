import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from SimpleAICV.detection.models import backbones
from SimpleAICV.detection.models.backbones.dinov3vit import VitPyramidNeck

__all__ = [
    'dinov3_vit_small_patch16_pfan_matting',
    'dinov3_vit_small_plus_patch16_pfan_matting',
    'dinov3_vit_base_patch16_pfan_matting',
    'dinov3_vit_large_patch16_pfan_matting',
    'dinov3_vit_large_plus_patch16_pfan_matting',
    'dinov3_vit_huge_plus_patch16_pfan_matting',
]


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 dilation=1,
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
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class CPFE(nn.Module):

    def __init__(self, inplanes=512, planes=32, dilation_rate_list=[3, 5, 7]):
        super(CPFE, self).__init__()

        # Define layers
        self.conv_1_1 = nn.Conv2d(inplanes,
                                  planes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False)
        self.conv_dil_3 = nn.Conv2d(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    dilation=dilation_rate_list[0],
                                    padding=dilation_rate_list[0],
                                    bias=False)
        self.conv_dil_5 = nn.Conv2d(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    dilation=dilation_rate_list[1],
                                    padding=dilation_rate_list[1],
                                    bias=False)
        self.conv_dil_7 = nn.Conv2d(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    dilation=dilation_rate_list[2],
                                    padding=dilation_rate_list[2],
                                    bias=False)

        self.conv = ConvBnActBlock(planes * 4,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   dilation=1,
                                   has_bn=True,
                                   has_act=True)

    def forward(self, x):
        x_1x1 = self.conv_1_1(x)
        x_dilate_3 = self.conv_dil_3(x)
        x_dilate_5 = self.conv_dil_5(x)
        x_dilate_7 = self.conv_dil_7(x)

        x = torch.cat((x_1x1, x_dilate_3, x_dilate_5, x_dilate_7), dim=1)

        x = self.conv(x)

        return x


class ConvTransposeBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvTransposeBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(inplanes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               groups=groups,
                               bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class PFANMatting(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 cpfe_planes=32,
                 use_gradient_checkpoint=False):
        super(PFANMatting, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.neck = VitPyramidNeck(inplanes=self.backbone.out_channels,
                                   planes=planes)

        self.global_high_level_cpfe_3 = CPFE(inplanes=planes,
                                             planes=cpfe_planes,
                                             dilation_rate_list=[3, 5, 7])
        self.global_high_level_cpfe_4 = CPFE(inplanes=planes,
                                             planes=cpfe_planes,
                                             dilation_rate_list=[3, 5, 7])

        self.global_high_level_conv = ConvBnActBlock(int(2 * cpfe_planes),
                                                     cpfe_planes,
                                                     kernel_size=1,
                                                     stride=1,
                                                     padding=0,
                                                     groups=1,
                                                     dilation=1,
                                                     has_bn=True,
                                                     has_act=False)

        # processing low level (ll) feature
        self.global_low_level_conv_1 = ConvBnActBlock(planes,
                                                      cpfe_planes,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      groups=1,
                                                      dilation=1,
                                                      has_bn=True,
                                                      has_act=True)

        self.global_low_level_conv_2 = ConvBnActBlock(planes,
                                                      cpfe_planes,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      groups=1,
                                                      dilation=1,
                                                      has_bn=True,
                                                      has_act=True)

        self.global_low_level_conv = ConvBnActBlock(int(2 * cpfe_planes),
                                                    cpfe_planes,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    dilation=1,
                                                    has_bn=True,
                                                    has_act=False)

        self.global_reduce_conv1 = ConvBnActBlock(int(2 * cpfe_planes),
                                                  cpfe_planes,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  groups=1,
                                                  dilation=1,
                                                  has_bn=True,
                                                  has_act=False)

        self.global_upsample_conv1 = ConvTransposeBnActBlock(cpfe_planes,
                                                             cpfe_planes,
                                                             kernel_size=2,
                                                             stride=2,
                                                             groups=1,
                                                             has_bn=True,
                                                             has_act=True)
        self.global_upsample_conv2 = ConvBnActBlock(cpfe_planes,
                                                    cpfe_planes,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    groups=1,
                                                    dilation=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.global_upsample_conv3 = ConvTransposeBnActBlock(cpfe_planes,
                                                             cpfe_planes,
                                                             kernel_size=2,
                                                             stride=2,
                                                             groups=1,
                                                             has_bn=True,
                                                             has_act=True)

        self.global_pred_conv = nn.Conv2d(cpfe_planes,
                                          3,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)

        self.local_high_level_cpfe_3 = CPFE(inplanes=planes,
                                            planes=cpfe_planes,
                                            dilation_rate_list=[3, 5, 7])
        self.local_high_level_cpfe_4 = CPFE(inplanes=planes,
                                            planes=cpfe_planes,
                                            dilation_rate_list=[3, 5, 7])

        self.local_high_level_conv = ConvBnActBlock(int(2 * cpfe_planes),
                                                    cpfe_planes,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    dilation=1,
                                                    has_bn=True,
                                                    has_act=False)

        self.local_low_level_conv_1 = ConvBnActBlock(planes,
                                                     cpfe_planes,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     groups=1,
                                                     dilation=1,
                                                     has_bn=True,
                                                     has_act=True)

        self.local_low_level_conv_2 = ConvBnActBlock(planes,
                                                     cpfe_planes,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     groups=1,
                                                     dilation=1,
                                                     has_bn=True,
                                                     has_act=True)

        self.local_low_level_conv = ConvBnActBlock(int(2 * cpfe_planes),
                                                   cpfe_planes,
                                                   kernel_size=1,
                                                   stride=1,
                                                   padding=0,
                                                   groups=1,
                                                   dilation=1,
                                                   has_bn=True,
                                                   has_act=False)

        self.local_reduce_conv1 = ConvBnActBlock(int(4 * cpfe_planes),
                                                 cpfe_planes,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 groups=1,
                                                 dilation=1,
                                                 has_bn=True,
                                                 has_act=False)

        self.local_upsample_conv1 = ConvTransposeBnActBlock(cpfe_planes,
                                                            cpfe_planes,
                                                            kernel_size=2,
                                                            stride=2,
                                                            groups=1,
                                                            has_bn=True,
                                                            has_act=True)
        self.local_upsample_conv2 = ConvBnActBlock(cpfe_planes,
                                                   cpfe_planes,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   groups=1,
                                                   dilation=1,
                                                   has_bn=True,
                                                   has_act=True)
        self.local_upsample_conv3 = ConvTransposeBnActBlock(cpfe_planes,
                                                            cpfe_planes,
                                                            kernel_size=2,
                                                            stride=2,
                                                            groups=1,
                                                            has_bn=True,
                                                            has_act=True)

        self.local_pred_conv = nn.Conv2d(cpfe_planes,
                                         1,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x1, x2, x3, x4 = self.neck(x)

        ##########################
        ### Decoder part - Global
        ##########################
        # Process high level features
        x4_feat_g = self.global_high_level_cpfe_4(x4)
        x3_feat_g = self.global_high_level_cpfe_3(x3)
        x4_feat_g = F.interpolate(x4_feat_g,
                                  size=(x3.shape[2], x3.shape[3]),
                                  mode='bilinear')

        conv_34_feats_g = torch.cat((x3_feat_g, x4_feat_g), dim=1)
        conv_34_feats_g = self.global_high_level_conv(conv_34_feats_g)
        conv_34_feats_g = F.interpolate(conv_34_feats_g,
                                        size=(x1.shape[2], x1.shape[3]),
                                        mode='bilinear')

        # Process low level features
        x1_feat_g = self.global_low_level_conv_1(x1)
        x2_feat_g = self.global_low_level_conv_2(x2)
        x2_feat_g = F.interpolate(x2_feat_g,
                                  size=(x1.shape[2], x1.shape[3]),
                                  mode='bilinear')

        conv_12_feats_g = torch.cat((x1_feat_g, x2_feat_g), dim=1)
        conv_12_feats_g = self.global_low_level_conv(conv_12_feats_g)

        conv_0_feats_g = torch.cat((conv_12_feats_g, conv_34_feats_g), dim=1)
        conv_0_feats_g = self.global_reduce_conv1(conv_0_feats_g)
        conv_0_feats_g = self.global_upsample_conv1(conv_0_feats_g)
        conv_0_feats_g = self.global_upsample_conv2(conv_0_feats_g)
        conv_0_feats_g = self.global_upsample_conv3(conv_0_feats_g)

        global_pred = self.global_pred_conv(conv_0_feats_g)

        ##########################
        ### Decoder part - Local
        ##########################
        # Process high level features
        x3_feat_f = self.local_high_level_cpfe_3(x3)
        x4_feat_f = self.local_high_level_cpfe_4(x4)
        x4_feat_f = F.interpolate(x4_feat_f,
                                  size=(x3.shape[2], x3.shape[3]),
                                  mode='bilinear')

        conv_34_feats_f = torch.cat((x3_feat_f, x4_feat_f), dim=1)
        conv_34_feats_f = self.local_high_level_conv(conv_34_feats_f)
        conv_34_feats_f = F.interpolate(conv_34_feats_f,
                                        size=(x1.shape[2], x1.shape[3]),
                                        mode='bilinear')
        conv_34_feats_f = torch.cat((conv_34_feats_f, conv_34_feats_g), dim=1)

        x1_feat_f = self.local_low_level_conv_1(x1)
        x2_feat_f = self.local_low_level_conv_2(x2)
        x2_feat_f = F.interpolate(x2_feat_f,
                                  size=(x1.shape[2], x1.shape[3]),
                                  mode='bilinear')

        conv_12_feats_f = torch.cat((x1_feat_f, x2_feat_f), dim=1)
        conv_12_feats_f = self.local_low_level_conv(conv_12_feats_f)
        conv_12_feats_f = torch.cat(
            (conv_12_feats_f, conv_12_feats_g, conv_34_feats_f), dim=1)

        conv_0_feats_f = self.local_reduce_conv1(conv_12_feats_f)
        conv_0_feats_f = self.local_upsample_conv1(conv_0_feats_f)
        conv_0_feats_f = self.local_upsample_conv2(conv_0_feats_f)
        conv_0_feats_f = self.local_upsample_conv3(conv_0_feats_f)

        local_pred = self.local_pred_conv(conv_0_feats_f)

        global_pred = global_pred.float()
        local_pred = local_pred.float()
        global_pred = self.sigmoid(global_pred)
        local_pred = self.sigmoid(local_pred)

        fused_pred = self.collaborative_matting(global_pred, local_pred)

        return global_pred, local_pred, fused_pred

    def collaborative_matting(self, global_pred, local_pred):
        # 0为背景区域，1为local区域，2为global区域
        device = global_pred.device
        # max_cls_idxs <===> [0, 1, 2]
        # max_cls_idxs:[b,h,w] -> [b,1,h,w]
        _, max_cls_idxs = torch.max(global_pred, dim=1)
        max_cls_idxs = torch.unsqueeze(max_cls_idxs.float(), dim=1)

        # trimap_mask:[0, 1, 2] ===> [0, 1, 0],保留local区域
        trimap_mask = max_cls_idxs.clone().to(device)
        trimap_mask[trimap_mask == 2] = 0

        # fg_mask: [0, 1, 2] ===> [0, 0, 1]，保留global区域
        fg_mask = max_cls_idxs.clone().to(device)
        fg_mask[fg_mask == 1] = 0
        fg_mask[fg_mask == 2] = 1

        # fused_pred只保留预测为128区域
        fused_pred = local_pred * trimap_mask + fg_mask

        return fused_pred


def _pfan_matting(backbone_type, backbone_pretrained_path, planes, **kwargs):
    model = PFANMatting(backbone_type=backbone_type,
                        backbone_pretrained_path=backbone_pretrained_path,
                        planes=planes,
                        **kwargs)

    return model


def dinov3_vit_small_patch16_pfan_matting(backbone_pretrained_path='',
                                          **kwargs):
    return _pfan_matting('dinov3_vit_small_patch16_backbone',
                         backbone_pretrained_path, 256, **kwargs)


def dinov3_vit_small_plus_patch16_pfan_matting(backbone_pretrained_path='',
                                               **kwargs):
    return _pfan_matting('dinov3_vit_small_plus_patch16_backbone',
                         backbone_pretrained_path, 256, **kwargs)


def dinov3_vit_base_patch16_pfan_matting(backbone_pretrained_path='',
                                         **kwargs):
    return _pfan_matting('dinov3_vit_base_patch16_backbone',
                         backbone_pretrained_path, 256, **kwargs)


def dinov3_vit_large_patch16_pfan_matting(backbone_pretrained_path='',
                                          **kwargs):
    return _pfan_matting('dinov3_vit_large_patch16_backbone',
                         backbone_pretrained_path, 256, **kwargs)


def dinov3_vit_large_plus_patch16_pfan_matting(backbone_pretrained_path='',
                                               **kwargs):
    return _pfan_matting('dinov3_vit_large_plus_patch16_backbone',
                         backbone_pretrained_path, 256, **kwargs)


def dinov3_vit_huge_plus_patch16_pfan_matting(backbone_pretrained_path='',
                                              **kwargs):
    return _pfan_matting('dinov3_vit_huge_plus_patch16_backbone',
                         backbone_pretrained_path, 256, **kwargs)


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

    net = dinov3_vit_small_patch16_pfan_matting()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = dinov3_vit_small_plus_patch16_pfan_matting()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = dinov3_vit_base_patch16_pfan_matting()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = dinov3_vit_large_patch16_pfan_matting()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = dinov3_vit_large_plus_patch16_pfan_matting()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = dinov3_vit_huge_plus_patch16_pfan_matting()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'1111, flops: {flops}, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))
