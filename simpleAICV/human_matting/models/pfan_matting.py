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

from simpleAICV.detection.models import backbones

__all__ = [
    'resnet18_pfan_matting',
    'resnet34_pfan_matting',
    'resnet50_pfan_matting',
    'resnet101_pfan_matting',
    'resnet152_pfan_matting',
    'vanb0_pfan_matting',
    'vanb1_pfan_matting',
    'vanb2_pfan_matting',
    'vanb3_pfan_matting',
    'convformers18_pfan_matting',
    'convformers36_pfan_matting',
    'convformerm36_pfan_matting',
    'convformerb36_pfan_matting',
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
                 planes=[32, 64, 160, 256],
                 cpfe_planes=32,
                 use_gradient_checkpoint=False):
        super(PFANMatting, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        self.global_high_level_cpfe_3 = CPFE(inplanes=planes[-2],
                                             planes=cpfe_planes,
                                             dilation_rate_list=[3, 5, 7])
        self.global_high_level_cpfe_4 = CPFE(inplanes=planes[-1],
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
        self.global_low_level_conv_1 = ConvBnActBlock(planes[-4],
                                                      cpfe_planes,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      groups=1,
                                                      dilation=1,
                                                      has_bn=True,
                                                      has_act=True)

        self.global_low_level_conv_2 = ConvBnActBlock(planes[-3],
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

        self.local_high_level_cpfe_3 = CPFE(inplanes=planes[-2],
                                            planes=cpfe_planes,
                                            dilation_rate_list=[3, 5, 7])
        self.local_high_level_cpfe_4 = CPFE(inplanes=planes[-1],
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

        self.local_low_level_conv_1 = ConvBnActBlock(planes[-4],
                                                     cpfe_planes,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     groups=1,
                                                     dilation=1,
                                                     has_bn=True,
                                                     has_act=True)

        self.local_low_level_conv_2 = ConvBnActBlock(planes[-3],
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
        # x: [b,3,832,832]
        # torch.Size([1, 32, 208, 208]) torch.Size([1, 64, 104, 104]) torch.Size([1, 160, 52, 52]) torch.Size([1, 256, 26, 26])
        x1, x2, x3, x4 = self.backbone(x)

        ##########################
        ### Decoder part - Global
        ##########################
        # Process high level features
        x4_feat_g = self.global_high_level_cpfe_4(x4)
        x3_feat_g = self.global_high_level_cpfe_3(x3)
        # torch.Size([1, 32, 52, 52]) torch.Size([1, 32, 26, 26])

        x4_feat_g = F.interpolate(x4_feat_g,
                                  size=(x3.shape[2], x3.shape[3]),
                                  mode='bilinear',
                                  align_corners=True)
        # torch.Size([1, 32, 52, 52])
        conv_34_feats_g = torch.cat((x3_feat_g, x4_feat_g), dim=1)
        # torch.Size([1, 64, 52, 52])

        conv_34_feats_g = self.global_high_level_conv(conv_34_feats_g)
        # conv_34_feats_g:[1, 32, 52, 52]
        conv_34_feats_g = F.interpolate(conv_34_feats_g,
                                        size=(x1.shape[2], x1.shape[3]),
                                        mode='bilinear',
                                        align_corners=True)
        # torch.Size([1, 32, 208, 208])

        # Process low level features
        x1_feat_g = self.global_low_level_conv_1(x1)
        x2_feat_g = self.global_low_level_conv_2(x2)
        # torch.Size([1, 32, 208, 208]) torch.Size([1, 32, 104, 104])
        x2_feat_g = F.interpolate(x2_feat_g,
                                  size=(x1.shape[2], x1.shape[3]),
                                  mode='bilinear',
                                  align_corners=True)
        # torch.Size([1, 32, 208, 208])
        conv_12_feats_g = torch.cat((x1_feat_g, x2_feat_g), dim=1)
        # torch.Size([1, 64, 208, 208])
        conv_12_feats_g = self.global_low_level_conv(conv_12_feats_g)
        # torch.Size([1, 32, 208, 208])

        conv_0_feats_g = torch.cat((conv_12_feats_g, conv_34_feats_g), dim=1)

        # torch.Size([1, 64, 208, 208])
        conv_0_feats_g = self.global_reduce_conv1(conv_0_feats_g)
        # torch.Size([1, 32, 208, 208])

        conv_0_feats_g = self.global_upsample_conv1(conv_0_feats_g)
        # torch.Size([1, 32, 416, 416])
        conv_0_feats_g = self.global_upsample_conv2(conv_0_feats_g)
        # torch.Size([1, 32, 416, 416])
        conv_0_feats_g = self.global_upsample_conv3(conv_0_feats_g)
        # [1, 32, 832, 832]

        global_pred = self.global_pred_conv(conv_0_feats_g)
        # torch.Size([1, 3, 832, 832])
        # global_pred:[1, 3, 832, 832],3:0为背景区域，1为local区域，2为global区域

        ##########################
        ### Decoder part - Local
        ##########################
        # Process high level features
        x3_feat_f = self.local_high_level_cpfe_3(x3)
        x4_feat_f = self.local_high_level_cpfe_4(x4)
        # torch.Size([1, 32, 52, 52]) torch.Size([1, 32, 26, 26])

        x4_feat_f = F.interpolate(x4_feat_f,
                                  size=(x3.shape[2], x3.shape[3]),
                                  mode='bilinear',
                                  align_corners=True)
        # torch.Size([1, 32, 52, 52])

        conv_34_feats_f = torch.cat((x3_feat_f, x4_feat_f), dim=1)
        # torch.Size([1, 64, 52, 52])

        conv_34_feats_f = self.local_high_level_conv(conv_34_feats_f)
        # torch.Size([1, 32, 52, 52])

        conv_34_feats_f = F.interpolate(conv_34_feats_f,
                                        size=(x1.shape[2], x1.shape[3]),
                                        mode='bilinear',
                                        align_corners=True)
        # torch.Size([1, 32, 208, 208])

        conv_34_feats_f = torch.cat((conv_34_feats_f, conv_34_feats_g), dim=1)
        # torch.Size([1, 64, 208, 208])

        x1_feat_f = self.local_low_level_conv_1(x1)
        x2_feat_f = self.local_low_level_conv_2(x2)
        # torch.Size([1, 32, 208, 208]) torch.Size([1, 32, 104, 104])

        x2_feat_f = F.interpolate(x2_feat_f,
                                  size=(x1.shape[2], x1.shape[3]),
                                  mode='bilinear',
                                  align_corners=True)
        # torch.Size([1, 32, 208, 208])

        conv_12_feats_f = torch.cat((x1_feat_f, x2_feat_f), dim=1)
        # torch.Size([1, 64, 208, 208])
        conv_12_feats_f = self.local_low_level_conv(conv_12_feats_f)
        # torch.Size([1, 32, 208, 208])

        conv_12_feats_f = torch.cat(
            (conv_12_feats_f, conv_12_feats_g, conv_34_feats_f), dim=1)
        # torch.Size([1, 128, 208, 208])

        conv_0_feats_f = self.local_reduce_conv1(conv_12_feats_f)
        # torch.Size([1, 32, 208, 208])

        conv_0_feats_f = self.local_upsample_conv1(conv_0_feats_f)
        # torch.Size([1, 32, 416, 416])
        conv_0_feats_f = self.local_upsample_conv2(conv_0_feats_f)
        # torch.Size([1, 32, 416, 416])
        conv_0_feats_f = self.local_upsample_conv3(conv_0_feats_f)
        # [1, 32, 832, 832]

        local_pred = self.local_pred_conv(conv_0_feats_f)
        # torch.Size([1, 1, 832, 832])
        # local_pred:[1, 1, 832, 832]

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


def resnet18_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('resnet18backbone', backbone_pretrained_path,
                         [64, 128, 256, 512], **kwargs)


def resnet34_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('resnet34backbone', backbone_pretrained_path,
                         [64, 128, 256, 512], **kwargs)


def resnet50_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('resnet50backbone', backbone_pretrained_path,
                         [256, 512, 1024, 2048], **kwargs)


def resnet101_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('resnet101backbone', backbone_pretrained_path,
                         [256, 512, 1024, 2048], **kwargs)


def resnet152_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('resnet152backbone', backbone_pretrained_path,
                         [256, 512, 1024, 2048], **kwargs)


def vanb0_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('vanb0backbone', backbone_pretrained_path,
                         [32, 64, 160, 256], **kwargs)


def vanb1_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('vanb1backbone', backbone_pretrained_path,
                         [64, 128, 320, 512], **kwargs)


def vanb2_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('vanb2backbone', backbone_pretrained_path,
                         [64, 128, 320, 512], **kwargs)


def vanb3_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('vanb3backbone', backbone_pretrained_path,
                         [64, 128, 320, 512], **kwargs)


def convformers18_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('convformers18backbone', backbone_pretrained_path,
                         [64, 128, 320, 512], **kwargs)


def convformers36_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('convformers36backbone', backbone_pretrained_path,
                         [64, 128, 320, 512], **kwargs)


def convformerm36_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('convformerm36backbone', backbone_pretrained_path,
                         [96, 192, 384, 576], **kwargs)


def convformerb36_pfan_matting(backbone_pretrained_path='', **kwargs):
    return _pfan_matting('convformerb36backbone', backbone_pretrained_path,
                         [128, 256, 512, 768], **kwargs)


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

    net = resnet18_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = resnet34_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = resnet50_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = resnet101_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = resnet152_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = vanb0_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = vanb1_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = vanb2_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = vanb3_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = vanb3_pfan_matting(use_gradient_checkpoint=True)
    image_h, image_w = 832, 832
    outs = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = convformers18_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = convformers36_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = convformerm36_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))

    net = convformerb36_pfan_matting()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.rand(1, 3, image_h, image_w))
    for per_out in outs:
        print('2222', per_out.shape, torch.max(per_out), torch.min(per_out))
