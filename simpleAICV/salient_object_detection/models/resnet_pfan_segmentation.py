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

from simpleAICV.salient_object_detection.models import backbones

__all__ = [
    'resnet18_pfan_segmentation',
    'resnet34_pfan_segmentation',
    'resnet50_pfan_segmentation',
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


class ResNetPFANSegmentation(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=[32, 64, 160, 256],
                 cpfe_planes=32):
        super(ResNetPFANSegmentation, self).__init__()
        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
            })

        self.high_level_cpfe_3 = CPFE(inplanes=planes[-2],
                                      planes=cpfe_planes,
                                      dilation_rate_list=[3, 5, 7])
        self.high_level_cpfe_4 = CPFE(inplanes=planes[-1],
                                      planes=cpfe_planes,
                                      dilation_rate_list=[3, 5, 7])

        self.high_level_conv = ConvBnActBlock(int(2 * cpfe_planes),
                                              cpfe_planes,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              groups=1,
                                              dilation=1,
                                              has_bn=True,
                                              has_act=False)

        # processing low level (ll) feature
        self.low_level_conv_1 = ConvBnActBlock(planes[-4],
                                               cpfe_planes,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               groups=1,
                                               dilation=1,
                                               has_bn=True,
                                               has_act=True)

        self.low_level_conv_2 = ConvBnActBlock(planes[-3],
                                               cpfe_planes,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               groups=1,
                                               dilation=1,
                                               has_bn=True,
                                               has_act=True)

        self.low_level_conv = ConvBnActBlock(int(2 * cpfe_planes),
                                             cpfe_planes,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             dilation=1,
                                             has_bn=True,
                                             has_act=False)

        self.reduce_conv1 = ConvBnActBlock(int(2 * cpfe_planes),
                                           cpfe_planes,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           groups=1,
                                           dilation=1,
                                           has_bn=True,
                                           has_act=False)

        self.upsample_conv1 = ConvTransposeBnActBlock(cpfe_planes,
                                                      cpfe_planes,
                                                      kernel_size=2,
                                                      stride=2,
                                                      groups=1,
                                                      has_bn=True,
                                                      has_act=True)
        self.upsample_conv2 = ConvBnActBlock(cpfe_planes,
                                             cpfe_planes,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             groups=1,
                                             dilation=1,
                                             has_bn=True,
                                             has_act=True)
        self.upsample_conv3 = ConvTransposeBnActBlock(cpfe_planes,
                                                      cpfe_planes,
                                                      kernel_size=2,
                                                      stride=2,
                                                      groups=1,
                                                      has_bn=True,
                                                      has_act=True)

        self.pred_conv = nn.Conv2d(cpfe_planes,
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

        # Process high level features
        x4_feat_g = self.high_level_cpfe_4(x4)
        x3_feat_g = self.high_level_cpfe_3(x3)
        # torch.Size([1, 32, 52, 52]) torch.Size([1, 32, 26, 26])

        x4_feat_g = F.upsample(x4_feat_g,
                               size=(x3.shape[2], x3.shape[3]),
                               mode='bilinear',
                               align_corners=True)
        # torch.Size([1, 32, 52, 52])
        conv_34_feats = torch.cat((x3_feat_g, x4_feat_g), dim=1)
        # torch.Size([1, 64, 52, 52])

        conv_34_feats = self.high_level_conv(conv_34_feats)
        # conv_34_feats:[1, 32, 52, 52]
        conv_34_feats = F.upsample(conv_34_feats,
                                   size=(x1.shape[2], x1.shape[3]),
                                   mode='bilinear',
                                   align_corners=True)
        # torch.Size([1, 32, 208, 208])

        # Process low level features
        x1_feat_g = self.low_level_conv_1(x1)
        x2_feat_g = self.low_level_conv_2(x2)
        # torch.Size([1, 32, 208, 208]) torch.Size([1, 32, 104, 104])
        x2_feat_g = F.upsample(x2_feat_g,
                               size=(x1.shape[2], x1.shape[3]),
                               mode='bilinear',
                               align_corners=True)
        # torch.Size([1, 32, 208, 208])
        conv_12_feats = torch.cat((x1_feat_g, x2_feat_g), dim=1)
        # torch.Size([1, 64, 208, 208])
        conv_12_feats = self.low_level_conv(conv_12_feats)
        # torch.Size([1, 32, 208, 208])

        conv_0_feats = torch.cat((conv_12_feats, conv_34_feats), dim=1)
        # torch.Size([1, 64, 208, 208])
        conv_0_feats = self.reduce_conv1(conv_0_feats)
        # torch.Size([1, 32, 208, 208])

        conv_0_feats = self.upsample_conv1(conv_0_feats)
        # torch.Size([1, 32, 416, 416])
        conv_0_feats = self.upsample_conv2(conv_0_feats)
        # torch.Size([1, 32, 416, 416])
        conv_0_feats = self.upsample_conv3(conv_0_feats)
        # [1, 32, 832, 832]

        pred = self.pred_conv(conv_0_feats)
        # torch.Size([1, 1, 832, 832])

        pred = self.sigmoid(pred)

        return pred


def _resnet_pfan_segmentation(backbone_type, backbone_pretrained_path, planes,
                              **kwargs):
    model = ResNetPFANSegmentation(
        backbone_type=backbone_type,
        backbone_pretrained_path=backbone_pretrained_path,
        planes=planes,
        **kwargs)

    return model


def resnet18_pfan_segmentation(backbone_pretrained_path='', **kwargs):
    return _resnet_pfan_segmentation('resnet18backbone',
                                     backbone_pretrained_path,
                                     [64, 128, 256, 512], **kwargs)


def resnet34_pfan_segmentation(backbone_pretrained_path='', **kwargs):
    return _resnet_pfan_segmentation('resnet34backbone',
                                     backbone_pretrained_path,
                                     [64, 128, 256, 512], **kwargs)


def resnet50_pfan_segmentation(backbone_pretrained_path='', **kwargs):
    return _resnet_pfan_segmentation('resnet50backbone',
                                     backbone_pretrained_path,
                                     [256, 512, 1024, 2048], **kwargs)


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

    net = resnet50_pfan_segmentation()
    image_h, image_w = 832, 832
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(torch.rand(1, 3, image_h, image_w))
    print('2222', out.shape, torch.max(out), torch.min(out))
