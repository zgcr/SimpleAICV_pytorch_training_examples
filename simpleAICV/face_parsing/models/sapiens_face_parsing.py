import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from simpleAICV.detection.models import backbones

__all__ = [
    'sapiens_0_3b_face_parsing',
    'sapiens_0_6b_face_parsing',
    'sapiens_1_0b_face_parsing',
    'sapiens_2_0b_face_parsing',
]


class VitHead(nn.Module):

    def __init__(self,
                 inplanes=1024,
                 deconv_planes=(512, 256, 128, 64),
                 deconv_kernel_sizes=(4, 4, 4, 4),
                 conv_planes=(64, 64, 32, 32),
                 conv_kernel_sizes=(1, 1, 1, 1),
                 num_classes=19):
        super(VitHead, self).__init__()
        assert len(deconv_planes) == len(deconv_kernel_sizes)
        assert len(conv_planes) == len(conv_kernel_sizes)

        deconv_layers = []
        for planes, kernel_size in zip(deconv_planes, deconv_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0

            deconv_layers.append(
                nn.ConvTranspose2d(inplanes,
                                   planes,
                                   kernel_size=kernel_size,
                                   stride=2,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=False))
            deconv_layers.append(nn.InstanceNorm2d(planes))
            deconv_layers.append(nn.SiLU(inplace=True))
            inplanes = planes

        self.deconv_layers = nn.Sequential(*deconv_layers)

        inplanes = deconv_planes[-1]

        conv_layers = []
        for planes, kernel_size in zip(conv_planes, conv_kernel_sizes):
            padding = (kernel_size - 1) // 2
            conv_layers.append(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size,
                          stride=1,
                          padding=padding,
                          bias=False))
            conv_layers.append(nn.InstanceNorm2d(planes))
            conv_layers.append(nn.SiLU(inplace=True))
            inplanes = planes
        self.conv_layers = nn.Sequential(*conv_layers)

        inplanes = conv_planes[-1]

        self.pred_conv = nn.Conv2d(inplanes,
                                   num_classes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.pred_conv(x)

        return x


class ViTParsing(nn.Module):
    """
    num_classes数量必须包含背景类
    """

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 image_size=512,
                 planes=1024,
                 deconv_planes=(512, 256, 128, 64),
                 deconv_kernel_sizes=(4, 4, 4, 4),
                 conv_planes=(64, 64, 32, 32),
                 conv_kernel_sizes=(1, 1, 1, 1),
                 num_classes=19,
                 use_gradient_checkpoint=False):
        super(ViTParsing, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'image_size': image_size,
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        self.head = VitHead(inplanes=planes,
                            deconv_planes=deconv_planes,
                            deconv_kernel_sizes=deconv_kernel_sizes,
                            conv_planes=conv_planes,
                            conv_kernel_sizes=conv_kernel_sizes,
                            num_classes=num_classes)

    def forward(self, x):
        # x: [b,3,512,512]
        # torch.Size([1, 1024, 32, 32])
        x = self.backbone(x)

        # torch.Size([1, 19, 512, 512])
        x = self.head(x)

        return x


def _vit_face_parsing(backbone_type, backbone_pretrained_path, planes,
                      **kwargs):
    model = ViTParsing(backbone_type=backbone_type,
                       backbone_pretrained_path=backbone_pretrained_path,
                       planes=planes,
                       **kwargs)

    return model


def sapiens_0_3b_face_parsing(backbone_pretrained_path='', **kwargs):
    return _vit_face_parsing('sapiens_0_3b_backbone', backbone_pretrained_path,
                             1024, **kwargs)


def sapiens_0_6b_face_parsing(backbone_pretrained_path='', **kwargs):
    return _vit_face_parsing('sapiens_0_6b_backbone', backbone_pretrained_path,
                             1280, **kwargs)


def sapiens_1_0b_face_parsing(backbone_pretrained_path='', **kwargs):
    return _vit_face_parsing('sapiens_1_0b_backbone', backbone_pretrained_path,
                             1536, **kwargs)


def sapiens_2_0b_face_parsing(backbone_pretrained_path='', **kwargs):
    return _vit_face_parsing('sapiens_2_0b_backbone', backbone_pretrained_path,
                             1920, **kwargs)


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

    net = sapiens_0_3b_face_parsing(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(torch.rand(1, 3, image_h, image_w))
    print('2222', out.shape, torch.max(out), torch.min(out))

    net = sapiens_0_6b_face_parsing(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(torch.rand(1, 3, image_h, image_w))
    print('2222', out.shape, torch.max(out), torch.min(out))

    net = sapiens_1_0b_face_parsing(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(torch.rand(1, 3, image_h, image_w))
    print('2222', out.shape, torch.max(out), torch.min(out))

    net = sapiens_2_0b_face_parsing(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.rand(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(torch.rand(1, 3, image_h, image_w))
    print('2222', out.shape, torch.max(out), torch.min(out))

    net = sapiens_2_0b_face_parsing(image_size=512,
                                    use_gradient_checkpoint=True)
    image_h, image_w = 512, 512
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    out = net(torch.rand(1, 3, image_h, image_w))
    print('2222', out.shape, torch.max(out), torch.min(out))
