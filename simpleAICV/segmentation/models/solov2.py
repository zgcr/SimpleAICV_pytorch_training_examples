import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import pretrained_models_path

from simpleAICV.segmentation.common import load_state_dict
from simpleAICV.segmentation.models.backbone import ResNetBackbone
from simpleAICV.segmentation.models.fpn import Solov2FPN
from simpleAICV.segmentation.models.head import Solov2InsHead, Solov2MaskHead

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_solov2',
    'resnet34_solov2',
    'resnet50_solov2',
    'resnet101_solov2',
    'resnet152_solov2',
]

model_urls = {
    'resnet18_solov2': 'empty',
    'resnet34_solov2': 'empty',
    'resnet50_solov2': 'empty',
    'resnet101_solov2': 'empty',
    'resnet152_solov2': 'empty',
}


# assert input annotations are[x_min,y_min,x_max,y_max]
class SOLOV2(nn.Module):
    def __init__(self,
                 resnet_type='resnet50',
                 num_grids=[40, 36, 24, 16, 12],
                 num_classes=80,
                 planes=256):
        super(SOLOV2, self).__init__()
        self.backbone = ResNetBackbone(resnet_type=resnet_type,
                                       pretrained=True)
        expand_ratio = {
            'resnet18': 1,
            'resnet34': 1,
            'resnet50': 4,
            'resnet101': 4,
            'resnet152': 4,
        }
        C2_inplanes, C3_inplanes, C4_inplanes, C5_inplanes = int(
            64 * expand_ratio[resnet_type]), int(
                128 * expand_ratio[resnet_type]), int(
                    256 * expand_ratio[resnet_type]), int(
                        512 * expand_ratio[resnet_type])

        self.fpn = Solov2FPN(C2_inplanes, C3_inplanes, C4_inplanes,
                             C5_inplanes, planes)
        self.inshead = Solov2InsHead(planes,
                                     planes=512,
                                     num_classes=num_classes,
                                     num_kernels=256,
                                     num_grids=[40, 36, 24, 16, 12],
                                     num_layers=4,
                                     prior=0.01,
                                     use_gn=True)
        self.maskhead = Solov2MaskHead(planes,
                                       planes=128,
                                       num_masks=256,
                                       num_layers=4,
                                       use_gn=True)

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device

        outs = self.backbone(inputs)

        del inputs

        features = self.fpn(outs)

        del outs

        mask_out = self.maskhead(features[:-1])
        cate_outs, kernel_outs = self.inshead(features)

        del features

        return cate_outs, kernel_outs, mask_out


def _solov2(arch, pretrained, **kwargs):
    model = SOLOV2(arch, **kwargs)

    if pretrained:
        load_state_dict(
            torch.load(model_urls[arch + '_solov2'],
                       map_location=torch.device('cpu')), model)

    return model


def resnet18_solov2(pretrained=False, **kwargs):
    return _solov2('resnet18', pretrained, **kwargs)


def resnet34_solov2(pretrained=False, **kwargs):
    return _solov2('resnet34', pretrained, **kwargs)


def resnet50_solov2(pretrained=False, **kwargs):
    return _solov2('resnet50', pretrained, **kwargs)


def resnet101_solov2(pretrained=False, **kwargs):
    return _solov2('resnet101', pretrained, **kwargs)


def resnet152_solov2(pretrained=False, **kwargs):
    return _solov2('resnet152', pretrained, **kwargs)


if __name__ == '__main__':
    net = SOLOV2(resnet_type='resnet50')
    image_h, image_w = 640, 640
    cate_outs, kernel_outs, mask_out = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    for x, y in zip(cate_outs, kernel_outs):
        print('1111', x.shape, y.shape)
    print("2222", mask_out.shape)

    from thop import profile
    from thop import clever_format
    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    print(f"3333, flops: {flops}, params: {params}")