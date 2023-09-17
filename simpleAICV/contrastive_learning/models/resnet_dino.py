'''
Emerging Properties in Self-Supervised Vision Transformers
https://github.com/facebookresearch/dino
'''
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

from simpleAICV.classification.backbones.resnet import ConvBnActBlock, BasicBlock, Bottleneck

__all__ = [
    'resnet18_dino_pretrain_model',
    'resnet34_dino_pretrain_model',
    'resnet50_dino_pretrain_model',
]


class DINOPretrainModelHead(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 hidden_planes=2048,
                 bottleneck_planes=256,
                 use_bn=False,
                 use_norm_last_layer=True,
                 layer_nums=3):
        super(DINOPretrainModelHead, self).__init__()
        layers = []
        self.inplanes = inplanes
        for i in range(layer_nums):
            if i == 0 and layer_nums == 1:
                layers.append(nn.Linear(self.inplanes, bottleneck_planes))
                break
            elif i < layer_nums - 1:
                layers.append(nn.Linear(self.inplanes, hidden_planes))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_planes))
                layers.append(nn.GELU())
                self.inplanes = hidden_planes
            else:
                layers.append(nn.Linear(hidden_planes, bottleneck_planes))
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_planes, planes, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if use_norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.layers(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)

        return x


class ResNetDINOPretrainModel(nn.Module):

    def __init__(self,
                 block,
                 layer_nums,
                 inplanes=64,
                 head_planes=65535,
                 head_hidden_planes=2048,
                 head_bottleneck_planes=256,
                 head_use_bn=False,
                 head_use_norm_last_layer=True,
                 head_layer_nums=3,
                 global_crop_number=2,
                 local_crops_number=8):
        super(ResNetDINOPretrainModel, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4
        self.head_planes = head_planes
        self.head_hidden_planes = head_hidden_planes
        self.head_bottleneck_planes = head_bottleneck_planes
        self.head_use_bn = head_use_bn
        self.head_use_norm_last_layer = head_use_norm_last_layer
        self.head_layer_nums = head_layer_nums
        self.global_crop_number = global_crop_number
        self.local_crops_number = local_crops_number

        self.conv1 = ConvBnActBlock(3,
                                    self.inplanes,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block,
                                      self.planes[0],
                                      self.layer_nums[0],
                                      stride=1)
        self.layer2 = self.make_layer(self.block,
                                      self.planes[1],
                                      self.layer_nums[1],
                                      stride=2)
        self.layer3 = self.make_layer(self.block,
                                      self.planes[2],
                                      self.layer_nums[2],
                                      stride=2)
        self.layer4 = self.make_layer(self.block,
                                      self.planes[3],
                                      self.layer_nums[3],
                                      stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = DINOPretrainModelHead(
            self.planes[3] * self.expansion,
            self.head_planes,
            hidden_planes=self.head_hidden_planes,
            bottleneck_planes=self.head_bottleneck_planes,
            use_bn=self.head_use_bn,
            use_norm_last_layer=self.head_use_norm_last_layer,
            layer_nums=self.head_layer_nums)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, planes, layer_nums, stride):
        layers = []
        for i in range(0, layer_nums):
            if i == 0:
                layers.append(block(self.inplanes, planes, stride))
            else:
                layers.append(block(self.inplanes, planes))
            self.inplanes = planes * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x) == self.global_crop_number:
            crop_idxs = [self.global_crop_number]
        elif len(x) == self.global_crop_number + self.local_crops_number:
            crop_idxs = [
                self.global_crop_number,
                self.global_crop_number + self.local_crops_number,
            ]

        start_idx, outputs = 0, None
        for i, end_idx in enumerate(crop_idxs):
            inputs = torch.cat(x[start_idx:end_idx], dim=0)
            output = self.conv1(inputs)
            output = self.maxpool1(output)

            output = self.layer1(output)
            output = self.layer2(output)
            output = self.layer3(output)
            output = self.layer4(output)

            output = self.avgpool(output)
            output = output.view(output.size(0), -1)

            if i == 0:
                outputs = output
            else:
                outputs = torch.cat((outputs, output))

            start_idx = end_idx

        outputs = self.head(outputs)

        return outputs

    def get_attention_map(self, x):
        assert not isinstance(x, list)
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnetdinopretrainmodel(**kwargs):
    model = ResNetDINOPretrainModel(**kwargs)

    return model


def resnet18_dino_pretrain_model(**kwargs):
    return _resnetdinopretrainmodel(block=BasicBlock,
                                    layer_nums=[2, 2, 2, 2],
                                    inplanes=64,
                                    **kwargs)


def resnet34_dino_pretrain_model(**kwargs):
    return _resnetdinopretrainmodel(block=BasicBlock,
                                    layer_nums=[3, 4, 6, 3],
                                    inplanes=64,
                                    **kwargs)


def resnet50_dino_pretrain_model(**kwargs):
    return _resnetdinopretrainmodel(block=Bottleneck,
                                    layer_nums=[3, 4, 6, 3],
                                    inplanes=64,
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

    net = resnet50_dino_pretrain_model()
    inputs = [
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
    ]
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(inputs, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(inputs)
    print(f'1111, macs: {macs}, params: {params}, out_shape: {out.shape}')

    net = resnet50_dino_pretrain_model()
    inputs = [
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 96, 96),
        torch.randn(1, 3, 96, 96),
        torch.randn(1, 3, 96, 96),
        torch.randn(1, 3, 96, 96),
        torch.randn(1, 3, 96, 96),
        torch.randn(1, 3, 96, 96),
        torch.randn(1, 3, 96, 96),
        torch.randn(1, 3, 96, 96),
    ]
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(inputs, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(inputs)
    print(f'2222, macs: {macs}, params: {params}, out_shape: {out.shape}')
