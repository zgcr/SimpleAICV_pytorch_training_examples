import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import math

import torch
import torch.nn as nn

from simpleAICV.detection.common import load_state_dict

__all__ = [
    'detr_resnet18backbone',
    'detr_resnet34backbone',
    'detr_resnet50backbone',
    'detr_resnet101backbone',
    'detr_resnet152backbone',
]


class PositionEmbeddingBlock(nn.Module):

    def __init__(self, inplanes=128, temperature=10000, eps=1e-6):
        super(PositionEmbeddingBlock, self).__init__()
        self.inplanes = inplanes
        self.temperature = temperature
        self.eps = eps
        self.scale = 2 * math.pi

    def forward(self, masks):
        assert masks is not None
        device = masks.device

        not_masks = ~masks
        y_embed = torch.cumsum(not_masks, 1, dtype=torch.float32)
        x_embed = torch.cumsum(not_masks, 2, dtype=torch.float32)

        # normalize
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.inplanes, dtype=torch.float32, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.inplanes)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (torch.sin(pos_x[:, :, :, 0::2]), torch.cos(pos_x[:, :, :, 1::2])),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (torch.sin(pos_y[:, :, :, 0::2]), torch.cos(pos_y[:, :, :, 1::2])),
            dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class DINOPositionEmbeddingBlock(nn.Module):

    def __init__(self,
                 inplanes=128,
                 temperature_h=10000,
                 temperature_w=10000,
                 eps=1e-6):
        super(DINOPositionEmbeddingBlock, self).__init__()
        self.inplanes = inplanes
        self.temperature_h = temperature_h
        self.temperature_w = temperature_w
        self.eps = eps
        self.scale = 2 * math.pi

    def forward(self, masks):
        assert masks is not None
        device = masks.device

        not_masks = ~masks
        y_embed = torch.cumsum(not_masks, 1, dtype=torch.float32)
        x_embed = torch.cumsum(not_masks, 2, dtype=torch.float32)

        # normalize
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_tx = torch.arange(self.inplanes,
                              dtype=torch.float32,
                              device=device)
        dim_tx = self.temperature_w**(2 * (dim_tx // 2) / self.inplanes)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.inplanes,
                              dtype=torch.float32,
                              device=device)
        dim_ty = self.temperature_h**(2 * (dim_ty // 2) / self.inplanes)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack(
            (torch.sin(pos_x[:, :, :, 0::2]), torch.cos(pos_x[:, :, :, 1::2])),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (torch.sin(pos_y[:, :, :, 0::2]), torch.cos(pos_y[:, :, :, 1::2])),
            dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, planes):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(planes))
        self.register_buffer("bias", torch.zeros(planes))
        self.register_buffer("running_mean", torch.zeros(planes))
        self.register_buffer("running_var", torch.ones(planes))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale

        return x * scale + bias


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
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
                      bias=bias),
            FrozenBatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 1 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 4 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(planes,
                                    planes * 4,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes * 4,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class ResNetBackbone(nn.Module):

    def __init__(self, block, layer_nums, inplanes=64):
        super(ResNetBackbone, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4

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

        self.out_channels = [
            self.planes[0] * self.expansion,
            self.planes[1] * self.expansion,
            self.planes[2] * self.expansion,
            self.planes[3] * self.expansion,
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (FrozenBatchNorm2d)):
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
        x = self.conv1(x)
        x = self.maxpool1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def _resnetbackbone(block, layers, inplanes, pretrained_path=''):
    model = ResNetBackbone(block, layers, inplanes)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def detr_resnet18backbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [2, 2, 2, 2],
                            64,
                            pretrained_path=pretrained_path)

    return model


def detr_resnet34backbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def detr_resnet50backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def detr_resnet101backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 23, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def detr_resnet152backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 8, 36, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


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

    net = detr_resnet50backbone()
    image_h, image_w = 800, 800
    from thop import profile
    from thop import clever_format
    x = torch.randn(1, 3, image_h, image_w)
    macs, params = profile(net, inputs=(x, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(torch.autograd.Variable(x))
    for per_out in out:
        print('2222', per_out.shape)