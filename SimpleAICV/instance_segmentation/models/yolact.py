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

__all__ = [
    'resnet18_yolact',
    'resnet34_yolact',
    'resnet50_yolact',
    'resnet101_yolact',
    'resnet152_yolact',
    'vanb0_yolact',
    'vanb1_yolact',
    'vanb2_yolact',
    'vanb3_yolact',
    'convformers18_yolact',
    'convformers36_yolact',
    'convformerm36_yolact',
    'convformerb36_yolact',
]


class YOLACTFPN(nn.Module):

    def __init__(self, inplanes, planes):
        super(YOLACTFPN, self).__init__()
        self.inplanes = inplanes

        self.lat_layer_p3 = nn.Conv2d(inplanes[0],
                                      planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.lat_layer_p4 = nn.Conv2d(inplanes[1],
                                      planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.lat_layer_p5 = nn.Conv2d(inplanes[2],
                                      planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        self.pred_layers_p3 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
        )
        self.pred_layers_p4 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
        )
        self.pred_layers_p5 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
        )

        self.downsample_layers_p6 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
        )
        self.downsample_layers_p7 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, inputs):
        assert len(inputs) == len(self.inplanes)
        [C3, C4, C5] = inputs

        P5 = self.lat_layer_p5(C5)
        P4 = self.lat_layer_p4(C4)
        P4 = F.interpolate(
            P5, size=(P4.shape[2], P4.shape[3]), mode='bilinear') + P4
        P3 = self.lat_layer_p3(C3)
        P3 = F.interpolate(
            P4, size=(P3.shape[2], P3.shape[3]), mode='bilinear') + P3

        del C3, C4

        P5 = self.pred_layers_p5(P5)
        P4 = self.pred_layers_p4(P4)
        P3 = self.pred_layers_p3(P3)

        P6 = self.downsample_layers_p6(P5)
        P7 = self.downsample_layers_p7(P6)

        return P3, P4, P5, P6, P7


class YOLACTHead(nn.Module):

    def __init__(self,
                 ratios=[1, 1 / 2, 2],
                 inplanes=256,
                 proto_planes=32,
                 num_classes=80):
        super(YOLACTHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
        )

        self.bbox_pred_conv = nn.Conv2d(inplanes,
                                        len(ratios) * 4,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True)
        self.conf_pred_conv = nn.Conv2d(inplanes,
                                        len(ratios) * num_classes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True)
        self.coef_pred_conv = nn.Sequential(
            nn.Conv2d(inplanes,
                      len(ratios) * proto_planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.conv1(x)

        conf_pred = self.conf_pred_conv(x)
        bbox_pred = self.bbox_pred_conv(x)
        coef_pred = self.coef_pred_conv(x)

        return conf_pred, bbox_pred, coef_pred


class ProtoNet(nn.Module):

    def __init__(self, inplanes, planes):
        super(ProtoNet, self).__init__()
        self.proto_layers1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
        )
        self.proto_layers2 = nn.Sequential(
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x, size):
        x = self.proto_layers1(x)
        x = F.interpolate(x, size=size, mode='bilinear')
        x = self.proto_layers2(x)

        return x


class YOLACT(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 fpn_planes=256,
                 proto_planes=32,
                 num_classes=81,
                 use_gradient_checkpoint=False):
        super(YOLACT, self).__init__()
        self.fpn_planes = fpn_planes
        self.proto_planes = proto_planes
        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.fpn = YOLACTFPN(self.backbone.out_channels[1:4], self.fpn_planes)
        self.proto_net = ProtoNet(self.fpn_planes, self.proto_planes)
        self.head = YOLACTHead(ratios=[1, 1 / 2, 2],
                               inplanes=self.fpn_planes,
                               proto_planes=self.proto_planes,
                               num_classes=self.num_classes)

        self.semantic_seg_conv = nn.Conv2d(self.fpn_planes,
                                           self.num_classes - 1,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=True)

        nn.init.normal_(self.semantic_seg_conv.weight, std=0.01)
        nn.init.constant_(self.semantic_seg_conv.bias, val=0)

    def forward(self, inputs):
        features = self.backbone(inputs)

        # 1111 torch.Size([16, 256, 136, 136])
        # 1111 torch.Size([16, 512, 68, 68])
        # 1111 torch.Size([16, 1024, 34, 34])
        # 1111 torch.Size([16, 2048, 17, 17])

        feature_sizes = [[
            per_level_features.shape[2], per_level_features.shape[3]
        ] for per_level_features in features]

        if self.use_gradient_checkpoint:
            features = checkpoint(self.fpn, features[1:4], use_reentrant=False)
        else:
            features = self.fpn(features[1:4])

        # 2222 torch.Size([16, 256, 68, 68])
        # 2222 torch.Size([16, 256, 34, 34])
        # 2222 torch.Size([16, 256, 17, 17])
        # 2222 torch.Size([16, 256, 9, 9])
        # 2222 torch.Size([16, 256, 5, 5])

        proto_out = self.proto_net(features[0], feature_sizes[0])
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        # 3333 torch.Size([16, 136, 136, 32])

        class_preds, box_preds, coef_preds = [], [], []

        for feature in features:
            class_pred, box_pred, coef_pred = self.head(feature)

            class_pred = class_pred.permute(0, 2, 3, 1)
            class_pred = class_pred.view(class_pred.shape[0],
                                         class_pred.shape[1],
                                         class_pred.shape[2], -1,
                                         self.num_classes).contiguous()
            box_pred = box_pred.permute(0, 2, 3, 1)
            box_pred = box_pred.view(box_pred.shape[0], box_pred.shape[1],
                                     box_pred.shape[2], -1, 4).contiguous()

            coef_pred = coef_pred.permute(0, 2, 3, 1)
            coef_pred = coef_pred.view(coef_pred.shape[0], coef_pred.shape[1],
                                       coef_pred.shape[2], -1,
                                       self.proto_planes).contiguous()

            # 1212 torch.Size([1, 68, 68, 3, 81]) torch.Size([1, 68, 68, 3, 4]) torch.Size([1, 68, 68, 3, 32])
            # 1212 torch.Size([1, 34, 34, 3, 81]) torch.Size([1, 34, 34, 3, 4]) torch.Size([1, 34, 34, 3, 32])
            # 1212 torch.Size([1, 17, 17, 3, 81]) torch.Size([1, 17, 17, 3, 4]) torch.Size([1, 17, 17, 3, 32])
            # 1212 torch.Size([1, 9, 9, 3, 81]) torch.Size([1, 9, 9, 3, 4]) torch.Size([1, 9, 9, 3, 32])
            # 1212 torch.Size([1, 5, 5, 3, 81]) torch.Size([1, 5, 5, 3, 4]) torch.Size([1, 5, 5, 3, 32])

            class_preds.append(class_pred)
            box_preds.append(box_pred)
            coef_preds.append(coef_pred)

        seg_pred = self.semantic_seg_conv(features[0])

        return class_preds, box_preds, coef_preds, proto_out, seg_pred


def _yolact(backbone_type, backbone_pretrained_path, **kwargs):
    model = YOLACT(backbone_type,
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)

    return model


def resnet18_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('resnet18backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet34_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('resnet34backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet50_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('resnet50backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet101_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('resnet101backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def resnet152_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('resnet152backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def vanb0_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('vanb0backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def vanb1_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('vanb1backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def vanb2_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('vanb2backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def vanb3_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('vanb3backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def convformers18_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('convformers18backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def convformers36_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('convformers36backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def convformerm36_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('convformerm36backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def convformerb36_yolact(backbone_pretrained_path='', **kwargs):
    return _yolact('convformerb36backbone',
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

    net = resnet18_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = resnet34_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = resnet50_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = resnet101_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = resnet152_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = resnet152_yolact(use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = vanb0_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = vanb1_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = vanb2_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = vanb3_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = convformers18_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = convformers36_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = convformerm36_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)

    net = convformerb36_yolact()
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
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    for per_out in preds[0]:
        print('3333', per_out.shape)
    for per_out in preds[1]:
        print('4444', per_out.shape)
    for per_out in preds[2]:
        print('5555', per_out.shape)
    print('6666', preds[3].shape)
    print('7777', preds[4].shape)
