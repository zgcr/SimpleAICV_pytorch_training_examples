import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from SimpleAICV.detection.models import backbones
from SimpleAICV.face_detection.models.fpn import RetinaFaceFPN, RetinaFaceSSH
from SimpleAICV.face_detection.models.head import RetinaFaceClassHead, RetinaFaceBoxHead

__all__ = [
    'resnet18_retinaface',
    'resnet34_retinaface',
    'resnet50_retinaface',
    'resnet101_retinaface',
    'resnet152_retinaface',
]


class RetinaFace(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 fpn_feature_num=3,
                 planes=256,
                 anchor_num=3,
                 use_gradient_checkpoint=False):
        super(RetinaFace, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })
        self.fpn = RetinaFaceFPN(inplanes=self.backbone.out_channels[1:4],
                                 planes=planes)
        self.ssh1 = RetinaFaceSSH(inplanes=planes, planes=planes)
        self.ssh2 = RetinaFaceSSH(inplanes=planes, planes=planes)
        self.ssh3 = RetinaFaceSSH(inplanes=planes, planes=planes)

        self.cls_head_list = nn.ModuleList()
        for _ in range(fpn_feature_num):
            self.cls_head_list.append(
                RetinaFaceClassHead(inplanes=planes, anchor_num=anchor_num))

        self.box_head_list = nn.ModuleList()
        for _ in range(fpn_feature_num):
            self.box_head_list.append(
                RetinaFaceBoxHead(inplanes=planes, anchor_num=anchor_num))

    def forward(self, inputs):
        features = self.backbone(inputs)
        features = features[1:4]

        del inputs

        if self.use_gradient_checkpoint:
            features = checkpoint(self.fpn, features, use_reentrant=False)
            feature1 = checkpoint(self.ssh1, features[0], use_reentrant=False)
            feature2 = checkpoint(self.ssh2, features[1], use_reentrant=False)
            feature3 = checkpoint(self.ssh3, features[2], use_reentrant=False)
        else:
            features = self.fpn(features)
            feature1 = self.ssh1(features[0])
            feature2 = self.ssh2(features[1])
            feature3 = self.ssh3(features[2])

        features = [feature1, feature2, feature3]

        cls_heads, box_heads = [], []
        for idx, feature in enumerate(features):
            cls_head = self.cls_head_list[idx](feature)
            # [N,anchor_num*2,H,W] -> [N,H,W,anchor_num*2] -> [N,H,W,anchor_num,2]
            cls_head = cls_head.permute(0, 2, 3, 1).contiguous()
            cls_head = cls_head.view(cls_head.shape[0], cls_head.shape[1],
                                     cls_head.shape[2], -1, 1)
            cls_heads.append(cls_head)

            box_head = self.box_head_list[idx](feature)
            # [N,anchor_num*4,H,W] -> [N,H,W,anchor_num*4] -> [N,H,W,anchor_num,4]
            box_head = box_head.permute(0, 2, 3, 1).contiguous()
            box_head = box_head.view(box_head.shape[0], box_head.shape[1],
                                     box_head.shape[2], -1, 4)
            box_heads.append(box_head)

        del features

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20]]
        # cls_heads shape:[[B, 80, 80, 3, 1],[B, 40, 40, 3, 1],[B, 20, 20, 3, 1]]
        # box_heads shape:[[B, 80, 80, 3, 4],[B, 40, 40, 3, 4],[B, 20, 20, 3, 4]]
        return cls_heads, box_heads


def _retinaface(backbone_type, backbone_pretrained_path, **kwargs):
    model = RetinaFace(backbone_type,
                       backbone_pretrained_path=backbone_pretrained_path,
                       **kwargs)

    return model


def resnet18_retinaface(backbone_pretrained_path='', **kwargs):
    return _retinaface('resnet18backbone',
                       backbone_pretrained_path=backbone_pretrained_path,
                       **kwargs)


def resnet34_retinaface(backbone_pretrained_path='', **kwargs):
    return _retinaface('resnet34backbone',
                       backbone_pretrained_path=backbone_pretrained_path,
                       **kwargs)


def resnet50_retinaface(backbone_pretrained_path='', **kwargs):
    return _retinaface('resnet50backbone',
                       backbone_pretrained_path=backbone_pretrained_path,
                       **kwargs)


def resnet101_retinaface(backbone_pretrained_path='', **kwargs):
    return _retinaface('resnet101backbone',
                       backbone_pretrained_path=backbone_pretrained_path,
                       **kwargs)


def resnet152_retinaface(backbone_pretrained_path='', **kwargs):
    return _retinaface('resnet152backbone',
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

    net = resnet18_retinaface()
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = resnet34_retinaface()
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = resnet50_retinaface()
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = resnet101_retinaface()
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = resnet152_retinaface()
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
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = resnet152_retinaface(use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    outs = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)
