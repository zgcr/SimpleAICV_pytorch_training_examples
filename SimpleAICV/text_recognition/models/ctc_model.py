import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from SimpleAICV.text_recognition.models import backbones
from SimpleAICV.text_recognition.models.encoder import BiLSTMEncoder
from SimpleAICV.text_recognition.models.predictor import CTCPredictor

__all__ = [
    'CTCModel',
]


class CTCModel(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 num_classes=12114,
                 use_gradient_checkpoint=False):
        super(CTCModel, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        self.encoder = BiLSTMEncoder(inplanes=self.backbone.out_channels[-1],
                                     hidden_planes=planes)

        self.predictor = CTCPredictor(inplanes=planes,
                                      hidden_planes=planes,
                                      num_classes=num_classes)

    def forward(self, x):
        # [B,C,H,W]
        x = self.backbone(x)
        x = x[-1]

        # [B,C,W]
        x = torch.mean(x, dim=2)
        # [B,C,W]->[B,W,C]
        x = x.permute(0, 2, 1)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            # [B,W,C]
            x = self.encoder(x)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.predictor, x, use_reentrant=False)
        else:
            # [B,W,C]->[B,W,num_classes]
            x = self.predictor(x)

        return x


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

    net = CTCModel(backbone_type='resnet50backbone',
                   backbone_pretrained_path='',
                   planes=256,
                   num_classes=12114,
                   use_gradient_checkpoint=False)
    image_h, image_w = 32, 512
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
    outs = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', outs.shape)

    net = CTCModel(backbone_type='convformerm36backbone',
                   backbone_pretrained_path='',
                   planes=256,
                   num_classes=12114,
                   use_gradient_checkpoint=False)
    image_h, image_w = 32, 512
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
    outs = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', outs.shape)

    net = CTCModel(backbone_type='convformerm36backbone',
                   backbone_pretrained_path='',
                   planes=256,
                   num_classes=12114,
                   use_gradient_checkpoint=True)
    image_h, image_w = 32, 512
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print('2222', outs.shape)
