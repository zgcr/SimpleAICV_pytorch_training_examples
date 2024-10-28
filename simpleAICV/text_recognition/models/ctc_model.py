import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from simpleAICV.text_recognition.models import backbones
from simpleAICV.text_recognition.models import encoder
from simpleAICV.text_recognition.models import predictor

__all__ = [
    'CTCModel',
]


class CTCModel(nn.Module):

    def __init__(self, model_config, use_gradient_checkpoint=False):
        super(CTCModel, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        model_config['backbone']['param'][
            'use_gradient_checkpoint'] = use_gradient_checkpoint

        self.backbone = backbones.__dict__[model_config['backbone']['name']](
            **model_config['backbone']['param'])

        model_config['encoder']['param'][
            'inplanes'] = self.backbone.out_channels[-1]

        self.encoder = encoder.__dict__[model_config['encoder']['name']](
            **model_config['encoder']['param'])

        model_config['predictor']['param'][
            'inplanes'] = self.encoder.out_channels

        self.predictor = predictor.__dict__[model_config['predictor']['name']](
            **model_config['predictor']['param'])

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

    model_config = {
        'backbone': {
            'name': 'resnet50backbone',
            'param': {
                'pretrained_path': '',
            }
        },
        'encoder': {
            'name': 'BiLSTMEncoder',
            'param': {},
        },
        'predictor': {
            'name': 'CTCPredictor',
            'param': {
                'hidden_planes': 512,
                'num_classes': 12114,
            }
        },
    }

    net = CTCModel(model_config)
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', outs.shape)

    model_config = {
        'backbone': {
            'name': 'convformerm36backbone',
            'param': {
                'pretrained_path': '',
            }
        },
        'encoder': {
            'name': 'BiLSTMEncoder',
            'param': {},
        },
        'predictor': {
            'name': 'CTCPredictor',
            'param': {
                'hidden_planes': 512,
                'num_classes': 12114,
            }
        },
    }

    net = CTCModel(model_config)
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print('2222', outs.shape)

    net = CTCModel(model_config, use_gradient_checkpoint=True)
    image_h, image_w = 32, 512
    outs = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print('2222', outs.shape)
