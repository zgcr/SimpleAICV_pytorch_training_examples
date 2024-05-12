import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.text_recognition.models import backbones
from simpleAICV.text_recognition.models import encoder
from simpleAICV.text_recognition.models import predictor

__all__ = [
    'CTCModel',
]


class CTCModel(nn.Module):

    def __init__(self, model_config):
        super(CTCModel, self).__init__()
        self.backbone = backbones.__dict__[model_config['backbone']['name']](
            **model_config['backbone']['param'])

        model_config['encoder']['param'][
            'inplanes'] = self.backbone.out_channels

        self.encoder = encoder.__dict__[model_config['encoder']['name']](
            **model_config['encoder']['param'])

        model_config['predictor']['param'][
            'inplanes'] = self.encoder.out_channels

        self.predictor = predictor.__dict__[model_config['predictor']['name']](
            **model_config['predictor']['param'])

    def forward(self, x):
        # [B,C,H,W]
        x = self.backbone(x)
        # [B,C,W]
        x = torch.mean(x, dim=2)
        # [B,C,W]->[B,W,C]
        x = x.permute(0, 2, 1)
        # [B,W,C]
        x = self.encoder(x)
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
            'name': 'RepVGGEnhanceNetBackbone',
            'param': {
                'inplanes': 1,
                'planes': [32, 64, 128, 256],
                'k': 4,
                'deploy': True,
                'pretrained_path': '',
            }
        },
        'encoder': {
            'name': 'BiLSTMEncoder',
            'param': {},
        },
        'predictor': {
            'name': 'CTCEnhancePredictor',
            'param': {
                'hidden_planes': 192,
                'num_classes': 12114,
            }
        },
    }

    net = CTCModel(model_config)
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 1, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 1, image_h, image_w)))
    print('2222', outs.shape)

    model_config = {
        'backbone': {
            'name': 'resnet50backbone',
            'param': {
                'inplanes': 1,
                'pretrained_path': '',
            }
        },
        'encoder': {
            'name': 'BiLSTMEncoder',
            'param': {},
        },
        'predictor': {
            'name': 'CTCEnhancePredictor',
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
                           inputs=(torch.randn(1, 1, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 1, image_h, image_w)))
    print('2222', outs.shape)

    model_config = {
        'backbone': {
            'name': 'van_b1_backbone',
            'param': {
                'inplanes': 1,
            }
        },
        'encoder': {
            'name': 'TransformerEncoder',
            'param': {
                'encoder_layer_nums': 4,
                'head_nums': 4,
                'feedforward_ratio': 4,
                'encoding_width': 200,
            },
        },
        'predictor': {
            'name': 'CTCEnhancePredictor',
            'param': {
                'hidden_planes': 256,
                'num_classes': 12114,
            }
        },
    }

    net = CTCModel(model_config)
    image_h, image_w = 32, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 1, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(3, 1, image_h, image_w)))
    print('2222', outs.shape)
