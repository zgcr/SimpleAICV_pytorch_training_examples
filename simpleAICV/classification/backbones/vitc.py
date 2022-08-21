'''
Early Convolutions Help Transformers See Better
https://arxiv.org/pdf/2106.14881.pdf
https://github.com/Jack-Etheredge/early_convolutions_vit_pytorch/blob/main/vitc/early_convolutions.py
'''
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.classification.backbones.vit import TransformerEncoderLayer

__all__ = [
    'vitc_1GF',
    'vitc_4GF',
    'vitc_18GF',
    'vitc_36GF',
]


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
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class ViTC(nn.Module):

    def __init__(self,
                 planes_list,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 image_size=224,
                 dropout_prob=0.1,
                 drop_path_prob=0.1,
                 global_pool=True,
                 num_classes=1000):
        super(ViTC, self).__init__()
        self.image_size = image_size
        self.planes_list = planes_list
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.global_pool = global_pool
        self.num_classes = num_classes

        convs = []
        self.inplanes = 3
        for i in range(len(self.planes_list)):
            stride = 2 if self.inplanes != self.planes_list[i] else 1
            convs.append(
                ConvBnActBlock(self.inplanes,
                               self.planes_list[i],
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=1,
                               has_bn=False,
                               has_act=False))
            self.inplanes = self.planes_list[i]
        convs.append(
            ConvBnActBlock(self.planes_list[-1],
                           self.embedding_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=False,
                           has_act=False))
        self.convs = nn.Sequential(*convs)

        # downsample 16x
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_planes))
        self.position_encoding = nn.Parameter(
            torch.ones(1, (self.image_size // 16)**2 + 1,
                       self.embedding_planes))
        self.embedding_dropout = nn.Dropout(dropout_prob)

        drop_path_prob_list = []
        for block_idx in range(self.block_nums):
            if drop_path_prob == 0.:
                drop_path_prob_list.append(0.)
            else:
                per_layer_drop_path_prob = drop_path_prob * (
                    block_idx / (self.block_nums - 1))
                drop_path_prob_list.append(per_layer_drop_path_prob)

        blocks = []
        for i in range(self.block_nums):
            blocks.append(
                TransformerEncoderLayer(
                    self.embedding_planes,
                    self.head_nums,
                    feedforward_ratio=self.feedforward_ratio,
                    dropout_prob=dropout_prob,
                    drop_path_prob=drop_path_prob_list[i]))
        self.blocks = nn.Sequential(*blocks)

        self.norm = nn.LayerNorm(self.embedding_planes, eps=1e-6)
        self.fc = nn.Linear(self.embedding_planes, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=.02)

        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.position_encoding
        x = self.embedding_dropout(x)

        x = self.blocks(x)

        if self.global_pool:
            # global pool without cls token
            x = x[:, 1:, :].mean(dim=1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]

        x = self.fc(x)

        return x


def _vitc(planes_list, embedding_planes, block_nums, head_nums,
          feedforward_ratio, **kwargs):
    model = ViTC(planes_list, embedding_planes, block_nums, head_nums,
                 feedforward_ratio, **kwargs)

    return model


def vitc_1GF(**kwargs):
    return _vitc([24, 48, 96, 192], 192, 11, 3, 3, **kwargs)


def vitc_4GF(**kwargs):
    return _vitc([48, 96, 192, 384], 384, 11, 6, 3, **kwargs)


def vitc_18GF(**kwargs):
    return _vitc([64, 128, 128, 256, 256, 512], 768, 11, 12, 4, **kwargs)


def vitc_36GF(**kwargs):
    return _vitc([64, 128, 128, 256, 256, 512], 1024, 13, 16, 4, **kwargs)


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

    net = vitc_1GF(image_size=224, num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vitc_4GF(image_size=224, num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vitc_18GF(image_size=224, num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vitc_36GF(image_size=224, num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')