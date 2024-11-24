import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from simpleAICV.classification.backbones.vit import TransformerEncoderLayer
from simpleAICV.detection.common import load_state_dict

__all__ = [
    'vit_base_patch16_backbone',
    'vit_large_patch16_backbone',
    'vit_huge_patch14_backbone',
    'vit_small_patch14_backbone',
    'vit_base_patch14_backbone',
    'vit_large_patch14_backbone',
    'vit_giant_patch14_backbone',
    'sapiens_0_3b_backbone',
    'sapiens_0_6b_backbone',
    'sapiens_1_0b_backbone',
    'sapiens_2_0b_backbone',
]


class PatchEmbeddingBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_norm=False):
        super(PatchEmbeddingBlock, self).__init__()
        bias = False if has_norm else True

        self.proj = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.norm = nn.LayerNorm(inplanes,
                                 eps=1e-6) if has_norm else nn.Identity()

    def forward(self, x):
        x = self.proj(x)

        [b, c, h, w] = x.shape

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)

        return x, [b, c, h, w]


class ViTBackbone(nn.Module):

    def __init__(self,
                 patch_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 image_size=224,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 use_gradient_checkpoint=False):
        super(ViTBackbone, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.patch_embed = PatchEmbeddingBlock(3,
                                               self.embedding_planes,
                                               kernel_size=self.patch_size,
                                               stride=self.patch_size,
                                               padding=0,
                                               groups=1,
                                               has_norm=False)

        self.pos_embed = nn.Parameter(
            torch.ones(1, (self.image_size // self.patch_size)**2,
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
        self.blocks = nn.ModuleList(blocks)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x, [b, c, h, w] = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.embedding_dropout(x)

        for block in self.blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return x


def _vitbackbone(patch_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 pretrained_path='',
                 **kwargs):
    model = ViTBackbone(patch_size, embedding_planes, block_nums, head_nums,
                        feedforward_ratio, **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def vit_base_patch16_backbone(**kwargs):
    return _vitbackbone(16, 768, 12, 12, 4, **kwargs)


def vit_large_patch16_backbone(**kwargs):
    return _vitbackbone(16, 1024, 24, 16, 4, **kwargs)


def vit_huge_patch14_backbone(**kwargs):
    return _vitbackbone(14, 1280, 32, 16, 4, **kwargs)


def vit_small_patch14_backbone(**kwargs):
    return _vitbackbone(14, 384, 12, 6, 4, **kwargs)


def vit_base_patch14_backbone(**kwargs):
    return _vitbackbone(14, 768, 12, 12, 4, **kwargs)


def vit_large_patch14_backbone(**kwargs):
    return _vitbackbone(14, 1024, 24, 16, 4, **kwargs)


def vit_giant_patch14_backbone(**kwargs):
    return _vitbackbone(14, 1536, 40, 24, 4, **kwargs)


# 1024x1024 pretrain 1024x768 seg
def sapiens_0_3b_backbone(**kwargs):
    return _vitbackbone(16, 1024, 24, 16, 4, **kwargs)


# 1024x1024 pretrain 1024x768 seg
def sapiens_0_6b_backbone(**kwargs):
    return _vitbackbone(16, 1280, 32, 16, 4, **kwargs)


# 1024x1024 pretrain 1024x768 seg
def sapiens_1_0b_backbone(**kwargs):
    return _vitbackbone(16, 1536, 40, 24, 4, **kwargs)


# 1024x1024 pretrain 1024x768 seg
def sapiens_2_0b_backbone(**kwargs):
    return _vitbackbone(16, 1920, 48, 32, 4, **kwargs)


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

    net = vit_base_patch16_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_large_patch16_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_huge_patch14_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = sapiens_0_3b_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = sapiens_0_6b_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = sapiens_1_0b_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = sapiens_2_0b_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_small_patch14_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_base_patch14_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_large_patch14_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_giant_patch14_backbone(image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')
