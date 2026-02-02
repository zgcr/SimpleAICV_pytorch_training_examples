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

from SimpleAICV.classification.backbones.vit import TransformerEncoderLayer
from SimpleAICV.detection.common import load_state_dict

__all__ = [
    'vit_base_patch16_backbone',
    'vit_large_patch16_backbone',
    'vit_huge_patch14_backbone',
]


class VitPyramidNeck(nn.Module):

    def __init__(self, inplanes, planes):
        super(VitPyramidNeck, self).__init__()
        self.P2 = nn.Sequential(
            nn.ConvTranspose2d(inplanes,
                               planes,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               output_padding=0,
                               bias=True),
            nn.GELU(),
            nn.ConvTranspose2d(planes,
                               planes,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               output_padding=0,
                               bias=True),
            nn.GELU(),
        )

        self.P3 = nn.Sequential(
            nn.ConvTranspose2d(inplanes,
                               planes,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               output_padding=0,
                               bias=True),
            nn.GELU(),
        )
        self.P4 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.GELU(),
        )
        self.P5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
        )

    def forward(self, x):
        P2 = self.P2(x)
        P3 = self.P3(x)
        P4 = self.P4(x)
        P5 = self.P5(P4)

        return [P2, P3, P4, P5]


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

        self.out_channels = embedding_planes

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
    return _vitbackbone(16, 512, 24, 16, 4, **kwargs)


def vit_huge_patch14_backbone(**kwargs):
    return _vitbackbone(14, 1280, 32, 16, 4, **kwargs)


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
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )

    net = vit_base_patch16_backbone(image_size=512,
                                    use_gradient_checkpoint=True)
    image_h, image_w = 512, 512
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print('1111', out.shape)

    net = vit_large_patch16_backbone(image_size=512)
    image_h, image_w = 512, 512
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
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )

    net = vit_huge_patch14_backbone(image_size=512)
    image_h, image_w = 512, 512
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
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )
