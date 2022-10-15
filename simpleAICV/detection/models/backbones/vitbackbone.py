import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.detection.common import load_state_dict

__all__ = [
    'vit_tiny_backbone_patch16',
    'vit_small_backbone_patch16',
    'vit_base_backbone_patch16',
    'vit_large_backbone_patch16',
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

        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.norm = nn.LayerNorm(inplanes) if has_norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, inplanes, head_nums=8, dropout_prob=0.):
        super(MultiHeadAttention, self).__init__()
        self.head_nums = head_nums
        self.scale = (inplanes // head_nums)**-0.5

        self.qkv_linear = nn.Linear(inplanes, inplanes * 3)
        self.out_linear = nn.Linear(inplanes, inplanes)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape

        # [b,n,c] -> [b,n,3,head_num,c//head_num] -> [3,b,head_num,n,c//head_num]
        x = self.qkv_linear(x).view(b, n, 3, self.head_nums,
                                    c // self.head_nums).permute(
                                        2, 0, 3, 1, 4)
        # [3,b,head_num,n,c//head_num] -> 3个 [b,head_num,n,c//head_num]
        q, k, v = torch.unbind(x, dim=0)

        # [b,head_num,n,c//head_num] -> [b,head_num,n,n]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out_linear(x)
        x = self.dropout(x)

        return x


class FeedForward(nn.Module):

    def __init__(self, inplanes, feedforward_planes, dropout_prob=0.):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(inplanes, feedforward_planes)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(feedforward_planes, inplanes)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DropPathBlock(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    if drop_path_prob = 0. ,not use DropPath
    """

    def __init__(self, drop_path_prob=0., scale_by_keep=True):
        super(DropPathBlock, self).__init__()
        assert drop_path_prob >= 0.

        self.drop_path_prob = drop_path_prob
        self.keep_path_prob = 1 - drop_path_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_path_prob == 0. or not self.training:
            return x

        b = x.shape[0]
        device = x.device

        # work with diff dim tensors, not just 2D ConvNets
        shape = (b, ) + (1, ) * (len(x.shape) - 1)
        random_weight = torch.empty(shape).to(device).bernoulli_(
            self.keep_path_prob)

        if self.keep_path_prob > 0. and self.scale_by_keep:
            random_weight.div_(self.keep_path_prob)

        x = random_weight * x

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 feedforward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(inplanes)
        self.attention = MultiHeadAttention(inplanes,
                                            head_nums,
                                            dropout_prob=dropout_prob)
        self.norm2 = nn.LayerNorm(inplanes)
        self.feed_forward = FeedForward(inplanes,
                                        int(inplanes * feedforward_ratio),
                                        dropout_prob=dropout_prob)
        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.feed_forward(self.norm2(x)))

        return x


class ViT(nn.Module):

    def __init__(self,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 patch_size=16,
                 image_size=224,
                 dropout_prob=0.,
                 drop_path_prob=0.):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio

        self.patch_embedding = PatchEmbeddingBlock(3,
                                                   self.embedding_planes,
                                                   kernel_size=self.patch_size,
                                                   stride=self.patch_size,
                                                   padding=0,
                                                   groups=1,
                                                   has_norm=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_planes))
        self.position_encoding = nn.Parameter(
            torch.ones(1, (self.image_size // self.patch_size)**2 + 1,
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

        self.out_channels = self.embedding_planes

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.trunc_normal_(self.position_encoding, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x):
        out_h, out_w = x.shape[2] // self.patch_size, x.shape[
            3] // self.patch_size

        x = self.patch_embedding(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.position_encoding
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = x[:, 1:, :]

        x = torch.einsum('bcn->bnc', x)
        x = x.reshape(x.shape[0], x.shape[1], out_h, out_w)

        return x


def _vitbakbone(pretrained_path='', **kwargs):
    model = ViT(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path,
                        model,
                        loading_new_input_size_position_encoding_weight=True)
    else:
        print('no backbone pretrained model!')

    return model


def vit_tiny_backbone_patch16(image_size, pretrained_path='', **kwargs):
    return _vitbakbone(pretrained_path=pretrained_path,
                       image_size=image_size,
                       embedding_planes=192,
                       block_nums=12,
                       head_nums=3,
                       feedforward_ratio=4,
                       **kwargs)


def vit_small_backbone_patch16(image_size, pretrained_path='', **kwargs):
    return _vitbakbone(pretrained_path=pretrained_path,
                       image_size=image_size,
                       embedding_planes=384,
                       block_nums=12,
                       head_nums=6,
                       feedforward_ratio=4,
                       **kwargs)


def vit_base_backbone_patch16(image_size, pretrained_path='', **kwargs):
    return _vitbakbone(pretrained_path=pretrained_path,
                       image_size=image_size,
                       embedding_planes=768,
                       block_nums=12,
                       head_nums=12,
                       feedforward_ratio=4,
                       **kwargs)


def vit_large_backbone_patch16(image_size, pretrained_path='', **kwargs):
    return _vitbakbone(pretrained_path=pretrained_path,
                       image_size=image_size,
                       embedding_planes=1024,
                       block_nums=24,
                       head_nums=16,
                       feedforward_ratio=4,
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

    net = vit_tiny_backbone_patch16(image_size=512, pretrained_path='')
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_small_backbone_patch16(image_size=512, pretrained_path='')
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_base_backbone_patch16(image_size=512, pretrained_path='')
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_large_backbone_patch16(image_size=512, pretrained_path='')
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')