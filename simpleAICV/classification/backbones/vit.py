'''
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
https://arxiv.org/pdf/2010.11929.pdf
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import torch
import torch.nn as nn

__all__ = [
    'vit_tiny_patch16',
    'vit_small_patch16',
    'vit_base_patch16',
    'vit_large_patch16',
    'vit_huge_patch14',
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
                 patch_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 image_size=224,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 global_pool=False,
                 num_classes=1000):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.global_pool = global_pool
        self.num_classes = num_classes

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

        self.norm = nn.LayerNorm(self.embedding_planes)
        self.fc = nn.Linear(self.embedding_planes, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.trunc_normal_(self.position_encoding, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.trunc_normal_(self.fc.weight, std=2e-5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.patch_embedding(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.position_encoding
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            # global pool without cls token
            x = x[:, 1:, :].mean(dim=1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]

        x = self.fc(x)

        return x


def _vit(patch_size, embedding_planes, block_nums, head_nums,
         feedforward_ratio, **kwargs):
    model = ViT(patch_size, embedding_planes, block_nums, head_nums,
                feedforward_ratio, **kwargs)

    return model


def vit_tiny_patch16(**kwargs):
    return _vit(16, 192, 12, 3, 4, **kwargs)


def vit_small_patch16(**kwargs):
    return _vit(16, 384, 12, 6, 4, **kwargs)


def vit_base_patch16(**kwargs):
    return _vit(16, 768, 12, 12, 4, **kwargs)


def vit_large_patch16(**kwargs):
    return _vit(16, 1024, 24, 16, 4, **kwargs)


def vit_huge_patch14(**kwargs):
    return _vit(14, 1280, 32, 16, 4, **kwargs)


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

    net = vit_tiny_patch16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_small_patch16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_base_patch16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_large_patch16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_huge_patch14(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params},out_shape: {out.shape}')