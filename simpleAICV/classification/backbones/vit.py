'''
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
https://arxiv.org/pdf/2010.11929.pdf
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import torch
import torch.nn as nn

__all__ = [
    'vit_tiny_patch16_224',
    'vit_small_patch16_224',
    'vit_base_patch16_224',
    'vit_large_patch16_224',
]


class PatchEmbeddingBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(PatchEmbeddingBlock, self).__init__()
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

        # [B,C,H,W] -> [B,N,C]
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, inplanes, head_nums=8):
        super(MultiHeadAttention, self).__init__()
        self.head_nums = head_nums
        self.scale = (inplanes // head_nums)**-0.5

        self.head_linear = nn.Linear(inplanes, inplanes * 3)
        self.proj_linear = nn.Linear(inplanes, inplanes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape

        # [b,n,c] -> [b,n,3,head_num,c//head_num] -> [3,b,head_num,n,c//head_num]
        x = self.head_linear(x).view(b, n, 3, self.head_nums,
                                     c // self.head_nums).permute(
                                         2, 0, 3, 1, 4)
        # [3,b,head_num,n,c//head_num] -> 3个 [b,head_num,n,c//head_num]
        q, k, v = torch.unbind(x, dim=0)

        # [b,head_num,n,c//head_num] -> [b,head_num,n,n]
        x = self.softmax(q @ k.permute(0, 1, 3, 2) / self.scale)
        # [b,head_num,n,n] -> [b,head_num,n,c//head_num] -> [b,n,head_num,c//head_num] -> [b,n,c]
        x = (x @ v).permute(0, 2, 1, 3).reshape(b, n, c)

        x = self.proj_linear(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, inplanes, feedforward_planes):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(inplanes, feedforward_planes)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(feedforward_planes, inplanes)

    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))

        return x


class LayerNorm(nn.Module):
    def __init__(self, inplanes, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(inplanes))
        self.b_2 = nn.Parameter(torch.zeros(inplanes))
        self.eps = eps

    def forward(self, x):
        device = x.device
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2.to(device) * (x - mean) / (std +
                                                self.eps) + self.b_2.to(device)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, inplanes, head_nums, feedforward_ratio=4):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(inplanes, head_nums)
        self.attention_layer_norm = LayerNorm(inplanes)
        self.feed_forward = FeedForward(inplanes,
                                        int(inplanes * feedforward_ratio))
        self.feed_forward_layer_norm = LayerNorm(inplanes)

    def forward(self, x):
        x = x + self.attention(self.attention_layer_norm(x))
        x = x + self.feed_forward(self.feed_forward_layer_norm(x))

        return x


class ViT(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio=4,
                 num_classes=1000):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbeddingBlock(3,
                                                   self.embedding_planes,
                                                   kernel_size=self.patch_size,
                                                   stride=self.patch_size,
                                                   padding=0,
                                                   groups=1,
                                                   has_bn=False,
                                                   has_act=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_planes))
        self.position_encoding = nn.Parameter(
            torch.ones(1, (self.image_size // self.patch_size)**2 + 1,
                       self.embedding_planes))

        blocks = []
        for _ in range(self.block_nums):
            blocks.append(
                TransformerEncoderLayer(
                    self.embedding_planes,
                    self.head_nums,
                    feedforward_ratio=self.feedforward_ratio))
        self.blocks = nn.Sequential(*blocks)

        self.norm = LayerNorm(self.embedding_planes)

        self.fc = nn.Linear(self.embedding_planes, self.num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)

        x = torch.cat((self.cls_token.repeat(x.shape[0], 1, 1), x), dim=1)
        x = x + self.position_encoding

        x = self.blocks(x)

        x = x[:, 0]
        x = self.norm(x)
        x = self.fc(x)

        return x


def _vit(image_size, patch_size, embedding_planes, block_nums, head_nums,
         **kwargs):
    model = ViT(image_size, patch_size, embedding_planes, block_nums,
                head_nums, **kwargs)

    return model


def vit_tiny_patch16_224(**kwargs):
    return _vit(224, 16, 192, 12, 3, **kwargs)


def vit_small_patch16_224(**kwargs):
    return _vit(224, 16, 384, 12, 6, **kwargs)


def vit_base_patch16_224(**kwargs):
    return _vit(224, 16, 768, 12, 12, **kwargs)


def vit_large_patch16_224(**kwargs):
    return _vit(224, 16, 1024, 24, 16, **kwargs)


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

    net = vit_tiny_patch16_224(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_small_patch16_224(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_base_patch16_224(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = vit_large_patch16_224(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')
