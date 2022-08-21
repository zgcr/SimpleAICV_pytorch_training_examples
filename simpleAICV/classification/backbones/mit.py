import math

import torch
import torch.nn as nn


class DWConv(nn.Module):

    def __init__(self, inplanes):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(inplanes,
                                inplanes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=inplanes,
                                bias=True)

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        x = x.view(b, c, -1).transpose(1, 2)

        return x


class MlpBlock(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, dropout_prob=0.):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(inplanes, hidden_planes)
        self.dwconv = DWConv(hidden_planes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_planes, planes)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):

    def __init__(self, inplanes, num_heads=8, sr_ratio=1, dropout_prob=0.):
        super(Attention, self).__init__()
        assert inplanes % num_heads == 0

        self.num_heads = num_heads
        head_planes = inplanes // num_heads
        self.scale = head_planes**(-0.5)

        self.q = nn.Linear(inplanes, inplanes, bias=True)
        self.kv = nn.Linear(inplanes, inplanes * 2, bias=True)
        self.proj = nn.Linear(inplanes, inplanes)

        self.drop = nn.Dropout(dropout_prob)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(inplanes,
                                inplanes,
                                kernel_size=sr_ratio,
                                stride=sr_ratio)
            self.norm = nn.LayerNorm(inplanes)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads,
                              c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads,
                                     c // self.num_heads).permute(
                                         2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads,
                                    c // self.num_heads).permute(
                                        2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.drop(x)

        return x


class MitBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 num_heads,
                 mlp_ratio=4.,
                 sr_ratio=1,
                 dropout_prob=0.):
        super(MitBlock, self).__init__()
        self.norm1 = nn.LayerNorm(inplanes)
        self.attn = Attention(inplanes,
                              num_heads=num_heads,
                              sr_ratio=sr_ratio,
                              dropout_prob=dropout_prob)
        self.norm2 = nn.LayerNorm(inplanes)
        self.mlp = MlpBlock(inplanes=inplanes,
                            hidden_planes=int(inplanes * mlp_ratio),
                            planes=inplanes,
                            dropout_prob=dropout_prob)

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)

        return x


class PatchEmbeddingBlock(nn.Module):

    def __init__(self, inplanes, embedding_planes, patch_size=7, stride=4):
        super(PatchEmbeddingBlock, self).__init__()
        self.proj = nn.Conv2d(inplanes,
                              embedding_planes,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embedding_planes)

    def forward(self, x):
        x = self.proj(x)

        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.norm(x)

        return x, h, w


class MiT(nn.Module):

    def __init__(self,
                 inplanes=3,
                 embedding_planes=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 dropout_prob=0.,
                 num_classes=1000):
        super(MiT, self).__init__()

        self.patch_embed1 = PatchEmbeddingBlock(
            inplanes=inplanes,
            embedding_planes=embedding_planes[0],
            patch_size=7,
            stride=4)
        self.patch_embed2 = PatchEmbeddingBlock(
            inplanes=embedding_planes[0],
            embedding_planes=embedding_planes[1],
            patch_size=3,
            stride=2)
        self.patch_embed3 = PatchEmbeddingBlock(
            inplanes=embedding_planes[1],
            embedding_planes=embedding_planes[2],
            patch_size=3,
            stride=2)
        self.patch_embed4 = PatchEmbeddingBlock(
            inplanes=embedding_planes[2],
            embedding_planes=embedding_planes[3],
            patch_size=3,
            stride=2)

        self.block1 = nn.ModuleList([
            MitBlock(inplanes=embedding_planes[0],
                     num_heads=num_heads[0],
                     mlp_ratio=mlp_ratios[0],
                     sr_ratio=sr_ratios[0],
                     dropout_prob=dropout_prob) for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embedding_planes[0])

        self.block2 = nn.ModuleList([
            MitBlock(inplanes=embedding_planes[1],
                     num_heads=num_heads[1],
                     mlp_ratio=mlp_ratios[1],
                     sr_ratio=sr_ratios[1],
                     dropout_prob=dropout_prob) for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embedding_planes[1])

        self.block3 = nn.ModuleList([
            MitBlock(inplanes=embedding_planes[2],
                     num_heads=num_heads[2],
                     mlp_ratio=mlp_ratios[2],
                     sr_ratio=sr_ratios[2],
                     dropout_prob=dropout_prob) for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embedding_planes[2])

        self.block4 = nn.ModuleList([
            MitBlock(inplanes=embedding_planes[3],
                     num_heads=num_heads[3],
                     mlp_ratio=mlp_ratios[3],
                     sr_ratio=sr_ratios[3],
                     dropout_prob=dropout_prob) for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embedding_planes[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embedding_planes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, _, _, _ = x.shape

        # stage 1
        x, h, w = self.patch_embed1(x)
        for block in self.block1:
            x = block(x, h, w)
        x = self.norm1(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, h, w = self.patch_embed2(x)
        for block in self.block2:
            x = block(x, h, w)
        x = self.norm2(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, h, w = self.patch_embed3(x)
        for block in self.block3:
            x = block(x, h, w)
        x = self.norm3(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, h, w = self.patch_embed4(x)
        for block in self.block4:
            x = block(x, h, w)
        x = self.norm4(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _mit(**kwargs):
    model = MiT(**kwargs)

    return model


def mit_b0(**kwargs):
    return _mit(embedding_planes=[32, 64, 160, 256],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                depths=[2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                **kwargs)


def mit_b1(**kwargs):
    return _mit(embedding_planes=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                depths=[2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                **kwargs)


def mit_b2(**kwargs):
    return _mit(embedding_planes=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                depths=[3, 4, 6, 3],
                sr_ratios=[8, 4, 2, 1],
                **kwargs)


def mit_b3(**kwargs):
    return _mit(embedding_planes=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                depths=[3, 4, 18, 3],
                sr_ratios=[8, 4, 2, 1],
                **kwargs)


def mit_b4(**kwargs):
    return _mit(embedding_planes=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                depths=[3, 8, 27, 3],
                sr_ratios=[8, 4, 2, 1],
                **kwargs)


def mit_b5(**kwargs):
    return _mit(embedding_planes=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                depths=[3, 6, 40, 3],
                sr_ratios=[8, 4, 2, 1],
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

    net = mit_b0(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = mit_b1(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = mit_b2(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = mit_b3(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = mit_b4(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = mit_b5(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'6666, macs: {macs}, params: {params},out_shape: {out.shape}')