'''
PoolFormer: MetaFormer is Actually What You Need
https://arxiv.org/pdf/2111.11418.pdf
'''
import torch
import torch.nn as nn

__all__ = [
    'poolformer_s12',
    'poolformer_s24',
    'poolformer_s36',
    'poolformer_m36',
    'poolformer_m48',
    'convformer_s8',
    'convformer_s16',
    'convformer_m8',
    'convformer_m16',
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

        return x


class ChannelLayerNormBlock(nn.Module):
    '''
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    '''
    def __init__(self, inplanes):
        super(ChannelLayerNormBlock, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, inplanes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, inplanes, 1, 1))

    def forward(self, x):
        u = torch.mean(x, 1, keepdim=True)
        s = torch.pow(x - u, 2).mean(1, keepdim=True)

        x = (x - u) / torch.sqrt(s + 1e-4)
        x = self.weight * x + self.bias

        return x


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=1,
                              padding=kernel_size // 2,
                              groups=1,
                              bias=True)

    def forward(self, x):
        x = self.conv(x) - x

        return x


class PoolingBlock(nn.Module):
    def __init__(self, pool_size=3):
        super(PoolingBlock, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size,
                                 stride=1,
                                 padding=pool_size // 2,
                                 count_include_pad=False)

    def forward(self, x):
        x = self.pool(x) - x

        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, inplanes, hidden_planes, planes):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Conv2d(inplanes,
                             hidden_planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)
        nn.Conv2d(inplanes, hidden_planes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_planes,
                             planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class PoolFormerBlock(nn.Module):
    def __init__(self, inplanes, pool_size=3, feed_forward_ratio=4):
        super(PoolFormerBlock, self).__init__()
        self.token_mixer = PoolingBlock(pool_size=pool_size)
        self.norm1 = ChannelLayerNormBlock(inplanes)
        self.norm2 = ChannelLayerNormBlock(inplanes)
        self.feed_forward = FeedForwardBlock(
            inplanes, int(inplanes * feed_forward_ratio), inplanes)
        self.layer_scale_1 = nn.Parameter(torch.ones(1, inplanes, 1, 1))
        self.layer_scale_2 = nn.Parameter(torch.ones(1, inplanes, 1, 1))

    def forward(self, x):
        x = x + self.layer_scale_1 * self.token_mixer(self.norm1(x))
        x = x + self.layer_scale_2 * self.feed_forward(self.norm2(x))

        return x


class ConvFormerBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, feed_forward_ratio=4):
        super(ConvFormerBlock, self).__init__()
        self.token_mixer = ConvBlock(inplanes, planes, kernel_size=kernel_size)
        self.norm1 = ChannelLayerNormBlock(inplanes)
        self.norm2 = ChannelLayerNormBlock(inplanes)
        self.feed_forward = FeedForwardBlock(
            inplanes, int(inplanes * feed_forward_ratio), inplanes)
        self.layer_scale_1 = nn.Parameter(torch.ones(1, inplanes, 1, 1))
        self.layer_scale_2 = nn.Parameter(torch.ones(1, inplanes, 1, 1))

    def forward(self, x):
        x = x + self.layer_scale_1 * self.token_mixer(self.norm1(x))
        x = x + self.layer_scale_2 * self.feed_forward(self.norm2(x))

        return x


class PoolFormer(nn.Module):
    def __init__(self,
                 layer_nums,
                 planes,
                 pool_size=3,
                 feed_forward_ratio=4,
                 num_classes=1000):
        super(PoolFormer, self).__init__()
        self.num_classes = num_classes

        self.patch_embedding1 = PatchEmbeddingBlock(3,
                                                    planes[0],
                                                    kernel_size=7,
                                                    stride=4,
                                                    padding=2,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage2 = self.make_layer(planes[0],
                                      layer_nums[0],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio)
        self.patch_embedding2 = PatchEmbeddingBlock(planes[0],
                                                    planes[1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage3 = self.make_layer(planes[1],
                                      layer_nums[1],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio)
        self.patch_embedding3 = PatchEmbeddingBlock(planes[1],
                                                    planes[2],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage4 = self.make_layer(planes[2],
                                      layer_nums[2],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio)
        self.patch_embedding4 = PatchEmbeddingBlock(planes[2],
                                                    planes[3],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage5 = self.make_layer(planes[3],
                                      layer_nums[3],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3], self.num_classes)

    def make_layer(self,
                   inplanes,
                   layer_nums,
                   pool_size=3,
                   feed_forward_ratio=4):
        blocks = []
        for _ in range(layer_nums):
            blocks.append(
                PoolFormerBlock(inplanes,
                                pool_size=pool_size,
                                feed_forward_ratio=feed_forward_ratio))
        blocks = nn.Sequential(*blocks)

        return blocks

    def forward(self, x):
        x = self.patch_embedding1(x)
        x = self.stage2(x)
        x = self.patch_embedding2(x)
        x = self.stage3(x)
        x = self.patch_embedding3(x)
        x = self.stage4(x)
        x = self.patch_embedding4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ConvFormer(nn.Module):
    def __init__(self,
                 layer_nums,
                 planes,
                 feed_forward_ratio=1,
                 num_classes=1000):
        super(ConvFormer, self).__init__()
        self.num_classes = num_classes

        self.patch_embedding1 = PatchEmbeddingBlock(3,
                                                    planes[0],
                                                    kernel_size=7,
                                                    stride=2,
                                                    padding=3,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.patch_embedding2 = PatchEmbeddingBlock(planes[0],
                                                    planes[0],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage2 = self.make_layer(planes[0],
                                      planes[0],
                                      layer_nums[0],
                                      feed_forward_ratio=feed_forward_ratio)
        self.patch_embedding3 = PatchEmbeddingBlock(planes[0],
                                                    planes[1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage3 = self.make_layer(planes[1],
                                      planes[1],
                                      layer_nums[1],
                                      feed_forward_ratio=feed_forward_ratio)
        self.patch_embedding4 = PatchEmbeddingBlock(planes[1],
                                                    planes[2],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage4 = self.make_layer(planes[2],
                                      planes[2],
                                      layer_nums[2],
                                      feed_forward_ratio=feed_forward_ratio)
        self.patch_embedding5 = PatchEmbeddingBlock(planes[2],
                                                    planes[3],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.stage5 = self.make_layer(planes[3],
                                      planes[3],
                                      layer_nums[3],
                                      feed_forward_ratio=feed_forward_ratio)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3], self.num_classes)

    def make_layer(self, inplanes, planes, layer_nums, feed_forward_ratio=4):
        blocks = []
        for _ in range(layer_nums):
            blocks.append(
                ConvFormerBlock(inplanes,
                                planes,
                                kernel_size=3,
                                feed_forward_ratio=feed_forward_ratio))
            inplanes = planes
        blocks = nn.Sequential(*blocks)

        return blocks

    def forward(self, x):
        x = self.patch_embedding1(x)
        x = self.patch_embedding2(x)
        x = self.stage2(x)
        x = self.patch_embedding3(x)
        x = self.stage3(x)
        x = self.patch_embedding4(x)
        x = self.stage4(x)
        x = self.patch_embedding5(x)
        x = self.stage5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _poolformer(layer_nums, planes, **kwargs):
    model = PoolFormer(layer_nums, planes, **kwargs)

    return model


def poolformer_s12(**kwargs):
    return _poolformer([2, 2, 6, 2], [64, 128, 320, 512], **kwargs)


def poolformer_s24(**kwargs):
    return _poolformer([4, 4, 12, 4], [64, 128, 320, 512], **kwargs)


def poolformer_s36(**kwargs):
    return _poolformer([6, 6, 18, 6], [64, 128, 320, 512], **kwargs)


def poolformer_m36(**kwargs):
    return _poolformer([6, 6, 18, 6], [96, 192, 384, 768], **kwargs)


def poolformer_m48(**kwargs):
    return _poolformer([8, 8, 24, 8], [96, 192, 384, 768], **kwargs)


def _convformer(layer_nums, planes, **kwargs):
    model = ConvFormer(layer_nums, planes, **kwargs)

    return model


def convformer_s8(**kwargs):
    return _convformer([2, 2, 2, 2], [64, 128, 256, 512],
                       feed_forward_ratio=1,
                       **kwargs)


def convformer_s16(**kwargs):
    return _convformer([3, 4, 6, 3], [64, 128, 256, 512],
                       feed_forward_ratio=1,
                       **kwargs)


def convformer_m8(**kwargs):
    return _convformer([2, 2, 2, 2], [128, 256, 512, 1024],
                       feed_forward_ratio=1,
                       **kwargs)


def convformer_m16(**kwargs):
    return _convformer([3, 4, 6, 3], [128, 256, 512, 1024],
                       feed_forward_ratio=1,
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

    net = poolformer_s12(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_s24(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_s36(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_m36(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_m48(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_s8(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'6666, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_s16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'7777, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_m8(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'8888, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_m16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'9999, macs: {macs}, params: {params},out_shape: {out.shape}')