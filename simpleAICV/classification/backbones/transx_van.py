import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'transxvan_b0',
    'transxvan_b1',
    'transxvan_b2',
]


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 dilation=1,
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
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class LKA(nn.Module):

    def __init__(self, inplanes):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(inplanes,
                               inplanes,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               groups=inplanes,
                               bias=True)
        self.conv_spatial = nn.Conv2d(inplanes,
                                      inplanes,
                                      kernel_size=7,
                                      stride=1,
                                      padding=9,
                                      dilation=3,
                                      groups=inplanes,
                                      bias=True)
        self.conv1 = nn.Conv2d(inplanes,
                               inplanes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):

    def __init__(self, inplanes, head_nums=1, sr_ratio=1):
        super(Attention, self).__init__()
        assert inplanes % head_nums == 0
        self.head_nums = head_nums

        head_inplanes = inplanes // head_nums
        self.scale = head_inplanes**-0.5

        self.q = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.kv = nn.Conv2d(inplanes, inplanes * 2, kernel_size=1)

        self.sr = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes,
                           kernel_size=sr_ratio,
                           stride=sr_ratio,
                           padding=sr_ratio // 2,
                           groups=inplanes,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=inplanes,
                           dilation=1,
                           has_bn=True,
                           has_act=False),
        )

        self.local_conv = nn.Conv2d(inplanes,
                                    inplanes,
                                    kernel_size=3,
                                    padding=1,
                                    groups=inplanes)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x)
        q = q.reshape(B, self.head_nums, C // self.head_nums,
                      -1).transpose(-1, -2)

        kv = self.sr(x)
        kv = self.local_conv(kv) + kv

        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.head_nums, C // self.head_nums, -1)
        v = v.reshape(B, self.head_nums, C // self.head_nums,
                      -1).transpose(-1, -2)

        attn = (q @ k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        x = (attn @ v).transpose(-1, -2)
        x = x.reshape(B, C, H, W)

        return x


class HybridTokenMixer(nn.Module):
    '''
    D-Mixer
    '''

    def __init__(self, inplanes, head_nums=1, sr_ratio=1, reduction_ratio=8):
        super(HybridTokenMixer, self).__init__()
        assert inplanes % 2 == 0

        self.local_unit = LKA(inplanes=inplanes // 2)
        self.global_unit = Attention(inplanes=inplanes // 2,
                                     head_nums=head_nums,
                                     sr_ratio=sr_ratio)

        inter_planes = max(16, inplanes // reduction_ratio)
        self.proj = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=inplanes,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(inplanes,
                           inter_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(inter_planes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           dilation=1,
                           has_bn=True,
                           has_act=False),
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x

        return x


class MultiScaleDWConv(nn.Module):

    def __init__(self, inplanes, scales=(1, 3, 5, 7)):
        super(MultiScaleDWConv, self).__init__()
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scales)):
            if i == 0:
                channels = inplanes - inplanes // len(scales) * (len(scales) -
                                                                 1)
            else:
                channels = inplanes // len(scales)

            conv = nn.Conv2d(channels,
                             channels,
                             kernel_size=scales[i],
                             padding=scales[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)

        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)

        return x


class Mlp(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, dropout_prob=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Conv2d(inplanes, hidden_planes, 1)
        self.dwconv = MultiScaleDWConv(inplanes=hidden_planes,
                                       scales=(1, 3, 5, 7))
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_planes, planes, 1)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.act(x)
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


class Block(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums=1,
                 sr_ratio=1,
                 mlp_ratio=4.,
                 dropout_prob=0.,
                 drop_path_prob=0.):
        super(Block, self).__init__()
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.attn = HybridTokenMixer(inplanes,
                                     head_nums=head_nums,
                                     sr_ratio=sr_ratio)

        self.norm2 = nn.BatchNorm2d(inplanes)
        self.mlp = Mlp(inplanes=inplanes,
                       hidden_planes=int(inplanes * mlp_ratio),
                       planes=inplanes,
                       dropout_prob=dropout_prob)
        self.layer_scale_1 = nn.Parameter(0.01 * torch.ones(
            (1, inplanes, 1, 1)),
                                          requires_grad=True)
        self.layer_scale_2 = nn.Parameter(0.01 * torch.ones(
            (1, inplanes, 1, 1)),
                                          requires_grad=True)

        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))

        return x


class OverlapPatchEmbed(nn.Module):

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 inplanes=3,
                 embedding_planes=768):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(inplanes,
                              embedding_planes,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.BatchNorm2d(embedding_planes)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        return x


class TransXVAN(nn.Module):

    def __init__(self,
                 inplanes=3,
                 embedding_planes=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 block_nums=[3, 4, 6, 3],
                 head_nums=[1, 2, 4, 8],
                 sr_ratio=[8, 4, 2, 1],
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 num_classes=1000):
        super(TransXVAN, self).__init__()
        assert len(embedding_planes) == len(mlp_ratios) == len(block_nums)

        self.block_nums = block_nums
        self.num_classes = num_classes

        drop_path_prob_list = [
            x for x in np.linspace(0, drop_path_prob, sum(block_nums))
        ]

        currnet_stage_idx = 0
        currnet_inplanes = inplanes
        for i in range(len(block_nums)):
            if i == 0:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7,
                    stride=4,
                    inplanes=currnet_inplanes,
                    embedding_planes=embedding_planes[i])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    inplanes=currnet_inplanes,
                    embedding_planes=embedding_planes[i])
            currnet_inplanes = embedding_planes[i]

            block = nn.ModuleList([
                Block(inplanes=embedding_planes[i],
                      head_nums=head_nums[i],
                      sr_ratio=sr_ratio[i],
                      mlp_ratio=mlp_ratios[i],
                      dropout_prob=dropout_prob,
                      drop_path_prob=drop_path_prob_list[currnet_stage_idx +
                                                         j])
                for j in range(block_nums[i])
            ])
            norm = nn.BatchNorm2d(embedding_planes[i])
            currnet_stage_idx += block_nums[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embedding_planes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
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
        for i in range(len(self.block_nums)):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x = patch_embed(x)

            for blk in block:
                x = blk(x)

            x = norm(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.head(x)

        return x


def _transxvan(embedding_planes, mlp_ratios, block_nums, **kwargs):
    model = TransXVAN(embedding_planes=embedding_planes,
                      mlp_ratios=mlp_ratios,
                      block_nums=block_nums,
                      **kwargs)

    return model


def transxvan_b0(**kwargs):
    return _transxvan(embedding_planes=[32, 64, 160, 256],
                      mlp_ratios=[8, 8, 4, 4],
                      block_nums=[3, 3, 5, 2],
                      head_nums=[1, 2, 4, 8],
                      sr_ratio=[7, 5, 3, 1],
                      **kwargs)


def transxvan_b1(**kwargs):
    return _transxvan(embedding_planes=[64, 128, 320, 512],
                      mlp_ratios=[8, 8, 4, 4],
                      block_nums=[2, 2, 4, 2],
                      head_nums=[1, 2, 4, 8],
                      sr_ratio=[7, 5, 3, 1],
                      **kwargs)


def transxvan_b2(**kwargs):
    return _transxvan(embedding_planes=[64, 128, 320, 512],
                      mlp_ratios=[8, 8, 4, 4],
                      block_nums=[3, 3, 12, 3],
                      head_nums=[2, 4, 8, 16],
                      sr_ratio=[7, 5, 3, 1],
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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    net = transxvan_b0(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = transxvan_b1(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = transxvan_b2(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')
