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
                 has_norm=False):
        super(PatchEmbeddingBlock, self).__init__()
        bias = False if has_norm else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=inplanes)
            if has_norm else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

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

    def __init__(self, inplanes, hidden_planes, planes, dropout_prob=0.):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Conv2d(inplanes,
                             hidden_planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_planes,
                             planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)
        self.drop = nn.Dropout(dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


class PoolFormerBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 pool_size=3,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5):
        super(PoolFormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=inplanes)
        self.token_mixer = PoolingBlock(pool_size=pool_size)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=inplanes)
        self.feed_forward = FeedForwardBlock(
            inplanes, int(inplanes * feed_forward_ratio), inplanes,
            dropout_prob)
        self.layer_scale_1 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        self.layer_scale_2 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1 * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2 * self.feed_forward(self.norm2(x)))

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


class PoolFormer(nn.Module):

    def __init__(self,
                 layer_nums,
                 planes,
                 pool_size=3,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5,
                 num_classes=1000):
        super(PoolFormer, self).__init__()
        assert len(layer_nums) == 4
        assert len(planes) == 4
        self.num_classes = num_classes

        drop_path_prob_list = [[], [], [], []]
        for i, per_layer_num in enumerate(layer_nums):
            for per_block_index in range(per_layer_num):
                if drop_path_prob == 0.:
                    drop_path_prob_list[i].append(0.)
                else:
                    per_layer_drop_path_prob = drop_path_prob * (
                        per_block_index +
                        sum(layer_nums[:i])) / (sum(layer_nums) - 1)
                    drop_path_prob_list[i].append(per_layer_drop_path_prob)

        self.patch_embedding1 = PatchEmbeddingBlock(3,
                                                    planes[0],
                                                    kernel_size=7,
                                                    stride=4,
                                                    padding=2,
                                                    groups=1,
                                                    has_norm=False)
        self.stage2 = self.make_layer(planes[0],
                                      layer_nums[0],
                                      drop_path_prob_list[0],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding2 = PatchEmbeddingBlock(planes[0],
                                                    planes[1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage3 = self.make_layer(planes[1],
                                      layer_nums[1],
                                      drop_path_prob_list[1],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding3 = PatchEmbeddingBlock(planes[1],
                                                    planes[2],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage4 = self.make_layer(planes[2],
                                      layer_nums[2],
                                      drop_path_prob_list[2],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding4 = PatchEmbeddingBlock(planes[2],
                                                    planes[3],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage5 = self.make_layer(planes[3],
                                      layer_nums[3],
                                      drop_path_prob_list[3],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=planes[3])
        self.fc = nn.Linear(planes[3], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self,
                   inplanes,
                   layer_nums,
                   drop_path_prob,
                   pool_size=3,
                   feed_forward_ratio=4,
                   dropout_prob=0.,
                   layer_scale_factor=1e-5):
        assert len(drop_path_prob) == layer_nums

        blocks = []
        for block_idx in range(layer_nums):
            blocks.append(
                PoolFormerBlock(inplanes,
                                pool_size=pool_size,
                                feed_forward_ratio=feed_forward_ratio,
                                dropout_prob=dropout_prob,
                                drop_path_prob=drop_path_prob[block_idx],
                                layer_scale_factor=layer_scale_factor))
        blocks = nn.ModuleList(blocks)

        return blocks

    def forward(self, x):
        x = self.patch_embedding1(x)
        for per_block in self.stage2:
            x = per_block(x)

        x = self.patch_embedding2(x)
        for per_block in self.stage3:
            x = per_block(x)

        x = self.patch_embedding3(x)
        for per_block in self.stage4:
            x = per_block(x)

        x = self.patch_embedding4(x)
        for per_block in self.stage5:
            x = per_block(x)

        x = self.norm(x)
        x = x.mean([-2, -1])
        x = self.fc(x)

        return x


class ConvFormerBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5):
        super(ConvFormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=inplanes)
        self.token_mixer = ConvBlock(inplanes, planes, kernel_size=kernel_size)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=inplanes)
        self.feed_forward = FeedForwardBlock(
            inplanes, int(inplanes * feed_forward_ratio), inplanes,
            dropout_prob)
        self.layer_scale_1 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        self.layer_scale_2 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1 * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2 * self.feed_forward(self.norm2(x)))

        return x


class ConvFormer(nn.Module):

    def __init__(self,
                 layer_nums,
                 planes,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5,
                 num_classes=1000):
        super(ConvFormer, self).__init__()
        assert len(layer_nums) == 4
        assert len(planes) == 4
        self.num_classes = num_classes

        drop_path_prob_list = [[], [], [], []]
        for i, per_layer_num in enumerate(layer_nums):
            for per_block_index in range(per_layer_num):
                if drop_path_prob == 0.:
                    drop_path_prob_list[i].append(0.)
                else:
                    per_layer_drop_path_prob = drop_path_prob * (
                        per_block_index +
                        sum(layer_nums[:i])) / (sum(layer_nums) - 1)
                    drop_path_prob_list[i].append(per_layer_drop_path_prob)

        self.patch_embedding1 = PatchEmbeddingBlock(3,
                                                    planes[0],
                                                    kernel_size=7,
                                                    stride=4,
                                                    padding=2,
                                                    groups=1,
                                                    has_norm=False)

        self.stage2 = self.make_layer(planes[0],
                                      planes[0],
                                      layer_nums[0],
                                      drop_path_prob_list[0],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding2 = PatchEmbeddingBlock(planes[0],
                                                    planes[1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage3 = self.make_layer(planes[1],
                                      planes[1],
                                      layer_nums[1],
                                      drop_path_prob_list[1],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding3 = PatchEmbeddingBlock(planes[1],
                                                    planes[2],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage4 = self.make_layer(planes[2],
                                      planes[2],
                                      layer_nums[2],
                                      drop_path_prob_list[2],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding4 = PatchEmbeddingBlock(planes[2],
                                                    planes[3],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage5 = self.make_layer(planes[3],
                                      planes[3],
                                      layer_nums[3],
                                      drop_path_prob_list[3],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=planes[3])
        self.fc = nn.Linear(planes[3], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self,
                   inplanes,
                   planes,
                   layer_nums,
                   drop_path_prob,
                   feed_forward_ratio=4,
                   dropout_prob=0.,
                   layer_scale_factor=1e-5):
        assert len(drop_path_prob) == layer_nums

        blocks = []
        for block_idx in range(layer_nums):
            blocks.append(
                ConvFormerBlock(inplanes,
                                planes,
                                kernel_size=3,
                                feed_forward_ratio=feed_forward_ratio,
                                dropout_prob=dropout_prob,
                                drop_path_prob=drop_path_prob[block_idx],
                                layer_scale_factor=layer_scale_factor))
            inplanes = planes
        blocks = nn.ModuleList(blocks)

        return blocks

    def forward(self, x):
        x = self.patch_embedding1(x)
        for per_block in self.stage2:
            x = per_block(x)

        x = self.patch_embedding2(x)
        for per_block in self.stage3:
            x = per_block(x)

        x = self.patch_embedding3(x)
        for per_block in self.stage4:
            x = per_block(x)

        x = self.patch_embedding4(x)
        for per_block in self.stage5:
            x = per_block(x)

        x = self.norm(x)
        x = x.mean([-2, -1])
        x = self.fc(x)

        return x


class PatchEmbeddingBNBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_norm=False):
        super(PatchEmbeddingBNBlock, self).__init__()
        bias = False if has_norm else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(inplanes) if has_norm else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class FeedForwardReluBlock(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, dropout_prob=0.):
        super(FeedForwardReluBlock, self).__init__()
        self.fc1 = nn.Conv2d(inplanes,
                             hidden_planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_planes,
                             planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)
        self.drop = nn.Dropout(dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class ConvReluBNFormerBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5):
        super(ConvReluBNFormerBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.token_mixer = ConvBlock(inplanes, planes, kernel_size=kernel_size)
        self.norm2 = nn.BatchNorm2d(inplanes)
        self.feed_forward = FeedForwardReluBlock(
            inplanes, int(inplanes * feed_forward_ratio), inplanes,
            dropout_prob)
        self.layer_scale_1 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        self.layer_scale_2 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1 * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2 * self.feed_forward(self.norm2(x)))

        return x


class ConvReluBNFormer(nn.Module):

    def __init__(self,
                 layer_nums,
                 planes,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5,
                 num_classes=1000):
        super(ConvReluBNFormer, self).__init__()
        assert len(layer_nums) == 4
        assert len(planes) == 4
        self.num_classes = num_classes

        drop_path_prob_list = [[], [], [], []]
        for i, per_layer_num in enumerate(layer_nums):
            for per_block_index in range(per_layer_num):
                if drop_path_prob == 0.:
                    drop_path_prob_list[i].append(0.)
                else:
                    per_layer_drop_path_prob = drop_path_prob * (
                        per_block_index +
                        sum(layer_nums[:i])) / (sum(layer_nums) - 1)
                    drop_path_prob_list[i].append(per_layer_drop_path_prob)

        self.patch_embedding1 = PatchEmbeddingBNBlock(3,
                                                      planes[0],
                                                      kernel_size=7,
                                                      stride=4,
                                                      padding=2,
                                                      groups=1,
                                                      has_norm=False)

        self.stage2 = self.make_layer(planes[0],
                                      planes[0],
                                      layer_nums[0],
                                      drop_path_prob_list[0],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding2 = PatchEmbeddingBNBlock(planes[0],
                                                      planes[1],
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      groups=1,
                                                      has_norm=False)
        self.stage3 = self.make_layer(planes[1],
                                      planes[1],
                                      layer_nums[1],
                                      drop_path_prob_list[1],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding3 = PatchEmbeddingBNBlock(planes[1],
                                                      planes[2],
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      groups=1,
                                                      has_norm=False)
        self.stage4 = self.make_layer(planes[2],
                                      planes[2],
                                      layer_nums[2],
                                      drop_path_prob_list[2],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding4 = PatchEmbeddingBNBlock(planes[2],
                                                      planes[3],
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      groups=1,
                                                      has_norm=False)
        self.stage5 = self.make_layer(planes[3],
                                      planes[3],
                                      layer_nums[3],
                                      drop_path_prob_list[3],
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)

        self.norm = nn.BatchNorm2d(planes[3])
        self.fc = nn.Linear(planes[3], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self,
                   inplanes,
                   planes,
                   layer_nums,
                   drop_path_prob,
                   feed_forward_ratio=4,
                   dropout_prob=0.,
                   layer_scale_factor=1e-5):
        assert len(drop_path_prob) == layer_nums

        blocks = []
        for block_idx in range(layer_nums):
            blocks.append(
                ConvReluBNFormerBlock(inplanes,
                                      planes,
                                      kernel_size=3,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      drop_path_prob=drop_path_prob[block_idx],
                                      layer_scale_factor=layer_scale_factor))
            inplanes = planes
        blocks = nn.ModuleList(blocks)

        return blocks

    def forward(self, x):
        x = self.patch_embedding1(x)
        for per_block in self.stage2:
            x = per_block(x)

        x = self.patch_embedding2(x)
        for per_block in self.stage3:
            x = per_block(x)

        x = self.patch_embedding3(x)
        for per_block in self.stage4:
            x = per_block(x)

        x = self.patch_embedding4(x)
        for per_block in self.stage5:
            x = per_block(x)

        x = self.norm(x)
        x = x.mean([-2, -1])
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
    return _convformer([2, 2, 2, 2], [64, 128, 320, 512],
                       drop_path_prob=0.1,
                       **kwargs)


def convformer_s16(**kwargs):
    return _convformer([3, 4, 6, 3], [64, 128, 320, 512],
                       drop_path_prob=0.2,
                       feed_forward_ratio=4,
                       **kwargs)


def convformer_m8(**kwargs):
    return _convformer([2, 2, 2, 2], [96, 192, 384, 768],
                       drop_path_prob=0.2,
                       **kwargs)


def convformer_m16(**kwargs):
    return _convformer([3, 4, 6, 3], [96, 192, 384, 768],
                       drop_path_prob=0.3,
                       **kwargs)


def _convrelubnformer(layer_nums, planes, **kwargs):
    model = ConvReluBNFormer(layer_nums, planes, **kwargs)

    return model


def convrelubnformer_s8(**kwargs):
    return _convrelubnformer([2, 2, 2, 2], [64, 128, 320, 512],
                             drop_path_prob=0.1,
                             **kwargs)


def convrelubnformer_s16(**kwargs):
    return _convrelubnformer([3, 4, 6, 3], [64, 128, 320, 512],
                             drop_path_prob=0.2,
                             feed_forward_ratio=4,
                             **kwargs)


def convrelubnformer_m8(**kwargs):
    return _convrelubnformer([2, 2, 2, 2], [96, 192, 384, 768],
                             drop_path_prob=0.2,
                             **kwargs)


def convrelubnformer_m16(**kwargs):
    return _convrelubnformer([3, 4, 6, 3], [96, 192, 384, 768],
                             drop_path_prob=0.3,
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

    # when training, drop_path_prob=[0.1, 0.1, 0.2, 0.3, 0.4] for model [s12, s24, s36, m36, m48]
    # when testing, drop_path_prob=0. for model [s12, s24, s36, m36, m48]
    net = poolformer_s12(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_s24(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_s36(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'3333, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_m36(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'4444, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = poolformer_m48(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'5555, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_s8(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'6666, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_s16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'7777, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_m8(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'8888, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_m16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'9999, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convrelubnformer_s8(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'9191, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convrelubnformer_s16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'9292, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convrelubnformer_m8(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'9393, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convrelubnformer_m16(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'9494, macs: {macs}, params: {params},out_shape: {out.shape}')