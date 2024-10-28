import numpy as np

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

__all__ = [
    'convformer_s18',
    'convformer_s36',
    'convformer_m36',
    'convformer_b36',
]


class Downsampling(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 pre_norm=False,
                 post_norm=False):
        super(Downsampling, self).__init__()
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True)

        self.pre_norm = nn.BatchNorm2d(inplanes) if pre_norm else nn.Identity()
        self.post_norm = nn.BatchNorm2d(planes) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)

        x = self.conv(x)

        x = self.post_norm(x)

        return x


class SepConv(nn.Module):

    def __init__(self, inplanes, kernel_size=7, padding=3, expand_ratio=2):
        super(SepConv, self).__init__()
        middle_planes = int(expand_ratio * inplanes)

        self.pwconv1 = nn.Linear(inplanes, middle_planes, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.dwconv = nn.Conv2d(middle_planes,
                                middle_planes,
                                kernel_size=kernel_size,
                                padding=padding,
                                groups=middle_planes,
                                bias=False)
        self.act2 = nn.Identity()
        self.pwconv2 = nn.Linear(middle_planes, inplanes, bias=False)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)

        x = x.permute(0, 3, 1, 2)

        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)

        x = self.act2(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)

        return x


class Mlp(nn.Module):

    def __init__(self, inplanes, mlp_ratio=4, dropout_prob=0.):
        super(Mlp, self).__init__()
        hidden_planes = int(mlp_ratio * inplanes)

        self.fc1 = nn.Linear(inplanes, hidden_planes, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_planes, inplanes, bias=False)
        self.drop2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        x = x.permute(0, 3, 1, 2)

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


class MetaFormerBlock(nn.Module):

    def __init__(self, inplanes, dropout_prob=0., drop_path_prob=0.):
        super(MetaFormerBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.token_mixer = SepConv(inplanes=inplanes,
                                   kernel_size=7,
                                   padding=3,
                                   expand_ratio=2)

        self.norm2 = nn.BatchNorm2d(inplanes)
        self.mlp = Mlp(inplanes=inplanes,
                       mlp_ratio=4,
                       dropout_prob=dropout_prob)

        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(
            self.token_mixer(self.norm1(x).permute(0, 2, 3, 1)))
        x = x + self.drop_path(self.mlp(self.norm2(x).permute(0, 2, 3, 1)))

        return x


class MetaFormer(nn.Module):

    def __init__(self,
                 inplanes=3,
                 embedding_planes=[64, 128, 320, 512],
                 block_nums=[2, 2, 6, 2],
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 num_classes=1000,
                 use_gradient_checkpoint=False):
        super(MetaFormer, self).__init__()
        assert len(embedding_planes) == len(block_nums)

        self.block_nums = block_nums
        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        downsample_layers = []
        down_embedding_planes = [inplanes] + embedding_planes
        for i in range(len(block_nums)):
            if i == 0:
                per_downsample_layer = Downsampling(down_embedding_planes[i],
                                                    down_embedding_planes[i +
                                                                          1],
                                                    kernel_size=7,
                                                    stride=4,
                                                    padding=2,
                                                    pre_norm=False,
                                                    post_norm=True)
            else:
                per_downsample_layer = Downsampling(down_embedding_planes[i],
                                                    down_embedding_planes[i +
                                                                          1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    pre_norm=True,
                                                    post_norm=False)
            downsample_layers.append(per_downsample_layer)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        drop_path_prob_list = [
            x for x in np.linspace(0, drop_path_prob, sum(block_nums))
        ]

        stages = []
        currnet_stage_idx = 0
        for i in range(len(block_nums)):
            stage = nn.Sequential(*[
                MetaFormerBlock(inplanes=embedding_planes[i],
                                dropout_prob=dropout_prob,
                                drop_path_prob=drop_path_prob_list[
                                    currnet_stage_idx + j])
                for j in range(block_nums[i])
            ])
            stages.append(stage)
            currnet_stage_idx += block_nums[i]
        self.stages = nn.ModuleList(stages)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embedding_planes[3], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.block_nums)):
            if self.use_gradient_checkpoint:
                x = checkpoint(self.downsample_layers[i],
                               x,
                               use_reentrant=False)
                x = checkpoint(self.stages[i], x, use_reentrant=False)
            else:
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.head(x)

        return x


def _metaformer(block_nums, embedding_planes, **kwargs):
    model = MetaFormer(block_nums=block_nums,
                       embedding_planes=embedding_planes,
                       **kwargs)

    return model


def convformer_s18(**kwargs):
    model = _metaformer(block_nums=[3, 3, 9, 3],
                        embedding_planes=[64, 128, 320, 512],
                        **kwargs)

    return model


def convformer_s36(**kwargs):
    model = _metaformer(block_nums=[3, 12, 18, 3],
                        embedding_planes=[64, 128, 320, 512],
                        **kwargs)

    return model


def convformer_m36(**kwargs):
    model = _metaformer(block_nums=[3, 12, 18, 3],
                        embedding_planes=[96, 192, 384, 576],
                        **kwargs)

    return model


def convformer_b36(**kwargs):
    model = _metaformer(block_nums=[3, 12, 18, 3],
                        embedding_planes=[128, 256, 512, 768],
                        **kwargs)

    return model


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

    net = convformer_s18(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_s36(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_m36(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_b36(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = convformer_b36(num_classes=1000, use_gradient_checkpoint=True)
    image_h, image_w = 224, 224
    out = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')
