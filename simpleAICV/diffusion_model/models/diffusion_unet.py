'''
https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddpm_mnist.ipynb
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
https://github.com/ermongroup/ddim/blob/main/models/diffusion.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class TimeEmbedding(nn.Module):

    def __init__(self, inplanes):
        super(TimeEmbedding, self).__init__()
        self.inplanes = inplanes
        self.half_inplanes = inplanes // 2

    def forward(self, time):
        assert len(time.shape) == 1
        device = time.device

        embedded_time = torch.exp(-torch.arange(
            self.half_inplanes, dtype=torch.float32, device=device) *
                                  math.log(10000) / (self.half_inplanes - 1))
        embedded_time = time.float()[:, None] * embedded_time[None, :]
        embedded_time = torch.cat(
            [torch.sin(embedded_time),
             torch.cos(embedded_time)], dim=1)
        if self.inplanes % 2 == 1:  # zero pad
            embedded_time = F.pad(embedded_time, (0, 1, 0, 0))

        return embedded_time


class UpSampleBlock(nn.Module):

    def __init__(self, inplanes, use_conv=False):
        super(UpSampleBlock, self).__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.layer = nn.Conv2d(inplanes,
                                   inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   bias=True)

    def forward(self, x):
        x = F.interpolate(x,
                          size=(x.shape[2] * 2, x.shape[3] * 2),
                          mode='nearest')
        if self.use_conv:
            x = self.layer(x)

        return x


class DownSampleBlock(nn.Module):

    def __init__(self, inplanes, use_conv=False):
        super(DownSampleBlock, self).__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.layer = nn.Conv2d(inplanes,
                                   inplanes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=0)

    def forward(self, x):
        if self.use_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.layer(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

        return x


class AttentionBlock(nn.Module):

    def __init__(self, inplanes, num_groups=32):
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups, inplanes)
        self.q = nn.Conv2d(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True)
        self.k = nn.Conv2d(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True)
        self.v = nn.Conv2d(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True)
        self.proj_out = nn.Conv2d(inplanes,
                                  inplanes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=True)

    def forward(self, x):
        inputs = x
        b, c, h, w = x.shape

        x = self.norm(x)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.reshape(b, c, h * w)
        # b,hw,c
        q = q.permute(0, 2, 1)
        # b,c,hw
        k = k.reshape(b, c, h * w)

        # b,hw,hw
        weight = torch.bmm(q, k) * (int(c)**(-0.5))
        weight = F.softmax(weight, dim=2)

        v = v.reshape(b, c, h * w)
        # b,hw,hw (first hw of k, second of q)
        weight = weight.permute(0, 2, 1)
        x = torch.bmm(v, weight)
        x = x.reshape(b, c, h, w)

        x = self.proj_out(x) + inputs

        return x


class ResBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 time_planes,
                 num_groups=32,
                 num_classes_planes=None,
                 dropout_prob=0.,
                 use_conv_shortcut=False,
                 use_attention=False):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.num_classes_planes = num_classes_planes
        self.use_conv_shortcut = use_conv_shortcut
        self.use_attention = use_attention

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, inplanes), nn.SiLU(),
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=1,
                      bias=True))

        self.time_embedding_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(time_planes, planes))

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, planes), nn.SiLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=1,
                      bias=True))

        if self.num_classes_planes:
            self.class_embedding_proj = nn.Sequential(
                nn.SiLU(), nn.Linear(time_planes, planes))

        if self.inplanes != self.planes:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(inplanes,
                                               planes,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=True)
            else:
                self.nin_shortcut = nn.Conv2d(inplanes,
                                              planes,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              bias=True)

        if self.use_attention:
            self.attention = AttentionBlock(planes, num_groups=num_groups)

    def forward(self, x, embedded_time, embedded_class_label):
        inputs = x

        x = self.block1(x)

        x = x + self.time_embedding_proj(embedded_time).unsqueeze(
            -1).unsqueeze(-1)

        if self.num_classes_planes and embedded_class_label is not None:
            x = x + self.class_embedding_proj(embedded_class_label).unsqueeze(
                -1).unsqueeze(-1)

        x = self.block2(x)

        if self.inplanes != self.planes:
            if self.use_conv_shortcut:
                inputs = self.conv_shortcut(inputs)
            else:
                inputs = self.nin_shortcut(inputs)

        x = x + inputs

        if self.use_attention:
            x = self.attention(x)

        return x


class DiffusionUNet(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=128,
                 planes_multi=[1, 2, 2, 2],
                 time_embedding_ratio=4,
                 block_nums=2,
                 dropout_prob=0.,
                 num_groups=32,
                 use_attention_planes_multi_idx=[1],
                 num_classes=None,
                 use_gradient_checkpoint=False):
        super(DiffusionUNet, self).__init__()
        # class_label_condition
        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        time_planes = planes * time_embedding_ratio
        self.time_mlp = nn.Sequential(TimeEmbedding(planes),
                                      nn.Linear(planes, time_planes),
                                      nn.SiLU(),
                                      nn.Linear(time_planes, time_planes))

        if self.num_classes:
            self.class_mlp = nn.Sequential(
                nn.Embedding(num_embeddings=num_classes + 1,
                             embedding_dim=planes,
                             padding_idx=0), nn.Linear(planes, time_planes),
                nn.SiLU(), nn.Linear(time_planes, time_planes))

        self.stem = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=1,
                              bias=True)

        downsample_blocks = []
        downsample_planes = [planes]
        current_planes = planes
        for idx, per_planes_multi in enumerate(planes_multi):
            per_planes = int(planes * per_planes_multi)
            for _ in range(block_nums):
                downsample_blocks.append(
                    ResBlock(
                        current_planes,
                        per_planes,
                        time_planes,
                        num_groups=num_groups,
                        num_classes_planes=num_classes,
                        dropout_prob=dropout_prob,
                        use_conv_shortcut=False,
                        use_attention=(idx in use_attention_planes_multi_idx)))
                current_planes = per_planes
                downsample_planes.append(current_planes)
            if idx != len(planes_multi) - 1:
                downsample_blocks.append(
                    DownSampleBlock(current_planes, use_conv=True))
                downsample_planes.append(current_planes)
        self.downsample_blocks = nn.ModuleList(downsample_blocks)

        self.middle_blocks = nn.ModuleList([
            ResBlock(current_planes,
                     current_planes,
                     time_planes,
                     num_groups=num_groups,
                     num_classes_planes=num_classes,
                     dropout_prob=dropout_prob,
                     use_conv_shortcut=False,
                     use_attention=True),
            ResBlock(current_planes,
                     current_planes,
                     time_planes,
                     num_groups=num_groups,
                     num_classes_planes=num_classes,
                     dropout_prob=dropout_prob,
                     use_conv_shortcut=False,
                     use_attention=False),
        ])

        upsample_use_attention_planes_multi_idx = [
            len(planes_multi) - 1
        ] * len(use_attention_planes_multi_idx)
        upsample_use_attention_planes_multi_idx = [
            upsample_use_attention_planes_multi_idx[i] -
            use_attention_planes_multi_idx[i]
            for i in range(len(upsample_use_attention_planes_multi_idx))
        ]

        upsample_blocks = []
        for idx, per_planes_multi in reversed(list(enumerate(planes_multi))):
            per_planes = int(planes * per_planes_multi)
            for _ in range(block_nums + 1):
                upsample_blocks.append(
                    ResBlock(downsample_planes.pop() + current_planes,
                             per_planes,
                             time_planes,
                             num_groups=num_groups,
                             num_classes_planes=num_classes,
                             dropout_prob=dropout_prob,
                             use_conv_shortcut=False,
                             use_attention=(
                                 idx
                                 in upsample_use_attention_planes_multi_idx)))
                current_planes = per_planes
            if idx != 0:
                upsample_blocks.append(
                    UpSampleBlock(current_planes, use_conv=True))
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        assert len(downsample_planes) == 0

        self.last_norm = nn.GroupNorm(num_groups, planes)
        self.last_act = nn.SiLU()
        self.last_conv = nn.Conv2d(planes,
                                   inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   bias=True)

    def forward(self, x, time, class_label=None):
        embedded_time = self.time_mlp(time)
        embedded_class = class_label

        if self.num_classes and class_label is not None:
            embedded_class = self.class_mlp(embedded_class)

        x = self.stem(x)

        # 下采样
        downsample_features = [x]
        for per_layer in self.downsample_blocks:
            if isinstance(per_layer, ResBlock):
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x, embedded_time, embedded_class)
                else:
                    x = per_layer(x, embedded_time, embedded_class)
            else:
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x)
                else:
                    x = per_layer(x)
            downsample_features.append(x)

        # 中间层
        for per_layer in self.middle_blocks:
            if isinstance(per_layer, ResBlock):
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x, embedded_time, embedded_class)
                else:
                    x = per_layer(x, embedded_time, embedded_class)
            else:
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x)
                else:
                    x = per_layer(x)

        # 上采样
        for per_layer in self.upsample_blocks:
            if isinstance(per_layer, ResBlock):
                x = torch.cat([x, downsample_features.pop()], dim=1)
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x, embedded_time, embedded_class)
                else:
                    x = per_layer(x, embedded_time, embedded_class)
            else:
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x)
                else:
                    x = per_layer(x)

        x = self.last_norm(x)
        x = self.last_act(x)
        x = self.last_conv(x)

        return x


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

    time_step = 1000
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=None,
                        use_gradient_checkpoint=False)
    batch, channel, image_h, image_w = 1, 3, 32, 32
    images = torch.randn(batch, channel, image_h, image_w).float()
    times = torch.randint(time_step, (batch, )).float()

    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, times), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times))
    print('2222', outs.shape)

    time_step = 1000
    num_classes = 100
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=num_classes,
                        use_gradient_checkpoint=False)
    batch, channel, image_h, image_w = 1, 3, 32, 32
    images = torch.randn(batch, channel, image_h, image_w).float()
    times = torch.randint(time_step, (batch, )).float()
    labels = torch.randint(num_classes, (batch, )).long()

    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, times, labels), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'3333, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times),
               torch.autograd.Variable(labels))
    print('4444', outs.shape)

    time_step = 1000
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=None,
                        use_gradient_checkpoint=False)
    batch, channel, image_h, image_w = 1, 3, 64, 64
    images = torch.randn(batch, channel, image_h, image_w).float()
    times = torch.randint(time_step, (batch, )).float()

    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, times), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times))
    print('2222', outs.shape)

    time_step = 1000
    num_classes = 100
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=num_classes,
                        use_gradient_checkpoint=False)
    batch, channel, image_h, image_w = 1, 3, 64, 64
    images = torch.randn(batch, channel, image_h, image_w).float()
    times = torch.randint(time_step, (batch, )).float()
    labels = torch.randint(num_classes, (batch, )).long()

    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(images, times, labels), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'3333, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times),
               torch.autograd.Variable(labels))
    print('4444', outs.shape)

    time_step = 1000
    # num_classes = None
    num_classes = 100
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=num_classes,
                        use_gradient_checkpoint=True)
    batch, channel, image_h, image_w = 1, 3, 64, 64
    images = torch.randn(batch, channel, image_h, image_w).float()
    times = torch.randint(time_step, (batch, )).float()
    # labels = None
    labels = torch.randint(num_classes, (batch, )).long()
    # outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times))
    outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times),
               torch.autograd.Variable(labels))
    print('5555', outs.shape)