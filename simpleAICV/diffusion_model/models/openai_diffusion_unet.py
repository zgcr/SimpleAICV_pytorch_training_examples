'''
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from attention import SpatialTransformer


class TimeEmbedding(nn.Module):

    def __init__(self, inplanes, max_period=10000):
        super(TimeEmbedding, self).__init__()
        self.inplanes = inplanes
        self.half_inplanes = inplanes // 2
        self.max_period = max_period

    def forward(self, time):
        assert len(time.shape) == 1
        device = time.device

        embedded_time = torch.exp(-torch.arange(
            0, self.half_inplanes, dtype=torch.float32, device=device) *
                                  math.log(self.max_period) /
                                  self.half_inplanes)
        embedded_time = time.float()[:, None] * embedded_time[None, :]
        embedded_time = torch.cat(
            [torch.cos(embedded_time),
             torch.sin(embedded_time)], dim=1)
        # zero pad
        if self.inplanes % 2 == 1:
            embedded_time = torch.cat(
                [embedded_time,
                 torch.zeros_like(embedded_time[:, :1])], dim=1)

        return embedded_time


class UpSampleBlock(nn.Module):

    def __init__(self, inplanes, use_conv=False):
        super(UpSampleBlock, self).__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(inplanes,
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
            x = self.conv(x)

        return x


class DownSampleBlock(nn.Module):

    def __init__(self, inplanes, use_conv=False):
        super(DownSampleBlock, self).__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.op = torch.nn.Conv2d(inplanes,
                                      inplanes,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.op(x)

        return x


class QKVAttention(nn.Module):

    def __init__(self, num_heads):
        '''
        split heads before split qkv
        '''
        super(QKVAttention, self).__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        b, n, l = qkv.shape
        assert n % (self.num_heads * 3) == 0
        c = n // (self.num_heads * 3)

        # q,k,v:[b * self.num_heads, c, l]
        q, k, v = qkv.reshape(b * self.num_heads, c * 3, l).split(c, dim=1)
        scale = 1 / math.sqrt(math.sqrt(c))
        # More stable with f16 than dividing afterwards
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale).float()
        weight = F.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        a = a.reshape(b, -1, l)

        return a


class AttentionBlock(nn.Module):

    def __init__(self, inplanes, num_groups=32, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups, inplanes)
        self.qkv = nn.Conv1d(inplanes,
                             inplanes * 3,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.attention = QKVAttention(num_heads)
        self.proj_out = self.zero_module(
            nn.Conv1d(inplanes,
                      inplanes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True))

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()

        return module

    def forward(self, x):
        inputs = x
        b, c, h, w = x.shape

        x = x.reshape(b, c, -1)
        x = self.norm(x)
        x = self.qkv(x)

        x = self.attention(x)
        x = self.proj_out(x)
        x = x.reshape(b, c, h, w)

        x = x + inputs

        return x


class ResBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 embedding_planes,
                 num_groups=32,
                 dropout_prob=0.,
                 use_conv_shortcut=False,
                 up=False,
                 down=False):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.use_conv_shortcut = use_conv_shortcut

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, inplanes), nn.SiLU(),
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=1,
                      bias=True))

        if up:
            self.h_upd = UpSampleBlock(inplanes, False)
            self.x_upd = UpSampleBlock(inplanes, False)
        elif down:
            self.h_upd = DownSampleBlock(inplanes, False)
            self.x_upd = DownSampleBlock(inplanes, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.updown = up or down

        self.embedding_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_planes, planes))

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, planes), nn.SiLU(),
            nn.Dropout(dropout_prob),
            self.zero_module(
                nn.Conv2d(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          bias=True)))

        if self.inplanes == self.planes:
            self.shortcut = nn.Identity()
        elif use_conv_shortcut:
            self.shortcut = nn.Conv2d(inplanes,
                                      planes,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=True)
        else:
            self.shortcut = nn.Conv2d(inplanes,
                                      planes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()

        return module

    def forward(self, x, embedded_feature):
        inputs = x
        if self.updown:
            pre_layers, last_conv_layer = self.block1[:-1], self.block1[-1]
            x = pre_layers(x)
            x = self.h_upd(x)
            inputs = self.x_upd(inputs)
            x = last_conv_layer(x)
        else:
            x = self.block1(x)

        x = x + self.embedding_proj(embedded_feature).unsqueeze(-1).unsqueeze(
            -1)

        x = self.block2(x)

        x = x + self.shortcut(inputs)

        return x


# attention_resolutions = [4, 2, 1],
# use_spatial_transformer = True,  # custom transformer support


class OpenaiDiffusionUNet(nn.Module):

    def __init__(self,
                 inplanes=4,
                 planes=320,
                 planes_multi=[1, 2, 4, 4],
                 time_embedding_ratio=4,
                 block_nums=2,
                 head_nums=8,
                 dropout_prob=0.,
                 num_groups=32,
                 use_attention_planes_multi_idx=[1, 2, 3],
                 resblock_updown=False,
                 num_classes=None,
                 use_context_transformer_planes_multi_idx=[],
                 transformer_block_nums=1,
                 context_planes=768,
                 use_gradient_checkpoint=False):
        super(OpenaiDiffusionUNet, self).__init__()
        # set out_planes == inplanes
        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        time_planes = planes * time_embedding_ratio
        self.time_mlp = nn.Sequential(TimeEmbedding(planes),
                                      nn.Linear(planes, time_planes),
                                      nn.SiLU(),
                                      nn.Linear(time_planes, time_planes))

        if self.num_classes:
            self.class_mlp = nn.Embedding(num_classes, time_planes)

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
                    ResBlock(current_planes,
                             per_planes,
                             time_planes,
                             num_groups=num_groups,
                             dropout_prob=dropout_prob,
                             use_conv_shortcut=False,
                             up=False,
                             down=False,
                             use_context_transformer=(
                                 idx
                                 in use_context_transformer_planes_multi_idx),
                             transformer_block_nums=transformer_block_nums,
                             context_planes=context_planes,
                             use_attention=(idx
                                            in use_attention_planes_multi_idx),
                             head_nums=head_nums))
                current_planes = per_planes
                downsample_planes.append(current_planes)
            if idx != len(planes_multi) - 1:
                if resblock_updown:
                    downsample_blocks.append(
                        ResBlock(current_planes,
                                 current_planes,
                                 time_planes,
                                 num_groups=num_groups,
                                 dropout_prob=dropout_prob,
                                 use_conv_shortcut=False,
                                 up=False,
                                 down=True,
                                 use_context_transformer=False,
                                 transformer_block_nums=transformer_block_nums,
                                 context_planes=context_planes,
                                 use_attention=False,
                                 head_nums=head_nums))
                else:
                    downsample_blocks.append(
                        DownSampleBlock(current_planes, use_conv=True))
                downsample_planes.append(current_planes)
        self.downsample_blocks = nn.ModuleList(downsample_blocks)

        self.middle_blocks = nn.ModuleList([
            ResBlock(current_planes,
                     current_planes,
                     time_planes,
                     num_groups=num_groups,
                     dropout_prob=dropout_prob,
                     use_conv_shortcut=False,
                     up=False,
                     down=False,
                     use_context_transformer=(
                         idx in use_context_transformer_planes_multi_idx),
                     transformer_block_nums=transformer_block_nums,
                     context_planes=context_planes,
                     use_attention=(idx in use_attention_planes_multi_idx),
                     head_nums=head_nums),
            ResBlock(current_planes,
                     current_planes,
                     time_planes,
                     num_groups=num_groups,
                     dropout_prob=dropout_prob,
                     use_conv_shortcut=False,
                     up=False,
                     down=False),
        ])

        upsample_use_context_transformer_planes_multi_idx = [
            len(planes_multi) - 1
        ] * len(use_context_transformer_planes_multi_idx)
        upsample_use_context_transformer_planes_multi_idx = [
            upsample_use_context_transformer_planes_multi_idx[i] -
            use_context_transformer_planes_multi_idx[i] for i in range(
                len(upsample_use_context_transformer_planes_multi_idx))
        ]

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
                    ResBlock(
                        downsample_planes.pop() + current_planes,
                        per_planes,
                        time_planes,
                        num_groups=num_groups,
                        dropout_prob=dropout_prob,
                        use_conv_shortcut=False,
                        up=False,
                        down=False,
                        use_context_transformer=(
                            idx in
                            upsample_use_context_transformer_planes_multi_idx),
                        transformer_block_nums=transformer_block_nums,
                        context_planes=context_planes,
                        use_attention=(
                            idx in upsample_use_attention_planes_multi_idx),
                        head_nums=head_nums))
                current_planes = per_planes
            if idx:
                if resblock_updown:
                    upsample_blocks.append(
                        ResBlock(current_planes,
                                 current_planes,
                                 time_planes,
                                 num_groups=num_groups,
                                 dropout_prob=dropout_prob,
                                 use_conv_shortcut=False,
                                 up=True,
                                 down=False,
                                 use_context_transformer=False,
                                 transformer_block_nums=transformer_block_nums,
                                 context_planes=context_planes,
                                 use_attention=False,
                                 head_nums=head_nums))
                else:
                    upsample_blocks.append(
                        UpSampleBlock(current_planes, use_conv=True))
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        assert len(downsample_planes) == 0

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, planes), nn.SiLU(),
            self.zero_module(
                nn.Conv2d(planes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          bias=True)))

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()

        return module

    def forward(self, x, time, class_label=None, context=None):
        embedded_features = self.time_mlp(time)
        embedded_class = class_label
        context_features = context

        if self.num_classes and class_label is not None:
            embedded_features = embedded_features + self.class_mlp(
                embedded_class)

        x = self.stem(x)

        # 下采样
        downsample_features = [x]
        for per_layer in self.downsample_blocks:
            if isinstance(per_layer, ResBlock):
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x, embedded_features,
                                   context_features)
                else:
                    x = per_layer(x, embedded_features, context_features)
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
                    x = checkpoint(per_layer, x, embedded_features,
                                   context_features)
                else:
                    x = per_layer(x, embedded_features, context_features)
            else:
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x)
                else:
                    x = per_layer(x)

        # 上采样
        for per_layer in self.upsample_blocks:
            if x.shape[2] == downsample_features[-1].shape[2] and x.shape[
                    3] == downsample_features[-1].shape[3]:
                x = torch.cat([x, downsample_features.pop()], dim=1)
            if isinstance(per_layer, ResBlock):
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x, embedded_features,
                                   context_features)
                else:
                    x = per_layer(x, embedded_features, context_features)
            else:
                if self.use_gradient_checkpoint:
                    x = checkpoint(per_layer, x)
                else:
                    x = per_layer(x)

        x = self.out(x)

        return x


class Upsample(nn.Module):

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels,
                                  self.out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(self.channels,
                                self.out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=True)

        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.op(x)

        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self,
                 channels,
                 emb_channels,
                 dropout,
                 num_groups=32,
                 out_channels=None,
                 use_conv=False,
                 up=False,
                 down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups, channels), nn.SiLU(),
            nn.Conv2d(channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            self.zero_module(
                nn.Conv2d(self.out_channels,
                          self.out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels,
                                             self.out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=True)

        else:
            self.skip_connection = nn.Conv2d(channels,
                                             self.out_channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias=True)

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()

        return module

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels,
                             channels * 3,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = self.zero_module(
            nn.Conv1d(channels,
                      channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True))

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()

        return module

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):

    def __init__(self, n_heads):
        super(QKVAttentionLegacy, self).__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        a = a.reshape(bs, -1, length)

        return a


class OpenaiDiffusionUNet(nn.Module):

    def __init__(self,
                 inplanes=4,
                 out_planes=4,
                 inter_planes=320,
                 inter_planes_multi=[1, 2, 4, 4],
                 time_embedding_ratio=4,
                 res_block_nums=2,
                 attention_resolutions=[4, 2, 1],
                 dropout_prob=0.,
                 num_groups=32,
                 num_heads=8,
                 num_head_planes=-1,
                 use_spatial_transformer=True,
                 transformer_depth=1,
                 context_planes=768,
                 resblock_updown=False,
                 use_conv=True,
                 num_classes=None,
                 use_gradient_checkpoint=False):
        super(OpenaiDiffusionUNet, self).__init__()
        # when using spatial transformer,context_planes is not None
        assert (use_spatial_transformer) and (context_planes is not None)
        assert (not use_spatial_transformer) and (context_planes is None)

        # num_heads and num_head_planes only have one variable != -1
        assert (num_heads == -1
                and num_head_planes != -1) or (num_heads != -1
                                               and num_head_planes == -1)

        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        time_planes = inter_planes * time_embedding_ratio
        # self.time_embed
        self.time_mlp = nn.Sequential(TimeEmbedding(inter_planes),
                                      nn.Linear(inter_planes, time_planes),
                                      nn.SiLU(),
                                      nn.Linear(time_planes, time_planes))

        if self.num_classes is not None:
            # self.label_emb
            self.class_mlp = nn.Embedding(num_classes, time_planes)

        # self.input_blocks
        self.stem = nn.Conv2d(inplanes,
                              inter_planes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=1,
                              bias=True)

        # time_planes = inter_planes * time_embedding_ratio
        # self.time_embed = nn.Sequential(
        #     nn.Linear(inter_planes, time_planes),
        #     nn.SiLU(),
        #     nn.Linear(time_planes, time_planes),
        # )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_planes)

        # self.input_blocks = nn.ModuleList([
        #     TimestepEmbedSequential(
        #         nn.Conv2d(inplanes,
        #                   inter_planes,
        #                   kernel_size=3,
        #                   stride=1,
        #                   padding=1,
        #                   bias=True))
        # ])

        self._feature_size = inter_planes
        input_block_chans = [inter_planes]
        ch = inter_planes
        ds = 1
        for level, mult in enumerate(inter_planes_multi):
            for _ in range(res_block_nums):
                layers = [
                    ResBlock(
                        ch,
                        time_planes,
                        dropout_prob,
                        out_channels=mult * inter_planes,
                    )
                ]
                ch = mult * inter_planes
                if ds in attention_resolutions:
                    if num_head_planes == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_planes
                        dim_head = num_head_planes
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                        ) if not use_spatial_transformer else
                        SpatialTransformer(ch,
                                           num_heads,
                                           dim_head,
                                           depth=transformer_depth,
                                           context_dim=context_planes))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(inter_planes_multi) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_planes,
                            dropout_prob,
                            out_channels=out_ch,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, use_conv, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_planes == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_planes
            dim_head = num_head_planes
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_planes,
                dropout_prob,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_planes),
            ResBlock(
                ch,
                time_planes,
                dropout_prob,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(inter_planes_multi))[::-1]:
            for i in range(res_block_nums + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_planes,
                        dropout_prob,
                        out_channels=inter_planes * mult,
                    )
                ]
                ch = inter_planes * mult
                if ds in attention_resolutions:
                    if num_head_planes == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_planes
                        dim_head = num_head_planes
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                        ) if not use_spatial_transformer else
                        SpatialTransformer(ch,
                                           num_heads,
                                           dim_head,
                                           depth=transformer_depth,
                                           context_dim=context_planes))
                if level and i == res_block_nums:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_planes,
                            dropout_prob,
                            out_channels=out_ch,
                            up=True,
                        ) if resblock_updown else Upsample(
                            ch, use_conv, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            self.zero_module(
                nn.Conv2d(inter_planes,
                          out_planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True)),
        )

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()

        return module

    def forward(self, x, timesteps=None, context=None, class_label=None):
        assert (class_label is not None) == (self.num_classes is not None)

        embedded_features = self.time_mlp(timesteps)
        embedded_class = class_label
        context_features = context

        if self.num_classes is not None and class_label is not None:
            embedded_features = embedded_features + self.class_mlp(
                embedded_class)

        # t_emb = timestep_embedding(timesteps,
        #                            self.model_channels,
        #                            repeat_only=False)
        # emb = self.time_embed(t_emb)

        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0], )
        #     emb = emb + self.label_emb(y)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, embedded_features, context_features)
            hs.append(h)
        h = self.middle_block(h, embedded_features, context_features)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, embedded_features, context_features)
        h = h

        return self.out(h)


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

    # time_step = 1000
    # net = OpenaiDiffusionUNet(inplanes=3,
    #                           planes=256,
    #                           planes_multi=[1, 2, 3, 4],
    #                           time_embedding_ratio=4,
    #                           block_nums=2,
    #                           head_nums=8,
    #                           dropout_prob=0.,
    #                           num_groups=32,
    #                           use_attention_planes_multi_idx=[1, 2, 3],
    #                           resblock_updown=True,
    #                           num_classes=None,
    #                           use_context_transformer_planes_multi_idx=[],
    #                           transformer_block_nums=1,
    #                           context_planes=None,
    #                           use_gradient_checkpoint=False)
    # batch, channel, image_h, image_w = 1, 3, 64, 64
    # images = torch.randn(batch, channel, image_h, image_w).float()
    # times = torch.randint(time_step, (batch, )).float()

    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net, inputs=(images, times), verbose=False)
    # macs, params = clever_format([macs, params], '%.3f')
    # print(f'1111, macs: {macs}, params: {params}')
    # outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times))
    # print('2222', outs.shape)

    # time_step = 1000
    # num_classes = 100
    # net = OpenaiDiffusionUNet(inplanes=3,
    #                           planes=256,
    #                           planes_multi=[1, 2, 3, 4],
    #                           time_embedding_ratio=4,
    #                           block_nums=2,
    #                           head_nums=8,
    #                           dropout_prob=0.,
    #                           num_groups=32,
    #                           use_attention_planes_multi_idx=[1, 2, 3],
    #                           resblock_updown=True,
    #                           num_classes=num_classes,
    #                           use_context_transformer_planes_multi_idx=[],
    #                           transformer_block_nums=1,
    #                           context_planes=None,
    #                           use_gradient_checkpoint=False)
    # batch, channel, image_h, image_w = 1, 3, 64, 64
    # images = torch.randn(batch, channel, image_h, image_w).float()
    # times = torch.randint(time_step, (batch, )).float()
    # labels = torch.randint(num_classes, (batch, )).long()

    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net, inputs=(images, times, labels), verbose=False)
    # macs, params = clever_format([macs, params], '%.3f')
    # print(f'3333, macs: {macs}, params: {params}')
    # outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times),
    #            torch.autograd.Variable(labels))
    # print('4444', outs.shape)

    # time_step = 1000
    # num_classes = 100
    # net = OpenaiDiffusionUNet(inplanes=3,
    #                           planes=256,
    #                           planes_multi=[1, 2, 3, 4],
    #                           time_embedding_ratio=4,
    #                           block_nums=2,
    #                           head_nums=8,
    #                           dropout_prob=0.,
    #                           num_groups=32,
    #                           use_attention_planes_multi_idx=[1, 2, 3],
    #                           resblock_updown=True,
    #                           num_classes=num_classes,
    #                           use_context_transformer_planes_multi_idx=[],
    #                           transformer_block_nums=1,
    #                           context_planes=None,
    #                           use_gradient_checkpoint=True)
    # batch, channel, image_h, image_w = 1, 3, 64, 64
    # images = torch.randn(batch, channel, image_h, image_w).float()
    # times = torch.randint(time_step, (batch, )).float()
    # # labels = None
    # labels = torch.randint(num_classes, (batch, )).long()
    # # outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times))
    # outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times),
    #            torch.autograd.Variable(labels))
    # print('5555', outs.shape)

    # time_step = 1000
    # net = OpenaiDiffusionUNet(
    #     inplanes=3,
    #     planes=192,
    #     planes_multi=[1, 2, 3, 5],
    #     time_embedding_ratio=4,
    #     block_nums=2,
    #     head_nums=1,
    #     dropout_prob=0.,
    #     num_groups=32,
    #     use_attention_planes_multi_idx=[],
    #     resblock_updown=False,
    #     num_classes=None,
    #     use_context_transformer_planes_multi_idx=[1, 2, 3],
    #     transformer_block_nums=1,
    #     context_planes=512,
    #     use_gradient_checkpoint=False)
    # batch, channel, image_h, image_w = 1, 3, 64, 64
    # images = torch.randn(batch, channel, image_h, image_w).float()
    # times = torch.randint(time_step, (batch, )).float()
    # contexts = torch.randn(batch, 1024, 512).float()

    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net,
    #                        inputs=(images, times, None, contexts),
    #                        verbose=False)
    # macs, params = clever_format([macs, params], '%.3f')
    # print(f'6666, macs: {macs}, params: {params}')
    # outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times),
    #            None, torch.autograd.Variable(contexts))
    # print('7777', outs.shape)

    # time_step = 1000
    # num_classes = 100
    # net = OpenaiDiffusionUNet(
    #     inplanes=3,
    #     planes=192,
    #     planes_multi=[1, 2, 3, 5],
    #     time_embedding_ratio=4,
    #     block_nums=2,
    #     head_nums=1,
    #     dropout_prob=0.,
    #     num_groups=32,
    #     use_attention_planes_multi_idx=[],
    #     resblock_updown=False,
    #     num_classes=num_classes,
    #     use_context_transformer_planes_multi_idx=[1, 2, 3],
    #     transformer_block_nums=1,
    #     context_planes=512,
    #     use_gradient_checkpoint=False)
    # batch, channel, image_h, image_w = 1, 3, 64, 64
    # images = torch.randn(batch, channel, image_h, image_w).float()
    # times = torch.randint(time_step, (batch, )).float()
    # labels = torch.randint(num_classes, (batch, )).long()
    # contexts = torch.randn(batch, 1024, 512).float()

    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net,
    #                        inputs=(images, times, labels, contexts),
    #                        verbose=False)
    # macs, params = clever_format([macs, params], '%.3f')
    # print(f'8888, macs: {macs}, params: {params}')
    # outs = net(torch.autograd.Variable(images), torch.autograd.Variable(times),
    #            torch.autograd.Variable(labels),
    #            torch.autograd.Variable(contexts))
    # print('9999', outs.shape)