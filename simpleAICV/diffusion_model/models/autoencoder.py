import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from simpleAICV.diffusion_model.models.diffusion_unet import UpSampleBlock, DownSampleBlock, AttentionBlock


class ResBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 num_groups=32,
                 dropout_prob=0.,
                 use_conv_shortcut=False,
                 use_attention=False):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
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

    def forward(self, x):
        inputs = x

        x = self.block1(x)

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


class Encoder(nn.Module):

    def __init__(self,
                 enccoder_inplanes=3,
                 planes=128,
                 encoder_outplanes=3,
                 double_encoder_out_planes=False,
                 planes_multi=[1, 2, 4, 4],
                 block_nums=2,
                 dropout_prob=0.,
                 num_groups=32,
                 use_attention_planes_multi_idx=[],
                 use_gradient_checkpoint=False):
        super(self, Encoder).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.stem = nn.Conv2d(enccoder_inplanes,
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
                        num_groups=num_groups,
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
                     num_groups=num_groups,
                     dropout_prob=dropout_prob,
                     use_conv_shortcut=False,
                     use_attention=True),
            ResBlock(current_planes,
                     current_planes,
                     num_groups=num_groups,
                     dropout_prob=dropout_prob,
                     use_conv_shortcut=False,
                     use_attention=False),
        ])

        self.last_norm = nn.GroupNorm(num_groups, current_planes)
        self.last_act = nn.SiLU()
        self.last_conv = nn.Conv2d(
            current_planes,
            2 * encoder_outplanes
            if double_encoder_out_planes else encoder_outplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True)

    def forward(self, x):
        x = self.stem(x)

        # 下采样
        downsample_features = [x]
        for per_layer in self.downsample_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(per_layer, x)
            else:
                x = per_layer(x)
            downsample_features.append(x)

        # 中间层
        for per_layer in self.middle_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(per_layer, x)
            else:
                x = per_layer(x)

        x = self.last_norm(x)
        x = self.last_act(x)
        x = self.last_conv(x)

        return x


class Decoder(nn.Module):

    def __init__(self,
                 decoder_inplanes=3,
                 planes=128,
                 decoder_outplanes=3,
                 planes_multi=[1, 2, 4, 4],
                 block_nums=2,
                 dropout_prob=0.,
                 num_groups=32,
                 use_attention_planes_multi_idx=[],
                 use_gradient_checkpoint=False):
        super(self, Decoder).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint
        current_planes = planes * planes_multi[-1]

        self.stem = nn.Conv2d(decoder_inplanes,
                              current_planes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=1,
                              bias=True)

        self.middle_blocks = nn.ModuleList([
            ResBlock(current_planes,
                     current_planes,
                     num_groups=num_groups,
                     dropout_prob=dropout_prob,
                     use_conv_shortcut=False,
                     use_attention=True),
            ResBlock(current_planes,
                     current_planes,
                     num_groups=num_groups,
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
                    ResBlock(current_planes,
                             per_planes,
                             num_groups=num_groups,
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

        self.last_norm = nn.GroupNorm(num_groups, planes)
        self.last_act = nn.SiLU()
        self.last_conv = nn.Conv2d(planes,
                                   decoder_outplanes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   bias=True)

    def forward(self, x):
        # 中间层
        for per_layer in self.middle_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(per_layer, x)
            else:
                x = per_layer(x)

        # 上采样
        for per_layer in self.upsample_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(per_layer, x)
            else:
                x = per_layer(x)

        x = self.last_norm(x)
        x = self.last_act(x)
        x = self.last_conv(x)

        return x


class VQModel(nn.Module):

    def __init__(self,
                 embedding_planes=3,
                 inplanes=3,
                 planes=128,
                 middle_planes=3,
                 double_middle_planes=False,
                 outplanes=3,
                 planes_multi=[1, 2, 4],
                 block_nums=2,
                 dropout_prob=0.,
                 num_groups=32,
                 use_attention_planes_multi_idx=[],
                 use_gradient_checkpoint=False):
        super(self, VQModel).__init__()
        self.encoder = Encoder(
            enccoder_inplanes=inplanes,
            planes=planes,
            encoder_outplanes=middle_planes,
            double_encoder_out_planes=double_middle_planes,
            planes_multi=planes_multi,
            block_nums=block_nums,
            dropout_prob=dropout_prob,
            num_groups=num_groups,
            use_attention_planes_multi_idx=use_attention_planes_multi_idx,
            use_gradient_checkpoint=use_gradient_checkpoint)
        self.decoder = Decoder(
            decoder_inplanes=middle_planes,
            planes=planes,
            decoder_outplanes=outplanes,
            planes_multi=planes_multi,
            block_nums=block_nums,
            dropout_prob=dropout_prob,
            num_groups=num_groups,
            use_attention_planes_multi_idx=use_attention_planes_multi_idx,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.quant_conv = nn.Conv2d(middle_planes,
                                    embedding_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)
        self.post_quant_conv = nn.Conv2d(embedding_planes,
                                         middle_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         bias=True)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)

        return x

    def decode(self, x):
        x = self.post_quant_conv(x)
        x = self.decoder(x)

        return x


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape).to(device=self.parameters.device)

        return x


class AutoencoderKLModel(nn.Module):

    def __init__(self,
                 embedding_planes=4,
                 inplanes=3,
                 planes=128,
                 middle_planes=4,
                 double_middle_planes=True,
                 outplanes=3,
                 planes_multi=[1, 2, 4, 4],
                 block_nums=2,
                 dropout_prob=0.,
                 num_groups=32,
                 use_attention_planes_multi_idx=[],
                 use_gradient_checkpoint=False):
        super(self, AutoencoderKLModel).__init__()
        self.encoder = Encoder(
            enccoder_inplanes=inplanes,
            planes=planes,
            encoder_outplanes=middle_planes,
            double_encoder_out_planes=double_middle_planes,
            planes_multi=planes_multi,
            block_nums=block_nums,
            dropout_prob=dropout_prob,
            num_groups=num_groups,
            use_attention_planes_multi_idx=use_attention_planes_multi_idx,
            use_gradient_checkpoint=use_gradient_checkpoint)
        self.decoder = Decoder(
            decoder_inplanes=middle_planes,
            planes=planes,
            decoder_outplanes=outplanes,
            planes_multi=planes_multi,
            block_nums=block_nums,
            dropout_prob=dropout_prob,
            num_groups=num_groups,
            use_attention_planes_multi_idx=use_attention_planes_multi_idx,
            use_gradient_checkpoint=use_gradient_checkpoint)

        self.quant_conv = nn.Conv2d(
            2 * middle_planes if double_middle_planes else middle_planes,
            2 * embedding_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True)
        self.post_quant_conv = nn.Conv2d(embedding_planes,
                                         middle_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         bias=True)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)
        x = DiagonalGaussianDistribution(x)

        return x

    def decode(self, x):
        x = self.post_quant_conv(x)
        x = self.decoder(x)

        return x
