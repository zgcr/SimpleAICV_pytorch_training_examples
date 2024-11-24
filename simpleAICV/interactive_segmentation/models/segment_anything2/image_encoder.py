import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

# sam2_hiera_t.yaml
#   image_encoder:
#     _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
#     scalp: 1
#     trunk:
#       _target_: sam2.modeling.backbones.hieradet.Hiera
#       embed_dim: 96
#       num_heads: 1
#       stages: [1, 2, 7, 2]
#       global_att_blocks: [5, 7, 9]
#       window_pos_embed_bkg_spatial_size: [7, 7]
#     neck:
#       _target_: sam2.modeling.backbones.image_encoder.FpnNeck
#       position_encoding:
#         _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
#         num_pos_feats: 256
#         normalize: true
#         scale: null
#         temperature: 10000
#       d_model: 256
#       backbone_channel_list: [768, 384, 192, 96]
#       fpn_top_down_levels: [2, 3]  # output level 0 and 1 directly use the backbone features
#       fpn_interp_model: nearest

# sam2_hiera_s.yaml
#   image_encoder:
#     _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
#     scalp: 1
#     trunk:
#       _target_: sam2.modeling.backbones.hieradet.Hiera
#       embed_dim: 96
#       num_heads: 1
#       stages: [1, 2, 11, 2]
#       global_att_blocks: [7, 10, 13]
#       window_pos_embed_bkg_spatial_size: [7, 7]
#     neck:
#       _target_: sam2.modeling.backbones.image_encoder.FpnNeck
#       position_encoding:
#         _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
#         num_pos_feats: 256
#         normalize: true
#         scale: null
#         temperature: 10000
#       d_model: 256
#       backbone_channel_list: [768, 384, 192, 96]
#       fpn_top_down_levels: [2, 3]  # output level 0 and 1 directly use the backbone features
#       fpn_interp_model: nearest

# sam2_hiera_b+.yaml
#   image_encoder:
#     _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
#     scalp: 1
#     trunk:
#       _target_: sam2.modeling.backbones.hieradet.Hiera
#       embed_dim: 112
#       num_heads: 2
#     neck:
#       _target_: sam2.modeling.backbones.image_encoder.FpnNeck
#       position_encoding:
#         _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
#         num_pos_feats: 256
#         normalize: true
#         scale: null
#         temperature: 10000
#       d_model: 256
#       backbone_channel_list: [896, 448, 224, 112]
#       fpn_top_down_levels: [2, 3]  # output level 0 and 1 directly use the backbone features
#       fpn_interp_model: nearest

# sam2_hiera_l.yaml
#   image_encoder:
#     _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
#     scalp: 1
#     trunk:
#       _target_: sam2.modeling.backbones.hieradet.Hiera
#       embed_dim: 144
#       num_heads: 2
#       stages: [2, 6, 36, 4]
#       global_att_blocks: [23, 33, 43]
#       window_pos_embed_bkg_spatial_size: [7, 7]
#       window_spec: [8, 4, 16, 8]
#     neck:
#       _target_: sam2.modeling.backbones.image_encoder.FpnNeck
#       position_encoding:
#         _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
#         num_pos_feats: 256
#         normalize: true
#         scale: null
#         temperature: 10000
#       d_model: 256
#       backbone_channel_list: [1152, 576, 288, 144]
#       fpn_top_down_levels: [2, 3]  # output level 0 and 1 directly use the backbone features
#       fpn_interp_model: nearest


class PatchEmbed(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=768,
                 kernel_size=7,
                 stride=4,
                 padding=3):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        x = self.proj(x)

        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)

        return x


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x


class MLP(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, layer_nums):
        super(MLP, self).__init__()
        self.layer_nums = layer_nums

        h = [hidden_planes] * (layer_nums - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([inplanes] + h, h + [planes]))

        self.act = nn.GELU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.layer_nums - 1 else layer(x)

        return x


class MultiScaleAttention(nn.Module):

    def __init__(self, inplanes, planes, head_nums, pool_stride=None):
        super(MultiScaleAttention, self).__init__()
        self.head_nums = head_nums
        head_planes = planes // head_nums
        self.scale = head_planes**-0.5

        self.pool, self.pool_stride = None, pool_stride
        if self.pool_stride:
            self.pool = nn.MaxPool2d(kernel_size=pool_stride,
                                     stride=pool_stride,
                                     ceil_mode=False)

        self.qkv = nn.Linear(inplanes, planes * 3)
        self.proj = nn.Linear(planes, planes)

    def forward(self, x):
        B, H, W, _ = x.shape

        qkv = self.qkv(x)
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = qkv.reshape(B, H * W, 3, self.head_nums, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.pool:
            q = q.reshape(B, H, W, -1)
            # (B, H, W, C) -> (B, C, H, W)
            q = q.permute(0, 3, 1, 2)
            q = self.pool(q)
            # (B, C, H', W') -> (B, H', W', C)
            q = q.permute(0, 2, 3, 1)
            H, W = q.shape[1], q.shape[2]
            q = q.reshape(B, H * W, self.head_nums, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 head_nums,
                 mlp_ratio=4.0,
                 pool_stride=None,
                 window_size=0):
        super(MultiScaleBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(inplanes, eps=1e-6)

        self.pool, self.pool_stride = None, pool_stride
        if self.pool_stride:
            self.pool = nn.MaxPool2d(kernel_size=pool_stride,
                                     stride=pool_stride,
                                     ceil_mode=False)

        self.attn = MultiScaleAttention(inplanes=inplanes,
                                        planes=planes,
                                        head_nums=head_nums,
                                        pool_stride=pool_stride)

        self.norm2 = nn.LayerNorm(planes, eps=1e-6)

        self.mlp = MLP(inplanes=planes,
                       hidden_planes=int(planes * mlp_ratio),
                       planes=planes,
                       layer_nums=2)

        if inplanes != planes:
            self.proj = nn.Linear(inplanes, planes)

    def forward(self, x):
        # B, H, W, C
        shortcut = x

        x = self.norm1(x)

        # Skip connection
        if self.inplanes != self.planes and self.pool:
            shortcut = self.proj(x)
            # (B, H, W, C) -> (B, C, H, W)
            shortcut = shortcut.permute(0, 3, 1, 2)
            shortcut = self.pool(shortcut)
            # (B, C, H', W') -> (B, H', W', C)
            shortcut = shortcut.permute(0, 2, 3, 1)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.pool_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.pool_stride
            H, W = shortcut.shape[1], shortcut.shape[2]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Hiera(nn.Module):
    """
    https://github.com/facebookresearch/hiera
    """

    def __init__(self,
                 inplanes=3,
                 embedding_planes=112,
                 head_nums=2,
                 block_nums=[2, 3, 16, 3],
                 window_position_embedding_bkg_spatial_size=[14, 14],
                 window_specification=[8, 4, 14, 7],
                 global_attention_blocks=[12, 16, 20],
                 use_gradient_checkpoint=False):
        super(Hiera, self).__init__()
        assert len(block_nums) == 4
        assert len(block_nums) == len(window_specification)
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.stage_end_idxs = [
            sum(block_nums[:i]) - 1 for i in range(1,
                                                   len(block_nums) + 1)
        ]
        self.pool_block_idxs = [x + 1 for x in self.stage_end_idxs[:-1]]

        self.patch_embed = PatchEmbed(inplanes=inplanes,
                                      planes=embedding_planes,
                                      kernel_size=7,
                                      stride=4,
                                      padding=3)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, embedding_planes,
                        window_position_embedding_bkg_spatial_size[0],
                        window_position_embedding_bkg_spatial_size[1]))
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embedding_planes, window_specification[0],
                        window_specification[0]))

        current_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(sum(block_nums)):
            out_planes = embedding_planes
            block_head_nums = head_nums

            # first block of next stage use an initial window size of previous stage and final window size of current stage
            window_size = window_specification[current_stage - 1]
            window_size = 0 if i in global_attention_blocks else window_size

            if i - 1 in self.stage_end_idxs:
                # for per stage,out_planes mul 2 and block head nums mul 2
                out_planes = int(embedding_planes * 2)
                block_head_nums = int(block_head_nums * 2)
                current_stage += 1

            block = MultiScaleBlock(
                inplanes=embedding_planes,
                planes=out_planes,
                head_nums=block_head_nums,
                mlp_ratio=4.0,
                pool_stride=2 if i in self.pool_block_idxs else None,
                window_size=window_size)
            self.blocks.append(block)

            embedding_planes = out_planes
            head_nums = block_head_nums

        self.out_channels = [
            self.blocks[i].planes for i in self.stage_end_idxs[::-1]
        ]

    def forward(self, x):
        x = self.patch_embed(x)

        h, w = x.shape[1], x.shape[2]
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + self.pos_embed_window.tile([
            x // y
            for x, y in zip(pos_embed.shape, self.pos_embed_window.shape)
        ])
        pos_embed = pos_embed.permute(0, 2, 3, 1)

        x = x + pos_embed

        outs = []
        for idx, block in enumerate(self.blocks):
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, use_reentrant=True)
            else:
                x = block(x)

            if idx in self.stage_end_idxs:
                feature = x.permute(0, 3, 1, 2)
                outs.append(feature)

        return outs


class PositionEmbeddingBlock(nn.Module):

    def __init__(self, inplanes=128, temperature=10000, eps=1e-6):
        super(PositionEmbeddingBlock, self).__init__()
        self.inplanes = inplanes
        self.temperature = temperature
        self.eps = eps
        self.scale = 2 * math.pi

    def forward(self, x):
        device = x.device

        y_embed = (torch.arange(1, x.shape[-2] + 1,
                                dtype=torch.float32).to(device).view(
                                    1, -1, 1).repeat(x.shape[0], 1,
                                                     x.shape[-1]))
        x_embed = (torch.arange(1, x.shape[-1] + 1,
                                dtype=torch.float32).to(device).view(
                                    1, 1, -1).repeat(x.shape[0], x.shape[-2],
                                                     1))
        # normalize
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        planes_t = torch.arange(self.inplanes,
                                dtype=torch.float32,
                                device=device)
        planes_t = self.temperature**(2 * (planes_t // 2) / self.inplanes)

        pos_x = x_embed[:, :, :, None] / planes_t
        pos_y = y_embed[:, :, :, None] / planes_t

        pos_x = torch.stack(
            (torch.sin(pos_x[:, :, :, 0::2]), torch.cos(pos_x[:, :, :, 1::2])),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (torch.sin(pos_y[:, :, :, 0::2]), torch.cos(pos_y[:, :, :, 1::2])),
            dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        pos = pos.to(x.dtype)

        return pos


class FpnNeck(nn.Module):

    def __init__(self, inplanes_list=[896, 448, 224, 112], planes=256):
        super(FpnNeck, self).__init__()
        self.position_encoding = PositionEmbeddingBlock(inplanes=planes // 2,
                                                        temperature=10000,
                                                        eps=1e-6)

        self.convs = nn.ModuleList()
        for inplanes in inplanes_list:
            current_conv = nn.Sequential()
            current_conv.add_module(
                "conv",
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True))
            self.convs.append(current_conv)

    def forward(self, inputs):
        x0, x1, x2, x3 = inputs

        x3 = self.convs[0](x3)
        position_x3 = self.position_encoding(x3)

        x2 = self.convs[1](x2)
        x3_up = F.interpolate(x3,
                              size=(x2.shape[2], x2.shape[3]),
                              mode='nearest')
        x2 = x2 + x3_up
        position_x2 = self.position_encoding(x2)

        x1 = self.convs[2](x1)
        position_x1 = self.position_encoding(x1)

        x0 = self.convs[3](x0)
        position_x0 = self.position_encoding(x0)

        features = [x0, x1, x2, x3]
        positions = [position_x0, position_x1, position_x2, position_x3]

        return features, positions


class ImageEncoder(nn.Module):

    def __init__(self,
                 inplanes=3,
                 embedding_planes=112,
                 head_nums=2,
                 block_nums=[2, 3, 16, 3],
                 window_position_embedding_bkg_spatial_size=[14, 14],
                 window_specification=[8, 4, 14, 7],
                 global_attention_blocks=[12, 16, 20],
                 fpn_planes=256,
                 use_gradient_checkpoint=False):
        super(ImageEncoder, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.trunk = Hiera(inplanes=inplanes,
                           embedding_planes=embedding_planes,
                           head_nums=head_nums,
                           block_nums=block_nums,
                           window_position_embedding_bkg_spatial_size=
                           window_position_embedding_bkg_spatial_size,
                           window_specification=window_specification,
                           global_attention_blocks=global_attention_blocks,
                           use_gradient_checkpoint=use_gradient_checkpoint)
        self.neck = FpnNeck(inplanes_list=self.trunk.out_channels,
                            planes=fpn_planes)

    def forward(self, inputs):
        if self.use_gradient_checkpoint:
            features = checkpoint(self.trunk, inputs, use_reentrant=True)
        else:
            features = self.trunk(inputs)

        if self.use_gradient_checkpoint:
            features, positions = checkpoint(self.neck,
                                             features,
                                             use_reentrant=True)
        else:
            features, positions = self.neck(features)

        features, positions = features[:-1], positions[:-1]

        return features, positions


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

    net = ImageEncoder(inplanes=3,
                       embedding_planes=96,
                       head_nums=1,
                       block_nums=[1, 2, 7, 2],
                       window_position_embedding_bkg_spatial_size=[7, 7],
                       window_specification=[8, 4, 14, 7],
                       global_attention_blocks=[5, 7, 9],
                       fpn_planes=256,
                       use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = ImageEncoder(inplanes=3,
                       embedding_planes=96,
                       head_nums=1,
                       block_nums=[1, 2, 7, 2],
                       window_position_embedding_bkg_spatial_size=[7, 7],
                       window_specification=[8, 4, 14, 7],
                       global_attention_blocks=[5, 7, 9],
                       fpn_planes=256,
                       use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = ImageEncoder(inplanes=3,
                       embedding_planes=96,
                       head_nums=1,
                       block_nums=[1, 2, 11, 2],
                       window_position_embedding_bkg_spatial_size=[7, 7],
                       window_specification=[8, 4, 14, 7],
                       global_attention_blocks=[7, 10, 13],
                       fpn_planes=256,
                       use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = ImageEncoder(inplanes=3,
                       embedding_planes=112,
                       head_nums=2,
                       block_nums=[2, 3, 16, 3],
                       window_position_embedding_bkg_spatial_size=[14, 14],
                       window_specification=[8, 4, 14, 7],
                       global_attention_blocks=[12, 16, 20],
                       fpn_planes=256,
                       use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)

    net = ImageEncoder(inplanes=3,
                       embedding_planes=144,
                       head_nums=2,
                       block_nums=[2, 6, 36, 4],
                       window_position_embedding_bkg_spatial_size=[7, 7],
                       window_specification=[8, 4, 16, 8],
                       global_attention_blocks=[23, 33, 43],
                       fpn_planes=256,
                       use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    for per_out1, per_out2 in zip(outs[0], outs[1]):
        print(f'2222', per_out1.shape, per_out2.shape)
