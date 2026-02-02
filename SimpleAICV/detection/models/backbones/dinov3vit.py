import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import math
import numpy as np

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from SimpleAICV.detection.models.backbones.vit import VitPyramidNeck
from SimpleAICV.detection.common import load_state_dict

__all__ = [
    'dinov3_vit_small_patch16_backbone',
    'dinov3_vit_small_plus_patch16_backbone',
    'dinov3_vit_base_patch16_backbone',
    'dinov3_vit_large_patch16_backbone',
    'dinov3_vit_large_plus_patch16_backbone',
    'dinov3_vit_huge_plus_patch16_backbone',
    'dinov3_vit_7b_patch16_backbone',
]


class PatchEmbed(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=768,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 has_norm=False):
        super(PatchEmbed, self).__init__()
        self.planes = planes

        self.proj = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = nn.LayerNorm(planes,
                                 eps=1e-6) if has_norm else nn.Identity()

        k = 1 / (inplanes * (kernel_size**2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))

    def forward(self, x):
        # B C H W
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        # B HW C
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # B H W C
        x = x.reshape(-1, H, W, self.planes)

        return x


class LayerScale(nn.Module):

    def __init__(self, inplanes, init_values=1e-5, inplace=False):
        super(LayerScale, self).__init__()
        self.inplace = inplace

        self.gamma = nn.Parameter(torch.empty(inplanes))

        nn.init.constant_(self.gamma, init_values)

    def forward(self, x):
        if self.inplace:
            x = x.mul_(self.gamma)
        else:
            x = x * self.gamma

        return x


class Mlp(nn.Module):

    def __init__(self,
                 inplanes,
                 hidden_planes,
                 planes,
                 drop_prob=0.0,
                 bias=True):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(inplanes, hidden_planes, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_planes, planes, bias=bias)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class SwiGLUFFN(nn.Module):

    def __init__(self,
                 inplanes,
                 hidden_planes,
                 planes,
                 drop_prob=0.0,
                 bias=True,
                 align_to=8):
        super(SwiGLUFFN, self).__init__()
        d = int(hidden_planes * 2 / 3)
        swiglu_hidden_planes = d + (-d % align_to)

        self.w1 = nn.Linear(inplanes, swiglu_hidden_planes, bias=bias)
        self.w2 = nn.Linear(inplanes, swiglu_hidden_planes, bias=bias)
        self.w3 = nn.Linear(swiglu_hidden_planes, planes, bias=bias)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2

        out = self.w3(hidden)

        return out


class RopePositionEmbedding(nn.Module):

    def __init__(self,
                 embedding_planes,
                 head_nums,
                 base=100.0,
                 min_period=None,
                 max_period=None,
                 normalize_coords="separate",
                 shift_coords=None,
                 jitter_coords=None,
                 rescale_coords=None):
        super(RopePositionEmbedding, self).__init__()
        assert normalize_coords in ["min", "max", "separate"]
        assert embedding_planes % (4 * head_nums) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None
                                                   and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided.")

        head_planes = embedding_planes // head_nums

        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.register_buffer("periods",
                             torch.empty(head_planes // 4),
                             persistent=True)

        if base is not None:
            # [D//4]
            periods = base**(2 * torch.arange(head_planes // 4) /
                             (head_planes // 2))
        else:
            base = self.max_period / self.min_period
            # [D//4] range [0, 1]
            exponents = torch.linspace(0, 1, head_planes // 4)
            # range [1, max_period / min_period]
            periods = base**exponents
            # range [min_period / max_period, 1]
            periods = periods / base
            # range [min_period, max_period]
            periods = periods * self.max_period

        self.periods.data = periods

    def forward(self, H, W):
        device = self.periods.device

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            # [H]
            coords_h = torch.arange(0.5, H) / max_HW
            # [W]
            coords_w = torch.arange(0.5, W) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            # [H]
            coords_h = torch.arange(0.5, H) / min_HW
            # [W]
            coords_w = torch.arange(0.5, W) / min_HW
        elif self.normalize_coords == "separate":
            # [H]
            coords_h = torch.arange(0.5, H) / H
            # [W]
            coords_w = torch.arange(0.5, W) / W

        coords_h = coords_h.to(device)
        coords_w = coords_w.to(device)

        # [H, W, 2]
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"),
                             dim=-1)
        # [HW, 2]
        coords = coords.flatten(0, 1)
        # Shift range [0, 1] to [-1, +1]
        coords = 2.0 * coords - 1.0

        coords = coords.to(device)

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2).uniform_(-self.shift_coords,
                                               self.shift_coords)
            coords += shift_hw[None, :]
            shift_hw = shift_hw.to(device)

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]
            jitter_hw = jitter_hw.to(device)

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1).uniform_(rescale_min,
                                                 rescale_max).exp()
            rescale_hw = rescale_hw.to(device)
            coords *= rescale_hw

        # Prepare angles and sin/cos
        # [HW, 2, D//4]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        # [HW, D//2]
        angles = angles.flatten(1, 2)
        # [HW, D]
        angles = angles.tile(2)
        # [HW, D]
        cos = torch.cos(angles)
        # [HW, D]
        sin = torch.sin(angles)
        # 2 * [HW, D]

        return (sin, cos)


def rope_rotate_half(x):
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x, sin, cos):
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):

    def __init__(self, *args, **kwargs):
        super(LinearKMaskedBias, self).__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask",
                                 torch.full_like(self.bias, fill_value=1))
            self.bias_mask[o // 3:2 * o // 3].fill_(0)

    def forward(self, input):
        masked_bias = self.bias * self.bias_mask.to(
            self.bias.dtype) if self.bias is not None else None

        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums=8,
                 qkv_bias=False,
                 proj_bias=True,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super(SelfAttention, self).__init__()
        self.head_nums = head_nums
        head_planes = inplanes // head_nums
        self.scale = head_planes**-0.5

        self.qkv = LinearKMaskedBias(inplanes, inplanes * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inplanes, inplanes, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None, rope=None):
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv,
                                        attn_bias=attn_bias,
                                        rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)

        return x

    def compute_attention(self, qkv, attn_bias=None, rope=None):
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.head_nums, C // self.head_nums)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)

        return x.reshape([B, N, C])

    def apply_rope(self, q, k, rope):
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)

        return q, k


class SelfAttentionBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 ffn_ratio=4.0,
                 qkv_bias=False,
                 proj_bias=True,
                 ffn_bias=True,
                 drop=0.0,
                 attn_drop=0.0,
                 init_values=None,
                 drop_path=0.0,
                 ffn_layer=Mlp):
        super(SelfAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(inplanes, eps=1e-6)
        self.attn = SelfAttention(inplanes,
                                  head_nums=head_nums,
                                  qkv_bias=qkv_bias,
                                  proj_bias=proj_bias,
                                  attn_drop=attn_drop,
                                  proj_drop=drop)

        self.ls1 = LayerScale(
            inplanes=inplanes,
            init_values=init_values) if init_values else nn.Identity()

        self.norm2 = nn.LayerNorm(inplanes, eps=1e-6)
        mlp_hidden_planes = int(inplanes * ffn_ratio)
        self.mlp = ffn_layer(inplanes=inplanes,
                             hidden_planes=mlp_hidden_planes,
                             planes=inplanes,
                             drop_prob=drop,
                             bias=ffn_bias)

        self.ls2 = LayerScale(
            inplanes=inplanes,
            init_values=init_values) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope, indices):
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            # [batch, heads, patches, embed_dim]
            return sin[indices], cos[indices]
        else:
            # No batch dimension, do not index
            # [heads, patches, embed_dim] or [patches, embed_dim]
            return sin, cos

    def forward(self, x, rope=None):
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b,
                                        device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b,
                                        device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn


class DinoVisionTransformer(nn.Module):

    def __init__(self,
                 patch_size=16,
                 inplanes=3,
                 embedding_planes=768,
                 pos_embed_rope_base=100.0,
                 pos_embed_rope_min_period=None,
                 pos_embed_rope_max_period=None,
                 pos_embed_rope_normalize_coords="separate",
                 pos_embed_rope_shift_coords=None,
                 pos_embed_rope_jitter_coords=None,
                 pos_embed_rope_rescale_coords=None,
                 block_nums=12,
                 head_nums=12,
                 ffn_ratio=4.0,
                 qkv_bias=True,
                 drop_path_rate=0.,
                 layerscale_init=1e-5,
                 ffn_layer="mlp",
                 ffn_bias=True,
                 proj_bias=True,
                 use_gradient_checkpoint=False):
        super(DinoVisionTransformer, self).__init__()
        assert pos_embed_rope_normalize_coords in ["min", "max", "separate"]
        self.patch_size = patch_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.patch_embed = PatchEmbed(inplanes=inplanes,
                                      planes=embedding_planes,
                                      kernel_size=patch_size,
                                      stride=patch_size,
                                      padding=0,
                                      has_norm=False)

        self.rope_embed = RopePositionEmbedding(
            embedding_planes=embedding_planes,
            head_nums=head_nums,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords)

        ffn_layer_dict = {
            "mlp": Mlp,
            "swiglu": SwiGLUFFN,
            "swiglu64": partial(SwiGLUFFN, align_to=64),
        }
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * block_nums
        blocks_list = [
            SelfAttentionBlock(inplanes=embedding_planes,
                               head_nums=head_nums,
                               ffn_ratio=ffn_ratio_sequence[i],
                               qkv_bias=qkv_bias,
                               proj_bias=proj_bias,
                               ffn_bias=ffn_bias,
                               init_values=layerscale_init,
                               drop_path=drop_path_rate,
                               ffn_layer=ffn_layer_cls)
            for i in range(block_nums)
        ]
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = nn.LayerNorm(embedding_planes, eps=1e-6)

        self.out_channels = embedding_planes

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch, _, origin_h, origin_w = x.shape

        x = self.patch_embed(x)

        _, H, W, _ = x.shape
        x = x.flatten(1, 2)

        rope_sincos = self.rope_embed(H=H, W=W)

        for _, block in enumerate(self.blocks):
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, rope_sincos, use_reentrant=False)
            else:
                x = block(x, rope_sincos)

        x = self.norm(x)

        x = x.reshape(batch, origin_h // self.patch_size,
                      origin_w // self.patch_size, -1).permute(0, 3, 1,
                                                               2).contiguous()

        return x


def _dinov3vitbackbone(patch_size,
                       embedding_planes,
                       pos_embed_rope_normalize_coords,
                       pos_embed_rope_rescale_coords,
                       block_nums,
                       head_nums,
                       ffn_ratio,
                       qkv_bias,
                       ffn_layer,
                       pretrained_path='',
                       **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embedding_planes=embedding_planes,
        pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
        pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        block_nums=block_nums,
        head_nums=head_nums,
        ffn_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        ffn_layer=ffn_layer,
        **kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def dinov3_vit_small_patch16_backbone(patch_size=16,
                                      pretrained_path='',
                                      **kwargs):
    return _dinov3vitbackbone(patch_size=patch_size,
                              embedding_planes=384,
                              pos_embed_rope_normalize_coords="separate",
                              pos_embed_rope_rescale_coords=2,
                              block_nums=12,
                              head_nums=6,
                              ffn_ratio=4,
                              qkv_bias=True,
                              ffn_layer="mlp",
                              pretrained_path=pretrained_path,
                              **kwargs)


def dinov3_vit_small_plus_patch16_backbone(patch_size=16,
                                           pretrained_path='',
                                           **kwargs):
    return _dinov3vitbackbone(patch_size=patch_size,
                              embedding_planes=384,
                              pos_embed_rope_normalize_coords="separate",
                              pos_embed_rope_rescale_coords=2,
                              block_nums=12,
                              head_nums=6,
                              ffn_ratio=6,
                              qkv_bias=True,
                              ffn_layer="swiglu",
                              pretrained_path=pretrained_path,
                              **kwargs)


def dinov3_vit_base_patch16_backbone(patch_size=16,
                                     pretrained_path='',
                                     **kwargs):
    return _dinov3vitbackbone(patch_size=patch_size,
                              embedding_planes=768,
                              pos_embed_rope_normalize_coords="separate",
                              pos_embed_rope_rescale_coords=2,
                              block_nums=12,
                              head_nums=12,
                              ffn_ratio=4,
                              qkv_bias=True,
                              ffn_layer="mlp",
                              pretrained_path=pretrained_path,
                              **kwargs)


def dinov3_vit_large_patch16_backbone(patch_size=16,
                                      pretrained_path='',
                                      **kwargs):
    return _dinov3vitbackbone(patch_size=patch_size,
                              embedding_planes=1024,
                              pos_embed_rope_normalize_coords="separate",
                              pos_embed_rope_rescale_coords=2,
                              block_nums=24,
                              head_nums=16,
                              ffn_ratio=4,
                              qkv_bias=True,
                              ffn_layer="mlp",
                              pretrained_path=pretrained_path,
                              **kwargs)


def dinov3_vit_large_plus_patch16_backbone(patch_size=16,
                                           pretrained_path='',
                                           **kwargs):
    return _dinov3vitbackbone(patch_size=patch_size,
                              embedding_planes=1024,
                              pos_embed_rope_normalize_coords="separate",
                              pos_embed_rope_rescale_coords=2,
                              block_nums=24,
                              head_nums=16,
                              ffn_ratio=6,
                              qkv_bias=True,
                              ffn_layer="swiglu",
                              pretrained_path=pretrained_path,
                              **kwargs)


def dinov3_vit_huge_plus_patch16_backbone(patch_size=16,
                                          pretrained_path='',
                                          **kwargs):
    return _dinov3vitbackbone(patch_size=patch_size,
                              embedding_planes=1280,
                              pos_embed_rope_normalize_coords="separate",
                              pos_embed_rope_rescale_coords=2,
                              block_nums=32,
                              head_nums=20,
                              ffn_ratio=6,
                              qkv_bias=True,
                              ffn_layer="swiglu",
                              pretrained_path=pretrained_path,
                              **kwargs)


def dinov3_vit_7b_patch16_backbone(patch_size=16,
                                   pretrained_path='',
                                   **kwargs):
    return _dinov3vitbackbone(patch_size=patch_size,
                              embedding_planes=4096,
                              pos_embed_rope_normalize_coords="separate",
                              pos_embed_rope_rescale_coords=2,
                              block_nums=40,
                              head_nums=32,
                              ffn_ratio=3,
                              qkv_bias=False,
                              ffn_layer="swiglu64",
                              pretrained_path=pretrained_path,
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

    net = dinov3_vit_small_patch16_backbone()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )

    net = dinov3_vit_small_plus_patch16_backbone()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )

    net = dinov3_vit_base_patch16_backbone()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )

    net = dinov3_vit_base_patch16_backbone(use_gradient_checkpoint=True)
    image_h, image_w = 1024, 1024
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print('1111', out.shape)

    net = dinov3_vit_large_patch16_backbone()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )

    net = dinov3_vit_large_plus_patch16_backbone()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )

    net = dinov3_vit_huge_plus_patch16_backbone()
    image_h, image_w = 1024, 1024
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {out.shape}'
    )
