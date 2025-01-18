import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class PatchEmbed(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=768,
                 kernel_size=16,
                 stride=16,
                 padding=0):
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


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):

    def __init__(self, inplanes, head_nums=8, input_size=None):
        super(Attention, self).__init__()
        self.head_nums = head_nums
        head_planes = inplanes // head_nums
        self.scale = head_planes**-0.5

        self.qkv = nn.Linear(inplanes, inplanes * 3)
        self.proj = nn.Linear(inplanes, inplanes)

        assert (
            input_size is not None
        ), "Input size must be provided if using relative positional encoding."
        # initialize relative positional embeddings
        self.rel_pos_h = nn.Parameter(
            torch.zeros(2 * input_size[0] - 1, head_planes))
        self.rel_pos_w = nn.Parameter(
            torch.zeros(2 * input_size[1] - 1, head_planes))

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.head_nums,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.head_nums, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w,
                                      (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.head_nums, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MLPBlock(nn.Module):

    def __init__(self, inplanes, mlp_planes):
        super(MLPBlock, self).__init__()
        self.lin1 = nn.Linear(inplanes, mlp_planes)
        self.lin2 = nn.Linear(mlp_planes, inplanes)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.lin2(self.act(self.lin1(x)))

        return x


class Block(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 mlp_ratio=4.0,
                 input_size=None,
                 window_size=0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(inplanes, eps=1e-6)
        self.attn = Attention(inplanes=inplanes,
                              head_nums=head_nums,
                              input_size=input_size if window_size == 0 else
                              (window_size, window_size))

        self.norm2 = nn.LayerNorm(inplanes, eps=1e-6)
        self.mlp = MLPBlock(inplanes=inplanes,
                            mlp_planes=int(inplanes * mlp_ratio))

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class LayerNorm2d(nn.Module):

    def __init__(self, inplanes, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(inplanes))
        self.bias = nn.Parameter(torch.zeros(inplanes))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        return x


class ViTImageEncoder(nn.Module):

    def __init__(self,
                 image_size=1024,
                 patch_size=16,
                 inplanes=3,
                 embedding_planes=768,
                 block_nums=12,
                 head_nums=12,
                 mlp_ratio=4,
                 out_planes=256,
                 window_size=0,
                 global_attn_indexes=(),
                 use_gradient_checkpoint=False):
        super(ViTImageEncoder, self).__init__()
        self.image_size = image_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.patch_embed = PatchEmbed(inplanes=inplanes,
                                      planes=embedding_planes,
                                      kernel_size=patch_size,
                                      stride=patch_size,
                                      padding=0)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, image_size // patch_size, image_size // patch_size,
                        embedding_planes))

        blocks = []
        for i in range(block_nums):
            block = Block(
                inplanes=embedding_planes,
                head_nums=head_nums,
                mlp_ratio=mlp_ratio,
                input_size=(image_size // patch_size,
                            image_size // patch_size),
                window_size=window_size if i not in global_attn_indexes else 0)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.neck = nn.Sequential(
            nn.Conv2d(embedding_planes,
                      out_planes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), LayerNorm2d(out_planes),
            nn.Conv2d(out_planes,
                      out_planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), LayerNorm2d(out_planes))

    def forward(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed

        for block in self.blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, use_reentrant=True)
            else:
                x = block(x)

        x = x.permute(0, 3, 1, 2)

        if self.use_gradient_checkpoint:
            x = checkpoint(self.neck, x, use_reentrant=True)
        else:
            x = self.neck(x)

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

    net = ViTImageEncoder(image_size=1024,
                          patch_size=16,
                          inplanes=3,
                          embedding_planes=768,
                          block_nums=12,
                          head_nums=12,
                          mlp_ratio=4,
                          out_planes=256,
                          window_size=14,
                          global_attn_indexes=[2, 5, 8, 11],
                          use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = ViTImageEncoder(image_size=1024,
                          patch_size=16,
                          inplanes=3,
                          embedding_planes=768,
                          block_nums=12,
                          head_nums=12,
                          mlp_ratio=4,
                          out_planes=256,
                          window_size=14,
                          global_attn_indexes=[2, 5, 8, 11],
                          use_gradient_checkpoint=True)
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'2222, out_shape: {out.shape}')

    net = ViTImageEncoder(image_size=1024,
                          patch_size=16,
                          inplanes=3,
                          embedding_planes=1024,
                          block_nums=24,
                          head_nums=16,
                          mlp_ratio=4,
                          out_planes=256,
                          window_size=14,
                          global_attn_indexes=[5, 11, 17, 23],
                          use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = ViTImageEncoder(image_size=1024,
                          patch_size=16,
                          inplanes=3,
                          embedding_planes=1280,
                          block_nums=32,
                          head_nums=16,
                          mlp_ratio=4,
                          out_planes=256,
                          window_size=14,
                          global_attn_indexes=[7, 15, 23, 31],
                          use_gradient_checkpoint=False)
    image_h, image_w = 1024, 1024
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')
