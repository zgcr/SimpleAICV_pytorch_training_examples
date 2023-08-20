'''
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class GEGLU(nn.Module):

    def __init__(self, inplanes, planes):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(inplanes, planes * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        x * F.gelu(gate)

        return x


class FeedForward(nn.Module):

    def __init__(self,
                 inplanes,
                 planes=None,
                 feed_forward_ratio=4,
                 glu=False,
                 dropout_prob=0.):
        super(FeedForward, self).__init__()
        inter_planes = int(inplanes * feed_forward_ratio)
        planes = planes if planes else inplanes

        layers = []
        if not glu:
            layers.append(nn.Linear(inplanes, inter_planes))
            layers.append(nn.GELU())
        else:
            layers.append(GEGLU(inplanes, inter_planes))
        layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(inter_planes, planes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x


class CrossAttention(nn.Module):

    def __init__(self,
                 query_planes,
                 context_planes=None,
                 head_nums=8,
                 head_planes=64,
                 dropout_prob=0.):
        super(CrossAttention, self).__init__()
        inter_planes = head_planes * head_nums
        context_planes = context_planes if context_planes else query_planes

        self.scale = head_planes**-0.5
        self.head_nums = head_nums

        self.to_q = nn.Linear(query_planes, inter_planes, bias=False)
        self.to_k = nn.Linear(context_planes, inter_planes, bias=False)
        self.to_v = nn.Linear(context_planes, inter_planes, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inter_planes, query_planes),
                                    nn.Dropout(dropout_prob))

    def forward(self, x, context):
        h = self.head_nums

        q = self.to_q(x)

        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = F.softmax(sim, dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)

        return out


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 planes,
                 head_nums,
                 head_planes,
                 context_planes,
                 dropout_prob=0.,
                 gated_ff=True):
        super(BasicTransformerBlock, self).__init__()
        # is a self-attention
        self.attn1 = CrossAttention(query_planes=planes,
                                    head_nums=head_nums,
                                    head_planes=head_planes,
                                    dropout_prob=dropout_prob)
        self.ff = FeedForward(planes, dropout_prob=dropout_prob, glu=gated_ff)
        # is self-attn if context is none
        self.attn2 = CrossAttention(query_planes=planes,
                                    context_planes=context_planes,
                                    head_nums=head_nums,
                                    head_planes=head_planes,
                                    dropout_prob=dropout_prob)
        self.norm1 = nn.LayerNorm(planes)
        self.norm2 = nn.LayerNorm(planes)
        self.norm3 = nn.LayerNorm(planes)

    def forward(self, x, context):
        x = self.attn1(self.norm1(x), None) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 head_planes,
                 context_planes,
                 block_nums=1,
                 dropout_prob=0.,
                 num_groups=32):
        super(SpatialTransformer, self).__init__()
        self.inplanes = inplanes
        inter_planes = head_nums * head_planes

        self.norm = nn.GroupNorm(num_groups, inplanes)
        self.proj_in = nn.Conv2d(inplanes,
                                 inter_planes,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        transformer_blocks = []
        for _ in range(block_nums):
            transformer_blocks.append(
                BasicTransformerBlock(inter_planes,
                                      head_nums,
                                      head_planes,
                                      dropout_prob=dropout_prob,
                                      context_planes=context_planes))
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        self.proj_out = self.zero_module(
            nn.Conv2d(inter_planes,
                      inplanes,
                      kernel_size=1,
                      stride=1,
                      padding=0))

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()

        return module

    def forward(self, x, context):
        # note: if context is None, cross-attention defaults to self-attention
        b, c, h, w = x.shape

        inputs = x

        x = self.norm(x)
        x = self.proj_in(x)

        x = rearrange(x, 'b c h w -> b (h w) c')

        for transformer_blocks in self.transformer_blocks:
            x = transformer_blocks(x, context)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x = self.proj_out(x)

        x = x + inputs

        return x