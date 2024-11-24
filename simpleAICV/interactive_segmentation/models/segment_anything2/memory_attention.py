import math

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# sam2_hiera_t.yaml/sam2_hiera_s.yaml/sam2_hiera_b+.yaml/sam2_hiera_l.yaml

# memory_attention:
# _target_: sam2.modeling.memory_attention.MemoryAttention
# d_model: 256
# pos_enc_at_input: true
# layer:
#     _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
#     activation: relu
#     dim_feedforward: 2048
#     dropout: 0.1
#     pos_enc_at_attn: false
#     self_attention:
#     _target_: sam2.modeling.sam.transformer.RoPEAttention
#     rope_theta: 10000.0
#     feat_sizes: [32, 32]
#     embedding_dim: 256
#     num_heads: 1
#     downsample_rate: 1
#     dropout: 0.1
#     d_model: 256
#     pos_enc_at_cross_attn_keys: true
#     pos_enc_at_cross_attn_queries: false
#     cross_attention:
#     _target_: sam2.modeling.sam.transformer.RoPEAttention
#     rope_theta: 10000.0
#     feat_sizes: [32, 32]
#     rope_k_repeat: True
#     embedding_dim: 256
#     num_heads: 1
#     downsample_rate: 1
#     dropout: 0.1
#     kv_in_dim: 64
# num_layers: 4


def init_t_xy(end_x, end_y):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()

    return t_x, t_y


def compute_axial_cis(dim, end_x, end_y, theta=10000.0):
    freqs_x = 1.0 / (theta
                     **(torch.arange(0, dim, 4)[:(dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta
                     **(torch.arange(0, dim, 4)[:(dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis, x):
    ndim = len(x.shape)
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


def apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
           if xk.shape[-2] != 0 else None)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPEAttention(nn.Module):

    def __init__(self,
                 inplanes=256,
                 head_nums=1,
                 downsample_rate=1,
                 kv_inplanes=None,
                 feat_sizes=(32, 32),
                 rope_theta=10000.0,
                 rope_k_repeat=False):
        super(RoPEAttention, self).__init__()
        inter_planes = inplanes // downsample_rate
        kv_inplanes = kv_inplanes if kv_inplanes is not None else inplanes
        self.head_nums = head_nums
        assert inter_planes % head_nums == 0, "head_nums must divide inplanes."

        self.q_proj = nn.Linear(inplanes, inter_planes)
        self.k_proj = nn.Linear(kv_inplanes, inter_planes)
        self.v_proj = nn.Linear(kv_inplanes, inter_planes)
        self.out_proj = nn.Linear(inter_planes, inplanes)

        self.compute_cis = partial(compute_axial_cis,
                                   dim=inter_planes // head_nums,
                                   theta=rope_theta)
        self.freqs_cis = self.compute_cis(end_x=feat_sizes[0],
                                          end_y=feat_sizes[1])
        self.rope_k_repeat = rope_k_repeat

    def separate_heads(self, x, head_nums):
        b, n, c = x.shape
        x = x.reshape(b, n, head_nums, c // head_nums)
        # B x N_heads x N_tokens x C_per_head
        return x.transpose(1, 2)

    def recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        # B x N_tokens x C
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q, k, v, num_k_exclude_rope=0):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self.separate_heads(q, self.head_nums)
        k = self.separate_heads(k, self.head_nums)
        v = self.separate_heads(v, self.head_nums)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.shape[-2] - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat)

        out = F.scaled_dot_product_attention(q, k, v)

        out = self.recombine_heads(out)
        out = self.out_proj(out)

        return out


class MemoryAttentionLayer(nn.Module):

    def __init__(self,
                 inplanes=256,
                 head_nums=1,
                 downsample_rate=1,
                 feedforward_planes=2048,
                 dropout_prob=0.):
        super(MemoryAttentionLayer, self).__init__()
        self.self_attn = RoPEAttention(inplanes=inplanes,
                                       head_nums=head_nums,
                                       downsample_rate=downsample_rate,
                                       kv_inplanes=None,
                                       feat_sizes=(32, 32),
                                       rope_theta=10000.0,
                                       rope_k_repeat=False)

        self.cross_attn_image = RoPEAttention(inplanes=inplanes,
                                              head_nums=head_nums,
                                              downsample_rate=downsample_rate,
                                              kv_inplanes=64,
                                              feat_sizes=(32, 32),
                                              rope_theta=10000.0,
                                              rope_k_repeat=True)

        self.linear1 = nn.Linear(inplanes, feedforward_planes)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(feedforward_planes, inplanes)

        self.norm1 = nn.LayerNorm(inplanes)
        self.norm2 = nn.LayerNorm(inplanes)
        self.norm3 = nn.LayerNorm(inplanes)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                tgt,
                memory,
                pos=None,
                query_pos=None,
                num_k_exclude_rope=0):

        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(q=tgt2,
                                     k=memory + pos,
                                     v=memory,
                                     num_k_exclude_rope=num_k_exclude_rope)
        tgt = tgt + self.dropout2(tgt2)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class MemoryAttention(nn.Module):

    def __init__(self, inplanes=256, layer_nums=4):
        super(MemoryAttention, self).__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionLayer(inplanes=inplanes,
                                 head_nums=1,
                                 downsample_rate=1,
                                 feedforward_planes=2048,
                                 dropout_prob=0.) for _ in range(layer_nums)
        ])
        self.norm = nn.LayerNorm(inplanes)

    def forward(self,
                curr,
                memory,
                curr_pos=None,
                memory_pos=None,
                num_obj_ptr_tokens=0):
        # curr:self-attention inputs
        # memory:cross-attention inputs
        # curr_pos:pos_enc for self-attention inputs
        # memory_pos:pos_enc for cross-attention inputs
        # num_obj_ptr_tokens:number of object pointer *tokens*
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        assert curr.shape[1] == memory.shape[
            1], "Batch size must be the same for curr and memory"

        output = curr
        if curr_pos is not None:
            output = output + 0.1 * curr_pos

        # Convert to batch first
        output = output.transpose(0, 1)
        curr_pos = curr_pos.transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            output = layer(tgt=output,
                           memory=memory,
                           pos=memory_pos,
                           query_pos=curr_pos,
                           num_k_exclude_rope=num_obj_ptr_tokens)
        normed_output = self.norm(output)

        # Convert back to seq first
        normed_output = normed_output.transpose(0, 1)
        curr_pos = curr_pos.transpose(0, 1)

        return normed_output
