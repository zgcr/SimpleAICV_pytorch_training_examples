import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, layer_nums):
        super(MLP, self).__init__()
        self.layer_nums = layer_nums

        h = [hidden_planes] * (layer_nums - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([inplanes] + h, h + [planes]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.layer_nums - 1 else layer(x)

        return x


class Attention(nn.Module):

    def __init__(self, inplanes, head_nums, downsample_rate=1):
        super(Attention, self).__init__()
        inter_planes = inplanes // downsample_rate
        self.head_nums = head_nums
        assert inter_planes % head_nums == 0, "head_nums must divide inplanes."

        self.q_proj = nn.Linear(inplanes, inter_planes)
        self.k_proj = nn.Linear(inplanes, inter_planes)
        self.v_proj = nn.Linear(inplanes, inter_planes)
        self.out_proj = nn.Linear(inter_planes, inplanes)

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

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self.separate_heads(q, self.head_nums)
        k = self.separate_heads(k, self.head_nums)
        v = self.separate_heads(v, self.head_nums)

        out = F.scaled_dot_product_attention(q, k, v)

        out = self.recombine_heads(out)
        out = self.out_proj(out)

        return out


class TwoWayAttentionBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 mlp_planes=2048,
                 attention_downsample_rate=2,
                 skip_first_layer_pe=False):
        super(TwoWayAttentionBlock, self).__init__()
        self.self_attn = Attention(inplanes, head_nums)
        self.norm1 = nn.LayerNorm(inplanes)

        self.cross_attn_token_to_image = Attention(
            inplanes, head_nums, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(inplanes)

        self.mlp = MLP(inplanes, mlp_planes, inplanes, layer_nums=2)
        self.norm3 = nn.LayerNorm(inplanes)

        self.norm4 = nn.LayerNorm(inplanes)
        self.cross_attn_image_to_token = Attention(
            inplanes, head_nums, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out

        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):

    def __init__(self,
                 block_nums=2,
                 embedding_planes=256,
                 head_nums=8,
                 mlp_planes=2048,
                 attention_downsample_rate=2):
        super(TwoWayTransformer, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(block_nums):
            self.layers.append(
                TwoWayAttentionBlock(
                    inplanes=embedding_planes,
                    head_nums=head_nums,
                    mlp_planes=mlp_planes,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = Attention(
            embedding_planes,
            head_nums,
            downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_planes)

    def forward(self, image_embedding, image_pe, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(queries=queries,
                                  keys=keys,
                                  query_pe=point_embedding,
                                  key_pe=image_pe)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
