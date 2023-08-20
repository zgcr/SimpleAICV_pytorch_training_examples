import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import math
import copy

import torch
import torch.nn as nn

from simpleAICV.detection.models.multiscale_deformable_attention import MSDeformAttn


class ActivationBlock(nn.Module):

    def __init__(self, act_type='relu'):
        super(ActivationBlock, self).__init__()
        assert act_type in [
            'relu',
            'gelu',
            'glu',
            'prelu',
            'selu',
        ], 'Unsupport activation function!'
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'glu':
            self.act = nn.GLU()
        elif act_type == 'prelu':
            self.act = nn.PReLU()
        elif act_type == 'selu':
            self.act = nn.SELU()

    def forward(self, x):
        x = self.act(x)

        return x


class DINODETRClsHead(nn.Module):

    def __init__(self, hidden_inplanes, num_classes):
        super(DINODETRClsHead, self).__init__()
        self.cls_head = nn.Linear(hidden_inplanes, num_classes)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_head.bias.data = torch.ones(num_classes) * bias_value

    def forward(self, x):
        x = self.cls_head(x)

        return x


class DINODETRRegHead(nn.Module):

    def __init__(self, hidden_inplanes, num_layers=3):
        super(DINODETRRegHead, self).__init__()

        reg_layers = []
        for _ in range(num_layers - 1):
            reg_layers.append(nn.Linear(hidden_inplanes, hidden_inplanes))
            reg_layers.append(nn.ReLU(inplace=True))
        reg_layers.append(nn.Linear(hidden_inplanes, 4))
        self.reg_head = nn.Sequential(*reg_layers)

        nn.init.constant_(self.reg_head[-1].weight.data, 0)
        nn.init.constant_(self.reg_head[-1].bias.data, 0)

    def forward(self, x):
        x = self.reg_head(x)

        return x


class DINODETRMLPHead(nn.Module):

    def __init__(self, inplanes, hidden_inplanes, planes, num_layers=3):
        super(DINODETRMLPHead, self).__init__()

        layers = []
        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(inplanes, hidden_inplanes))
            else:
                layers.append(nn.Linear(hidden_inplanes, hidden_inplanes))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_inplanes, planes))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp_head(x)

        return x


class DeformableTransformer(nn.Module):

    def __init__(self,
                 inplanes=256,
                 dropout_prob=0.0,
                 head_nums=8,
                 query_nums=900,
                 feedforward_planes=2048,
                 encoder_layer_nums=6,
                 decoder_layer_nums=6,
                 activation='relu',
                 feature_level_nums=5,
                 encoder_point_nums=4,
                 decoder_point_nums=4,
                 module_seq=['sa', 'ca', 'ffn'],
                 num_classes=80):
        super(DeformableTransformer, self).__init__()
        self.inplanes = inplanes
        self.query_nums = query_nums

        assert encoder_layer_nums > 0
        assert decoder_layer_nums > 0
        assert feature_level_nums > 1

        for per_module_seq in module_seq:
            assert per_module_seq in ['sa', 'ca', 'ffn']

        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_inplanes=inplanes,
            feedforward_planes=feedforward_planes,
            dropout_prob=dropout_prob,
            activation=activation,
            feature_level_nums=feature_level_nums,
            head_nums=head_nums,
            point_nums=encoder_point_nums)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          layer_nums=encoder_layer_nums)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_inplanes=inplanes,
            feedforward_planes=feedforward_planes,
            dropout_prob=dropout_prob,
            activation=activation,
            feature_level_nums=feature_level_nums,
            head_nums=head_nums,
            point_nums=decoder_point_nums,
            module_seq=module_seq)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer,
                                          layer_nums=decoder_layer_nums,
                                          hidden_inplanes=inplanes)

        self.level_embed = nn.Parameter(
            torch.Tensor(feature_level_nums, inplanes))
        self.tgt_embed = nn.Embedding(query_nums, inplanes)

        # anchor selection at the output of encoder
        self.enc_output = nn.Linear(inplanes, inplanes)
        self.enc_output_norm = nn.LayerNorm(inplanes)
        self.enc_out_class_embed = DINODETRClsHead(inplanes, num_classes)
        self.enc_out_bbox_embed = DINODETRRegHead(hidden_inplanes=inplanes,
                                                  num_layers=3)

        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.tgt_embed.weight.data)

        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self,
                                     memory,
                                     memory_padding_mask,
                                     spatial_shapes,
                                     learnedwh=None):
        N_, _, _ = memory.shape
        device = memory.device

        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(
                N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32).to(device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32).to(device),
                indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1),
                              grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

            if learnedwh is not None:
                wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
            else:
                wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(-1,
                                                                 keepdim=True)
        output_proposals = torch.log(output_proposals /
                                     (1 - output_proposals))  # unsigmoid
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))

        return output_memory, output_proposals

    def forward(self,
                srcs,
                masks,
                refpoint_embed,
                pos_embeds,
                tgt,
                attn_mask=None):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask,
                  pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c

            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}

        device = src_flatten.device

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten,
                                          1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes,
                                         dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        #########################################################
        # Begin Encoder
        #########################################################
        memory = self.encoder(src_flatten,
                              pos=lvl_pos_embed_flatten,
                              level_start_index=level_start_index,
                              spatial_shapes=spatial_shapes,
                              valid_ratios=valid_ratios,
                              key_padding_mask=mask_flatten)

        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        #########################################################

        input_hw = None

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes, input_hw)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        enc_outputs_coord_unselected = self.enc_out_bbox_embed(
            output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
        topk = self.query_nums
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0],
                                    topk,
                                    dim=1)[1]  # bs, nq

        # gather boxes
        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unselected, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = torch.gather(
            output_proposals, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  # sigmoid

        # gather tgt
        tgt_undetach = torch.gather(
            output_memory, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, self.inplanes))

        tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs,
                                                        1).transpose(0, 1)

        if refpoint_embed is not None:
            refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_],
                                       dim=1)
            tgt = torch.cat([tgt, tgt_], dim=1)
        else:
            refpoint_embed, tgt = refpoint_embed_, tgt_

        #########################################################
        # End preparing tgt
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask)

        #########################################################
        # End Decoder
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################

        hs_enc = tgt_undetach.unsqueeze(0)
        ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)

        #########################################################
        # End postprocess
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, layer_nums):
        super(TransformerEncoder, self).__init__()
        assert layer_nums > 0
        self.layer_nums = layer_nums

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(layer_nums)])

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5,
                                                         H_ - 0.5,
                                                         H_,
                                                         dtype=torch.float32,
                                                         device=device),
                                          torch.linspace(0.5,
                                                         W_ - 0.5,
                                                         W_,
                                                         dtype=torch.float32,
                                                         device=device),
                                          indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] *
                                               H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] *
                                               W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, pos, spatial_shapes, level_start_index,
                valid_ratios, key_padding_mask):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios,
                                                     device=src.device)

        # main process
        for layer_id, layer in enumerate(self.layers):
            output = layer(src=output,
                           pos=pos,
                           reference_points=reference_points,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,
                           key_padding_mask=key_padding_mask)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, layer_nums, hidden_inplanes):
        super(TransformerDecoder, self).__init__()
        assert layer_nums > 0
        self.layer_nums = layer_nums

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(layer_nums)])

        self.norm = nn.LayerNorm(hidden_inplanes)
        self.ref_point_head = DINODETRMLPHead(2 * hidden_inplanes,
                                              hidden_inplanes,
                                              hidden_inplanes,
                                              num_layers=2)

        self.bbox_embed = None
        self.class_embed = None

    def gen_sineembed_for_position(self, pos_tensor):
        device = pos_tensor.device

        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float32).to(device)
        dim_t = 10000**(2 * (dim_t // 2) / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(
                pos_tensor.size(-1)))

        return pos

    def inverse_sigmoid(self, x, eps=1e-4):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)

        return torch.log(x1 / x2)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                refpoints_unsigmoid=None,
                level_start_index=None,
                spatial_shapes=None,
                valid_ratios=None):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                        * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :,
                                                          None] * valid_ratios[
                                                              None, :]
            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :])  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(
                query_sine_embed)  # nq, bs, 256
            pos_scale = 1
            query_pos = pos_scale * raw_query_pos

            output = layer(tgt=output,
                           tgt_query_pos=query_pos,
                           tgt_query_sine_embed=query_sine_embed,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           tgt_reference_points=reference_points_input,
                           memory=memory,
                           memory_key_padding_mask=memory_key_padding_mask,
                           memory_level_start_index=level_start_index,
                           memory_spatial_shapes=spatial_shapes,
                           memory_pos=pos,
                           self_attn_mask=tgt_mask,
                           cross_attn_mask=memory_mask)

            reference_before_sigmoid = self.inverse_sigmoid(reference_points)
            delta_unsig = self.bbox_embed[layer_id](output)
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = outputs_unsig.sigmoid()

            ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [[itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]]


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 hidden_inplanes=256,
                 feedforward_planes=2048,
                 dropout_prob=0.1,
                 activation="relu",
                 feature_level_nums=5,
                 head_nums=8,
                 point_nums=4):
        super(DeformableTransformerEncoderLayer, self).__init__()
        self.self_attn = MSDeformAttn(hidden_inplanes, feature_level_nums,
                                      head_nums, point_nums)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(hidden_inplanes)

        # ffn
        self.linear1 = nn.Linear(hidden_inplanes, feedforward_planes)
        self.activation = ActivationBlock(act_type=activation)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(feedforward_planes, hidden_inplanes)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.norm2 = nn.LayerNorm(hidden_inplanes)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                key_padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                              src, spatial_shapes, level_start_index,
                              key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 hidden_inplanes=256,
                 feedforward_planes=2048,
                 dropout_prob=0.1,
                 activation="relu",
                 feature_level_nums=5,
                 head_nums=8,
                 point_nums=4,
                 module_seq=['sa', 'ca', 'ffn']):
        super(DeformableTransformerDecoderLayer, self).__init__()
        for per_seq in module_seq:
            assert per_seq in ['sa', 'ca', 'ffn']

        self.module_seq = module_seq

        self.cross_attn = MSDeformAttn(hidden_inplanes, feature_level_nums,
                                       head_nums, point_nums)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(hidden_inplanes)

        # self attention
        self.self_attn = nn.MultiheadAttention(hidden_inplanes,
                                               head_nums,
                                               dropout=dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.norm2 = nn.LayerNorm(hidden_inplanes)

        # ffn
        self.linear1 = nn.Linear(hidden_inplanes, feedforward_planes)
        self.activation = ActivationBlock(act_type=activation)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(feedforward_planes, hidden_inplanes)
        self.dropout4 = nn.Dropout(dropout_prob)
        self.norm3 = nn.LayerNorm(hidden_inplanes)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self, tgt, tgt_query_pos=None, self_attn_mask=None):
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt

    def forward_ca(self,
                   tgt,
                   tgt_query_pos=None,
                   tgt_reference_points=None,
                   memory=None,
                   memory_key_padding_mask=None,
                   memory_level_start_index=None,
                   memory_spatial_shapes=None):

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1), memory_spatial_shapes,
            memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self,
                tgt,
                tgt_query_pos=None,
                tgt_query_sine_embed=None,
                tgt_key_padding_mask=None,
                tgt_reference_points=None,
                memory=None,
                memory_key_padding_mask=None,
                memory_level_start_index=None,
                memory_spatial_shapes=None,
                memory_pos=None,
                self_attn_mask=None,
                cross_attn_mask=None):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_reference_points,
                                      memory, memory_key_padding_mask,
                                      memory_level_start_index,
                                      memory_spatial_shapes)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, self_attn_mask)

        return tgt
