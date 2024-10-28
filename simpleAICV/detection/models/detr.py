import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from simpleAICV.detection.models import backbones
from simpleAICV.detection.models.backbones.detr_resnet import PositionEmbeddingBlock
from simpleAICV.detection.models.head import DETRClsRegHead

__all__ = [
    'resnet18_detr',
    'resnet34_detr',
    'resnet50_detr',
    'resnet101_detr',
    'resnet152_detr',
]


class ActivationBlock(nn.Module):

    def __init__(self, act_type='relu'):
        super(ActivationBlock, self).__init__()
        assert act_type in ['relu', 'gelu'], 'Unsupport activation function!'
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'gelu':
            self.act = nn.GELU()

    def forward(self, x):
        x = self.act(x)

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 hidden_planes,
                 head_nums,
                 feedforward_ratio=4,
                 dropout_prob=0.1,
                 act_type="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Attention模块
        self.attention = nn.MultiheadAttention(hidden_planes,
                                               head_nums,
                                               dropout=dropout_prob)
        # Feedforward模块
        self.linear1 = nn.Linear(hidden_planes,
                                 int(hidden_planes * feedforward_ratio))
        self.linear2 = nn.Linear(int(hidden_planes * feedforward_ratio),
                                 hidden_planes)
        self.norm1 = nn.LayerNorm(hidden_planes)
        self.norm2 = nn.LayerNorm(hidden_planes)

        self.act = ActivationBlock(act_type)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        # 添加位置信息
        if pos is not None:
            q = k = src + pos
        else:
            q = k = src

        # 使用Attention模块和残差结构
        src2 = self.attention(q,
                              k,
                              value=src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 使用Feedforward模块和残差结构
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 hidden_planes,
                 head_nums,
                 feedforward_ratio=4,
                 dropout_prob=0.1,
                 act_type="relu"):
        super(TransformerDecoderLayer, self).__init__()
        # q自己做一个self-attention
        self.attention = nn.MultiheadAttention(hidden_planes,
                                               head_nums,
                                               dropout=dropout_prob)
        # q、k、v联合做一个self-attention
        self.multihead_attention = nn.MultiheadAttention(hidden_planes,
                                                         head_nums,
                                                         dropout=dropout_prob)
        # Feedforward模块
        self.linear1 = nn.Linear(hidden_planes,
                                 int(hidden_planes * feedforward_ratio))
        self.linear2 = nn.Linear(int(hidden_planes * feedforward_ratio),
                                 hidden_planes)

        self.norm1 = nn.LayerNorm(hidden_planes)
        self.norm2 = nn.LayerNorm(hidden_planes)
        self.norm3 = nn.LayerNorm(hidden_planes)

        self.activation = ActivationBlock(act_type)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        # 添加位置信息
        if query_pos is not None:
            q = k = tgt + query_pos
        else:
            q = k = tgt

        #---------------------------------------------#
        #   q自己做一个self-attention
        #---------------------------------------------#
        # 使用Attention模块,tgt + query_embed
        tgt2 = self.attention(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        #---------------------------------------------#
        #   q、k、v联合做一个self-attention
        #---------------------------------------------#
        # 使用Multi-Attention模块
        if query_pos is not None:
            q = tgt + query_pos
        else:
            q = tgt

        if pos is not None:
            k = memory + pos
        else:
            k = memory

        tgt2 = self.multihead_attention(
            query=q,
            key=k,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # 使用Feedforward模块和残差结构
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DETRTransformer(nn.Module):

    def __init__(self,
                 inplanes=256,
                 head_nums=8,
                 feedforward_ratio=4,
                 encoder_layer_nums=6,
                 decoder_layer_nums=6,
                 dropout_prob=0.1,
                 act_type='relu'):
        super(DETRTransformer, self).__init__()
        self.inplanes = inplanes
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.encoder_layer_nums = encoder_layer_nums
        self.decoder_layer_nums = decoder_layer_nums
        self.dropout_prob = dropout_prob
        self.act_type = act_type

        encoder_blocks = []
        for _ in range(self.encoder_layer_nums):
            encoder_blocks.append(
                TransformerEncoderLayer(
                    self.inplanes,
                    self.head_nums,
                    feedforward_ratio=self.feedforward_ratio,
                    dropout_prob=dropout_prob,
                    act_type=self.act_type))
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        decoder_blocks = []
        for _ in range(self.decoder_layer_nums):
            decoder_blocks.append(
                TransformerDecoderLayer(
                    self.inplanes,
                    self.head_nums,
                    feedforward_ratio=self.feedforward_ratio,
                    dropout_prob=dropout_prob,
                    act_type=self.act_type))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_norm = nn.LayerNorm(self.inplanes)

        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def forward(self, src, mask, query_embed, pos_embed):
        # [b,256,25,25],[b,25,25],[100,256],[b,256,25,25]
        b, c, h, w = src.shape[0], src.shape[1], src.shape[2], src.shape[3]

        # [b,c,h,w]->[b,c,h*w]->[h*w,b,c]
        src = src.flatten(2).permute(2, 0, 1)

        # [query_nums,c]->[query_nums,1,c]->[query_nums,b,c]
        query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)

        # [b,c,h,w]->[b,c,h*w]->[h*w,b,c]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # [b,h,w]->[b,h*w]
        mask = mask.flatten(1)

        # [query_nums,b,c]
        tgt = torch.zeros_like(query_embed)

        # src:[h*w,b,c],mask:[b,h*w], query_embed:[query_nums,b,c], pos_embed:[h*w,b,c]
        # [h*w,b,c]->[h*w,b,c]
        memory = src
        for per_encoder_layer in self.encoder_blocks:
            memory = per_encoder_layer(memory,
                                       src_key_padding_mask=mask,
                                       pos=pos_embed)

        intermediate = []
        for per_decoder_layer in self.decoder_blocks:
            tgt = per_decoder_layer(tgt,
                                    memory,
                                    memory_key_padding_mask=mask,
                                    pos=pos_embed,
                                    query_pos=query_embed)
            intermediate.append(self.decoder_norm(tgt))

        hs = torch.stack(intermediate)
        hs = hs.transpose(1, 2)

        memory = memory.permute(1, 2, 0).view(b, c, h, w)

        return hs, memory


class DETR(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 hidden_inplanes=256,
                 query_nums=100,
                 num_classes=80,
                 use_gradient_checkpoint=False):
        super(DETR, self).__init__()
        self.hidden_inplanes = hidden_inplanes
        self.query_nums = query_nums
        self.num_classes = num_classes
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'pretrained_path': backbone_pretrained_path,
                'use_gradient_checkpoint': use_gradient_checkpoint,
            })

        self.position_embedding = PositionEmbeddingBlock(
            inplanes=self.hidden_inplanes // 2, temperature=10000, eps=1e-6)

        self.proj_conv = nn.Conv2d(self.backbone.out_channels[-1],
                                   self.hidden_inplanes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True)

        self.transformer = DETRTransformer(inplanes=self.hidden_inplanes,
                                           head_nums=8,
                                           feedforward_ratio=4,
                                           encoder_layer_nums=6,
                                           decoder_layer_nums=6,
                                           dropout_prob=0.1,
                                           act_type='relu')

        self.query_embed = nn.Embedding(self.query_nums, self.hidden_inplanes)

        self.head = DETRClsRegHead(self.hidden_inplanes,
                                   self.num_classes + 1,
                                   num_layers=3)

    def forward(self, inputs, masks):
        assert masks is not None

        # [b,c,h,w]=[b,3,800,800]
        # [b,2048,25,25],[b,256,25,25],[b,25,25]
        features = self.backbone(inputs)
        features = features[-1]

        masks = F.interpolate(masks.float().unsqueeze(1),
                              size=[features.shape[2], features.shape[3]
                                    ]).to(torch.bool).squeeze(1)
        positions = self.position_embedding(masks)

        assert masks is not None

        # [b,256,25,25]
        features = self.proj_conv(features)

        if self.use_gradient_checkpoint:
            # [b,256,25,25],[b,25,25],[100,256],[b,256,25,25]
            features, memory = checkpoint(self.transformer,
                                          features,
                                          masks.float(),
                                          self.query_embed.weight,
                                          positions,
                                          use_reentrant=False)
        else:
            # [b,256,25,25],[b,25,25],[100,256],[b,256,25,25]
            features, memory = self.transformer(features, masks.float(),
                                                self.query_embed.weight,
                                                positions)

        if self.use_gradient_checkpoint:
            # [6,b,100,256],[b,256,25,25]
            cls_outputs, reg_outputs = checkpoint(self.head,
                                                  features,
                                                  use_reentrant=False)
        else:
            # [6,b,100,256],[b,256,25,25]
            cls_outputs, reg_outputs = self.head(features)

        del features

        # if input size:[B,3,800,800]
        # cls_outputs shape:[6,B,query_nums,num_classes+1]
        # reg_outputs shape:[6,B,query_nums,4]
        return [cls_outputs, reg_outputs]


def _detr(backbone_type, backbone_pretrained_path, **kwargs):
    model = DETR(backbone_type,
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)

    return model


def resnet18_detr(backbone_pretrained_path='', **kwargs):
    return _detr('detr_resnet18backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet34_detr(backbone_pretrained_path='', **kwargs):
    return _detr('detr_resnet34backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet50_detr(backbone_pretrained_path='', **kwargs):
    return _detr('detr_resnet50backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet101_detr(backbone_pretrained_path='', **kwargs):
    return _detr('detr_resnet101backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
                 **kwargs)


def resnet152_detr(backbone_pretrained_path='', **kwargs):
    return _detr('detr_resnet152backbone',
                 backbone_pretrained_path=backbone_pretrained_path,
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

    from tools.path import COCO2017_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.detection.datasets.cocodataset import CocoDetection
    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionResize, DetectionCollater, DETRDetectionCollater

    cocodataset = CocoDetection(COCO2017_path,
                                set_name='train2017',
                                transform=transforms.Compose([
                                    RandomHorizontalFlip(prob=0.5),
                                    RandomCrop(prob=0.5),
                                    RandomTranslate(prob=0.5),
                                    DetectionResize(
                                        resize=1024,
                                        stride=32,
                                        resize_type='yolo_style',
                                        multi_scale=False,
                                        multi_scale_range=[0.8, 1.0]),
                                    Normalize(),
                                ]))

    from torch.utils.data import DataLoader
    collater = DETRDetectionCollater(resize=1024,
                                     resize_type='yolo_style',
                                     max_annots_num=100)
    train_loader = DataLoader(cocodataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    net = resnet50_detr()

    for data in tqdm(train_loader):
        images, annots, masks, scales, sizes = data['image'], data[
            'annots'], data['mask'], data['scale'], data['size']
        print('0000', images.shape, annots.shape, masks.shape, scales.shape,
              sizes.shape)
        print('0000', images.dtype, annots.dtype, masks.dtype, scales.dtype,
              sizes.dtype)

        image_h, image_w = 1024, 1024
        from thop import profile
        from thop import clever_format
        macs, params = profile(net, inputs=(images, masks), verbose=False)
        macs, params = clever_format([macs, params], '%.3f')
        print(f'1111, macs: {macs}, params: {params}')
        outs = net(torch.autograd.Variable(images),
                   torch.autograd.Variable(masks))
        for out in outs:
            print('2222', out.shape)

        break

    net = resnet50_detr(use_gradient_checkpoint=True)

    for data in tqdm(train_loader):
        images, annots, masks, scales, sizes = data['image'], data[
            'annots'], data['mask'], data['scale'], data['size']
        print('0000', images.shape, annots.shape, masks.shape, scales.shape,
              sizes.shape)
        print('0000', images.dtype, annots.dtype, masks.dtype, scales.dtype,
              sizes.dtype)

        image_h, image_w = 1024, 1024
        from thop import profile
        from thop import clever_format
        macs, params = profile(net, inputs=(images, masks), verbose=False)
        macs, params = clever_format([macs, params], '%.3f')
        print(f'1111, macs: {macs}, params: {params}')
        outs = net(torch.autograd.Variable(images),
                   torch.autograd.Variable(masks))
        for out in outs:
            print('2222', out.shape)

        break
