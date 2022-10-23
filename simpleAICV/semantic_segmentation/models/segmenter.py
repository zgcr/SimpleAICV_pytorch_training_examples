import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.semantic_segmentation.models import backbones
from simpleAICV.semantic_segmentation.models.backbones.vitbackbone import TransformerEncoderLayer

__all__ = [
    'segmenter_vit_tiny_patch16_mask',
    'segmenter_vit_small_patch16_mask',
    'segmenter_vit_base_patch16_mask',
    'segmenter_vit_large_patch16_mask',
]


class MaskTransformer(nn.Module):

    def __init__(self,
                 patch_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio=4,
                 dropout_prob=0.1,
                 drop_path_prob=0.,
                 num_classses=150):
        super(MaskTransformer, self).__init__()
        self.patch_size = patch_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.dropout_prob = dropout_prob
        self.drop_path_prob = drop_path_prob
        self.num_classses = num_classses

        self.scale = embedding_planes**-0.5

        drop_path_prob_list = []
        for block_idx in range(self.block_nums):
            if drop_path_prob == 0.:
                drop_path_prob_list.append(0.)
            else:
                per_layer_drop_path_prob = drop_path_prob * (
                    block_idx / (self.block_nums - 1))
                drop_path_prob_list.append(per_layer_drop_path_prob)

        blocks = []
        for i in range(self.block_nums):
            blocks.append(
                TransformerEncoderLayer(
                    self.embedding_planes,
                    self.head_nums,
                    feedforward_ratio=self.feedforward_ratio,
                    dropout_prob=dropout_prob,
                    drop_path_prob=drop_path_prob_list[i]))
        self.blocks = nn.ModuleList(blocks)

        self.block_norm = nn.LayerNorm(embedding_planes)

        self.decoder_linear = nn.Linear(embedding_planes, embedding_planes)
        self.cls_embedding = nn.Parameter(
            torch.randn(1, num_classses, embedding_planes))

        self.proj_patch = nn.Parameter(
            self.scale * torch.randn(embedding_planes, embedding_planes))
        self.proj_classes = nn.Parameter(
            self.scale * torch.randn(embedding_planes, embedding_planes))
        self.mask_norm = nn.LayerNorm(num_classses)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        nn.init.trunc_normal_(self.cls_embedding, std=0.02)

    def forward(self, x, h, w):
        x = self.decoder_linear(x)
        cls_emb = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_emb), dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.block_norm(x)

        patches, cls_seg_feat = x[:, :-self.num_classses], x[:, -self.
                                                             num_classses:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        gh, gw = h // self.patch_size, w // self.patch_size
        masks = torch.einsum('bcn->bnc', masks)
        masks = masks.reshape(masks.shape[0], masks.shape[1], gh, gw)

        masks = F.interpolate(masks,
                              size=(h, w),
                              mode='bilinear',
                              align_corners=True)

        return masks


class Segmenter(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 image_size=512,
                 patch_size=16,
                 block_nums=2,
                 head_nums=12,
                 feedforward_ratio=4,
                 dropout_prob=0.1,
                 drop_path_prob=0.,
                 num_classes=150):
        super(Segmenter, self).__init__()
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.dropout_prob = dropout_prob
        self.drop_path_prob = drop_path_prob
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type](
            **{
                'image_size': image_size,
                'patch_size': patch_size,
                'pretrained_path': backbone_pretrained_path,
            })
        self.head = MaskTransformer(
            patch_size=patch_size,
            embedding_planes=self.backbone.embedding_planes,
            block_nums=self.block_nums,
            head_nums=self.head_nums,
            feedforward_ratio=self.feedforward_ratio,
            dropout_prob=self.dropout_prob,
            drop_path_prob=self.drop_path_prob,
            num_classses=self.num_classes)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.backbone(x)
        x = self.head(x, h, w)

        return x


def _segmenter(backbone_type, backbone_pretrained_path, image_size, block_nums,
               head_nums, **kwargs):
    model = Segmenter(backbone_type,
                      backbone_pretrained_path=backbone_pretrained_path,
                      image_size=image_size,
                      block_nums=block_nums,
                      head_nums=head_nums,
                      **kwargs)

    return model


def segmenter_vit_tiny_patch16_mask(backbone_pretrained_path='',
                                    image_size=512,
                                    **kwargs):
    return _segmenter('vit_tiny_backbone_patch16',
                      backbone_pretrained_path=backbone_pretrained_path,
                      image_size=image_size,
                      block_nums=2,
                      head_nums=3,
                      **kwargs)


def segmenter_vit_small_patch16_mask(backbone_pretrained_path='',
                                     image_size=512,
                                     **kwargs):
    return _segmenter('vit_small_backbone_patch16',
                      backbone_pretrained_path=backbone_pretrained_path,
                      image_size=image_size,
                      block_nums=2,
                      head_nums=6,
                      **kwargs)


def segmenter_vit_base_patch16_mask(backbone_pretrained_path='',
                                    image_size=512,
                                    **kwargs):
    return _segmenter('vit_base_backbone_patch16',
                      backbone_pretrained_path=backbone_pretrained_path,
                      image_size=image_size,
                      block_nums=2,
                      head_nums=12,
                      **kwargs)


def segmenter_vit_large_patch16_mask(backbone_pretrained_path='',
                                     image_size=512,
                                     **kwargs):
    return _segmenter('vit_large_backbone_patch16',
                      backbone_pretrained_path=backbone_pretrained_path,
                      image_size=image_size,
                      block_nums=2,
                      head_nums=16,
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

    net = segmenter_vit_tiny_patch16_mask(backbone_pretrained_path='',
                                          image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = segmenter_vit_small_patch16_mask(backbone_pretrained_path='',
                                           image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = segmenter_vit_base_patch16_mask(backbone_pretrained_path='',
                                          image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = segmenter_vit_large_patch16_mask(backbone_pretrained_path='',
                                           image_size=512)
    image_h, image_w = 512, 512
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')
