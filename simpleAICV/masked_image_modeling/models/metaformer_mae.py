import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn

from simpleAICV.classification.backbones.vit import TransformerEncoderLayer

__all__ = [
    'poolformer_s36_patch32_224_mae_pretrain_model',
]


class PatchEmbeddingBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_norm=False):
        super(PatchEmbeddingBlock, self).__init__()
        bias = False if has_norm else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=inplanes)
            if has_norm else nn.Sequential(),
        )

    def forward(self, x, mask):
        x = x * mask
        x = self.layer(x)

        return x


class PoolingBlock(nn.Module):

    def __init__(self, pool_size=3):
        super(PoolingBlock, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size,
                                 stride=1,
                                 padding=pool_size // 2,
                                 count_include_pad=False)

    def forward(self, x, mask):
        x = x * mask
        x = self.pool(x) - x

        return x


class FeedForwardBlock(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, dropout_prob=0.):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Conv2d(inplanes,
                             hidden_planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_planes,
                             planes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             groups=1,
                             bias=True)
        self.drop = nn.Dropout(dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        x = x * mask
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DropPathBlock(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    if drop_path_prob = 0. ,not use DropPath
    """

    def __init__(self, drop_path_prob=0., scale_by_keep=True):
        super(DropPathBlock, self).__init__()
        assert drop_path_prob >= 0.

        self.drop_path_prob = drop_path_prob
        self.keep_path_prob = 1 - drop_path_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_path_prob == 0. or not self.training:
            return x

        b = x.shape[0]
        device = x.device

        # work with diff dim tensors, not just 2D ConvNets
        shape = (b, ) + (1, ) * (len(x.shape) - 1)
        random_weight = torch.empty(shape).to(device).bernoulli_(
            self.keep_path_prob)

        if self.keep_path_prob > 0. and self.scale_by_keep:
            random_weight.div_(self.keep_path_prob)

        x = random_weight * x

        return x


class PoolFormerBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 pool_size=3,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5):
        super(PoolFormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=inplanes)
        self.token_mixer = PoolingBlock(pool_size=pool_size)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=inplanes)
        self.feed_forward = FeedForwardBlock(
            inplanes, int(inplanes * feed_forward_ratio), inplanes,
            dropout_prob)
        self.layer_scale_1 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        self.layer_scale_2 = nn.Parameter(layer_scale_factor *
                                          torch.ones(1, inplanes, 1, 1))
        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x, mask):
        x = x * mask
        x = x + self.drop_path(
            self.layer_scale_1 * self.token_mixer(self.norm1(x), mask))
        x = x + self.drop_path(
            self.layer_scale_2 * self.feed_forward(self.norm2(x), mask))

        return x


class PoolFormerMAEPretrainModelEncoder(nn.Module):

    def __init__(self,
                 image_size,
                 patch_size,
                 layer_nums,
                 planes,
                 pool_size=3,
                 feed_forward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 layer_scale_factor=1e-5,
                 mask_ratio=0.75):
        super(PoolFormerMAEPretrainModelEncoder, self).__init__()
        assert len(layer_nums) == 4
        assert len(planes) == 4

        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_nums = layer_nums
        self.planes = planes
        self.mask_ratio = mask_ratio

        drop_path_prob_list = [[], [], [], []]
        for i, per_layer_num in enumerate(layer_nums):
            for per_block_index in range(per_layer_num):
                if drop_path_prob == 0.:
                    drop_path_prob_list[i].append(0.)
                else:
                    per_layer_drop_path_prob = drop_path_prob * (
                        per_block_index +
                        sum(layer_nums[:i])) / (sum(layer_nums) - 1)
                    drop_path_prob_list[i].append(per_layer_drop_path_prob)

        self.patch_embedding1 = PatchEmbeddingBlock(3,
                                                    planes[0],
                                                    kernel_size=7,
                                                    stride=4,
                                                    padding=2,
                                                    groups=1,
                                                    has_norm=False)
        self.stage2 = self.make_layer(planes[0],
                                      layer_nums[0],
                                      drop_path_prob_list[0],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding2 = PatchEmbeddingBlock(planes[0],
                                                    planes[1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage3 = self.make_layer(planes[1],
                                      layer_nums[1],
                                      drop_path_prob_list[1],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding3 = PatchEmbeddingBlock(planes[1],
                                                    planes[2],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage4 = self.make_layer(planes[2],
                                      layer_nums[2],
                                      drop_path_prob_list[2],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)
        self.patch_embedding4 = PatchEmbeddingBlock(planes[2],
                                                    planes[3],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    groups=1,
                                                    has_norm=False)
        self.stage5 = self.make_layer(planes[3],
                                      layer_nums[3],
                                      drop_path_prob_list[3],
                                      pool_size=pool_size,
                                      feed_forward_ratio=feed_forward_ratio,
                                      dropout_prob=dropout_prob,
                                      layer_scale_factor=layer_scale_factor)

        self.stage2_decode_conv = nn.Conv2d(self.planes[0],
                                            self.planes[3],
                                            kernel_size=8,
                                            stride=8)
        self.stage3_decode_conv = nn.Conv2d(self.planes[1],
                                            self.planes[3],
                                            kernel_size=4,
                                            stride=4)
        self.stage4_decode_conv = nn.Conv2d(self.planes[2],
                                            self.planes[3],
                                            kernel_size=2,
                                            stride=2)

        self.out_planes = self.planes[3]
        self.final_embedding = nn.Linear(self.planes[3], self.planes[3])
        self.position_encoding = nn.Parameter(
            torch.ones(1, 3, self.image_size, self.image_size))
        self.norm = nn.LayerNorm(self.planes[-1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def make_layer(self,
                   inplanes,
                   layer_nums,
                   drop_path_prob,
                   pool_size=3,
                   feed_forward_ratio=4,
                   dropout_prob=0.,
                   layer_scale_factor=1e-5):
        assert len(drop_path_prob) == layer_nums

        blocks = []
        for block_idx in range(layer_nums):
            blocks.append(
                PoolFormerBlock(inplanes,
                                pool_size=pool_size,
                                feed_forward_ratio=feed_forward_ratio,
                                dropout_prob=dropout_prob,
                                drop_path_prob=drop_path_prob[block_idx],
                                layer_scale_factor=layer_scale_factor))
        blocks = nn.ModuleList(blocks)

        return blocks

    def forward(self, x):
        # final mask
        keep_ids, final_mask, restore_ids = self.random_masking(x)

        # mask for origin image
        mask_for_origin_image = final_mask.reshape(
            -1, 7, 7).unsqueeze(-1).repeat(1, 1, 1, 1024).reshape(
                -1, 7, 7, 32, 32).permute(0, 1, 3, 2,
                                          4).reshape(x.shape[0], 224,
                                                     224).unsqueeze(1)

        # mask for downsample 4x
        mask_for_downsample_4x = final_mask.reshape(
            -1, 7, 7).unsqueeze(-1).repeat(1, 1, 1, 64).reshape(
                -1, 7, 7, 8, 8).permute(0, 1, 3, 2,
                                        4).reshape(x.shape[0], 56,
                                                   56).unsqueeze(1)
        # mask for downsample 8x
        mask_for_downsample_8x = final_mask.reshape(
            -1, 7, 7).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(
                -1, 7, 7, 4, 4).permute(0, 1, 3, 2,
                                        4).reshape(x.shape[0], 28,
                                                   28).unsqueeze(1)
        # for downsample 16x
        mask_for_downsample_16x = final_mask.reshape(
            -1, 7, 7).unsqueeze(-1).repeat(1, 1, 1,
                                           4).reshape(-1, 7, 7, 2, 2).permute(
                                               0, 1, 3, 2,
                                               4).reshape(x.shape[0], 14,
                                                          14).unsqueeze(1)
        # for downsample 32x
        mask_for_downsample_32x = final_mask.reshape(
            -1, 7, 7).unsqueeze(-1).repeat(1, 1, 1,
                                           1).reshape(-1, 7, 7, 1, 1).permute(
                                               0, 1, 3, 2,
                                               4).reshape(x.shape[0], 7,
                                                          7).unsqueeze(1)

        x = x + self.position_encoding

        x = self.patch_embedding1(x, mask_for_origin_image)
        for per_block in self.stage2:
            x = per_block(x, mask_for_downsample_4x)
        stage2_x = self.stage2_decode_conv(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embedding2(x, mask_for_downsample_4x)
        for per_block in self.stage3:
            x = per_block(x, mask_for_downsample_8x)
        stage3_x = self.stage3_decode_conv(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embedding3(x, mask_for_downsample_8x)
        for per_block in self.stage4:
            x = per_block(x, mask_for_downsample_16x)
        stage4_x = self.stage4_decode_conv(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embedding4(x, mask_for_downsample_16x)
        for per_block in self.stage5:
            x = per_block(x, mask_for_downsample_32x)
        x = x.flatten(2).permute(0, 2, 1)

        x = self.final_embedding(x)

        x = torch.gather(x,
                         dim=1,
                         index=keep_ids.unsqueeze(-1).repeat(
                             1, 1, x.shape[-1]))
        stage2_x = torch.gather(stage2_x,
                                dim=1,
                                index=keep_ids.unsqueeze(-1).repeat(
                                    1, 1, stage2_x.shape[-1]))
        stage3_x = torch.gather(stage3_x,
                                dim=1,
                                index=keep_ids.unsqueeze(-1).repeat(
                                    1, 1, stage3_x.shape[-1]))
        stage4_x = torch.gather(stage4_x,
                                dim=1,
                                index=keep_ids.unsqueeze(-1).repeat(
                                    1, 1, stage4_x.shape[-1]))

        x = stage2_x + stage3_x + stage4_x + x
        x = self.norm(x)

        return x, final_mask, restore_ids

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, C, H, W]
        """
        b = x.shape[0]
        n = int((self.image_size // self.patch_size)**2)
        keep_length = int(n * (1 - self.mask_ratio))

        # noise in [0, 1]
        noise = torch.rand(b, n, device=x.device)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        shuffle_ids = torch.argsort(noise, dim=1)
        restore_ids = torch.argsort(shuffle_ids, dim=1)

        # keep the first subset
        keep_ids = shuffle_ids[:, :keep_length]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, n], device=x.device)
        mask[:, :keep_length] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=restore_ids)

        return keep_ids, mask, restore_ids


class PoolFormerMAEPretrainModelDecoder(nn.Module):

    def __init__(self,
                 patch_size,
                 image_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 dropout_prob=0.1):
        super(PoolFormerMAEPretrainModelDecoder, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio

        self.mask_token = nn.Parameter(torch.zeros(1, 1,
                                                   self.embedding_planes))

        self.position_encoding = nn.Parameter(torch.zeros(
            1, (self.image_size // self.patch_size)**2, self.embedding_planes),
                                              requires_grad=False)

        blocks = []
        for _ in range(self.block_nums):
            blocks.append(
                TransformerEncoderLayer(
                    self.embedding_planes,
                    self.head_nums,
                    feedforward_ratio=self.feedforward_ratio,
                    dropout_prob=dropout_prob,
                    drop_path_prob=0.))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(self.embedding_planes)

        self.fc = nn.Linear(self.embedding_planes,
                            self.patch_size * self.patch_size * 3)

        position_encoding_init = self.get_2d_sincos_position_encoding_init(
            self.position_encoding.shape[-1],
            int(image_size // patch_size),
            cls_token=False)
        self.position_encoding.data.copy_(
            torch.from_numpy(position_encoding_init).float().unsqueeze(0))

        nn.init.normal_(self.mask_token, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_2d_sincos_position_encoding_init(self,
                                             embedding_planes,
                                             patch_nums,
                                             cls_token=False):
        """
        return:
        position_encoding_init: [patch_nums*patch_nums, embedding_planes] or [1+patch_nums*patch_nums, embedding_planes] (w/ or w/o cls_token)
        """
        grid_h = np.arange(patch_nums, dtype=np.float32)
        grid_w = np.arange(patch_nums, dtype=np.float32)
        # here w goes first
        grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)

        grid = grid.reshape([2, 1, patch_nums, patch_nums])
        position_encoding_init = self.get_2d_sincos_position_encoding_from_grid(
            embedding_planes, grid)
        if cls_token:
            position_encoding_init = np.concatenate(
                [np.zeros([1, embedding_planes]), position_encoding_init],
                axis=0)

        return position_encoding_init

    def get_2d_sincos_position_encoding_from_grid(self, embedding_planes,
                                                  grid):
        assert embedding_planes % 2 == 0

        # use half of dimensions to encode grid_h
        # (H*W, D/2)
        position_encoding_init_h = self.get_1d_sincos_position_encoding_from_grid(
            embedding_planes // 2, grid[0])
        # (H*W, D/2)
        position_encoding_init_w = self.get_1d_sincos_position_encoding_from_grid(
            embedding_planes // 2, grid[1])
        # (H*W, D)
        position_encoding_init = np.concatenate(
            [position_encoding_init_h, position_encoding_init_w], axis=1)

        return position_encoding_init

    def get_1d_sincos_position_encoding_from_grid(self, embedding_planes,
                                                  grid):
        omega = np.arange(embedding_planes // 2, dtype=np.float32)
        omega /= embedding_planes / 2.
        # (D/2,)
        omega = 1. / 10000**omega
        # (M,)
        grid = grid.reshape(-1)
        # (M, D/2), outer product
        out = np.einsum('m,d->md', grid, omega)
        # (M, D/2)
        position_encoding_init_sin = np.sin(out)
        # (M, D/2)
        position_encoding_init_cos = np.cos(out)
        # (M, D)
        position_encoding_init = np.concatenate(
            [position_encoding_init_sin, position_encoding_init_cos], axis=1)

        return position_encoding_init

    def forward(self, x, restore_ids):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0],
                                             restore_ids.shape[1] - x.shape[1],
                                             1)
        # no inclued cls token
        x = torch.cat([x, mask_tokens], dim=1)
        # unshuffle
        x = torch.gather(x,
                         dim=1,
                         index=restore_ids.unsqueeze(-1).repeat(
                             1, 1, x.shape[2]))

        x = x + self.position_encoding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.fc(x)

        return x


class PoolFormerMAEPretrainModel(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=32,
                 encoder_layer_nums=[6, 6, 18, 6],
                 encoder_planes=[64, 128, 320, 512],
                 encoder_pool_size=3,
                 encoder_feed_forward_ratio=4,
                 encoder_dropout_prob=0.,
                 encoder_drop_path_prob=0.,
                 encoder_layer_scale_factor=1e-5,
                 mask_ratio=0.75,
                 decoder_embedding_planes=384,
                 decoder_block_nums=4,
                 decoder_head_nums=6,
                 decoder_feedforward_ratio=4,
                 decoder_dropout_prob=0.):
        super(PoolFormerMAEPretrainModel, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        assert self.image_size % self.patch_size == 0

        self.encoder = PoolFormerMAEPretrainModelEncoder(
            image_size=image_size,
            patch_size=patch_size,
            layer_nums=encoder_layer_nums,
            planes=encoder_planes,
            pool_size=encoder_pool_size,
            feed_forward_ratio=encoder_feed_forward_ratio,
            dropout_prob=encoder_dropout_prob,
            drop_path_prob=encoder_drop_path_prob,
            layer_scale_factor=encoder_layer_scale_factor,
            mask_ratio=mask_ratio)

        self.decoder = PoolFormerMAEPretrainModelDecoder(
            patch_size=patch_size,
            image_size=image_size,
            embedding_planes=decoder_embedding_planes,
            block_nums=decoder_block_nums,
            head_nums=decoder_head_nums,
            feedforward_ratio=decoder_feedforward_ratio,
            dropout_prob=decoder_dropout_prob)

        self.encoder_to_decoder = nn.Linear(self.encoder.out_planes,
                                            decoder_embedding_planes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, mask, restore_ids = self.encoder(x)
        x = self.encoder_to_decoder(x)
        x = self.decoder(x, restore_ids)

        return x, mask

    def images_to_patch(self, images):
        """
        images: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        patch_nums = self.image_size // self.patch_size
        x = images.reshape(images.shape[0], 3, patch_nums, self.patch_size,
                           patch_nums, self.patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(x.shape[0], patch_nums * patch_nums,
                             self.patch_size * self.patch_size * 3))

        return x

    def patch_to_images(self, x):
        """
        x: (N, L, patch_size**2 *3)
        images: (N, 3, H, W)
        """
        h, w = int(x.shape[1]**0.5), int(x.shape[1]**0.5)

        images = x.reshape(shape=(x.shape[0], h, w, self.patch_size,
                                  self.patch_size, 3))
        images = torch.einsum('nhwpqc->nchpwq', images)
        images = images.reshape(shape=(images.shape[0], 3, h * self.patch_size,
                                       h * self.patch_size))

        return images


def _poolformermaepretrainmodel(**kwargs):
    model = PoolFormerMAEPretrainModel(**kwargs)

    return model


def poolformer_s36_patch32_224_mae_pretrain_model(**kwargs):
    return _poolformermaepretrainmodel(image_size=224,
                                       patch_size=32,
                                       encoder_layer_nums=[6, 6, 18, 6],
                                       encoder_planes=[64, 128, 320, 512],
                                       encoder_pool_size=3,
                                       encoder_feed_forward_ratio=4,
                                       encoder_dropout_prob=0.,
                                       encoder_drop_path_prob=0.,
                                       encoder_layer_scale_factor=1e-5,
                                       mask_ratio=0.75,
                                       decoder_embedding_planes=512,
                                       decoder_block_nums=8,
                                       decoder_head_nums=16,
                                       decoder_feedforward_ratio=4,
                                       decoder_dropout_prob=0.,
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

    net = poolformer_s36_patch32_224_mae_pretrain_model()
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out, mask = net(
        torch.autograd.Variable(torch.randn(1, 3, image_h, image_w)))
    print(
        f'1111, macs: {macs}, params: {params}, out_shape: {out.shape}, mask_shape: {mask.shape}'
    )