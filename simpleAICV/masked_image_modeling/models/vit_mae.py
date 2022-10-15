import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn

from simpleAICV.classification.backbones.vit import PatchEmbeddingBlock, TransformerEncoderLayer

__all__ = [
    'vit_tiny_patch16_224_mae_pretrain_model',
    'vit_small_patch16_224_mae_pretrain_model',
    'vit_base_patch16_224_mae_pretrain_model',
    'vit_large_patch16_224_mae_pretrain_model',
    'vit_huge_patch14_224_mae_pretrain_model',
]


class VITMAEPretrainModelEncoder(nn.Module):

    def __init__(self,
                 patch_size,
                 image_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 mask_ratio=0.75,
                 dropout_prob=0.):
        super(VITMAEPretrainModelEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio
        self.mask_ratio = mask_ratio

        self.patch_embedding = PatchEmbeddingBlock(3,
                                                   self.embedding_planes,
                                                   kernel_size=self.patch_size,
                                                   stride=self.patch_size,
                                                   padding=0,
                                                   groups=1,
                                                   has_norm=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_planes))
        self.position_encoding = nn.Parameter(torch.zeros(
            1, (self.image_size // self.patch_size)**2 + 1,
            self.embedding_planes),
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

        # initialize (and freeze) position encoding by sin-cos embedding
        position_encoding_init = self.get_2d_sincos_position_encoding_init(
            self.position_encoding.shape[-1],
            int(image_size // patch_size),
            cls_token=True)
        self.position_encoding.data.copy_(
            torch.from_numpy(position_encoding_init).float().unsqueeze(0))

        for m in self.patch_embedding.modules():
            if isinstance(m, nn.Conv2d):
                # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
                w = m.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=.02)

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

    def forward(self, x):
        x = self.patch_embedding(x)

        x = x + self.position_encoding[:, 1:, :]

        keep_ids, mask, restore_ids = self.random_masking(x)

        # mask x
        x = torch.gather(x,
                         dim=1,
                         index=keep_ids.unsqueeze(-1).repeat(
                             1, 1, x.shape[-1]))

        # append cls token
        cls_token = self.cls_token + self.position_encoding[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x, mask, restore_ids

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, N, C], B is batch_size, C is embedding_planes
        """
        b, n, c = x.shape  # batch, length, dim
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


class VITMAEPretrainModelDecoder(nn.Module):

    def __init__(self,
                 patch_size,
                 image_size,
                 embedding_planes,
                 block_nums,
                 head_nums,
                 feedforward_ratio,
                 dropout_prob=0.1):
        super(VITMAEPretrainModelDecoder, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.embedding_planes = embedding_planes
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.feedforward_ratio = feedforward_ratio

        self.mask_token = nn.Parameter(torch.zeros(1, 1,
                                                   self.embedding_planes))

        self.position_encoding = nn.Parameter(torch.zeros(
            1, (self.image_size // self.patch_size)**2 + 1,
            self.embedding_planes),
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
            cls_token=True)
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
        mask_tokens = self.mask_token.repeat(
            x.shape[0], restore_ids.shape[1] + 1 - x.shape[1], 1)
        # no inclued cls token
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        # unshuffle
        x_ = torch.gather(x_,
                          dim=1,
                          index=restore_ids.unsqueeze(-1).repeat(
                              1, 1, x.shape[2]))
        # append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.position_encoding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.fc(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class VITMAEPretrainModel(nn.Module):

    def __init__(self,
                 patch_size=16,
                 image_size=224,
                 mask_ratio=0.75,
                 encoder_embedding_planes=768,
                 encoder_block_nums=12,
                 encoder_head_nums=12,
                 encoder_feedforward_ratio=4,
                 encoder_dropout_prob=0.,
                 decoder_embedding_planes=384,
                 decoder_block_nums=4,
                 decoder_head_nums=6,
                 decoder_feedforward_ratio=4,
                 decoder_dropout_prob=0.):
        super(VITMAEPretrainModel, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        assert self.image_size % self.patch_size == 0

        self.encoder = VITMAEPretrainModelEncoder(
            patch_size=patch_size,
            image_size=image_size,
            embedding_planes=encoder_embedding_planes,
            block_nums=encoder_block_nums,
            head_nums=encoder_head_nums,
            feedforward_ratio=encoder_feedforward_ratio,
            mask_ratio=mask_ratio,
            dropout_prob=encoder_dropout_prob)

        self.decoder = VITMAEPretrainModelDecoder(
            patch_size=patch_size,
            image_size=image_size,
            embedding_planes=decoder_embedding_planes,
            block_nums=decoder_block_nums,
            head_nums=decoder_head_nums,
            feedforward_ratio=decoder_feedforward_ratio,
            dropout_prob=decoder_dropout_prob)

        self.encoder_to_decoder = nn.Linear(encoder_embedding_planes,
                                            decoder_embedding_planes)

        for m in self.encoder_to_decoder.modules():
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


def _vitmaepretrainmodel(**kwargs):
    model = VITMAEPretrainModel(**kwargs)

    return model


def vit_tiny_patch16_224_mae_pretrain_model(**kwargs):
    return _vitmaepretrainmodel(patch_size=16,
                                image_size=224,
                                encoder_embedding_planes=192,
                                encoder_block_nums=12,
                                encoder_head_nums=3,
                                encoder_feedforward_ratio=4,
                                encoder_dropout_prob=0.,
                                decoder_embedding_planes=192,
                                decoder_block_nums=4,
                                decoder_head_nums=3,
                                decoder_feedforward_ratio=4,
                                decoder_dropout_prob=0.,
                                **kwargs)


def vit_small_patch16_224_mae_pretrain_model(**kwargs):
    return _vitmaepretrainmodel(patch_size=16,
                                image_size=224,
                                encoder_embedding_planes=384,
                                encoder_block_nums=12,
                                encoder_head_nums=6,
                                encoder_feedforward_ratio=4,
                                encoder_dropout_prob=0.,
                                decoder_embedding_planes=192,
                                decoder_block_nums=4,
                                decoder_head_nums=3,
                                decoder_feedforward_ratio=4,
                                decoder_dropout_prob=0.,
                                **kwargs)


def vit_base_patch16_224_mae_pretrain_model(**kwargs):
    return _vitmaepretrainmodel(patch_size=16,
                                image_size=224,
                                encoder_embedding_planes=768,
                                encoder_block_nums=12,
                                encoder_head_nums=12,
                                encoder_feedforward_ratio=4,
                                encoder_dropout_prob=0.,
                                decoder_embedding_planes=512,
                                decoder_block_nums=8,
                                decoder_head_nums=16,
                                decoder_feedforward_ratio=4,
                                decoder_dropout_prob=0.,
                                **kwargs)


def vit_large_patch16_224_mae_pretrain_model(**kwargs):
    return _vitmaepretrainmodel(patch_size=16,
                                image_size=224,
                                encoder_embedding_planes=1024,
                                encoder_block_nums=24,
                                encoder_head_nums=16,
                                encoder_feedforward_ratio=4,
                                encoder_dropout_prob=0.,
                                decoder_embedding_planes=512,
                                decoder_block_nums=8,
                                decoder_head_nums=16,
                                decoder_feedforward_ratio=4,
                                decoder_dropout_prob=0.,
                                **kwargs)


def vit_huge_patch14_224_mae_pretrain_model(**kwargs):
    return _vitmaepretrainmodel(patch_size=14,
                                image_size=224,
                                encoder_embedding_planes=1280,
                                encoder_block_nums=32,
                                encoder_head_nums=16,
                                encoder_feedforward_ratio=4,
                                encoder_dropout_prob=0.,
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

    net = vit_tiny_patch16_224_mae_pretrain_model()
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

    net = vit_small_patch16_224_mae_pretrain_model()
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
        f'2222, macs: {macs}, params: {params}, out_shape: {out.shape}, mask_shape: {mask.shape}'
    )

    net = vit_base_patch16_224_mae_pretrain_model()
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
        f'3333, macs: {macs}, params: {params}, out_shape: {out.shape}, mask_shape: {mask.shape}'
    )

    net = vit_large_patch16_224_mae_pretrain_model()
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
        f'4444, macs: {macs}, params: {params}, out_shape: {out.shape}, mask_shape: {mask.shape}'
    )

    net = vit_huge_patch14_224_mae_pretrain_model()
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
        f'5555, macs: {macs}, params: {params}, out_shape: {out.shape}, mask_shape: {mask.shape}'
    )
