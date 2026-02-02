import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MaskDownSampler(nn.Module):

    def __init__(self,
                 inplanes=256,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 total_stride=16):
        super(MaskDownSampler, self).__init__()
        layer_nums = int(math.log2(total_stride) // math.log2(stride))
        assert stride**layer_nums == total_stride

        self.encoder = nn.Sequential()
        mask_inplanes, mask_out_planes = 1, 1
        for _ in range(layer_nums):
            mask_out_planes = mask_inplanes * (stride**2)
            self.encoder.append(
                nn.Conv2d(mask_inplanes,
                          mask_out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding))
            self.encoder.append(LayerNorm2d(mask_out_planes))
            self.encoder.append(nn.GELU())
            mask_inplanes = mask_out_planes

        self.encoder.append(nn.Conv2d(mask_out_planes, inplanes,
                                      kernel_size=1))

    def forward(self, x):
        x = self.encoder(x)

        return x


class CXBlock(nn.Module):

    def __init__(self, inplanes=256):
        super(CXBlock, self).__init__()
        self.dwconv = nn.Conv2d(inplanes,
                                inplanes,
                                kernel_size=7,
                                padding=3,
                                groups=inplanes)
        self.norm = LayerNorm2d(inplanes, eps=1e-6)

        self.pwconv1 = nn.Linear(inplanes, 4 * inplanes)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * inplanes, inplanes)
        self.gamma = nn.Parameter(1e-6 * torch.ones((inplanes)),
                                  requires_grad=True)

    def forward(self, x):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.gamma * x

        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = input + x

        return x


class Fuser(nn.Module):

    def __init__(self, inplanes, layer_nums=2):
        super(Fuser, self).__init__()
        self.layers = nn.ModuleList(
            [CXBlock(inplanes=inplanes) for _ in range(layer_nums)])

    def forward(self, x):
        # normally x: (N, C, H, W)
        for layer in self.layers:
            x = layer(x)

        return x


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000):
        super(PositionEmbeddingSine, self).__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature

        self.scale = 2 * math.pi

    def forward(self, x):
        device = x.device

        y_embed = (torch.arange(1, x.shape[-2] + 1, dtype=torch.float32).view(
            1, -1, 1).repeat(x.shape[0], 1, x.shape[-1])).to(device)
        x_embed = (torch.arange(1, x.shape[-1] + 1, dtype=torch.float32).view(
            1, 1, -1).repeat(x.shape[0], x.shape[-2], 1)).to(device)

        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32).to(device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.to(x.dtype)

        return pos


class MemoryEncoder(nn.Module):

    def __init__(self, inplanes=256, planes=64):
        super(MemoryEncoder, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.mask_downsampler = MaskDownSampler(inplanes=inplanes,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                total_stride=16)

        self.pix_feat_proj = nn.Conv2d(inplanes,
                                       inplanes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        self.fuser = Fuser(inplanes=inplanes, layer_nums=2)

        self.position_encoding = PositionEmbeddingSine(num_pos_feats=planes,
                                                       temperature=10000)

        if self.inplanes != self.planes:
            self.out_proj = nn.Conv2d(inplanes, planes, kernel_size=1)

    def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
        device = masks.device

        # Process masks sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)

        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        pix_feat = pix_feat.to(device)

        features = self.pix_feat_proj(pix_feat)
        features = features + masks
        features = self.fuser(features)

        if self.inplanes != self.planes:
            features = self.out_proj(features)

        positions = self.position_encoding(features)

        return features, positions
