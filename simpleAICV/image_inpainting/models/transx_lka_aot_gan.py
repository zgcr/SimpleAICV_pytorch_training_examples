import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import spectral_norm

__all__ = [
    'TRANSXLKAAOTGANGeneratorModel',
    'TRANSXLKAAOTGANDiscriminatorModel',
]


class TRANSXLKAAOTGANGeneratorModel(nn.Module):

    def __init__(self, planes=[64, 128, 256, 512], block_num=8, head_num=4):
        super(TRANSXLKAAOTGANGeneratorModel, self).__init__()
        assert len(planes) == 4

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(4, planes[0], kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True))

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=2,
                      padding=1), nn.ReLU(inplace=True))

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=2,
                      padding=1), nn.ReLU(inplace=True))

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(planes[2], planes[3], kernel_size=3, stride=2,
                      padding=1), nn.ReLU(inplace=True))

        self.middle = nn.ModuleList([
            LKAAOTBlock(planes[3], head_num=head_num) for _ in range(block_num)
        ])

        self.decoder_layer4 = nn.Sequential(UpConv(planes[3], planes[2]),
                                            nn.ReLU(inplace=True))
        self.decoder_layer3 = nn.Sequential(UpConv(planes[2], planes[1]),
                                            nn.ReLU(inplace=True))
        self.decoder_layer2 = nn.Sequential(UpConv(planes[1], planes[0]),
                                            nn.ReLU(inplace=True))
        self.decoder_layer1 = nn.Conv2d(planes[0],
                                        3,
                                        kernel_size=7,
                                        stride=1,
                                        padding=3)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)

        x1_down = self.encoder_layer1(x)
        x2_down = self.encoder_layer2(x1_down)
        x3_down = self.encoder_layer3(x2_down)
        x4_down = self.encoder_layer4(x3_down)

        x_middle = x4_down
        for per_middle_layer in self.middle:
            x_middle = per_middle_layer(x_middle)

        x_middle = x_middle + x4_down

        x4_up = self.decoder_layer4(x_middle)
        x4_up = x4_up + x3_down

        x3_up = self.decoder_layer3(x4_up)
        x3_up = x3_up + x2_down

        x2_up = self.decoder_layer2(x3_up)
        x2_up = x2_up + x1_down

        x1_up = self.decoder_layer1(x2_up)

        x = torch.tanh(x1_up)

        return x


class UpConv(nn.Module):

    def __init__(self, inplanes, planes):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 dilation=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class LKA(nn.Module):

    def __init__(self, inplanes):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(inplanes,
                               inplanes,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               groups=inplanes)
        self.conv_spatial = nn.Conv2d(inplanes,
                                      inplanes,
                                      kernel_size=7,
                                      stride=1,
                                      padding=9,
                                      dilation=3,
                                      groups=inplanes)
        self.conv1 = nn.Conv2d(inplanes,
                               inplanes,
                               kernel_size=1,
                               stride=1,
                               padding=0)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):

    def __init__(self, inplanes, sr_ratio, head_num):
        super(Attention, self).__init__()
        assert inplanes % head_num == 0
        self.head_num = head_num

        head_inplanes = inplanes // head_num
        self.scale = head_inplanes**-0.5

        self.q = nn.Conv2d(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.kv = nn.Conv2d(inplanes,
                            inplanes * 2,
                            kernel_size=1,
                            stride=1,
                            padding=0)

        self.sr = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes,
                           kernel_size=sr_ratio,
                           stride=sr_ratio,
                           padding=sr_ratio // 2,
                           groups=1 if sr_ratio == 1 else inplanes,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           dilation=1,
                           has_bn=True,
                           has_act=False))

        self.local_conv = nn.Conv2d(inplanes,
                                    inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=inplanes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        origin_h, origin_w = x.shape[2], x.shape[3]

        x = F.interpolate(x,
                          size=(x.shape[2] // 4, x.shape[3] // 4),
                          mode='nearest')

        B, C, H, W = x.shape

        q = self.q(x)
        # [B,head_num,C//head_num,H*W]->[B,head_num,H*W,C//head_num]
        q = q.reshape(B, self.head_num, C // self.head_num,
                      -1).transpose(-1, -2)

        kv = self.sr(x)
        kv = self.local_conv(kv) + kv

        kv = self.kv(kv)
        k, v = torch.chunk(kv, chunks=2, dim=1)

        # [B,head_num,C//head_num,H*W]
        k = k.reshape(B, self.head_num, C // self.head_num, -1)
        # [B,head_num,C//head_num,H*W]->[B,head_num,H*W,C//head_num]
        v = v.reshape(B, self.head_num, C // self.head_num,
                      -1).transpose(-1, -2)

        # [B,1,H*W,H*W]
        attn = (q @ k) * self.scale
        attn = self.softmax(attn)

        x = (attn @ v).transpose(-1, -2)
        x = x.reshape(B, C, H, W)

        x = F.interpolate(x, size=(origin_w, origin_h), mode='nearest')

        return x


class HybridTokenMixer(nn.Module):
    '''
    D-Mixer
    '''

    def __init__(self, inplanes, sr_ratio, head_num, reduction_ratio=8):
        super(HybridTokenMixer, self).__init__()
        assert inplanes % 2 == 0

        self.local_unit = LKA(inplanes=inplanes // 2)
        self.global_unit = Attention(inplanes=inplanes // 2,
                                     sr_ratio=sr_ratio,
                                     head_num=head_num)

        inter_planes = max(16, inplanes // reduction_ratio)
        self.proj = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=inplanes,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(inplanes,
                           inter_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
            ConvBnActBlock(inter_planes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           dilation=1,
                           has_bn=True,
                           has_act=False))

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self.proj(x) + x

        return x


class Mlp(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes):
        super(Mlp, self).__init__()
        self.fc1 = nn.Conv2d(inplanes,
                             hidden_planes,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        self.dwconv_k1 = nn.Conv2d(hidden_planes // 4,
                                   hidden_planes // 4,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.dwconv_k3 = nn.Conv2d(hidden_planes // 4,
                                   hidden_planes // 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=hidden_planes // 4)
        self.dwconv_k5 = nn.Conv2d(hidden_planes // 4,
                                   hidden_planes // 4,
                                   kernel_size=5,
                                   stride=1,
                                   padding=2,
                                   groups=hidden_planes // 4)
        self.dwconv_k7 = nn.Conv2d(hidden_planes // 4,
                                   hidden_planes // 4,
                                   kernel_size=7,
                                   stride=1,
                                   padding=3,
                                   groups=hidden_planes // 4)

        self.act = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(hidden_planes,
                             planes,
                             kernel_size=1,
                             stride=1,
                             padding=0)

    def forward(self, x):
        x = self.fc1(x)

        x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)

        x1 = self.dwconv_k1(x1)
        x2 = self.dwconv_k3(x2)
        x3 = self.dwconv_k5(x3)
        x4 = self.dwconv_k7(x4)

        x_combine = torch.cat([x1, x2, x3, x4], dim=1)

        x = x_combine + x

        x = self.act(x)

        x = self.fc2(x)

        return x


class Block(nn.Module):

    def __init__(self, inplanes, sr_ratio, head_num, mlp_ratio=4.):
        super(Block, self).__init__()
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.attn = HybridTokenMixer(inplanes, sr_ratio, head_num)

        self.norm2 = nn.BatchNorm2d(inplanes)
        self.mlp = Mlp(inplanes=inplanes,
                       hidden_planes=int(inplanes * mlp_ratio),
                       planes=inplanes)
        self.layer_scale_1 = nn.Parameter(0.01 * torch.ones(
            (1, inplanes, 1, 1)),
                                          requires_grad=True)
        self.layer_scale_2 = nn.Parameter(0.01 * torch.ones(
            (1, inplanes, 1, 1)),
                                          requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1 * self.attn(self.norm1(x))
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))

        return x


class LKAAOTBlock(nn.Module):

    def __init__(self, inplanes, head_num):
        super(LKAAOTBlock, self).__init__()
        self.block = nn.Sequential(
            Block(inplanes=inplanes,
                  sr_ratio=1,
                  head_num=head_num,
                  mlp_ratio=4.), nn.ReLU(inplace=True))

        self.fuse = nn.Conv2d(inplanes,
                              inplanes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.gate = nn.Conv2d(inplanes,
                              inplanes,
                              kernel_size=1,
                              stride=1,
                              padding=0)

    def forward(self, x):
        out = self.block(x)

        out = self.fuse(out)
        mask = self.gate(x)
        mask = torch.sigmoid(mask)

        return x * (1 + mask * (-1)) + out * mask


class TRANSXLKAAOTGANDiscriminatorModel(nn.Module):

    def __init__(self, planes=[64, 128, 256, 512]):
        super(TRANSXLKAAOTGANDiscriminatorModel, self).__init__()
        assert len(planes) == 4

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(3,
                          planes[0],
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(planes[0],
                          planes[1],
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(planes[1],
                          planes[2],
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(planes[2],
                          planes[3],
                          kernel_size=4,
                          stride=1,
                          padding=1,
                          bias=False)), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes[3], 1, kernel_size=4, stride=1, padding=1))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        feat = self.conv(x)

        return feat


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

    generator_net = TRANSXLKAAOTGANGeneratorModel(planes=[64, 128, 256, 512],
                                                  block_num=8)
    batch, channel, image_h, image_w = 1, 3, 512, 512
    images = torch.randn(batch, channel, image_h, image_w).float()
    masks = torch.randint(0, 2, size=(batch, 1, image_h, image_w)).float()

    from thop import profile
    from thop import clever_format
    macs, params = profile(generator_net,
                           inputs=(images, masks),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = generator_net(torch.autograd.Variable(images),
                         torch.autograd.Variable(masks))
    print('2222', outs.shape)

    generator_net = TRANSXLKAAOTGANGeneratorModel(planes=[64, 96, 128, 256],
                                                  block_num=8)
    batch, channel, image_h, image_w = 1, 3, 832, 832
    images = torch.randn(batch, channel, image_h, image_w).float()
    masks = torch.randint(0, 2, size=(batch, 1, image_h, image_w)).float()

    from thop import profile
    from thop import clever_format
    macs, params = profile(generator_net,
                           inputs=(images, masks),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = generator_net(torch.autograd.Variable(images),
                         torch.autograd.Variable(masks))
    print('2222', outs.shape)

    discriminator_net = TRANSXLKAAOTGANDiscriminatorModel(
        planes=[64, 128, 256, 512])
    batch, channel, image_h, image_w = 1, 3, 512, 512
    images = torch.randn(batch, channel, image_h, image_w).float()

    from thop import profile
    from thop import clever_format
    macs, params = profile(discriminator_net, inputs=(images, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = discriminator_net(torch.autograd.Variable(images), )
    print('2222', outs.shape)
