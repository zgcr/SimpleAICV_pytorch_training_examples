import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import spectral_norm

__all__ = [
    'AOTGANGeneratorModel',
    'AOTGANDiscriminatorModel',
]


class AOTGANGeneratorModel(nn.Module):

    def __init__(self, planes=[64, 128, 256], rates=[1, 2, 4, 8], block_num=8):
        super(AOTGANGeneratorModel, self).__init__()
        assert len(planes) == 3

        self.encoder = nn.Sequential(
            nn.Conv2d(4, planes[0], kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=2,
                      padding=1),
            nn.ReLU(True),
            nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=2,
                      padding=1),
            nn.ReLU(True),
        )

        self.middle = nn.ModuleList(
            [AOTBlock(planes[2], rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(planes[2], planes[1]),
            nn.ReLU(True),
            UpConv(planes[1], planes[0]),
            nn.ReLU(True),
            nn.Conv2d(planes[0], 3, kernel_size=7, stride=1, padding=3),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)

        x = self.encoder(x)

        for per_middle_layer in self.middle:
            x = per_middle_layer(x)

        x = self.decoder(x)

        x = torch.tanh(x)

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


class AOTBlock(nn.Module):

    def __init__(self, inplanes, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.Conv2d(inplanes,
                              inplanes // 4,
                              kernel_size=3,
                              stride=1,
                              padding=rate,
                              dilation=rate), nn.ReLU(True)),
            )

        self.fuse = nn.Conv2d(inplanes,
                              inplanes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1)
        self.gate = nn.Conv2d(inplanes,
                              inplanes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1)

    def forward(self, x):
        out = [
            self.__getattr__(f'block{str(i).zfill(2)}')(x)
            for i in range(len(self.rates))
        ]

        out = torch.cat(out, dim=1)
        out = self.fuse(out)
        mask = self.gate(x)
        mask = torch.sigmoid(mask)

        return x * (1 - mask) + out * mask


class AOTGANDiscriminatorModel(nn.Module):

    def __init__(self, planes=[64, 128, 256, 512]):
        super(AOTGANDiscriminatorModel, self).__init__()
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

    generator_net = AOTGANGeneratorModel(planes=[64, 96, 128],
                                         rates=[1, 2, 4, 8],
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

    generator_net = AOTGANGeneratorModel(planes=[64, 128, 256],
                                         rates=[1, 2, 4, 8],
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

    discriminator_net = AOTGANDiscriminatorModel(planes=[64, 128, 256, 512])
    batch, channel, image_h, image_w = 1, 3, 512, 512
    images = torch.randn(batch, channel, image_h, image_w).float()

    from thop import profile
    from thop import clever_format
    macs, params = profile(discriminator_net, inputs=(images, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = discriminator_net(torch.autograd.Variable(images), )
    print('2222', outs.shape)
