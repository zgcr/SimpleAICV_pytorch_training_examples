import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMFPN(nn.Module):

    def __init__(self, inplanes, planes):
        super(SAMFPN, self).__init__()
        # inplanes:[C2_inplanes,C3_inplanes,C4_inplanes,C5_inplanes]
        self.P2_1 = nn.Conv2d(inplanes[0],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_1 = nn.Conv2d(inplanes[1],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_1 = nn.Conv2d(inplanes[2],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P5_1 = nn.Conv2d(inplanes[3],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)

        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        [C2, C3, C4, C5] = inputs

        P5 = self.P5_1(C5)

        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5,
                           size=(P4.shape[2], P4.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P4

        P3 = self.P3_1(C3)
        P4 = F.interpolate(P3,
                           size=(P4.shape[2], P4.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P4

        P2 = self.P2_1(C2)
        P4 = F.interpolate(P2,
                           size=(P4.shape[2], P4.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P4

        P4 = self.P4_2(P4)

        return P4


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

    net = SAMFPN([256, 512, 1024, 2048], 256)
    C2, C3, C4, C5 = torch.randn(3, 256, 256, 256), torch.randn(
        3, 512, 128, 128), torch.randn(3, 1024, 64,
                                       64), torch.randn(3, 2048, 32, 32)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=([C2, C3, C4, C5], ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net([C2, C3, C4, C5])
    print('2222', outs.shape)
