import torch
import torch.nn as nn
import torch.nn.functional as F


class Solov2FPN(nn.Module):
    def __init__(self, C2_inplanes, C3_inplanes, C4_inplanes, C5_inplanes,
                 planes):
        super(Solov2FPN, self).__init__()
        self.P5_1 = nn.Conv2d(C5_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=True)
        self.P4_1 = nn.Conv2d(C4_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=True)
        self.P3_1 = nn.Conv2d(C3_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=True)
        self.P2_1 = nn.Conv2d(C2_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=True)

        self.P5_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.P2_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.P6_1 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, inputs):
        [C2, C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        del C5
        P4 = self.P4_1(C4)
        del C4
        P3 = self.P3_1(C3)
        del C3
        P2 = self.P2_1(C2)
        del C2

        P4 = F.interpolate(P5, size=(P4.shape[2], P4.shape[3]),
                           mode='nearest') + P4
        P3 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3
        P2 = F.interpolate(P3, size=(P2.shape[2], P2.shape[3]),
                           mode='nearest') + P2

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)
        P2 = self.P2_2(P2)

        P6 = self.P6_1(P5)

        return [P2, P3, P4, P5, P6]


if __name__ == '__main__':
    image_h, image_w = 640, 640
    fpn = Solov2FPN(256, 512, 1024, 2048, 256)
    C2, C3, C4, C5 = torch.randn(3, 256, 160, 160), torch.randn(
        3, 512, 80, 80), torch.randn(3, 1024, 40,
                                     40), torch.randn(3, 2048, 20, 20)
    features = fpn([C2, C3, C4, C5])

    for feature in features:
        print('1111', feature.shape)