import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaFPN(nn.Module):
    def __init__(self, C3_inplanes, C4_inplanes, C5_inplanes, planes):
        super(RetinaFPN, self).__init__()
        self.P3_1 = nn.Conv2d(C3_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P4_1 = nn.Conv2d(C4_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P5_1 = nn.Conv2d(C5_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P5_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P6 = nn.Conv2d(C5_inplanes,
                            planes,
                            kernel_size=3,
                            stride=2,
                            padding=1)

        self.P7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5, size=(P4.shape[2], P4.shape[3]),
                           mode='nearest') + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3

        P6 = self.P6(C5)
        P7 = self.P7(P6)

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        del C3, C4, C5

        return [P3, P4, P5, P6, P7]


if __name__ == '__main__':
    image_h, image_w = 640, 640
    fpn = RetinaFPN(512, 1024, 2048, 256)
    C3, C4, C5 = torch.randn(3, 512, 80, 80), torch.randn(3, 1024, 40,
                                                          40), torch.randn(
                                                              3, 2048, 20, 20)
    features = fpn([C3, C4, C5])

    for feature in features:
        print("1111", feature.shape)