import torch
import torch.nn as nn


class PAN(nn.Module):
    def __init__(self, planes):
        super(PAN, self).__init__()
        self.P3_down = nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.P4_down = nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.P5_down = nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.P6_down = nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)

    def forward(self, inputs):
        [P3, P4, P5, P6, P7] = inputs

        P3_downsample = self.P3_down(P3)
        P4 = P3_downsample + P4

        P4_downsample = self.P4_down(P4)
        P5 = P4_downsample + P5

        P5_downsample = self.P5_down(P5)
        P6 = P5_downsample + P6

        P6_downsample = self.P6_down(P6)
        P7 = P6_downsample + P7

        del P3_downsample, P4_downsample, P5_downsample, P6_downsample

        return [P3, P4, P5, P6, P7]