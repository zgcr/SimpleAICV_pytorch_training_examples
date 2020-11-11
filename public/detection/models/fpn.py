import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaFPN(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 planes,
                 use_p5=False):
        super(RetinaFPN, self).__init__()
        self.use_p5 = use_p5
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
        if self.use_p5:
            self.P6 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        else:
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

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        if self.use_p5:
            P6 = self.P6(P5)
        else:
            P6 = self.P6(C5)

        del C3, C4, C5

        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=kernel_size // 2,
                              groups=groups,
                              bias=False)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(planes)
        if self.has_act:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.act(x)

        return x


class YOLOV3FPNHead(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 num_anchors=3,
                 num_classes=80):
        super(YOLOV3FPNHead, self).__init__()
        P5_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P5_1_layers.append(
                    ConvBnActBlock(C5_inplanes,
                                   C5_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            else:
                P5_1_layers.append(
                    ConvBnActBlock(C5_inplanes // 2,
                                   C5_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
        self.P5_1 = nn.Sequential(*P5_1_layers)
        self.P5_up_conv = ConvBnActBlock(C5_inplanes // 2,
                                         C4_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         has_bn=True,
                                         has_act=True)
        self.P5_2 = ConvBnActBlock(C5_inplanes // 2,
                                   C5_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)
        self.P5_pred_conv = nn.Conv2d(C5_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      bias=True)

        P4_1_layers = []
        for i in range(5):
            if i == 0:
                P4_1_layers.append(
                    ConvBnActBlock((C4_inplanes // 2) + C4_inplanes,
                                   C4_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 1:
                P4_1_layers.append(
                    ConvBnActBlock(C4_inplanes // 2,
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 0:
                P4_1_layers.append(
                    ConvBnActBlock(C4_inplanes,
                                   C4_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
        self.P4_1 = nn.Sequential(*P4_1_layers)
        self.P4_up_conv = ConvBnActBlock(C4_inplanes // 2,
                                         C3_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         has_bn=True,
                                         has_act=True)
        self.P4_2 = ConvBnActBlock(C4_inplanes // 2,
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)
        self.P4_pred_conv = nn.Conv2d(C4_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      bias=True)

        P3_1_layers = []
        for i in range(5):
            if i == 0:
                P3_1_layers.append(
                    ConvBnActBlock((C3_inplanes // 2) + C3_inplanes,
                                   C3_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 1:
                P3_1_layers.append(
                    ConvBnActBlock(C3_inplanes // 2,
                                   C3_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 0:
                P3_1_layers.append(
                    ConvBnActBlock(C3_inplanes,
                                   C3_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
        self.P3_1 = nn.Sequential(*P3_1_layers)
        self.P3_2 = ConvBnActBlock(C3_inplanes // 2,
                                   C3_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)
        self.P3_pred_conv = nn.Conv2d(C3_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      bias=True)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)

        del C5

        C5_upsample = self.P5_up_conv(P5)
        C5_upsample = F.interpolate(C5_upsample,
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')

        C4 = torch.cat([C4, C5_upsample], axis=1)
        del C5_upsample
        P4 = self.P4_1(C4)
        del C4
        C4_upsample = self.P4_up_conv(P4)
        C4_upsample = F.interpolate(C4_upsample,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='nearest')

        C3 = torch.cat([C3, C4_upsample], axis=1)
        del C4_upsample
        P3 = self.P3_1(C3)
        del C3

        P5 = self.P5_2(P5)
        P5 = self.P5_pred_conv(P5)

        P4 = self.P4_2(P4)
        P4 = self.P4_pred_conv(P4)

        P3 = self.P3_2(P3)
        P3 = self.P3_pred_conv(P3)

        return [P3, P4, P5]


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def hard_swish(self, x, inplace):
        inner = F.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if inplace else x.mul(inner)

    def forward(self, x):
        return self.hard_swish(x, self.inplace)


class SeparableConvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(inplanes,
                                        inplanes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=inplanes,
                                        bias=False)
        self.pointwise_conv = nn.Conv2d(inplanes,
                                        planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=True)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)

        return x


class EfficientDetBiFPN(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 planes,
                 first_time=False,
                 epsilon=1e-4):
        super(EfficientDetBiFPN, self).__init__()
        self.first_time = first_time
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(planes, planes)
        self.conv5_up = SeparableConvBlock(planes, planes)
        self.conv4_up = SeparableConvBlock(planes, planes)
        self.conv3_up = SeparableConvBlock(planes, planes)
        self.conv4_down = SeparableConvBlock(planes, planes)
        self.conv5_down = SeparableConvBlock(planes, planes)
        self.conv6_down = SeparableConvBlock(planes, planes)
        self.conv7_down = SeparableConvBlock(planes, planes)

        self.p4_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p5_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p6_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p7_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.hardswish = HardSwish(inplace=True)

        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                nn.Conv2d(C5_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p4_down_channel = nn.Sequential(
                nn.Conv2d(C4_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p3_down_channel = nn.Sequential(
                nn.Conv2d(C3_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p5_to_p6 = nn.Sequential(
                nn.Conv2d(C5_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.p6_to_p7 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1), )

            self.p4_down_channel_2 = nn.Sequential(
                nn.Conv2d(C4_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p5_down_channel_2 = nn.Sequential(
                nn.Conv2d(C5_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.first_time:
            [C3, C4, C5] = inputs

            P3 = self.p3_down_channel(C3)
            P4 = self.p4_down_channel(C4)
            P5 = self.p5_down_channel(C5)

            P6 = self.p5_to_p6(C5)
            P7 = self.p6_to_p7(P6)

        else:
            [P3, P4, P5, P6, P7] = inputs
        # P7_0 to P7_2
        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        P6_up = self.conv6_up(
            self.hardswish(weight[0] * P6 + weight[1] * F.interpolate(
                P7, size=(P6.shape[2], P6.shape[3]), mode='nearest')))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        P5_up = self.conv5_up(
            self.hardswish(weight[0] * P5 + weight[1] * F.interpolate(
                P6_up, size=(P5.shape[2], P5.shape[3]), mode='nearest')))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        P4_up = self.conv4_up(
            self.hardswish(weight[0] * P4 + weight[1] * F.interpolate(
                P5_up, size=(P4.shape[2], P4.shape[3]), mode='nearest')))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        P3_out = self.conv3_up(
            self.hardswish(weight[0] * P3 + weight[1] * F.interpolate(
                P4_up, size=(P3.shape[2], P3.shape[3]), mode='nearest')))

        if self.first_time:
            P4 = self.p4_down_channel_2(C4)
            P5 = self.p5_down_channel_2(C5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        P4_out = self.conv4_down(
            self.hardswish(weight[0] * P4 + weight[1] * P4_up +
                           weight[2] * self.p4_downsample(P3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        P5_out = self.conv5_down(
            self.hardswish(weight[0] * P5 + weight[1] * P5_up +
                           weight[2] * self.p5_downsample(P4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        P6_out = self.conv6_down(
            self.hardswish(weight[0] * P6 + weight[1] * P6_up +
                           weight[2] * self.p6_downsample(P5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        P7_out = self.conv7_down(
            self.hardswish(weight[0] * P7 +
                           weight[1] * self.p7_downsample(P6_out)))

        return [P3_out, P4_out, P5_out, P6_out, P7_out]


if __name__ == '__main__':
    image_h, image_w = 640, 640
    fpn = RetinaFPN(512, 1024, 2048, 256)
    C3, C4, C5 = torch.randn(3, 512, 80, 80), torch.randn(3, 1024, 40,
                                                          40), torch.randn(
                                                              3, 2048, 20, 20)
    features = fpn([C3, C4, C5])

    for feature in features:
        print("1111", feature.shape)

    image_h, image_w = 640, 640
    fpn = EfficientDetBiFPN(512,
                            1024,
                            2048,
                            256,
                            first_time=True,
                            epsilon=1e-4)
    C3, C4, C5 = torch.randn(3, 512, 80, 80), torch.randn(3, 1024, 40,
                                                          40), torch.randn(
                                                              3, 2048, 20, 20)
    features = fpn([C3, C4, C5])

    for feature in features:
        print("2222", feature.shape)

    image_h, image_w = 640, 640
    fpn = EfficientDetBiFPN(512,
                            1024,
                            2048,
                            256,
                            first_time=False,
                            epsilon=1e-4)
    P3, P4, P5, P6, P7 = torch.randn(3, 256, 80, 80), torch.randn(
        3, 256, 40,
        40), torch.randn(3, 256, 20,
                         20), torch.randn(3, 256, 10,
                                          10), torch.randn(3, 256, 5, 5)
    features = fpn([P3, P4, P5, P6, P7])

    for feature in features:
        print("3333", feature.shape)

    image_h, image_w = 416, 416
    fpn = YOLOV3FPNHead(256, 512, 1024, num_anchors=3, num_classes=80)
    C3, C4, C5 = torch.randn(3, 256, 52, 52), torch.randn(3, 512, 26,
                                                          26), torch.randn(
                                                              3, 1024, 13, 13)
    features = fpn([C3, C4, C5])

    for feature in features:
        print("4444", feature.shape)