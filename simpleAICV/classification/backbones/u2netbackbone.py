import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'u2netbackbone',
    'u2netsmallbackbone',
]


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


### RSU-7 ###
class RSU7Block(nn.Module):

    def __init__(self, inplanes=3, middle_planes=12, planes=3):
        super(RSU7Block, self).__init__()

        self.rebnconvin = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

        self.rebnconv1 = ConvBnActBlock(planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv2 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv3 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv4 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv5 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv6 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv7 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=2,
                                        groups=1,
                                        dilation=2,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv6d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv5d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv4d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv3d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv2d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv1d = ConvBnActBlock(middle_planes * 2,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

    def forward(self, x):
        xin = self.rebnconvin(x)

        x1 = self.rebnconv1(xin)
        x = self.pool1(x1)

        x2 = self.rebnconv2(x)
        x = self.pool2(x2)

        x3 = self.rebnconv3(x)
        x = self.pool3(x3)

        x4 = self.rebnconv4(x)
        x = self.pool4(x4)

        x5 = self.rebnconv5(x)
        x = self.pool5(x5)

        x6 = self.rebnconv6(x)

        x7 = self.rebnconv7(x6)

        x6d = self.rebnconv6d(torch.cat((x7, x6), dim=1))
        x6dup = F.interpolate(x6d,
                              size=(x5.shape[2], x5.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x5d = self.rebnconv5d(torch.cat((x6dup, x5), dim=1))
        x5dup = F.interpolate(x5d,
                              size=(x4.shape[2], x4.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x4d = self.rebnconv4d(torch.cat((x5dup, x4), dim=1))
        x4dup = F.interpolate(x4d,
                              size=(x3.shape[2], x3.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x3d = self.rebnconv3d(torch.cat((x4dup, x3), dim=1))
        x3dup = F.interpolate(x3d,
                              size=(x2.shape[2], x2.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x2d = self.rebnconv2d(torch.cat((x3dup, x2), dim=1))
        x2dup = F.interpolate(x2d,
                              size=(x1.shape[2], x1.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x1d = self.rebnconv1d(torch.cat((x2dup, x1), dim=1))

        x1d = x1d + xin

        return x1d


### RSU-6 ###
class RSU6Block(nn.Module):

    def __init__(self, inplanes=3, middle_planes=12, planes=3):
        super(RSU6Block, self).__init__()

        self.rebnconvin = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

        self.rebnconv1 = ConvBnActBlock(planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv2 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv3 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv4 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv5 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv6 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=2,
                                        groups=1,
                                        dilation=2,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv5d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv4d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv3d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv2d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv1d = ConvBnActBlock(middle_planes * 2,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

    def forward(self, x):
        xin = self.rebnconvin(x)

        x1 = self.rebnconv1(xin)
        x = self.pool1(x1)

        x2 = self.rebnconv2(x)
        x = self.pool2(x2)

        x3 = self.rebnconv3(x)
        x = self.pool3(x3)

        x4 = self.rebnconv4(x)
        x = self.pool4(x4)

        x5 = self.rebnconv5(x)

        x6 = self.rebnconv6(x5)

        x5d = self.rebnconv5d(torch.cat((x6, x5), dim=1))
        x5dup = F.interpolate(x5d,
                              size=(x4.shape[2], x4.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x4d = self.rebnconv4d(torch.cat((x5dup, x4), dim=1))
        x4dup = F.interpolate(x4d,
                              size=(x3.shape[2], x3.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x3d = self.rebnconv3d(torch.cat((x4dup, x3), dim=1))
        x3dup = F.interpolate(x3d,
                              size=(x2.shape[2], x2.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x2d = self.rebnconv2d(torch.cat((x3dup, x2), dim=1))
        x2dup = F.interpolate(x2d,
                              size=(x1.shape[2], x1.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x1d = self.rebnconv1d(torch.cat((x2dup, x1), dim=1))
        x1d = x1d + xin

        return x1d


### RSU-5 ###
class RSU5Block(nn.Module):

    def __init__(self, inplanes=3, middle_planes=12, planes=3):
        super(RSU5Block, self).__init__()

        self.rebnconvin = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

        self.rebnconv1 = ConvBnActBlock(planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv2 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv3 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv4 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv5 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=2,
                                        groups=1,
                                        dilation=2,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv4d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv3d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv2d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv1d = ConvBnActBlock(middle_planes * 2,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

    def forward(self, x):
        xin = self.rebnconvin(x)

        x1 = self.rebnconv1(xin)
        x = self.pool1(x1)

        x2 = self.rebnconv2(x)
        x = self.pool2(x2)

        x3 = self.rebnconv3(x)
        x = self.pool3(x3)

        x4 = self.rebnconv4(x)

        x5 = self.rebnconv5(x4)

        x4d = self.rebnconv4d(torch.cat((x5, x4), dim=1))
        x4dup = F.interpolate(x4d,
                              size=(x3.shape[2], x3.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x3d = self.rebnconv3d(torch.cat((x4dup, x3), dim=1))
        x3dup = F.interpolate(x3d,
                              size=(x2.shape[2], x2.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x2d = self.rebnconv2d(torch.cat((x3dup, x2), dim=1))
        x2dup = F.interpolate(x2d,
                              size=(x1.shape[2], x1.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x1d = self.rebnconv1d(torch.cat((x2dup, x1), dim=1))
        x1d = x1d + xin

        return x1d


### RSU-4 ###
class RSU4Block(nn.Module):

    def __init__(self, inplanes=3, middle_planes=12, planes=3):
        super(RSU4Block, self).__init__()

        self.rebnconvin = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

        self.rebnconv1 = ConvBnActBlock(planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv2 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.rebnconv3 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv4 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=2,
                                        groups=1,
                                        dilation=2,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv3d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=2,
                                         groups=1,
                                         dilation=2,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv2d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=2,
                                         groups=1,
                                         dilation=2,
                                         has_bn=True,
                                         has_act=True)
        self.rebnconv1d = ConvBnActBlock(middle_planes * 2,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=2,
                                         groups=1,
                                         dilation=2,
                                         has_bn=True,
                                         has_act=True)

    def forward(self, x):
        xin = self.rebnconvin(x)

        x1 = self.rebnconv1(xin)
        x = self.pool1(x1)

        x2 = self.rebnconv2(x)
        x = self.pool2(x2)

        x3 = self.rebnconv3(x)

        x4 = self.rebnconv4(x3)

        x3d = self.rebnconv3d(torch.cat((x4, x3), dim=1))
        x3dup = F.interpolate(x3d,
                              size=(x2.shape[2], x2.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x2d = self.rebnconv2d(torch.cat((x3dup, x2), dim=1))
        x2dup = F.interpolate(x2d,
                              size=(x1.shape[2], x1.shape[3]),
                              mode='bilinear',
                              align_corners=True)

        x1d = self.rebnconv1d(torch.cat((x2dup, x1), dim=1))
        x1d = x1d + xin

        return x1d


### RSU-4F ###
class RSU4FBlock(nn.Module):

    def __init__(self, inplanes=3, middle_planes=12, planes=3):
        super(RSU4FBlock, self).__init__()

        self.rebnconvin = ConvBnActBlock(inplanes,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

        self.rebnconv1 = ConvBnActBlock(planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        dilation=1,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv2 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=2,
                                        groups=1,
                                        dilation=2,
                                        has_bn=True,
                                        has_act=True)
        self.rebnconv3 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=4,
                                        groups=1,
                                        dilation=4,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv4 = ConvBnActBlock(middle_planes,
                                        middle_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=8,
                                        groups=1,
                                        dilation=8,
                                        has_bn=True,
                                        has_act=True)

        self.rebnconv3d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=4,
                                         groups=1,
                                         dilation=4,
                                         has_bn=True,
                                         has_act=True)

        self.rebnconv2d = ConvBnActBlock(middle_planes * 2,
                                         middle_planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=2,
                                         groups=1,
                                         dilation=2,
                                         has_bn=True,
                                         has_act=True)

        self.rebnconv1d = ConvBnActBlock(middle_planes * 2,
                                         planes,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=1,
                                         dilation=1,
                                         has_bn=True,
                                         has_act=True)

    def forward(self, x):
        xin = self.rebnconvin(x)

        x1 = self.rebnconv1(xin)
        x2 = self.rebnconv2(x1)
        x3 = self.rebnconv3(x2)

        x4 = self.rebnconv4(x3)

        x3d = self.rebnconv3d(torch.cat((x4, x3), dim=1))
        x2d = self.rebnconv2d(torch.cat((x3d, x2), dim=1))
        x1d = self.rebnconv1d(torch.cat((x2d, x1), dim=1))
        x1d = x1d + xin

        return x1d


class U2NetBackbone(nn.Module):

    def __init__(self, num_classes=1000):
        super(U2NetBackbone, self).__init__()
        self.num_classes = num_classes

        self.stage1 = RSU7Block(inplanes=3, middle_planes=32, planes=64)
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage2 = RSU6Block(inplanes=64, middle_planes=32, planes=128)
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage3 = RSU5Block(inplanes=128, middle_planes=64, planes=256)
        self.pool34 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage4 = RSU4Block(inplanes=256, middle_planes=128, planes=512)
        self.pool45 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage5 = RSU4FBlock(inplanes=512, middle_planes=256, planes=512)
        self.pool56 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage6 = RSU4FBlock(inplanes=512, middle_planes=256, planes=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        #stage 1
        x = self.stage1(x)
        x = self.pool12(x)

        #stage 2
        x = self.stage2(x)
        x = self.pool23(x)

        #stage 3
        x = self.stage3(x)
        x = self.pool34(x)

        #stage 4
        x = self.stage4(x)
        x = self.pool45(x)

        #stage 5
        x = self.stage5(x)
        x = self.pool56(x)

        #stage 6
        x = self.stage6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def u2netbackbone(**kwargs):
    model = U2NetBackbone(**kwargs)

    return model


class U2NetSmallBackbone(nn.Module):

    def __init__(self, num_classes=1000):
        super(U2NetSmallBackbone, self).__init__()
        self.num_classes = num_classes

        self.stage1 = RSU7Block(inplanes=3, middle_planes=16, planes=64)
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage2 = RSU6Block(inplanes=64, middle_planes=16, planes=64)
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage3 = RSU5Block(inplanes=64, middle_planes=16, planes=64)
        self.pool34 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage4 = RSU4Block(inplanes=64, middle_planes=16, planes=64)
        self.pool45 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage5 = RSU4FBlock(inplanes=64, middle_planes=16, planes=64)
        self.pool56 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.stage6 = RSU4FBlock(inplanes=64, middle_planes=16, planes=64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.num_classes)

    def forward(self, x):
        #stage 1
        x = self.stage1(x)
        x = self.pool12(x)

        #stage 2
        x = self.stage2(x)
        x = self.pool23(x)

        #stage 3
        x = self.stage3(x)
        x = self.pool34(x)

        #stage 4
        x = self.stage4(x)
        x = self.pool45(x)

        #stage 5
        x = self.stage5(x)
        x = self.pool56(x)

        #stage 6
        x = self.stage6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def u2netsmallbackbone(**kwargs):
    model = U2NetSmallBackbone(**kwargs)

    return model


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

    net = u2netbackbone(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    net = u2netsmallbackbone(num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')