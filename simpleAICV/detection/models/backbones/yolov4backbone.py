import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from simpleAICV.classification.backbones.yolov4backbone import ConvBnActBlock, CSPDarkNetTinyBlock, CSPDarkNetBlock
from simpleAICV.detection.common import load_state_dict

__all__ = [
    'yolov4cspdarknettinybackbone',
    'yolov4cspdarknet53backbone',
]


class CSPDarknetTiny(nn.Module):

    def __init__(self, planes=[64, 128, 256, 512], act_type='leakyrelu'):
        super(CSPDarknetTiny, self).__init__()
        self.conv1 = ConvBnActBlock(3,
                                    32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(32,
                                    planes[0],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetTinyBlock(planes[0],
                                          planes[0],
                                          act_type=act_type)
        self.block2 = CSPDarkNetTinyBlock(planes[1],
                                          planes[1],
                                          act_type=act_type)
        self.block3 = CSPDarkNetTinyBlock(planes[2],
                                          planes[2],
                                          act_type=act_type)
        self.conv3 = ConvBnActBlock(planes[3],
                                    planes[3],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)

        self.out_channels = [planes[2], planes[3]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, x1 = self.block3(x)
        x2 = self.conv3(x)

        return x1, x2


class CSPDarknet53(nn.Module):

    def __init__(self,
                 inplanes=32,
                 planes=[64, 128, 256, 512, 1024],
                 act_type='leakyrelu'):
        super(CSPDarknet53, self).__init__()
        self.conv1 = ConvBnActBlock(3,
                                    inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetBlock(inplanes,
                                      planes[0],
                                      num_blocks=1,
                                      reduction=False,
                                      act_type=act_type)
        self.block2 = CSPDarkNetBlock(planes[0],
                                      planes[1],
                                      num_blocks=2,
                                      reduction=True,
                                      act_type=act_type)
        self.block3 = CSPDarkNetBlock(planes[1],
                                      planes[2],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block4 = CSPDarkNetBlock(planes[2],
                                      planes[3],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block5 = CSPDarkNetBlock(planes[3],
                                      planes[4],
                                      num_blocks=4,
                                      reduction=True,
                                      act_type=act_type)

        self.out_channels = [planes[2], planes[3], planes[4]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        C3 = self.block3(x)
        C4 = self.block4(C3)
        C5 = self.block5(C4)

        return [C3, C4, C5]


def yolov4cspdarknettinybackbone(pretrained_path='', **kwargs):
    model = CSPDarknetTiny(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def yolov4cspdarknet53backbone(pretrained_path='', **kwargs):
    model = CSPDarknet53(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

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

    net = yolov4cspdarknettinybackbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)

    net = yolov4cspdarknet53backbone()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        print('2222', out.shape)