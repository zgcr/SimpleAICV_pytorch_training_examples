import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn


class RetinaFaceClassHead(nn.Module):

    def __init__(self, inplanes=256, anchor_num=3):
        super(RetinaFaceClassHead, self).__init__()
        # 1 is class_num
        self.conv1x1 = nn.Conv2d(inplanes,
                                 anchor_num * 1,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=1,
                                 bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1x1(x)
        x = x.float()
        x = self.sigmoid(x)

        return x


class RetinaFaceBoxHead(nn.Module):

    def __init__(self, inplanes=256, anchor_num=3):
        super(RetinaFaceBoxHead, self).__init__()
        # 4 is box coords
        self.conv1x1 = nn.Conv2d(inplanes,
                                 anchor_num * 4,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=1,
                                 bias=True)

    def forward(self, x):
        x = self.conv1x1(x)

        return x


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

    inputs = torch.randn(3, 256, 120, 120)
    net = RetinaFaceClassHead(inplanes=256)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(inputs, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(inputs)
    print('2222', out.shape)

    inputs = torch.randn(3, 256, 120, 120)
    net = RetinaFaceBoxHead(inplanes=256)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(inputs, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    out = net(inputs)
    print('2222', out.shape)
