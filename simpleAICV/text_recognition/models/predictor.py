import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

__all__ = [
    'CTCPredictor',
]


class CTCPredictor(nn.Module):

    def __init__(self, inplanes, hidden_planes, num_classes):
        super(CTCPredictor, self).__init__()
        self.linear1 = nn.Linear(inplanes, hidden_planes)
        self.linear2 = nn.Linear(hidden_planes, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

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

    net = CTCPredictor(inplanes=256, hidden_planes=192, num_classes=12114)
    x = torch.randn(3, 64, 256)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(x, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(x)
    print(f'1111, macs: {macs}, params: {params},out: {out.shape}')
