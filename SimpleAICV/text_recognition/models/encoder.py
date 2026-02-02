import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

__all__ = [
    'BiLSTMEncoder',
]


class BiLSTMEncoder(nn.Module):

    def __init__(self, inplanes, hidden_planes):
        super(BiLSTMEncoder, self).__init__()
        self.linear0 = nn.Linear(inplanes, hidden_planes)
        self.rnn1 = nn.LSTM(hidden_planes,
                            hidden_planes,
                            bidirectional=True,
                            batch_first=True)
        self.linear1 = nn.Linear(hidden_planes * 2, hidden_planes)
        self.rnn2 = nn.LSTM(hidden_planes,
                            hidden_planes,
                            bidirectional=True,
                            batch_first=True)
        self.linear2 = nn.Linear(hidden_planes * 2, hidden_planes)

    def forward(self, x):
        """
        input:shape:[B,W,C],W=time steps
        output : contextual feature [batch_size x T x output_size]
        """
        x = self.linear0(x)
        # batch_size x T x inplanes -> batch_size x T x (2*hidden_planes)
        self.rnn1.flatten_parameters()
        x, _ = self.rnn1(x)
        x = self.linear1(x)  # batch_size x T x output_size
        self.rnn2.flatten_parameters()
        x, _ = self.rnn2(x)
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

    net = BiLSTMEncoder(inplanes=512, hidden_planes=256)
    x = torch.randn(3, 64, 512)
    out = net(x)
    print(f'1111, out: {out.shape}')
