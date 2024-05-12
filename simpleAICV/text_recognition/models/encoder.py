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
    'TransformerEncoder',
]


class BiLSTMEncoder(nn.Module):

    def __init__(self, inplanes):
        super(BiLSTMEncoder, self).__init__()
        self.rnn1 = nn.LSTM(inplanes,
                            inplanes,
                            bidirectional=True,
                            batch_first=True)
        self.linear1 = nn.Linear(inplanes * 2, inplanes)
        self.rnn2 = nn.LSTM(inplanes,
                            inplanes,
                            bidirectional=True,
                            batch_first=True)
        self.linear2 = nn.Linear(inplanes * 2, inplanes)

        self.out_channels = inplanes

    def forward(self, x):
        """
        input:shape:[B,W,C],W=time steps
        output : contextual feature [batch_size x T x output_size]
        """
        # batch_size x T x inplanes -> batch_size x T x (2*hidden_planes)
        self.rnn1.flatten_parameters()
        x, _ = self.rnn1(x)
        x = self.linear1(x)  # batch_size x T x output_size
        self.rnn2.flatten_parameters()
        x, _ = self.rnn2(x)
        x = self.linear2(x)

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, inplanes, head_nums=8, dropout_prob=0.):
        super(MultiHeadAttention, self).__init__()
        self.head_nums = head_nums
        self.scale = (inplanes // head_nums)**-0.5

        self.qkv_linear = nn.Linear(inplanes, inplanes * 3)
        self.out_linear = nn.Linear(inplanes, inplanes)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape

        # [b,n,c] -> [b,n,3,head_num,c//head_num] -> [3,b,head_num,n,c//head_num]
        x = self.qkv_linear(x).view(b, n, 3, self.head_nums,
                                    c // self.head_nums).permute(
                                        2, 0, 3, 1, 4)
        # [3,b,head_num,n,c//head_num] -> 3个 [b,head_num,n,c//head_num]
        q, k, v = torch.unbind(x, dim=0)

        # [b,head_num,n,c//head_num] -> [b,head_num,n,n]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out_linear(x)
        x = self.dropout(x)

        return x


class FeedForward(nn.Module):

    def __init__(self, inplanes, feedforward_planes, dropout_prob=0.):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(inplanes, feedforward_planes)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(feedforward_planes, inplanes)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DropPathBlock(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    if drop_path_prob = 0. ,not use DropPath
    """

    def __init__(self, drop_path_prob=0., scale_by_keep=True):
        super(DropPathBlock, self).__init__()
        assert drop_path_prob >= 0.

        self.drop_path_prob = drop_path_prob
        self.keep_path_prob = 1 - drop_path_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_path_prob == 0. or not self.training:
            return x

        b = x.shape[0]
        device = x.device

        # work with diff dim tensors, not just 2D ConvNets
        shape = (b, ) + (1, ) * (len(x.shape) - 1)
        random_weight = torch.empty(shape).to(device).bernoulli_(
            self.keep_path_prob)

        if self.keep_path_prob > 0. and self.scale_by_keep:
            random_weight.div_(self.keep_path_prob)

        x = random_weight * x

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 inplanes,
                 head_nums,
                 feedforward_ratio=4,
                 dropout_prob=0.,
                 drop_path_prob=0.):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(inplanes)
        self.attention = MultiHeadAttention(inplanes,
                                            head_nums,
                                            dropout_prob=dropout_prob)
        self.norm2 = nn.LayerNorm(inplanes)
        self.feed_forward = FeedForward(inplanes,
                                        int(inplanes * feedforward_ratio),
                                        dropout_prob=dropout_prob)
        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.feed_forward(self.norm2(x)))

        return x


class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer_nums,
                 inplanes,
                 head_nums,
                 feedforward_ratio=1,
                 dropout_prob=0.1,
                 encoding_width=80):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(inplanes, head_nums, feedforward_ratio,
                                    dropout_prob)
            for _ in range(encoder_layer_nums)
        ])
        self.encoding = nn.Parameter(torch.ones(1, encoding_width, inplanes))

        self.out_channels = inplanes

    def forward(self, x):
        _, w, _ = x.shape
        encoding = self.encoding[:, 0:w, :].to(x.device)

        x += encoding
        for per_layer in self.layers:
            x = per_layer(x)

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

    net = BiLSTMEncoder(inplanes=256)
    x = torch.randn(3, 64, 256)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(x, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(x)
    print(f'1111, macs: {macs}, params: {params},out: {out.shape}')
    print(net.out_channels)

    net = TransformerEncoder(encoder_layer_nums=2,
                             inplanes=256,
                             head_nums=2,
                             feedforward_ratio=1,
                             dropout_prob=0.,
                             encoding_width=80)
    x = torch.randn(3, 64, 256)
    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(x, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(x)
    print(f'2222, macs: {macs}, params: {params},out: {out.shape}')
    print(net.out_channels)
