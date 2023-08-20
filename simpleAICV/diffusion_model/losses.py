import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MSELoss',
    'L1Loss',
]


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        pass

    def forward(self, pred, label):
        batch_size = pred.shape[0]

        loss = (pred - label)**2
        loss = torch.mean(loss, dim=[1, 2, 3])
        loss = loss.sum() / batch_size

        return loss


class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        pass

    def forward(self, pred, label):
        batch_size = pred.shape[0]

        loss = torch.abs(pred - label)
        loss = torch.mean(loss, dim=[1, 2, 3])
        loss = loss.sum() / batch_size

        return loss


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

    pred = torch.autograd.Variable(torch.randn(1, 3, 32, 32))
    label = torch.autograd.Variable(torch.randn(1, 3, 32, 32))
    loss1 = MSELoss()
    out = loss1(pred, label)
    print('1111', out)

    pred = torch.autograd.Variable(torch.randn(1, 3, 32, 32))
    label = torch.autograd.Variable(torch.randn(1, 3, 32, 32))
    loss2 = L1Loss()
    out = loss2(pred, label)
    print('2222', out)
