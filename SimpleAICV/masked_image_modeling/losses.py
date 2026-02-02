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

    def forward(self, pred, label, mask):
        pred = pred.float()
        label = label.float()
        mask = mask.float()

        loss = (pred - label)**2
        # [N, L], mean loss per patch
        loss = loss.mean(dim=-1)
        # mean loss on removed patches
        loss = (loss * mask).sum() / (mask.sum() + 1e-4)

        return loss


class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        pass

    def forward(self, pred, label, mask):
        pred = pred.float()
        label = label.float()
        mask = mask.float()

        loss = torch.abs(pred - label)
        # mean loss on removed patches
        loss = (loss * mask).sum() / (mask.sum() + 1e-4)

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

    pred = torch.autograd.Variable(torch.randn(1, 192, 768))
    label = torch.autograd.Variable(torch.randn(1, 192, 768))
    mask = torch.randint(low=0, high=2, size=(1, 192))
    loss1 = MSELoss()
    out = loss1(pred, label, mask)
    print('1111', out)

    pred = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    label = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    mask = torch.randint(low=0, high=2, size=(1, 3, 224, 224))
    loss2 = L1Loss()
    out = loss2(pred, label, mask)
    print('2222', out)
