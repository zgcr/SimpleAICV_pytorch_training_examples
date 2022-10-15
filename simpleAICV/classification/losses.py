import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CELoss',
    'FocalCELoss',
    'LabelSmoothCELoss',
    'OneHotLabelCELoss',
]


class CELoss(nn.Module):
    '''
    Cross Entropy Loss
    '''

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred, label):
        loss = self.loss(pred, label)

        return loss


class FocalCELoss(nn.Module):

    def __init__(self, gamma=2.0):
        super(FocalCELoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()

        pt = torch.where(torch.eq(one_hot_label, 1.), pred, 1. - pred)
        focal_weight = torch.pow((1. - pt), self.gamma)

        loss = (-torch.log(pred)) * one_hot_label
        loss = focal_weight * loss
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss


class LabelSmoothCELoss(nn.Module):
    '''
    Label Smooth Cross Entropy Loss
    '''

    def __init__(self, smoothing=0.1):
        super(LabelSmoothCELoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 -
            self.smoothing) * one_hot_label + self.smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss


class OneHotLabelCELoss(nn.Module):
    '''
    Cross Entropy Loss,input label is one-hot format(include soft label)
    '''

    def __init__(self):
        super(OneHotLabelCELoss, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)

        return loss.mean()


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

    pred = torch.autograd.Variable(torch.randn(5, 5))
    label = torch.tensor(np.array([0, 1, 2, 3, 4]))
    print('1111', pred.shape, label.shape)

    loss1 = CELoss()
    out = loss1(pred, label)
    print('1111', out)

    loss2 = FocalCELoss(gamma=2.0)
    out = loss2(pred, label)
    print('2222', out)

    loss3 = LabelSmoothCELoss(smoothing=0.1)
    out = loss3(pred, label)
    print('3333', out)

    label = torch.tensor(
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]))
    loss4 = OneHotLabelCELoss()
    out = loss4(pred, label)
    print('4444', out)
