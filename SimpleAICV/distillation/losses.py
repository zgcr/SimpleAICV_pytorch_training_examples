import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CELoss',
    'OneHotLabelCELoss',
    'KDLoss',
    'DMLLoss',
    'L2Loss',
]


class CELoss(nn.Module):
    '''
    Cross Entropy Loss
    '''

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred, label):
        pred = pred.float()

        loss = self.loss(pred, label)

        return loss


class OneHotLabelCELoss(nn.Module):
    '''
    Cross Entropy Loss,input label is one-hot format(include soft label)
    '''

    def __init__(self):
        super(OneHotLabelCELoss, self).__init__()

    def forward(self, pred, label):
        pred = pred.float()

        loss = torch.sum(-label * F.log_softmax(pred, dim=-1), dim=-1)

        return loss.mean()


class KDLoss(nn.Module):

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.t = T

    def forward(self, stu_preds, tea_preds):
        stu_preds = stu_preds.float()
        tea_preds = tea_preds.float()

        s = F.softmax(stu_preds / self.t, dim=1)
        s = torch.clamp(s, min=1e-4, max=1. - 1e-4)
        s = torch.log(s)

        t = F.softmax(tea_preds / self.t, dim=1)
        t = torch.clamp(t, min=1e-4, max=1. - 1e-4)

        loss = F.kl_div(s, t, reduction='batchmean') * (self.t**2)

        return loss


class DMLLoss(nn.Module):

    def __init__(self, T):
        super(DMLLoss, self).__init__()
        self.t = T

    def forward(self, stu_preds, tea_preds):
        stu_preds = stu_preds.float()
        tea_preds = tea_preds.float()

        stu_softmax = F.softmax(stu_preds / self.t, dim=1)
        stu_softmax = torch.clamp(stu_softmax, min=1e-4, max=1. - 1e-4)

        tea_softmax = F.softmax(tea_preds / self.t, dim=1)
        tea_softmax = torch.clamp(tea_softmax, min=1e-4, max=1. - 1e-4)

        stu_log_softmax = F.softmax(stu_preds / self.t, dim=1)
        stu_log_softmax = torch.clamp(stu_log_softmax, min=1e-4, max=1. - 1e-4)
        stu_log_softmax = torch.log(stu_log_softmax)

        tea_log_softmax = F.softmax(tea_preds / self.t, dim=1)
        tea_log_softmax = torch.clamp(tea_log_softmax, min=1e-4, max=1. - 1e-4)
        tea_log_softmax = torch.log(tea_log_softmax)

        loss = (F.kl_div(stu_log_softmax, tea_softmax, reduction='batchmean') *
                (self.t**2) +
                F.kl_div(tea_log_softmax, stu_softmax, reduction='batchmean') *
                (self.t**2)) / 2.0

        return loss


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, stu_preds, tea_preds):
        stu_preds = stu_preds.float()
        tea_preds = tea_preds.float()

        loss = self.loss(stu_preds, tea_preds)

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

    stu_pred = torch.autograd.Variable(torch.randn(5, 5))
    tea_pred = torch.autograd.Variable(torch.randn(5, 5))
    label = torch.tensor(np.array([0, 1, 2, 3, 4])).long()
    print('1111', stu_pred.shape, tea_pred.shape, label.shape)

    loss1 = CELoss()
    out = loss1(stu_pred, label)
    print('1111', out)

    loss2 = KDLoss(T=1.0)
    out = loss2(stu_pred, tea_pred)
    print('2222', out)

    loss3 = DMLLoss(T=1.0)
    out = loss3(stu_pred, tea_pred)
    print('3333', out)

    loss4 = L2Loss()
    out = loss4(stu_pred, tea_pred)
    print('4444', out)

    label = torch.tensor(
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]))
    loss4 = OneHotLabelCELoss()
    out = loss4(stu_pred, label)
    print('5555', out)
