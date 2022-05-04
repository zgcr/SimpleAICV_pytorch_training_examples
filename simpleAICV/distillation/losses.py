import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CELoss',
    'KDLoss',
    'DMLLoss',
    'L2Loss',
    'DKDLoss',
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


class KDLoss(nn.Module):

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.t = T

    def forward(self, stu_preds, tea_preds):
        s = F.log_softmax(stu_preds / self.t, dim=1)
        t = F.softmax(tea_preds / self.t, dim=1)

        loss = F.kl_div(s, t, reduction='batchmean') * (self.t**2)

        return loss


class DMLLoss(nn.Module):

    def __init__(self, T):
        super(DMLLoss, self).__init__()
        self.t = T

    def forward(self, stu_preds, tea_preds):
        stu_softmax = F.softmax(stu_preds / self.t, dim=1)
        tea_softmax = F.softmax(tea_preds / self.t, dim=1)
        stu_log_softmax = F.log_softmax(stu_preds / self.t, dim=1)
        tea_log_softmax = F.log_softmax(tea_preds / self.t, dim=1)

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
        loss = self.loss(stu_preds, tea_preds)

        return loss


class DKDLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=0.5, T=1.0):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.t = T

    def forward(self, stu_preds, tea_preds, label):
        stu_preds_before_softmax = stu_preds
        tea_preds_before_softmax = tea_preds

        gt_mask = torch.zeros_like(stu_preds).scatter_(1, label.unsqueeze(-1),
                                                       1).bool()
        other_mask = torch.zeros_like(stu_preds).scatter_(
            1, label.unsqueeze(-1), 0).bool()

        stu_preds = F.softmax(stu_preds / self.t, dim=1)
        stu_preds1 = (stu_preds * gt_mask).sum(dim=1, keepdims=True)
        stu_preds2 = (stu_preds * other_mask).sum(dim=1, keepdims=True)
        stu_preds = torch.cat([stu_preds1, stu_preds2], dim=1)
        log_stu_preds = torch.log(stu_preds + 1e-4)

        tea_preds = F.softmax(tea_preds / self.t, dim=1)
        tea_preds1 = (tea_preds * gt_mask).sum(dim=1, keepdims=True)
        tea_preds2 = (tea_preds * other_mask).sum(dim=1, keepdims=True)
        tea_preds = torch.cat([tea_preds1, tea_preds2], dim=1)

        tckd_loss = F.kl_div(log_stu_preds, tea_preds,
                             reduction='batchmean') * (self.t**2)

        tea_preds_part2 = F.softmax(tea_preds_before_softmax / self.t -
                                    1000.0 * gt_mask,
                                    dim=1)
        log_stu_preds_part2 = F.log_softmax(stu_preds_before_softmax / self.t -
                                            1000.0 * gt_mask,
                                            dim=1)
        nckd_loss = F.kl_div(log_stu_preds_part2,
                             tea_preds_part2,
                             reduction='batchmean') * (self.t**2)

        dkd_loss = self.alpha * tckd_loss + self.beta * nckd_loss

        return dkd_loss


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
    label = torch.tensor(np.array([0, 1, 2, 3, 4]))
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

    loss5 = DKDLoss(alpha=1.0, beta=0.5, T=1.0)
    out = loss5(stu_pred, tea_pred, label)
    print('5555', out)
