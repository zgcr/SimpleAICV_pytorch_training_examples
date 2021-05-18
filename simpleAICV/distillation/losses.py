import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CELoss',
    'KDLoss',
]


class CELoss(nn.Module):
    '''Cross Entropy Loss'''
    def __init__(self):
        super().__init__()

    def forward(self, stu_pred, label):
        loss = F.cross_entropy(stu_pred, label)
        return loss


class KDLoss(nn.Module):
    '''Knowledge Distillation Loss'''
    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        loss = F.kl_div(s, t, size_average=False) * (self.t**
                                                     2) / stu_pred.shape[0]
        return loss