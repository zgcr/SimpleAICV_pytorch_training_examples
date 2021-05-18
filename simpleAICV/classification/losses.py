import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CELoss',
    'LabelSmoothCELoss',
]


class CELoss(nn.Module):
    '''
    Cross Entropy Loss
    '''
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        loss = F.cross_entropy(pred, label)

        return loss


class LabelSmoothCELoss(nn.Module):
    '''
    Label Smooth Cross Entropy Loss
    '''
    def __init__(self, smoothing=0.1):
        super().__init__()
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