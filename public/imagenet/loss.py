import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    """Cross Entropy Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        loss = F.cross_entropy(pred, label)

        return loss


class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss