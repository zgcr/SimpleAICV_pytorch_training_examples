import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    """Cross Entropy Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, stu_pred, label):
        loss = F.cross_entropy(stu_pred, label)
        return loss


class CDLoss(nn.Module):
    """Channel Distillation Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, stu_features: list, tea_features: list):
        loss = 0.
        for s, t in zip(stu_features, tea_features):
            s = s.mean(dim=(2, 3), keepdim=False)
            t = t.mean(dim=(2, 3), keepdim=False)
            loss += torch.mean(torch.pow(s - t, 2))
        return loss


class KDLoss(nn.Module):
    """Knowledge Distillation Loss"""
    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        loss = F.kl_div(s, t, size_average=False) * (self.t**
                                                     2) / stu_pred.shape[0]
        return loss


class GKDLoss(nn.Module):
    """Knowledge Distillation Loss"""
    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred, label):
        stu_pred_log_softmax = F.log_softmax(stu_pred / self.t, dim=1)
        tea_pred_softmax = F.softmax(tea_pred / self.t, dim=1)
        tea_pred_argmax = torch.argmax(tea_pred_softmax, dim=1)
        mask = torch.eq(label, tea_pred_argmax).float()
        count = (mask[mask == 1]).size(0)
        mask = mask.unsqueeze(-1)
        only_correct_sample_stu_pred_log_softmax = stu_pred_log_softmax.mul(
            mask)
        only_correct_sample_tea_pred_softmax = tea_pred_softmax.mul(mask)
        only_correct_sample_tea_pred_softmax[
            only_correct_sample_tea_pred_softmax == 0.0] = 1.0

        loss = F.kl_div(only_correct_sample_stu_pred_log_softmax,
                        only_correct_sample_tea_pred_softmax,
                        reduction='sum') * (self.t**2) / count
        return loss
