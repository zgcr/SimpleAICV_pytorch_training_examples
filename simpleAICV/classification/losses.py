import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CELoss',
    'FocalCELoss',
    'LabelSmoothCELoss',
    'OneHotLabelCELoss',
    'SemanticSoftmaxLoss',
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


class SemanticSoftmaxLoss(torch.nn.Module):

    def __init__(self, normalization_factor_list, smoothing):
        super(SemanticSoftmaxLoss, self).__init__()
        self.normalization_factor_list = normalization_factor_list
        self.smoothing = smoothing

    def forward(self, semantic_outputs, semantic_labels):
        """
        Calculates the semantic cross-entropy loss distance between outputs and labels
        """
        losses = []

        # scanning hirarchy_level_list
        for i in range(len(semantic_outputs)):
            outputs_i = semantic_outputs[i]
            labels_i = semantic_labels[:, i]

            # generate probs
            log_preds = F.log_softmax(outputs_i, dim=1)

            # generate labels (with protections)
            labels_i_valid = labels_i.clone()
            labels_i_valid[labels_i_valid < 0] = 0
            num_classes = outputs_i.shape[-1]

            labels_classes = torch.zeros_like(outputs_i).scatter_(
                1, labels_i_valid.unsqueeze(1), 1)
            labels_classes.mul_(1 - self.smoothing).add_(self.smoothing /
                                                         num_classes)

            cross_entropy_loss_tot = -labels_classes.mul(log_preds)
            cross_entropy_loss_tot *= ((labels_i >= 0).unsqueeze(1))
            # sum over classes and mean over batch
            cross_entropy_loss = cross_entropy_loss_tot.sum(dim=-1)
            loss_i = cross_entropy_loss.mean()

            losses.append(loss_i)

        total_loss = 0
        # summing over hirarchies
        for i, loss_h in enumerate(losses):
            total_loss += loss_h * self.normalization_factor_list[i]

        return total_loss


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

    pred = torch.autograd.Variable(torch.randn(5, 10450))
    label = torch.tensor(np.array([0, 1, 2, 3, 4]))

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from tools.path import ImageNet21K_path

    import torchvision.transforms as transforms

    from simpleAICV.classification.datasets.imagenet21kdataset import ImageNet21KSemanticTreeLabelDataset
    from simpleAICV.classification.common import Opencv2PIL, PIL2Opencv, TorchRandomResizedCrop, TorchRandomHorizontalFlip, RandomErasing, TorchResize, TorchCenterCrop, Normalize, AutoAugment, RandAugment, ClassificationCollater

    imagenet21kdataset = ImageNet21KSemanticTreeLabelDataset(
        root_dir=ImageNet21K_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=224),
            TorchRandomHorizontalFlip(prob=0.5),
            PIL2Opencv(),
            Normalize(),
        ]))
    semantic_pred = imagenet21kdataset.convert_outputs_to_semantic_outputs(
        pred)
    semantic_label = imagenet21kdataset.convert_single_labels_to_semantic_labels(
        label)
    loss5 = SemanticSoftmaxLoss(
        normalization_factor_list=imagenet21kdataset.normalization_factor_list,
        smoothing=0.1)
    out = loss5(semantic_pred, semantic_label)
    print('5555', semantic_label.shape, out)
