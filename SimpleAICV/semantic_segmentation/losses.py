import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CELoss',
    'MultiClassBCELoss',
    'IoULoss',
    'DiceLoss',
]


class CELoss(nn.Module):
    '''
    Cross Entropy Loss
    '''

    def __init__(self):
        super(CELoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.softmax(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        one_hot_label = F.one_hot(label.long(),
                                  num_classes=num_classes).float()

        loss = (-torch.log(pred)) * one_hot_label

        loss = loss.sum(dim=-1)
        loss = loss.mean()

        return loss


class MultiClassBCELoss(nn.Module):
    '''
    Multi Class Binary Cross Entropy Loss
    '''

    def __init__(self):
        super(MultiClassBCELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        loss_ground_truth = F.one_hot(label.long(),
                                      num_classes=num_classes).float()

        bce_loss = -(loss_ground_truth * torch.log(pred) +
                     (1. - loss_ground_truth) * torch.log(1. - pred))

        bce_loss = bce_loss.mean()

        return bce_loss


class IoULoss(nn.Module):

    def __init__(self, logit_type='softmax'):
        super(IoULoss, self).__init__()
        assert logit_type in ['softmax', 'sigmoid']
        if logit_type == 'softmax':
            self.logit = nn.Softmax(dim=-1)
        elif logit_type == 'sigmoid':
            self.logit = nn.Sigmoid()

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.logit(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        loss_ground_truth = F.one_hot(label.long(),
                                      num_classes=num_classes).float()

        intersection = pred * loss_ground_truth

        iou_loss = 1. - torch.sum(intersection, dim=1) / torch.clamp(
            (torch.sum(pred, dim=1) + torch.sum(loss_ground_truth, dim=1) -
             torch.sum(intersection, dim=1)),
            min=1e-4)
        iou_loss = iou_loss.mean()

        return iou_loss


class DiceLoss(nn.Module):

    def __init__(self, logit_type='softmax'):
        super(DiceLoss, self).__init__()
        assert logit_type in ['softmax', 'sigmoid']
        if logit_type == 'softmax':
            self.logit = nn.Softmax(dim=-1)
        elif logit_type == 'sigmoid':
            self.logit = nn.Sigmoid()

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.logit(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        loss_ground_truth = F.one_hot(label.long(),
                                      num_classes=num_classes).float()

        intersection = pred * loss_ground_truth

        dice_loss = 1. - (2 * torch.sum(intersection, dim=1) +
                          1e-4) / (torch.sum(pred, dim=1) +
                                   torch.sum(loss_ground_truth, dim=1) + 1e-4)
        dice_loss = dice_loss.mean()

        return dice_loss


if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

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

    from tools.path import ADE20Kdataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.semantic_segmentation.datasets.ade20kdataset import ADE20KSemanticSegmentation
    from SimpleAICV.semantic_segmentation.common import YoloStyleResize, RandomHorizontalFlip, Normalize, SemanticSegmentationCollater

    ade20kdataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        transform=transforms.Compose([
            YoloStyleResize(resize=512),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = SemanticSegmentationCollater(resize=512)
    train_loader = DataLoader(ade20kdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.semantic_segmentation.models.pfan_semantic_segmentation import resnet50_pfan_semantic_segmentation
    net = resnet50_pfan_semantic_segmentation(backbone_pretrained_path='',
                                              num_classes=151)

    loss1 = CELoss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        out = loss1(preds, masks)
        print('2222', out)
        break

    loss2 = MultiClassBCELoss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        out = loss2(preds, masks)
        print('2222', out)
        break

    loss3 = IoULoss(logit_type='softmax')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        out = loss3(preds, masks)
        print('2222', out)
        break

    loss4 = IoULoss(logit_type='sigmoid')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        out = loss4(preds, masks)
        print('2222', out)
        break

    loss5 = DiceLoss(logit_type='softmax')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        out = loss5(preds, masks)
        print('2222', out)
        break

    loss6 = DiceLoss(logit_type='sigmoid')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        out = loss6(preds, masks)
        print('2222', out)
        break
