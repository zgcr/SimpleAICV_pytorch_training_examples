import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = [
    'CELoss',
    'MultiClassBCELoss',
    'IoULoss',
    'DiceLoss',
    'LovaszLoss',
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
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.softmax(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        one_hot_label = F.one_hot(label.long(),
                                  num_classes=num_classes).float()

        loss = (-torch.log(pred)) * one_hot_label

        loss = loss.sum(axis=-1)
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
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = pred.float()
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
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = pred.float()
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
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = pred.float()
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


class LovaszLoss(nn.Module):

    def __init__(self):
        super(LovaszLoss, self).__init__()
        self.logit = nn.Sigmoid()

    def forward(self, pred, label):
        # assumes pred of a sigmoid layer
        # pred shape:[b,c,h,w] -> [b,h,w,c] -> [b*h*w,c]
        # label shape:[b,h,w] -> [b*h*w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = pred.float()
        pred = self.logit(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        lovasz_loss, class_count = 0., 0
        for class_idx in range(1, num_classes + 1):
            per_class_target_mask = (label == class_idx).float()
            if per_class_target_mask.sum() == 0:
                continue
            class_count += 1
            per_class_pred = pred[:, class_idx]
            errors = (Variable(per_class_target_mask) - per_class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            per_class_target_mask_sorted = per_class_target_mask[perm.long()]
            per_class_loss = torch.dot(
                errors_sorted,
                Variable(self.lovasz_grad(per_class_target_mask_sorted)))
            lovasz_loss += per_class_loss

        lovasz_loss = lovasz_loss / float(class_count)

        return lovasz_loss

    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard


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

    from tools.path import face_parsing_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.face_parsing.datasets.face_parsing_dataset import FaceParsingDataset, CelebAMask_HQ_19_CLASSES
    from simpleAICV.face_parsing.common import RandomCrop, RandomShrink, RandomRGBToGRAY, RandomHorizontalFlip, RandomVerticalFlip, YoloStyleResize, Resize, Normalize, FaceParsingCollater

    face_parsing_trainset = FaceParsingDataset(face_parsing_dataset_path,
                                               set_name_list=[
                                                   'CelebAMask-HQ',
                                               ],
                                               set_type='train',
                                               cats=CelebAMask_HQ_19_CLASSES,
                                               transform=transforms.Compose([
                                                   YoloStyleResize(resize=512),
                                                   Normalize(),
                                               ]))

    from torch.utils.data import DataLoader
    collater = FaceParsingCollater(resize=512)
    train_loader = DataLoader(face_parsing_trainset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.face_parsing.models.pfan_face_parsing import resnet50_pfan_face_parsing
    net = resnet50_pfan_face_parsing(num_classes=19)

    loss1 = CELoss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('1212', preds.shape)
        out = loss1(preds, masks)
        print('1313', out)
        break

    loss2 = MultiClassBCELoss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('2222', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('2323', preds.shape)
        out = loss2(preds, masks)
        print('2424', out)
        break

    loss3 = IoULoss(logit_type='softmax')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('3333', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('3434', preds.shape)
        out = loss3(preds, masks)
        print('3535', out)
        break

    loss4 = IoULoss(logit_type='sigmoid')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('4444', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('4545', preds.shape)
        out = loss4(preds, masks)
        print('4646', out)
        break

    loss5 = DiceLoss(logit_type='softmax')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('5555', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('5656', preds.shape)
        out = loss5(preds, masks)
        print('5757', out)
        break

    loss6 = DiceLoss(logit_type='sigmoid')
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('6666', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('6767', preds.shape)
        out = loss6(preds, masks)
        print('6868', out)
        break

    loss7 = LovaszLoss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('7777', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('7878', preds.shape)
        out = loss7(preds, masks)
        print('7979', out)
        break
