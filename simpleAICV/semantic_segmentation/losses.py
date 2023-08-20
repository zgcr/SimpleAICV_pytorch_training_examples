import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = [
    'CELoss',
    'FocalCELoss',
    'MultiClassBCELoss',
    'MultiClassFocalBCELoss',
    'MultiClassOHEMBCELoss',
    'IoULoss',
    'DiceLoss',
    'LovaszLoss',
]


class CELoss(nn.Module):
    '''
    Cross Entropy Loss
    '''

    def __init__(self, ignore_index=None):
        super(CELoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.softmax(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

        one_hot_label = F.one_hot(label.long(),
                                  num_classes=num_classes).float()

        loss = (-torch.log(pred)) * one_hot_label

        loss = loss.sum(axis=-1)
        loss = loss.mean()

        return loss


class FocalCELoss(nn.Module):

    def __init__(self, gamma=2.0, ignore_index=None):
        super(FocalCELoss, self).__init__()
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=-1)
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.softmax(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

        one_hot_label = F.one_hot(label.long(),
                                  num_classes=num_classes).float()

        pt = torch.where(torch.eq(one_hot_label, 1.), pred, 1. - pred)
        focal_weight = torch.pow((1. - pt), self.gamma)

        loss = (-torch.log(pred)) * one_hot_label
        loss = focal_weight * loss

        loss = loss.sum(axis=-1)
        loss = loss.mean()

        return loss


class MultiClassBCELoss(nn.Module):
    '''
    Multi Class Binary Cross Entropy Loss
    '''

    def __init__(self, ignore_index=None):
        super(MultiClassBCELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

        loss_ground_truth = F.one_hot(label.long(),
                                      num_classes=num_classes).float()

        bce_loss = -(loss_ground_truth * torch.log(pred) +
                     (1. - loss_ground_truth) * torch.log(1. - pred))

        bce_loss = bce_loss.mean()

        return bce_loss


class MultiClassFocalBCELoss(nn.Module):
    '''
    Multi Class Focal Binary Cross Entropy Loss
    '''

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super(MultiClassFocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

        device = pred.device
        positive_points_num = (label >= 0).sum()

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        loss_ground_truth = F.one_hot(label.long(),
                                      num_classes=num_classes).float()

        alpha_factor = torch.ones_like(pred) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), pred, 1. - pred)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        bce_loss = -(loss_ground_truth * torch.log(pred) +
                     (1. - loss_ground_truth) * torch.log(1. - pred))

        bce_loss = focal_weight * bce_loss
        bce_loss = bce_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        bce_loss = bce_loss / positive_points_num

        return bce_loss


class MultiClassOHEMBCELoss(nn.Module):
    '''
    Multi Class Binary Cross Entropy Loss
    '''

    def __init__(self, negative_ratio=3.0, ignore_index=None):
        super(MultiClassOHEMBCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.sigmoid = nn.Sigmoid()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[-1]

        pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

        positive_point_mask = (label >= 0).float()

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(label.long(),
                                      num_classes=num_classes).float()

        positive_points_num = int(positive_point_mask.sum())
        negative_points_num = min(
            int((1. - positive_point_mask).sum()),
            int(positive_points_num * self.negative_ratio))

        bce_loss = -(loss_ground_truth * torch.log(pred) +
                     (1. - loss_ground_truth) * torch.log(1. - pred))
        bce_loss = torch.sum(bce_loss, axis=-1)

        positive_loss = bce_loss * positive_point_mask
        negative_loss = bce_loss * (1. - positive_point_mask)
        negative_loss, _ = torch.topk(negative_loss.view(-1),
                                      negative_points_num)

        ohem_bce_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_points_num + negative_points_num + 1e-4)

        return ohem_bce_loss


class IoULoss(nn.Module):

    def __init__(self, logit_type='softmax', ignore_index=None):
        super(IoULoss, self).__init__()
        assert logit_type in ['softmax', 'sigmoid']
        if logit_type == 'softmax':
            self.logit = nn.Softmax(dim=-1)
        elif logit_type == 'sigmoid':
            self.logit = nn.Sigmoid()

        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.logit(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

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

    def __init__(self, logit_type='softmax', smooth=1e-4, ignore_index=None):
        super(DiceLoss, self).__init__()
        assert logit_type in ['softmax', 'sigmoid']
        if logit_type == 'softmax':
            self.logit = nn.Softmax(dim=-1)
        elif logit_type == 'sigmoid':
            self.logit = nn.Sigmoid()

        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.logit(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

        loss_ground_truth = F.one_hot(label.long(),
                                      num_classes=num_classes).float()

        intersection = pred * loss_ground_truth

        dice_loss = 1. - (2 * torch.sum(intersection, dim=1) + self.smooth) / (
            torch.sum(pred, dim=1) + torch.sum(loss_ground_truth, dim=1) +
            self.smooth)
        dice_loss = dice_loss.mean()

        return dice_loss


class LovaszLoss(nn.Module):

    def __init__(self, ignore_index=None):
        super(LovaszLoss, self).__init__()
        self.logit = nn.Sigmoid()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # assumes pred of a sigmoid layer
        # pred shape:[b,c,h,w] -> [b,h,w,c] -> [b*h*w,c]
        # label shape:[b,h,w] -> [b*h*w]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]

        pred = self.logit(pred)
        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1, num_classes).contiguous()
        label = label.view(-1).contiguous()

        if self.ignore_index:
            filter_mask = (label >= 0) & (label != self.ignore_index)
            pred = (pred[filter_mask]).contiguous()
            label = (label[filter_mask]).contiguous()

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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from tools.path import ADE20Kdataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.semantic_segmentation.datasets.ade20kdataset import ADE20KSemanticSegmentation
    from simpleAICV.semantic_segmentation.common import RandomCropResize, RandomHorizontalFlip, PhotoMetricDistortion, Normalize, SemanticSegmentationCollater

    ade20kdataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        reduce_zero_label=True,
        transform=transforms.Compose([
            RandomCropResize(image_scale=(2048, 512),
                             multi_scale=True,
                             multi_scale_range=(0.5, 2.0),
                             crop_size=(512, 512),
                             cat_max_ratio=0.75,
                             ignore_index=255),
            RandomHorizontalFlip(prob=0.5),
            PhotoMetricDistortion(brightness_delta=32,
                                  contrast_range=(0.5, 1.5),
                                  saturation_range=(0.5, 1.5),
                                  hue_delta=18,
                                  prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = SemanticSegmentationCollater(resize=512, ignore_index=255)
    train_loader = DataLoader(ade20kdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.semantic_segmentation.models.deeplabv3plus import resnet50backbone_deeplabv3plus
    net = resnet50backbone_deeplabv3plus(backbone_pretrained_path='',
                                         num_classes=150)

    loss1 = CELoss(ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('1111', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss1(preds, masks)
        print('1212', out)
        break

    loss2 = FocalCELoss(gamma=2.0, ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('2222', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss2(preds, masks)
        print('2323', out)
        break

    loss3 = MultiClassBCELoss(ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('3333', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss3(preds, masks)
        print('3434', out)
        break

    loss4 = MultiClassFocalBCELoss(alpha=0.25, gamma=2.0, ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('4444', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss4(preds, masks)
        print('4545', out)
        break

    loss5 = MultiClassOHEMBCELoss(negative_ratio=3.0, ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('5555', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss5(preds, masks)
        print('5656', out)
        break

    loss6 = DiceLoss(logit_type='softmax', smooth=1e-4, ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('6666', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss6(preds, masks)
        print('6767', out)
        break

    loss7 = DiceLoss(logit_type='sigmoid', smooth=1e-4, ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('7777', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss7(preds, masks)
        print('7878', out)
        break

    loss8 = LovaszLoss(ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('8888', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss8(preds, masks)
        print('8989', out)
        break

    loss9 = IoULoss(logit_type='softmax', ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('9999', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss9(preds, masks)
        print('9191', out)
        break

    loss10 = IoULoss(logit_type='sigmoid', ignore_index=255)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('9999', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss10(preds, masks)
        print('9292', out)
        break

    ade20kdataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        reduce_zero_label=False,
        transform=transforms.Compose([
            RandomCropResize(image_scale=(2048, 512),
                             multi_scale=True,
                             multi_scale_range=(0.5, 2.0),
                             crop_size=(512, 512),
                             cat_max_ratio=0.75,
                             ignore_index=None),
            RandomHorizontalFlip(prob=0.5),
            PhotoMetricDistortion(brightness_delta=32,
                                  contrast_range=(0.5, 1.5),
                                  saturation_range=(0.5, 1.5),
                                  hue_delta=18,
                                  prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = SemanticSegmentationCollater(resize=512, ignore_index=None)
    train_loader = DataLoader(ade20kdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.semantic_segmentation.models.deeplabv3plus import resnet50backbone_deeplabv3plus
    net = resnet50backbone_deeplabv3plus(backbone_pretrained_path='',
                                         num_classes=150)

    loss1 = CELoss(ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('1111', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss1(preds, masks)
        print('1212', out)
        break

    loss2 = FocalCELoss(gamma=2.0, ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('2222', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss2(preds, masks)
        print('2323', out)
        break

    loss3 = MultiClassBCELoss(ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('3333', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss3(preds, masks)
        print('3434', out)
        break

    loss4 = MultiClassFocalBCELoss(alpha=0.25, gamma=2.0, ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('4444', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss4(preds, masks)
        print('4545', out)
        break

    loss5 = MultiClassOHEMBCELoss(negative_ratio=3.0, ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('5555', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss5(preds, masks)
        print('5656', out)
        break

    loss6 = DiceLoss(logit_type='softmax', smooth=1e-4, ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('6666', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss6(preds, masks)
        print('6767', out)
        break

    loss7 = DiceLoss(logit_type='sigmoid', smooth=1e-4, ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('7777', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss7(preds, masks)
        print('7878', out)
        break

    loss8 = LovaszLoss(ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('8888', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss8(preds, masks)
        print('8989', out)
        break

    loss9 = IoULoss(logit_type='softmax', ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('9999', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss9(preds, masks)
        print('9191', out)
        break

    loss10 = IoULoss(logit_type='sigmoid', ignore_index=None)
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('9999', images.shape, masks.shape, scales.shape, sizes.shape)
        preds = net(images)
        out = loss10(preds, masks)
        print('9292', out)
        break