import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'BCELoss',
    'OHEMBCELoss',
    'BCEIouloss',
    'BCEDiceLoss',
]


class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
        pass

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]
        assert num_classes == 1

        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1)
        label = label.view(-1)

        loss = -(label * torch.log(pred) + (1. - label) * torch.log(1. - pred))
        loss = loss.mean()

        return loss


class OHEMBCELoss(nn.Module):

    def __init__(self, negative_ratio=1.5):
        super(OHEMBCELoss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        num_classes = pred.shape[3]
        assert num_classes == 1

        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(-1)
        label = label.view(-1)

        positive_point_mask = (label > 0).float()

        positive_points_num = int(positive_point_mask.sum())
        negative_points_num = min(
            int((1. - positive_point_mask).sum()),
            int(positive_points_num * self.negative_ratio))

        loss = -(label * torch.log(pred) + (1. - label) * torch.log(1. - pred))

        positive_loss = loss * positive_point_mask
        negative_loss = loss * (1. - positive_point_mask)
        negative_loss, _ = torch.topk(negative_loss.view(-1),
                                      negative_points_num)

        ohem_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_points_num + negative_points_num + 1e-4)

        return ohem_loss


class BCEIouloss(nn.Module):

    def __init__(self, smooth=1e-4):
        super(BCEIouloss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        batch, num_classes = pred.shape[0], pred.shape[3]
        assert num_classes == 1

        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(batch, -1)
        label = label.view(batch, -1)

        intersection = pred * label

        iou_loss = 1. - (torch.sum(intersection, dim=1) + self.smooth) / (
            torch.sum(pred, dim=1) + torch.sum(label, dim=1) -
            torch.sum(intersection, dim=1) + self.smooth)
        iou_loss = iou_loss.mean()

        return iou_loss


class BCEDiceLoss(nn.Module):

    def __init__(self, smooth=1e-4):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, label):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        pred = pred.float()
        pred = pred.permute(0, 2, 3, 1).contiguous()
        batch, num_classes = pred.shape[0], pred.shape[3]
        assert num_classes == 1

        pred = torch.clamp(pred, min=1e-4, max=1. - 1e-4)

        pred = pred.view(batch, -1)
        label = label.view(batch, -1)

        intersection = pred * label

        dice_loss = 1. - (2 * torch.sum(intersection, dim=1) + self.smooth) / (
            torch.sum(pred, dim=1) + torch.sum(label, dim=1) + self.smooth)
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

    from tools.path import salient_object_detection_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.salient_object_detection.datasets.salient_object_detection_dataset import SalientObjectDetectionDataset
    from SimpleAICV.salient_object_detection.common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, SalientObjectDetectionSegmentationCollater

    salient_object_detection_dataset = SalientObjectDetectionDataset(
        salient_object_detection_dataset_path,
        set_name_list=[
            'AM2K',
            'DIS5K',
            'HRS10K',
            'HRSOD',
            'UHRSD',
        ],
        set_type='train',
        transform=transforms.Compose([
            YoloStyleResize(resize=1024),
            # Resize(resize=1024),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = SalientObjectDetectionSegmentationCollater(resize=1024)
    train_loader = DataLoader(salient_object_detection_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.salient_object_detection.models import vanb3_pfan_segmentation
    net = vanb3_pfan_segmentation()

    loss = BCELoss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('1212', preds.shape)
        out = loss(preds, masks)
        print('1313', out)
        break

    loss = OHEMBCELoss(negative_ratio=3.0)
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('1212', preds.shape)
        out = loss(preds, masks)
        print('1313', out)
        break

    loss = BCEIouloss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('1212', preds.shape)
        out = loss(preds, masks)
        print('1313', out)
        break

    loss = BCEDiceLoss()
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('1111', images.shape, masks.shape, sizes.shape)
        preds = net(images)
        print('1212', preds.shape)
        out = loss(preds, masks)
        print('1313', out)
        break
