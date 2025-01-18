import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'SAMMultiLevelLoss',
    'SAMMultiLevelIoUMaxLoss',
    'SAMMultiLevelAssignLoss',
]


class SAMMultiLevelLoss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAMMultiLevelLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        pred_masks, pred_ious = inputs

        focal_loss = self.focal_loss(pred_masks, targets)
        dice_loss = self.dice_loss(pred_masks, targets)
        iou_predict_loss = self.iou_predict_loss(pred_masks, targets,
                                                 pred_ious)

        focal_loss = self.focal_loss_weight * focal_loss
        dice_loss = self.dice_loss_weight * dice_loss
        iou_predict_loss = self.iou_predict_loss_weight * iou_predict_loss

        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def focal_loss(self, inputs, targets):
        idx_nums = inputs.shape[1]
        targets = targets.reshape(-1)

        inputs = inputs.float()

        total_focal_loss = 0.
        for per_idx in range(inputs.shape[1]):
            per_idx_inputs = inputs[:, per_idx:per_idx + 1, :, :]
            per_idx_inputs = per_idx_inputs.reshape(-1)

            assert per_idx_inputs.shape[0] == targets.shape[0]

            bce_loss = F.binary_cross_entropy_with_logits(per_idx_inputs,
                                                          targets,
                                                          reduction='none')
            focal_loss = self.alpha * (
                1 - torch.exp(-bce_loss))**self.gamma * bce_loss
            focal_loss = focal_loss.mean()
            total_focal_loss += focal_loss

        total_focal_loss = total_focal_loss / idx_nums

        return total_focal_loss

    def dice_loss(self, inputs, targets):
        idx_nums = inputs.shape[1]

        inputs = inputs.float()
        inputs = self.sigmoid(inputs)

        targets = targets.reshape(-1)

        total_dice_loss = 0.
        for per_idx in range(inputs.shape[1]):
            per_idx_inputs = inputs[:, per_idx:per_idx + 1, :, :]
            per_idx_inputs = per_idx_inputs.reshape(-1)

            assert per_idx_inputs.shape[0] == targets.shape[0]

            intersection = (per_idx_inputs * targets).sum()

            dice_loss = 1. - (
                (2. * intersection + self.smooth) /
                (per_idx_inputs.sum() + targets.sum() + self.smooth))
            total_dice_loss += dice_loss

        total_dice_loss = total_dice_loss / idx_nums

        return total_dice_loss

    def iou_predict_loss(self, inputs, targets, iou_predictions):
        inputs = (inputs >= self.mask_threshold).float()

        batch_size = inputs.shape[0]
        idx_nums = inputs.shape[1]

        targets = targets.reshape(batch_size, -1)

        total_iou_predict_loss = 0.
        for per_idx in range(inputs.shape[1]):
            per_idx_inputs = inputs[:, per_idx:per_idx + 1, :, :]
            per_idx_inputs = per_idx_inputs.reshape(batch_size, -1)

            intersection = per_idx_inputs * targets

            per_idx_iou_gt = (torch.sum(intersection, dim=1) + self.smooth) / (
                (torch.sum(per_idx_inputs, dim=1) + torch.sum(targets, dim=1) -
                 torch.sum(intersection, dim=1)) + self.smooth)

            per_idx_iou_predictions = iou_predictions[:, per_idx]
            iou_predict_loss = F.mse_loss(per_idx_iou_predictions,
                                          per_idx_iou_gt,
                                          reduction='sum') / batch_size
            total_iou_predict_loss += iou_predict_loss

        total_iou_predict_loss = total_iou_predict_loss / idx_nums

        return total_iou_predict_loss


class SAMMultiLevelIoUMaxLoss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAMMultiLevelIoUMaxLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        pred_masks, pred_ious = inputs

        pred_masks_0_1 = (pred_masks >= self.mask_threshold).float()
        intersection = (pred_masks_0_1 * targets).sum(dim=(2, 3))
        union = pred_masks_0_1.sum(dim=(2, 3)) + targets.sum(
            dim=(2, 3)) - intersection + 1e-4
        ious = intersection / union
        max_iou_idx = ious.argmax(dim=1)

        batch_range = torch.arange(pred_masks.shape[0],
                                   device=pred_masks.device)
        pred_masks = pred_masks[batch_range, max_iou_idx]
        pred_masks = pred_masks.unsqueeze(1)
        pred_ious = pred_ious[batch_range, max_iou_idx]
        pred_ious = pred_ious.unsqueeze(1)

        focal_loss = self.focal_loss(pred_masks, targets)
        dice_loss = self.dice_loss(pred_masks, targets)
        iou_predict_loss = self.iou_predict_loss(pred_masks, targets,
                                                 pred_ious)

        focal_loss = self.focal_loss_weight * focal_loss
        dice_loss = self.dice_loss_weight * dice_loss
        iou_predict_loss = self.iou_predict_loss_weight * iou_predict_loss

        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def focal_loss(self, inputs, targets):
        inputs = inputs.float()
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        assert inputs.shape[0] == targets.shape[0]

        bce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction='none')
        focal_loss = self.alpha * (1 -
                                   torch.exp(-bce_loss))**self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        return focal_loss

    def dice_loss(self, inputs, targets):
        inputs = inputs.float()
        inputs = self.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        assert inputs.shape[0] == targets.shape[0]

        intersection = (inputs * targets).sum()
        dice_loss = 1. - ((2. * intersection + self.smooth) /
                          (inputs.sum() + targets.sum() + self.smooth))

        return dice_loss

    def iou_predict_loss(self, inputs, targets, iou_predictions):
        inputs = (inputs >= self.mask_threshold).float()

        batch_size = inputs.shape[0]

        inputs = inputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        iou_predictions = iou_predictions.view(-1)

        assert inputs.shape[0] == targets.shape[0] == iou_predictions.shape[0]

        intersection = inputs * targets

        iou_gt = (torch.sum(intersection, dim=1) + self.smooth) / (
            (torch.sum(inputs, dim=1) + torch.sum(targets, dim=1) -
             torch.sum(intersection, dim=1)) + self.smooth)

        iou_predict_loss = F.mse_loss(iou_predictions, iou_gt,
                                      reduction='sum') / batch_size

        return iou_predict_loss


class SAMMultiLevelAssignLoss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0,
                 idx_nums=4,
                 area_ranges=[[0.04, 0.64], [0.0, 0.04], [0.01, 0.25],
                              [0.16, 1.0]]):
        super(SAMMultiLevelAssignLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold
        self.idx_nums = idx_nums
        self.area_ranges = area_ranges
        assert len(self.area_ranges) == self.idx_nums

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        pred_masks, pred_ious = inputs

        assert self.idx_nums == pred_masks.shape[1] == pred_ious.shape[1]

        focal_loss = self.focal_loss(pred_masks, targets)
        dice_loss = self.dice_loss(pred_masks, targets)
        iou_predict_loss = self.iou_predict_loss(pred_masks, targets,
                                                 pred_ious)

        focal_loss = self.focal_loss_weight * focal_loss
        dice_loss = self.dice_loss_weight * dice_loss
        iou_predict_loss = self.iou_predict_loss_weight * iou_predict_loss

        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def focal_loss(self, inputs, targets):
        # torch.Size([3, 4, 1024, 1024]) torch.Size([3, 1, 1024, 1024])
        batch_size = inputs.shape[0]

        inputs = inputs.float()

        total_focal_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_focal_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_inputs = inputs[per_sample_idx]
            per_sample_target = targets[per_sample_idx]

            per_sample_target_h, per_sample_target_w = per_sample_target.shape[
                1], per_sample_target.shape[2]
            per_sample_target_area_ratio = torch.sum(
                per_sample_target) / float(
                    per_sample_target_h * per_sample_target_w)

            per_sample_target = per_sample_target.reshape(-1)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_target_area_ratio < per_area_range2:
                    per_idx_inputs = per_sample_inputs[per_idx:per_idx +
                                                       1, :, :]
                    per_idx_inputs = per_idx_inputs.reshape(-1)

                    assert per_idx_inputs.shape[0] == per_sample_target.shape[
                        0]

                    per_idx_bce_loss = F.binary_cross_entropy_with_logits(
                        per_idx_inputs, per_sample_target, reduction='none')
                    per_idx_focal_loss = self.alpha * (1 - torch.exp(
                        -per_idx_bce_loss))**self.gamma * per_idx_bce_loss
                    per_idx_focal_loss = per_idx_focal_loss.mean()

                    per_sample_focal_loss += per_idx_focal_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_focal_loss = per_sample_focal_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_focal_loss += per_sample_focal_loss

        if valid_batch_size > 0:
            total_focal_loss = total_focal_loss / valid_batch_size

        return total_focal_loss

    def dice_loss(self, inputs, targets):
        batch_size = inputs.shape[0]

        inputs = inputs.float()
        inputs = self.sigmoid(inputs)

        total_dice_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_dice_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_inputs = inputs[per_sample_idx]
            per_sample_target = targets[per_sample_idx]

            per_sample_target_h, per_sample_target_w = per_sample_target.shape[
                1], per_sample_target.shape[2]
            per_sample_target_area_ratio = torch.sum(
                per_sample_target) / float(
                    per_sample_target_h * per_sample_target_w)

            per_sample_target = per_sample_target.reshape(-1)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_target_area_ratio < per_area_range2:
                    per_idx_inputs = per_sample_inputs[per_idx:per_idx +
                                                       1, :, :]
                    per_idx_inputs = per_idx_inputs.reshape(-1)

                    assert per_idx_inputs.shape[0] == per_sample_target.shape[
                        0]

                    intersection = (per_idx_inputs * per_sample_target).sum()
                    per_idx_dice_loss = 1. - (
                        (2. * intersection + self.smooth) /
                        (per_idx_inputs.sum() + per_sample_target.sum() +
                         self.smooth))

                    per_sample_dice_loss += per_idx_dice_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_dice_loss = per_sample_dice_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_dice_loss += per_sample_dice_loss

        if valid_batch_size > 0:
            total_dice_loss = total_dice_loss / valid_batch_size

        return total_dice_loss

    def iou_predict_loss(self, inputs, targets, iou_predictions):
        inputs = (inputs >= self.mask_threshold).float()

        batch_size = inputs.shape[0]

        total_iou_predict_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_iou_predict_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_inputs = inputs[per_sample_idx]
            per_sample_target = targets[per_sample_idx]
            per_sample_iou_predictions = iou_predictions[per_sample_idx]

            per_sample_target_h, per_sample_target_w = per_sample_target.shape[
                1], per_sample_target.shape[2]
            per_sample_target_area_ratio = torch.sum(
                per_sample_target) / float(
                    per_sample_target_h * per_sample_target_w)

            per_sample_target = per_sample_target.reshape(-1)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_target_area_ratio < per_area_range2:
                    per_idx_inputs = per_sample_inputs[per_idx:per_idx +
                                                       1, :, :]
                    per_idx_inputs = per_idx_inputs.reshape(-1)

                    assert per_idx_inputs.shape[0] == per_sample_target.shape[
                        0]

                    intersection = per_idx_inputs * per_sample_target

                    per_idx_iou_gt = (
                        torch.sum(intersection, dim=0) + self.smooth) / (
                            (torch.sum(per_idx_inputs, dim=0) +
                             torch.sum(per_sample_target, dim=0) -
                             torch.sum(intersection, dim=0)) + self.smooth)

                    per_idx_iou_predictions = per_sample_iou_predictions[
                        per_idx]
                    per_idx_iou_predict_loss = F.mse_loss(
                        per_idx_iou_predictions,
                        per_idx_iou_gt,
                        reduction='sum')
                    per_sample_iou_predict_loss += per_idx_iou_predict_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_iou_predict_loss = per_sample_iou_predict_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_iou_predict_loss += per_sample_iou_predict_loss

        if valid_batch_size > 0:
            total_iou_predict_loss = total_iou_predict_loss / valid_batch_size

        return total_iou_predict_loss


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

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from tools.path import interactive_segmentation_dataset_path

    from simpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

    samdataset = SAMSegmentationDataset(interactive_segmentation_dataset_path,
                                        set_name=[
                                            'sa_000020',
                                        ],
                                        set_type='train',
                                        per_set_image_choose_max_num={
                                            'sa_000020': 1000000,
                                        },
                                        per_image_mask_chosse_max_num=16,
                                        positive_points_num=9,
                                        negative_points_num=9,
                                        area_filter_ratio=0.0001,
                                        box_noise_wh_ratio=0.1,
                                        mask_noise_area_ratio=0.04,
                                        transform=transforms.Compose([
                                            SamResize(resize=1024),
                                            SamRandomHorizontalFlip(prob=0.5),
                                            SamNormalize(
                                                mean=[123.675, 116.28, 103.53],
                                                std=[58.395, 57.12, 57.375]),
                                        ]))

    from torch.utils.data import DataLoader

    collater = SAMBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from simpleAICV.interactive_segmentation.models.segment_anything.sam import sam_b
    net = sam_b(image_size=1024,
                frozen_image_encoder=False,
                frozen_prompt_encoder=False,
                frozen_mask_decoder=False,
                use_gradient_checkpoint=True,
                sigmoid_out=False,
                binary_mask_out=False,
                mask_threshold=0.0)
    load_state_dict(
        '/root/autodl-tmp/pretrained_models/sam_official_pytorch_weights/sam_vit_b_01ec64.pth',
        net)

    loss = SAMMultiLevelLoss(alpha=0.8,
                             gamma=2,
                             smooth=1e-4,
                             focal_loss_weight=20,
                             dice_loss_weight=1,
                             iou_predict_loss_weight=1,
                             mask_threshold=0.0)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('3333', preds[0].shape, preds[1].shape, preds[0].dtype,
              preds[1].dtype)

        loss_dict = loss(preds, input_masks)
        print('4444', loss_dict)

        break

    loss = SAMMultiLevelIoUMaxLoss(alpha=0.8,
                                   gamma=2,
                                   smooth=1e-4,
                                   focal_loss_weight=20,
                                   dice_loss_weight=1,
                                   iou_predict_loss_weight=1,
                                   mask_threshold=0.0)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('3333', preds[0].shape, preds[1].shape, preds[0].dtype,
              preds[1].dtype)

        loss_dict = loss(preds, input_masks)
        print('4444', loss_dict)

        break

    loss = SAMMultiLevelAssignLoss(alpha=0.8,
                                   gamma=2,
                                   smooth=1e-4,
                                   focal_loss_weight=20,
                                   dice_loss_weight=1,
                                   iou_predict_loss_weight=1,
                                   mask_threshold=0.0,
                                   idx_nums=4,
                                   area_ranges=[[0.04, 0.64], [0.0, 0.04],
                                                [0.01, 0.25], [0.16, 1.0]])

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('3333', preds[0].shape, preds[1].shape, preds[0].dtype,
              preds[1].dtype)

        loss_dict = loss(preds, input_masks)
        print('4444', loss_dict)

        break
