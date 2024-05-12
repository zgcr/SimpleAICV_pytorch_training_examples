import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'SAMLoss',
    'SAMMultiLevelLoss',
    'SAMMultiLevelAssignLoss',
    'SAML1Loss',
    'SAMAdvanceLoss',
    'SAMMultiLevelAdvanceLoss',
]


class SAMLoss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAMLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        pred_masks = []
        for per_mask in inputs[0]:
            pred_masks.append(per_mask)
        pred_masks = torch.cat(pred_masks, dim=0)

        pred_ious = []
        for per_iou in inputs[1]:
            pred_ious.append(per_iou)
        pred_ious = torch.cat(pred_ious, dim=0)

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
        inputs = self.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

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
        pred_masks = []
        for per_mask in inputs[0]:
            pred_masks.append(per_mask)
        pred_masks = torch.cat(pred_masks, dim=0)

        pred_ious = []
        for per_iou in inputs[1]:
            pred_ious.append(per_iou)
        pred_ious = torch.cat(pred_ious, dim=0)

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

        return focal_loss

    def dice_loss(self, inputs, targets):
        idx_nums = inputs.shape[1]

        inputs = self.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

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
                 area_ranges=[[0.0, 0.04], [0.01, 0.16], [0.09, 0.49],
                              [0.25, 1.0]]):
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
        pred_masks = []
        for per_mask in inputs[0]:
            pred_masks.append(per_mask)
        pred_masks = torch.cat(pred_masks, dim=0)

        pred_ious = []
        for per_iou in inputs[1]:
            pred_ious.append(per_iou)
        pred_ious = torch.cat(pred_ious, dim=0)

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
        batch_size = inputs.shape[0]

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

        inputs = self.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

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


class SAML1Loss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAML1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        pred_masks = []
        for per_mask in inputs[0]:
            pred_masks.append(per_mask)
        pred_masks = torch.cat(pred_masks, dim=0)

        pred_ious = []
        for per_iou in inputs[1]:
            pred_ious.append(per_iou)
        pred_ious = torch.cat(pred_ious, dim=0)

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
        inputs = self.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

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

        iou_predict_loss = torch.abs(iou_predictions - iou_gt)
        iou_predict_loss = torch.sum(iou_predict_loss) / batch_size

        return iou_predict_loss


class SAMAdvanceLoss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 bce_loss_weight=20,
                 iou_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAMAdvanceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce_loss_weight = bce_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        pred_masks = []
        for per_mask in inputs[0]:
            pred_masks.append(per_mask)
        pred_masks = torch.cat(pred_masks, dim=0)

        pred_ious = []
        for per_iou in inputs[1]:
            pred_ious.append(per_iou)
        pred_ious = torch.cat(pred_ious, dim=0)

        bce_loss = self.bce_loss(pred_masks, targets)
        iou_loss = self.iou_loss(pred_masks, targets)
        iou_predict_loss = self.iou_predict_loss(pred_masks, targets,
                                                 pred_ious)

        bce_loss = self.bce_loss_weight * bce_loss
        iou_loss = self.iou_loss_weight * iou_loss
        iou_predict_loss = self.iou_predict_loss_weight * iou_predict_loss

        loss_dict = {
            'bce_loss': bce_loss,
            'iou_loss': iou_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def bce_loss(self, inputs, targets):
        inputs = self.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        assert inputs.shape[0] == targets.shape[0]

        loss = -(targets * torch.log(inputs) +
                 (1. - targets) * torch.log(1. - inputs))
        loss = loss.mean()

        return loss

    def iou_loss(self, inputs, targets):
        inputs = self.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        assert inputs.shape[0] == targets.shape[0]

        intersection = (inputs * targets).sum()
        iou_loss = 1. - (intersection + self.smooth) / (
            inputs.sum() + targets.sum() - intersection + self.smooth)

        return iou_loss

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

        iou_predict_loss = torch.abs(iou_predictions - iou_gt)
        iou_predict_loss = torch.sum(iou_predict_loss) / batch_size

        return iou_predict_loss


class SAMMultiLevelAdvanceLoss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 bce_loss_weight=20,
                 iou_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAMMultiLevelAdvanceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce_loss_weight = bce_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        pred_masks = []
        for per_mask in inputs[0]:
            pred_masks.append(per_mask)
        pred_masks = torch.cat(pred_masks, dim=0)

        pred_ious = []
        for per_iou in inputs[1]:
            pred_ious.append(per_iou)
        pred_ious = torch.cat(pred_ious, dim=0)

        bce_loss = self.bce_loss(pred_masks, targets)
        iou_loss = self.iou_loss(pred_masks, targets)
        iou_predict_loss = self.iou_predict_loss(pred_masks, targets,
                                                 pred_ious)

        bce_loss = self.bce_loss_weight * bce_loss
        iou_loss = self.iou_loss_weight * iou_loss
        iou_predict_loss = self.iou_predict_loss_weight * iou_predict_loss

        loss_dict = {
            'bce_loss': bce_loss,
            'iou_loss': iou_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def bce_loss(self, inputs, targets):
        targets = targets.reshape(-1)
        out_idx_nums = inputs.shape[1]

        total_bce_loss = 0.
        for idx in range(inputs.shape[1]):
            per_idx_inputs = inputs[:, idx:idx + 1, :, :]
            per_idx_inputs = per_idx_inputs.reshape(-1)

            assert per_idx_inputs.shape[0] == targets.shape[0]

            bce_loss = F.binary_cross_entropy_with_logits(per_idx_inputs,
                                                          targets,
                                                          reduction='none')
            bce_loss = bce_loss.mean()
            total_bce_loss += bce_loss

        total_bce_loss = total_bce_loss / out_idx_nums

        return total_bce_loss

    def iou_loss(self, inputs, targets):
        out_idx_nums = inputs.shape[1]

        targets = targets.reshape(-1)

        total_iou_loss = 0.
        for idx in range(inputs.shape[1]):
            per_idx_inputs = inputs[:, idx:idx + 1, :, :]

            per_idx_inputs = self.sigmoid(per_idx_inputs)
            per_idx_inputs = torch.clamp(per_idx_inputs,
                                         min=1e-4,
                                         max=1. - 1e-4)

            per_idx_inputs = per_idx_inputs.reshape(-1)

            assert per_idx_inputs.shape[0] == targets.shape[0]

            intersection = (per_idx_inputs * targets).sum()
            iou_loss = 1. - (intersection + self.smooth) / (per_idx_inputs.sum(
            ) + targets.sum() - intersection + self.smooth)

            total_iou_loss += iou_loss

        total_iou_loss = total_iou_loss / out_idx_nums

        return total_iou_loss

    def iou_predict_loss(self, inputs, targets, iou_predictions):
        inputs = (inputs >= self.mask_threshold).float()

        batch_size = inputs.shape[0]
        out_idx_nums = inputs.shape[1]

        targets = targets.reshape(batch_size, -1)

        total_iou_predict_loss = 0.
        for idx in range(out_idx_nums):
            per_idx_inputs = inputs[:, idx:idx + 1, :, :]
            per_idx_iou_predictions = iou_predictions[:, idx:idx + 1]

            per_idx_inputs = per_idx_inputs.reshape(batch_size, -1)
            per_idx_iou_predictions = per_idx_iou_predictions.reshape(-1)

            assert per_idx_inputs.shape[0] == targets.shape[0]

            intersection = per_idx_inputs * targets

            iou_gt = (torch.sum(intersection, dim=1) + self.smooth) / (
                (torch.sum(per_idx_inputs, dim=1) + torch.sum(targets, dim=1) -
                 torch.sum(intersection, dim=1)) + self.smooth)

            assert per_idx_iou_predictions.shape[0] == iou_gt.shape[0]

            iou_predict_loss = torch.abs(per_idx_iou_predictions - iou_gt)
            iou_predict_loss = torch.sum(iou_predict_loss) / batch_size

            total_iou_predict_loss += iou_predict_loss

        total_iou_predict_loss = total_iou_predict_loss / out_idx_nums

        return total_iou_predict_loss


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

    from tools.path import COCO2017_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.interactive_segmentation.datasets.coco2017dataset import COCO2017dataset
    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMCollater, load_state_dict

    sam1bdataset = COCO2017dataset(COCO2017_path,
                                   set_name='train2017',
                                   positive_points_num=9,
                                   negative_points_num=9,
                                   area_filter_ratio=0.0025,
                                   box_noise_pixel=50,
                                   mask_noise_pixel=100,
                                   transform=transforms.Compose([
                                       SamResize(resize=1024),
                                       SamRandomHorizontalFlip(prob=0.5),
                                       SamNormalize(
                                           mean=[123.675, 116.28, 103.53],
                                           std=[58.395, 57.12, 57.375]),
                                   ]))

    from torch.utils.data import DataLoader
    collater = SAMCollater(resize=1024,
                           positive_point_num_range=[1, 5],
                           negative_point_num_range=0,
                           batch_align_random_point_num=False,
                           positive_negative_point_num_ratio=None)
    train_loader = DataLoader(sam1bdataset,
                              batch_size=5,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.interactive_segmentation.models.segment_anything import sam_b
    net = sam_b(image_size=1024,
                frozen_image_encoder=False,
                frozen_prompt_encoder=False,
                frozen_mask_decoder=False,
                use_gradient_checkpoint=True,
                sigmoid_out=False,
                binary_mask_out=False,
                mask_threshold=0.0)
    load_state_dict(
        '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/sam_official_pytorch_weights/sam_vit_b_01ec64.pth',
        net)

    loss = SAMLoss(alpha=0.8,
                   gamma=2,
                   smooth=1e-4,
                   focal_loss_weight=20,
                   dice_loss_weight=1,
                   iou_predict_loss_weight=1)

    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        net = net.cuda()
        batch_images = batch_images.cuda()
        batch_masks = batch_masks.cuda()
        print('1111', batch_images.shape, batch_masks.shape)

        preds = net(batch_images, batch_prompts, mask_out_idxs=[0])

        for per_pred1, per_pred2 in zip(preds[0], preds[1]):
            print('2222', per_pred1.shape, per_pred2.shape)

        loss_dict = loss(preds, batch_masks)
        print('3333', loss_dict)

        break

    loss = SAMMultiLevelLoss(alpha=0.8,
                             gamma=2,
                             smooth=1e-4,
                             focal_loss_weight=20,
                             dice_loss_weight=1,
                             iou_predict_loss_weight=1)

    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        net = net.cuda()
        batch_images = batch_images.cuda()
        batch_masks = batch_masks.cuda()
        print('1111', batch_images.shape, batch_masks.shape)

        preds = net(batch_images, batch_prompts, mask_out_idxs=[0, 1, 2, 3])

        for per_pred1, per_pred2 in zip(preds[0], preds[1]):
            print('2222', per_pred1.shape, per_pred2.shape)

        loss_dict = loss(preds, batch_masks)
        print('3333', loss_dict)

        break

    loss = SAMMultiLevelAssignLoss(alpha=0.8,
                                   gamma=2,
                                   smooth=1e-4,
                                   focal_loss_weight=20,
                                   dice_loss_weight=1,
                                   iou_predict_loss_weight=1,
                                   idx_nums=4,
                                   area_ranges=[[0.0, 0.04], [0.01, 0.16],
                                                [0.09, 0.49], [0.25, 1.0]])

    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        net = net.cuda()
        batch_images = batch_images.cuda()
        batch_masks = batch_masks.cuda()
        print('1111', batch_images.shape, batch_masks.shape)

        preds = net(batch_images, batch_prompts, mask_out_idxs=[0, 1, 2, 3])

        for per_pred1, per_pred2 in zip(preds[0], preds[1]):
            print('2222', per_pred1.shape, per_pred2.shape)

        loss_dict = loss(preds, batch_masks)
        print('3333', loss_dict)

        break

    loss = SAML1Loss(alpha=0.8,
                     gamma=2,
                     smooth=1e-4,
                     focal_loss_weight=20,
                     dice_loss_weight=1,
                     iou_predict_loss_weight=1)

    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        net = net.cuda()
        batch_images = batch_images.cuda()
        batch_masks = batch_masks.cuda()
        print('1111', batch_images.shape, batch_masks.shape)

        preds = net(batch_images, batch_prompts, mask_out_idxs=[0])

        for per_pred1, per_pred2 in zip(preds[0], preds[1]):
            print('2222', per_pred1.shape, per_pred2.shape)

        loss_dict = loss(preds, batch_masks)
        print('3333', loss_dict)

        break

    loss = SAMAdvanceLoss(alpha=0.8,
                          gamma=2,
                          smooth=1e-4,
                          bce_loss_weight=20,
                          iou_loss_weight=1,
                          iou_predict_loss_weight=1)

    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        net = net.cuda()
        batch_images = batch_images.cuda()
        batch_masks = batch_masks.cuda()
        print('1111', batch_images.shape, batch_masks.shape)

        preds = net(batch_images, batch_prompts, mask_out_idxs=[0])

        for per_pred1, per_pred2 in zip(preds[0], preds[1]):
            print('2222', per_pred1.shape, per_pred2.shape)

        loss_dict = loss(preds, batch_masks)
        print('3333', loss_dict)

        break

    loss = SAMMultiLevelAdvanceLoss(alpha=0.8,
                                    gamma=2,
                                    smooth=1e-4,
                                    bce_loss_weight=20,
                                    iou_loss_weight=1,
                                    iou_predict_loss_weight=1)

    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        net = net.cuda()
        batch_images = batch_images.cuda()
        batch_masks = batch_masks.cuda()
        print('1111', batch_images.shape, batch_masks.shape)

        preds = net(batch_images, batch_prompts, mask_out_idxs=[0, 1, 2, 3])

        for per_pred1, per_pred2 in zip(preds[0], preds[1]):
            print('2222', per_pred1.shape, per_pred2.shape)

        loss_dict = loss(preds, batch_masks)
        print('3333', loss_dict)

        break
