import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.detection.losses import IoUMethod

INF = 100000000


class Solov2Loss(nn.Module):
    def __init__(self,
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768),
                               (384, INF)),
                 strides=[8, 8, 16, 32, 32],
                 num_grids=[40, 36, 24, 16, 12],
                 mask_stride=4,
                 mask_loss_weight=3.0,
                 cls_loss_weight=1.0,
                 sigma=0.2,
                 alpha=0.25,
                 gamma=2.0,
                 epsilon=1e-4):
        super(Solov2Loss, self).__init__()
        self.scale_ranges = scale_ranges
        self.strides = strides
        self.num_grids = num_grids
        self.mask_stride = mask_stride
        self.mask_loss_weight = mask_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, annotations, cate_outs, kernel_outs, mask_out):
        '''
        compute dice loss and cate loss in one batch
        '''
        mask_hw = mask_out.shape[2:4]
        cate_targets, ins_targets, positive_grid_idxs = self.get_batch_targets(
            annotations, mask_hw)

        del annotations

        mask_loss, cls_loss = self.compute_batch_loss(cate_outs, kernel_outs,
                                                      mask_out, cate_targets,
                                                      ins_targets,
                                                      positive_grid_idxs)

        del cate_outs, kernel_outs, mask_out, cate_targets, ins_targets, positive_grid_idxs

        mask_loss = self.mask_loss_weight * mask_loss
        cls_loss = self.cls_loss_weight * cls_loss

        loss_dict = {
            'mask_loss': mask_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict

    def compute_batch_loss(self, cate_outs, kernel_outs, mask_out,
                           cate_targets, ins_targets, positive_grid_idxs):
        device = mask_out.device

        mask_sample_nums = sum([
            len(per_img_level_positive_grid_idxs)
            for per_img_positive_grid_idxs in positive_grid_idxs
            for per_img_level_positive_grid_idxs in per_img_positive_grid_idxs
        ])

        if mask_sample_nums == 0:
            # return zero mask loss and cls loss
            return torch.tensor(0.).to(device), torch.tensor(0.).to(device)

        # positive_grid_idxs:per img->per level->idxs
        # snap_positive_grid_idxs:per level->per img->idxs
        snap_positive_grid_idxs = []
        for i in range(len(self.num_grids)):
            per_level_positive_grid_idxs = []
            for per_img_positive_grid_idxs in positive_grid_idxs:
                per_level_positive_grid_idxs.append(
                    per_img_positive_grid_idxs[i])
            snap_positive_grid_idxs.append(per_level_positive_grid_idxs)

        kernel_preds = []
        for per_level_kernel_outs, per_level_positive_grid_idxs in zip(
                kernel_outs, snap_positive_grid_idxs):
            per_level_kernel_preds = []
            for per_level_img_kernel_outs, per_level_img_positive_grid_idxs in zip(
                    per_level_kernel_outs, per_level_positive_grid_idxs):
                per_level_img_kernel_outs = per_level_img_kernel_outs.view(
                    -1, per_level_img_kernel_outs.shape[-1])
                per_level_img_kernel_preds = per_level_img_kernel_outs[
                    per_level_img_positive_grid_idxs]
                per_level_kernel_preds.append(per_level_img_kernel_preds)
            kernel_preds.append(per_level_kernel_preds)

        # generate masks
        mask_preds = []
        H, W = mask_out.shape[-2:]
        for per_level_kernel_preds in kernel_preds:
            per_level_mask_pred = []
            for img_idx, per_level_img_kernel_preds in enumerate(
                    per_level_kernel_preds):
                if per_level_img_kernel_preds.shape[0] != 0:
                    per_img_mask_pred = mask_out[img_idx]
                    per_img_mask_pred = per_img_mask_pred.unsqueeze(0)

                    sample_nums, num_masks = per_level_img_kernel_preds.shape
                    per_level_img_kernel_preds = per_level_img_kernel_preds.unsqueeze(
                        -1).unsqueeze(-1)
                    per_img_mask_pred = F.conv2d(per_img_mask_pred,
                                                 per_level_img_kernel_preds,
                                                 stride=1).view(-1, H, W)
                    per_level_mask_pred.append(per_img_mask_pred)

            if len(per_level_mask_pred) != 0:
                per_level_mask_pred = torch.cat(per_level_mask_pred, dim=0)
            else:
                per_level_mask_pred = torch.zeros([0, H, W],
                                                  dtype=torch.float32,
                                                  device=device)
            mask_preds.append(per_level_mask_pred)
        mask_preds = torch.cat(mask_preds, dim=0)

        mask_targets = torch.cat([
            per_img_level_ins_targets for per_img_ins_targets in ins_targets
            for per_img_level_ins_targets in per_img_ins_targets
        ],
                                 dim=0)

        # compute mask dice loss
        mask_preds = mask_preds.view(mask_preds.shape[0], -1)
        mask_preds = torch.sigmoid(mask_preds)
        mask_targets = mask_targets.view(mask_targets.shape[0], -1)
        a = torch.sum(mask_preds * mask_targets, dim=1)
        b = torch.sum(mask_preds * mask_preds, dim=1)
        c = torch.sum(mask_targets * mask_targets, dim=1)
        mask_loss = 1 - (2 * a) / (b + c + 1e-4)
        mask_loss = mask_loss.sum() / mask_sample_nums

        combine_cate_targets = []
        for i in range(len(self.num_grids)):
            per_level_cate_targets = []
            for per_img_cate_targets in cate_targets:
                per_level_cate_targets.append(per_img_cate_targets[i])
            per_level_cate_targets = torch.stack(
                per_level_cate_targets).flatten()
            combine_cate_targets.append(per_level_cate_targets)
        combine_cate_targets = torch.cat(combine_cate_targets, dim=0)

        combine_cate_preds = [
            per_level_cate_out.view(-1, per_level_cate_out.shape[-1])
            for per_level_cate_out in cate_outs
        ]
        combine_cate_preds = torch.cat(combine_cate_preds, dim=0)

        # compute cls focal loss
        positive_idxs = (combine_cate_targets > -1)
        cate_sample_nums = len(positive_idxs[positive_idxs == True])
        positive_cate_preds = combine_cate_preds[positive_idxs]
        positive_cate_targets = combine_cate_targets[positive_idxs]
        positive_cate_preds = torch.clamp(positive_cate_preds,
                                          min=self.epsilon,
                                          max=1. - self.epsilon)
        num_classes = positive_cate_preds.shape[1]
        # generate 80 binary ground truth classes for each anchor
        cate_ground_truth = F.one_hot(positive_cate_targets.long(),
                                      num_classes=num_classes)
        cate_ground_truth = cate_ground_truth.float()
        alpha_factor = torch.ones_like(positive_cate_preds) * self.alpha
        alpha_factor = torch.where(torch.eq(cate_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(cate_ground_truth, 1.), positive_cate_preds,
                         1. - positive_cate_preds)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)
        cls_loss = -(
            cate_ground_truth * torch.log(positive_cate_preds) +
            (1. - cate_ground_truth) * torch.log(1. - positive_cate_preds))
        cls_loss = focal_weight * cls_loss
        cls_loss = cls_loss.sum() / cate_sample_nums

        return mask_loss, cls_loss

    def get_batch_targets(self, annotations, mask_size):
        annot_boxes, annot_masks, annot_classes = annotations[
            'box'], annotations['mask'], annotations['class']
        device = annot_boxes.device
        max_object_num = annot_boxes.shape[1]

        cate_targets, ins_targets, positive_grid_idxs = [], [], []
        for img_idx, (per_img_boxes, per_img_masks,
                      per_img_classes) in enumerate(
                          zip(annot_boxes, annot_masks, annot_classes)):
            per_img_boxes = per_img_boxes[per_img_classes[:, 0] > -1]
            per_img_masks = per_img_masks[per_img_classes[:, 0] > -1]
            per_img_classes = per_img_classes[per_img_classes[:, 0] > -1]

            valid_masks_flag = per_img_masks.sum(dim=-1).sum(dim=-1) > 0
            per_img_boxes = per_img_boxes[valid_masks_flag, :]
            per_img_masks = per_img_masks[valid_masks_flag, :]
            per_img_classes = per_img_classes[valid_masks_flag, :]

            per_img_ins_targets, per_img_cate_targets,per_img_positive_grid_idxs =[],[],[]
            for (lower_scale,
                 upper_scale), stride, num_grid in zip(self.scale_ranges,
                                                       self.strides,
                                                       self.num_grids):
                per_level_ins_targets = []
                per_level_positive_grid_idxs = []
                per_level_cate_targets = torch.ones([num_grid, num_grid],
                                                    dtype=torch.float32,
                                                    device=device) * (-1.)

                per_img_gt_areas = torch.sqrt(
                    (per_img_boxes[:, 2] - per_img_boxes[:, 0]) *
                    (per_img_boxes[:, 3] - per_img_boxes[:, 1]))
                positive_idxs = ((per_img_gt_areas >= lower_scale) &
                                 (per_img_gt_areas <= upper_scale)).nonzero(
                                     as_tuple=False).flatten()

                if per_img_boxes.shape[0] != 0 and len(positive_idxs) != 0:
                    gt_boxes = per_img_boxes[positive_idxs]
                    gt_masks = per_img_masks[positive_idxs]
                    gt_classes = per_img_classes[positive_idxs]

                    half_boxes_w = 0.5 * (gt_boxes[:, 2] -
                                          gt_boxes[:, 0]) * self.sigma
                    half_boxes_h = 0.5 * (gt_boxes[:, 3] -
                                          gt_boxes[:, 1]) * self.sigma

                    _, gt_masks_h, gt_masks_w = gt_masks.shape
                    ys = torch.arange(0,
                                      gt_masks_h,
                                      dtype=torch.float32,
                                      device=device)
                    xs = torch.arange(0,
                                      gt_masks_w,
                                      dtype=torch.float32,
                                      device=device)
                    m00 = gt_masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-4)
                    m10 = (gt_masks * xs).sum(dim=-1).sum(dim=-1)
                    m01 = (gt_masks * ys[:, None]).sum(dim=-1).sum(dim=-1)
                    mask_center_xs, mask_center_ys = m10 / m00, m01 / m00

                    scale = 1. / self.mask_stride
                    resize_gt_masks_h, resize_gt_masks_w = int(
                        gt_masks_h * scale + 0.5), int(gt_masks_w * scale +
                                                       0.5)
                    gt_masks = cv2.resize(
                        gt_masks.permute(1, 2, 0).cpu().numpy(),
                        (resize_gt_masks_w, resize_gt_masks_h))
                    gt_masks = torch.tensor(gt_masks,
                                            dtype=torch.float32,
                                            device=device)
                    if len(gt_masks.shape) == 2:
                        gt_masks = gt_masks.unsqueeze(-1)
                    gt_masks = gt_masks.permute(2, 0, 1)

                    for gt_idx, (per_gt_mask, per_gt_class, per_half_box_h,
                                 per_half_box_w, center_y,
                                 center_x) in enumerate(
                                     zip(gt_masks, gt_classes, half_boxes_h,
                                         half_boxes_w, mask_center_ys,
                                         mask_center_xs)):
                        # get mask input h,w
                        mask_input_size = (mask_size[0] * self.mask_stride,
                                           mask_size[1] * self.mask_stride)
                        # get center y/x idx in grid
                        coord_h = int(
                            (center_y / mask_input_size[0]) // (1. / num_grid))
                        coord_w = int(
                            (center_x / mask_input_size[1]) // (1. / num_grid))
                        # get left, top, right, down idx in grid
                        top_box = max(
                            0,
                            int(((center_y - per_half_box_h) /
                                 mask_input_size[0]) // (1. / num_grid)))
                        down_box = min(
                            num_grid - 1,
                            int(((center_y + per_half_box_h) /
                                 mask_input_size[0]) // (1. / num_grid)))
                        left_box = max(
                            0,
                            int(((center_x - per_half_box_w) /
                                 mask_input_size[1]) // (1. / num_grid)))
                        right_box = min(
                            num_grid - 1,
                            int(((center_x + per_half_box_w) /
                                 mask_input_size[1]) // (1. / num_grid)))
                        # get left, top, right, down idx in grid
                        # left, top, right, down points form a rectangle on grid
                        top = max(top_box, coord_h - 1)
                        down = min(down_box, coord_h + 1)
                        left = max(coord_w - 1, left_box)
                        right = min(right_box, coord_w + 1)

                        per_level_cate_targets[top:(down + 1),
                                               left:(right + 1)] = per_gt_class
                        for i in range(top, down + 1):
                            for j in range(left, right + 1):
                                # for each point in rectangle,create a ins_target(The mask that is downsampled by self.mask_stride)
                                grid_relative_idx = int(i * num_grid + j)
                                per_point_ins_target = torch.zeros(
                                    [1, mask_size[0], mask_size[1]],
                                    dtype=torch.float32,
                                    device=device)
                                per_point_ins_target[
                                    0, :per_gt_mask.shape[0], :per_gt_mask.
                                    shape[1]] = per_gt_mask
                                per_level_ins_targets.append(
                                    per_point_ins_target)
                                per_level_positive_grid_idxs.append(
                                    grid_relative_idx)

                    if len(per_level_ins_targets) != 0:
                        per_level_ins_targets = torch.cat(
                            per_level_ins_targets, dim=0)
                else:
                    per_level_ins_targets = torch.zeros(
                        [0, mask_size[0], mask_size[1]],
                        dtype=torch.float32,
                        device=device)

                per_img_cate_targets.append(per_level_cate_targets)
                per_img_ins_targets.append(per_level_ins_targets)
                per_img_positive_grid_idxs.append(per_level_positive_grid_idxs)

            cate_targets.append(per_img_cate_targets)
            ins_targets.append(per_img_ins_targets)
            positive_grid_idxs.append(per_img_positive_grid_idxs)

        return cate_targets, ins_targets, positive_grid_idxs


class CondInstLoss(nn.Module):
    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 mi=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]],
                 alpha=0.25,
                 gamma=2.,
                 cls_loss_weight=1.,
                 box_loss_weight=1.,
                 center_ness_loss_weight=1.,
                 mask_loss_weight=1.,
                 box_loss_iou_type='CIoU',
                 mask_preds_upsample_ratio=2,
                 num_masks=8,
                 center_sample_radius=1.5,
                 use_center_sample=True,
                 epsilon=1e-4):
        super(CondInstLoss, self).__init__()
        assert box_loss_iou_type in ['IoU', 'GIoU', 'DIoU',
                                     'CIoU'], 'wrong IoU type!'

        self.alpha = alpha
        self.gamma = gamma
        self.strides = strides
        self.mi = mi
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight
        self.center_ness_loss_weight = center_ness_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.box_loss_iou_type = box_loss_iou_type
        self.mask_preds_upsample_ratio = mask_preds_upsample_ratio
        self.num_masks = num_masks
        self.center_sample_radius = center_sample_radius
        self.use_center_sample = use_center_sample
        self.epsilon = epsilon
        self.iou_function = IoUMethod()

    def forward(self, annotations, cls_heads, reg_heads, center_heads,
                controllers_heads, mask_out, batch_positions):
        '''
        compute loss in one batch
        '''
        device = mask_out.device

        cls_preds, reg_preds, center_preds, controllers_preds, batch_detection_targets, batch_mask_targets, batch_positive_idxs = self.get_batch_targets(
            cls_heads,
            reg_heads,
            center_heads,
            controllers_heads,
            batch_positions,
            annotations,
            use_center_sample=self.use_center_sample)

        del cls_heads, reg_heads, center_heads, controllers_heads, batch_positions

        mask_loss = self.compute_batch_mask_loss(
            mask_out, controllers_preds, batch_detection_targets[:, :, 5:6],
            batch_mask_targets, batch_positive_idxs)

        del mask_out, controllers_preds, batch_mask_targets, batch_positive_idxs

        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
        reg_preds = reg_preds.view(-1, reg_preds.shape[-1])
        center_preds = center_preds.view(-1, center_preds.shape[-1])
        batch_detection_targets = batch_detection_targets.view(
            -1, batch_detection_targets.shape[-1])

        positive_points_num = batch_detection_targets[
            batch_detection_targets[:, 5] > 0].shape[0]

        if positive_points_num > 0:
            cls_loss = self.compute_batch_focal_loss(cls_preds,
                                                     batch_detection_targets)
            reg_loss = self.compute_batch_iou_loss(reg_preds,
                                                   batch_detection_targets)
            center_ness_loss = self.compute_batch_centerness_loss(
                center_preds, batch_detection_targets)
        else:
            cls_loss = torch.tensor(0.).to(device)
            reg_loss = torch.tensor(0.).to(device)
            center_ness_loss = torch.tensor(0.).to(device)

        cls_loss = self.cls_loss_weight * cls_loss
        reg_loss = self.box_loss_weight * reg_loss
        center_ness_loss = self.center_ness_loss_weight * center_ness_loss
        mask_loss = self.mask_loss_weight * mask_loss

        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'center_ness_loss': center_ness_loss,
            'mask_loss': mask_loss
        }

        return loss_dict

    def compute_batch_focal_loss(self, cls_preds, batch_targets):
        '''
        compute batch focal loss(cls loss)
        cls_preds:[batch_size*points_num,num_classes]
        batch_targets:[batch_size*points_num,8]
        '''
        device = cls_preds.device
        cls_preds = torch.clamp(cls_preds,
                                min=self.epsilon,
                                max=1. - self.epsilon)
        positive_points_num = batch_targets[batch_targets[:, 5] > 0].shape[0]
        num_classes = cls_preds.shape[1]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_targets[:, 5].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(cls_preds) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), cls_preds,
                         1. - cls_preds)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        batch_bce_loss = -(
            loss_ground_truth * torch.log(cls_preds) +
            (1. - loss_ground_truth) * torch.log(1. - cls_preds))

        batch_focal_loss = focal_weight * batch_bce_loss
        batch_focal_loss = batch_focal_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        batch_focal_loss = batch_focal_loss / positive_points_num

        return batch_focal_loss

    def compute_batch_iou_loss(self, reg_preds, batch_targets):
        '''
        compute batch giou loss(reg loss)
        reg_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        '''
        # only use positive points sample to compute reg loss
        device = reg_preds.device
        reg_preds = reg_preds[batch_targets[:, 5] > 0]
        batch_targets = batch_targets[batch_targets[:, 5] > 0]
        positive_points_num = batch_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        center_ness_targets = batch_targets[:, 4]

        pred_bboxes_xy_min = batch_targets[:, 6:8] - reg_preds[:, 0:2]
        pred_bboxes_xy_max = batch_targets[:, 6:8] + reg_preds[:, 2:4]
        gt_bboxes_xy_min = batch_targets[:, 6:8] - batch_targets[:, 0:2]
        gt_bboxes_xy_max = batch_targets[:, 6:8] + batch_targets[:, 2:4]

        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                dim=1)
        gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max], dim=1)

        ious = self.iou_function(pred_bboxes,
                                 gt_bboxes,
                                 iou_type=self.box_loss_iou_type,
                                 box_type='xyxy')
        ious_loss = 1 - ious
        # use center_ness_targets as the weight of gious loss
        ious_loss = ious_loss * center_ness_targets
        ious_loss = ious_loss.sum() / positive_points_num

        return ious_loss

    def compute_batch_centerness_loss(self, center_preds, batch_targets):
        '''
        compute batch center_ness loss(center ness loss)
        center_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        '''
        # only use positive points sample to compute center_ness loss
        device = center_preds.device
        center_preds = center_preds[batch_targets[:, 5] > 0]
        batch_targets = batch_targets[batch_targets[:, 5] > 0]
        positive_points_num = batch_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        center_preds = torch.clamp(center_preds,
                                   min=self.epsilon,
                                   max=1. - self.epsilon)
        center_ness_targets = batch_targets[:, 4:5]

        center_ness_loss = -(
            center_ness_targets * torch.log(center_preds) +
            (1. - center_ness_targets) * torch.log(1. - center_preds))

        center_ness_loss = center_ness_loss.sum() / positive_points_num

        return center_ness_loss

    def compute_batch_mask_loss(self, mask_out, controllers_preds,
                                batch_mask_classes, batch_mask_targets,
                                batch_positive_idxs):
        device = mask_out.device
        batch_size, mask_h, mask_w, _ = mask_out.shape
        relative_coord_x = torch.arange(mask_w).view(1, -1).float().repeat(
            mask_h, 1) / (mask_w - 1) * 2 - 1
        relative_coord_y = torch.arange(mask_h).view(-1, 1).float().repeat(
            1, mask_w) / (mask_h - 1) * 2 - 1
        relative_coord_x = relative_coord_x.view(1, 1, mask_h, mask_w).repeat(
            batch_size, 1, 1, 1)
        relative_coord_y = relative_coord_y.view(1, 1, mask_h, mask_w).repeat(
            batch_size, 1, 1, 1)
        relative_coord_x = relative_coord_x.permute(0, 2, 3, 1).to(device)
        relative_coord_y = relative_coord_y.permute(0, 2, 3, 1).to(device)
        mask_out = torch.cat([mask_out, relative_coord_x, relative_coord_y],
                             dim=-1)

        del relative_coord_x, relative_coord_y

        positive_controllers_preds = [
            per_image_controllers_preds[per_image_positive_idxs]
            for per_image_controllers_preds, per_image_positive_idxs in zip(
                controllers_preds, batch_positive_idxs)
        ]

        positive_mask_classes = [
            per_image_mask_classes[per_image_positive_idxs]
            for per_image_mask_classes, per_image_positive_idxs in zip(
                batch_mask_classes, batch_positive_idxs)
        ]

        del controllers_preds, batch_mask_classes, batch_positive_idxs

        # 3 fcn conv layers weight and bias params idx
        conv1_w_end = int((self.num_masks + 2) * self.num_masks)
        conv1_bias_end = int(conv1_w_end + self.num_masks)
        conv2_w_end = int(conv1_bias_end + self.num_masks * self.num_masks)
        conv2_bias_end = int(conv2_w_end + self.num_masks)
        conv3_w_end = int(conv2_bias_end + self.num_masks)
        conv3_bias_end = int(conv3_w_end + 1)
        final_mask_h, final_mask_w = int(
            mask_h * self.mask_preds_upsample_ratio), int(
                mask_w * self.mask_preds_upsample_ratio)

        mask_preds, mask_gts = [], []
        for per_image_mask_out, per_image_controllers, per_image_mask_classes, per_image_mask_targets in zip(
                mask_out, positive_controllers_preds, positive_mask_classes,
                batch_mask_targets):
            sample_nums = per_image_controllers.shape[0]

            if sample_nums == 0:
                per_image_mask_preds = torch.zeros(
                    [0, final_mask_h, final_mask_w],
                    dtype=torch.float32,
                    device=device)
                per_image_mask_gts = torch.zeros(
                    [0, final_mask_h, final_mask_w],
                    dtype=torch.float32,
                    device=device)
            else:
                # get mask fcn head 3 conv layers weight and bias
                conv1_weights = per_image_controllers[:, 0:conv1_w_end].reshape(
                    -1, int(self.num_masks), int(self.num_masks + 2)).reshape(
                        -1,
                        int(self.num_masks + 2)).unsqueeze(-1).unsqueeze(-1)
                conv1_bias = per_image_controllers[:, conv1_w_end:
                                                   conv1_bias_end].flatten()
                conv2_weights = per_image_controllers[:, conv1_bias_end:
                                                      conv2_w_end].reshape(
                                                          -1,
                                                          int(self.num_masks),
                                                          int(self.num_masks)
                                                      ).reshape(
                                                          -1,
                                                          int(self.num_masks)
                                                      ).unsqueeze(
                                                          -1).unsqueeze(-1)
                conv2_bias = per_image_controllers[:, conv2_w_end:
                                                   conv2_bias_end].flatten()
                conv3_weights = per_image_controllers[:, conv2_bias_end:
                                                      conv3_w_end].unsqueeze(
                                                          -1).unsqueeze(-1)
                conv3_bias = per_image_controllers[:, conv3_w_end:
                                                   conv3_bias_end].flatten()

                # get mask preds through mask fcn head(3 conv layers)
                per_image_mask_out = per_image_mask_out.permute(2, 0,
                                                                1).unsqueeze(0)
                per_image_mask_preds = F.conv2d(per_image_mask_out,
                                                conv1_weights, conv1_bias)
                per_image_mask_preds = F.relu(per_image_mask_preds)
                per_image_mask_preds = F.conv2d(per_image_mask_preds,
                                                conv2_weights,
                                                conv2_bias,
                                                groups=sample_nums)
                per_image_mask_preds = F.relu(per_image_mask_preds)
                per_image_mask_preds = F.conv2d(per_image_mask_preds,
                                                conv3_weights,
                                                conv3_bias,
                                                groups=sample_nums)

                # resize mask preds to upsample 2x
                per_image_mask_preds = F.pad(per_image_mask_preds,
                                             pad=(0, 1, 0, 1),
                                             mode="replicate")
                oh, ow = final_mask_h + 1, final_mask_w + 1
                per_image_mask_preds = F.interpolate(per_image_mask_preds,
                                                     size=(oh, ow),
                                                     mode='nearest')
                pad_factor = int(self.mask_preds_upsample_ratio / 2.)
                per_image_mask_preds = F.pad(per_image_mask_preds,
                                             pad=(pad_factor, 0, pad_factor,
                                                  0),
                                             mode="replicate")
                per_image_mask_preds = per_image_mask_preds[:, :, :oh -
                                                            1, :ow - 1]
                per_image_mask_preds = per_image_mask_preds.squeeze(0)

                # resize mask targets to downsample 4x
                per_image_mask_gts = cv2.resize(
                    per_image_mask_targets.permute(1, 2, 0).cpu().numpy(),
                    (final_mask_w, final_mask_h))
                per_image_mask_gts = torch.tensor(per_image_mask_gts,
                                                  dtype=torch.float32,
                                                  device=device)
                if len(per_image_mask_gts.shape) == 2:
                    per_image_mask_gts = per_image_mask_gts.unsqueeze(-1)
                per_image_mask_gts = per_image_mask_gts.permute(2, 0, 1)

            mask_preds.append(per_image_mask_preds)
            mask_gts.append(per_image_mask_gts)

        mask_preds = torch.cat(mask_preds, dim=0)
        mask_gts = torch.cat(mask_gts, dim=0)

        del mask_out, positive_controllers_preds, positive_mask_classes, batch_mask_targets

        # compute mask dice loss
        total_sample_nums = mask_gts.shape[0]

        if total_sample_nums == 0:
            return torch.tensor(0.).to(device)

        mask_preds = mask_preds.view(mask_preds.shape[0], -1)
        mask_preds = torch.sigmoid(mask_preds)
        mask_gts = mask_gts.view(mask_gts.shape[0], -1)
        a = torch.sum(mask_preds * mask_gts, dim=1)
        b = torch.sum(mask_preds * mask_preds, dim=1)
        c = torch.sum(mask_gts * mask_gts, dim=1)
        mask_loss = 1 - (2 * a) / (b + c + 1e-4)
        mask_loss = mask_loss.sum() / total_sample_nums

        return mask_loss

    def get_batch_targets(self,
                          cls_heads,
                          reg_heads,
                          center_heads,
                          controllers_heads,
                          batch_positions,
                          annotations,
                          use_center_sample=True):
        '''
        Assign a ground truth target for each position on feature map
        '''
        annot_boxes, annot_masks, annot_classes = annotations[
            'box'], annotations['mask'], annotations['class']
        device = annot_boxes.device
        mask_h, mask_w = annot_masks.shape[-2], annot_masks.shape[-1]

        batch_mi, batch_stride = [], []
        for reg_head, mi, stride in zip(reg_heads, self.mi, self.strides):
            mi = torch.tensor(mi).to(device)
            B, H, W, _ = reg_head.shape
            per_level_mi = torch.zeros([B, H, W, 2],
                                       dtype=torch.float32,
                                       device=device)
            per_level_mi = per_level_mi + mi
            batch_mi.append(per_level_mi)
            per_level_stride = torch.zeros([B, H, W, 1],
                                           dtype=torch.float32,
                                           device=device)
            per_level_stride = per_level_stride + stride
            batch_stride.append(per_level_stride)

        cls_preds,reg_preds,center_preds,controllers_preds,all_points_position,all_points_mi,all_points_stride=[],[],[],[],[],[],[]
        for cls_pred, reg_pred, center_pred, controllers_pred, per_level_position, per_level_mi, per_level_stride in zip(
                cls_heads, reg_heads, center_heads, controllers_heads,
                batch_positions, batch_mi, batch_stride):
            cls_pred = cls_pred.view(cls_pred.shape[0], -1, cls_pred.shape[-1])
            reg_pred = reg_pred.view(reg_pred.shape[0], -1, reg_pred.shape[-1])
            center_pred = center_pred.view(center_pred.shape[0], -1,
                                           center_pred.shape[-1])
            controllers_pred = controllers_pred.view(
                controllers_pred.shape[0], -1, controllers_pred.shape[-1])
            per_level_position = per_level_position.view(
                per_level_position.shape[0], -1, per_level_position.shape[-1])
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1,
                                             per_level_mi.shape[-1])
            per_level_stride = per_level_stride.view(
                per_level_stride.shape[0], -1, per_level_stride.shape[-1])

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            center_preds.append(center_pred)
            controllers_preds.append(controllers_pred)
            all_points_position.append(per_level_position)
            all_points_mi.append(per_level_mi)
            all_points_stride.append(per_level_stride)

        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)
        center_preds = torch.cat(center_preds, dim=1)
        controllers_preds = torch.cat(controllers_preds, dim=1)
        all_points_position = torch.cat(all_points_position, dim=1)
        all_points_mi = torch.cat(all_points_mi, dim=1)
        all_points_stride = torch.cat(all_points_stride, dim=1)

        del cls_heads, reg_heads, center_heads, controllers_heads, batch_positions, batch_mi, batch_stride

        batch_detection_targets,batch_mask_targets, batch_positive_idxs = [], [],[]
        for per_image_position, per_image_mi, per_image_stride, per_image_annot_boxes, per_image_annot_masks, per_image_annot_classes in zip(
                all_points_position, all_points_mi, all_points_stride,
                annot_boxes, annot_masks, annot_classes):
            per_image_annot_boxes = per_image_annot_boxes[
                per_image_annot_classes[:, 0] >= 0]
            per_image_annot_masks = per_image_annot_masks[
                per_image_annot_classes[:, 0] >= 0]
            per_image_annot_classes = per_image_annot_classes[
                per_image_annot_classes[:, 0] >= 0]

            valid_masks_flag = per_image_annot_masks.sum(dim=-1).sum(
                dim=-1) > 0
            per_image_annot_boxes = per_image_annot_boxes[valid_masks_flag, :]
            per_image_annot_masks = per_image_annot_masks[valid_masks_flag, :]
            per_image_annot_classes = per_image_annot_classes[
                valid_masks_flag, :]

            points_num = per_image_position.shape[0]

            if per_image_annot_classes.shape[0] == 0:
                # 6:l,t,r,b,center-ness_gt,class_index
                per_image_detection_targets = torch.zeros([points_num, 6],
                                                          dtype=torch.float32,
                                                          device=device)
                per_image_positive_idxs = []
                per_image_mask_targets = torch.zeros([0, mask_h, mask_w],
                                                     dtype=torch.float32,
                                                     device=device)
            else:
                annotaion_num = per_image_annot_classes.shape[0]
                candidates = torch.zeros([points_num, annotaion_num, 4],
                                         dtype=torch.float32,
                                         device=device)
                candidates = candidates + per_image_annot_boxes.unsqueeze(0)

                per_image_position = per_image_position.unsqueeze(1).repeat(
                    1, annotaion_num, 1)

                if use_center_sample:
                    candidates_center = (candidates[:, :, 2:4] +
                                         candidates[:, :, 0:2]) / 2
                    judge_distance = per_image_stride * self.center_sample_radius
                    judge_distance = judge_distance.repeat(1, annotaion_num)

                candidates[:, :,
                           0:2] = per_image_position[:, :,
                                                     0:2] - candidates[:, :,
                                                                       0:2]
                candidates[:, :,
                           2:4] = candidates[:, :,
                                             2:4] - per_image_position[:, :,
                                                                       0:2]

                candidates_min_value, _ = candidates.min(axis=-1, keepdim=True)
                sample_flag = (candidates_min_value[:, :, 0] >
                               0).int().unsqueeze(-1)
                # get all negative reg targets which points ctr out of gt box
                candidates = candidates * sample_flag

                # if use center sample get all negative reg targets which points not in center circle
                if use_center_sample:
                    compute_distance = torch.sqrt(
                        (per_image_position[:, :, 0] -
                         candidates_center[:, :, 0])**2 +
                        (per_image_position[:, :, 1] -
                         candidates_center[:, :, 1])**2)
                    center_sample_flag = (compute_distance <
                                          judge_distance).int().unsqueeze(-1)
                    candidates = candidates * center_sample_flag

                # get all negative reg targets which assign ground turth not in range of mi
                candidates_max_value, _ = candidates.max(axis=-1, keepdim=True)
                per_image_mi = per_image_mi.unsqueeze(1).repeat(
                    1, annotaion_num, 1)
                m1_negative_flag = (candidates_max_value[:, :, 0] >
                                    per_image_mi[:, :, 0]).int().unsqueeze(-1)
                candidates = candidates * m1_negative_flag
                m2_negative_flag = (candidates_max_value[:, :, 0] <
                                    per_image_mi[:, :, 1]).int().unsqueeze(-1)
                candidates = candidates * m2_negative_flag

                final_sample_flag = candidates.sum(axis=-1).sum(axis=-1)
                final_sample_flag = final_sample_flag > 0
                positive_index = (final_sample_flag == True).nonzero(
                    as_tuple=False).squeeze(dim=-1)

                # if no assign positive sample
                if len(positive_index) == 0:
                    del candidates
                    # 6:l,t,r,b,center-ness_gt,class_index
                    per_image_detection_targets = torch.zeros(
                        [points_num, 6], dtype=torch.float32, device=device)
                    per_image_positive_idxs = []
                    per_image_mask_targets = torch.zeros([0, mask_h, mask_w],
                                                         dtype=torch.float32,
                                                         device=device)
                else:
                    positive_candidates = candidates[positive_index]
                    per_image_positive_idxs = positive_index.cpu().numpy(
                    ).tolist()

                    del candidates

                    sample_box_gts = per_image_annot_boxes.unsqueeze(0)
                    sample_box_gts = sample_box_gts.repeat(
                        positive_candidates.shape[0], 1, 1)
                    sample_mask_gts = per_image_annot_masks.unsqueeze(0)
                    sample_mask_gts = sample_mask_gts.repeat(
                        positive_candidates.shape[0], 1, 1, 1)
                    sample_class_gts = per_image_annot_classes[:, 0].unsqueeze(
                        -1).unsqueeze(0)
                    sample_class_gts = sample_class_gts.repeat(
                        positive_candidates.shape[0], 1, 1)

                    # 6:l,t,r,b,center-ness_gt,class_index
                    per_image_detection_targets = torch.zeros(
                        [points_num, 6], dtype=torch.float32, device=device)

                    if positive_candidates.shape[1] == 1:
                        # if only one candidate for each positive sample
                        # assign l,t,r,b,center_ness_gt,class_index ground truth
                        # class_index value from 1 to 80 represent 80 positive classes
                        # class_index value 0 represenet negative class
                        positive_candidates = positive_candidates.squeeze(1)
                        sample_mask_gts = sample_mask_gts.squeeze(1)
                        sample_class_gts = sample_class_gts.squeeze(1)
                        per_image_detection_targets[positive_index,
                                                    0:4] = positive_candidates
                        l, t, r, b = per_image_detection_targets[
                            positive_index, 0:1], per_image_detection_targets[
                                positive_index,
                                1:2], per_image_detection_targets[
                                    positive_index,
                                    2:3], per_image_detection_targets[
                                        positive_index, 3:4]
                        per_image_detection_targets[
                            positive_index, 4:5] = torch.sqrt(
                                (torch.min(l, r) / torch.max(l, r)) *
                                (torch.min(t, b) / torch.max(t, b)))
                        per_image_detection_targets[positive_index,
                                                    5:6] = sample_class_gts + 1
                        per_image_mask_targets = sample_mask_gts
                    else:
                        # if a positive point sample have serveral object candidates,then choose the smallest area object candidate as the ground turth for this positive point sample
                        gts_w_h = sample_box_gts[:, :,
                                                 2:4] - sample_box_gts[:, :,
                                                                       0:2]
                        gts_area = gts_w_h[:, :, 0] * gts_w_h[:, :, 1]
                        positive_candidates_value = positive_candidates.sum(
                            axis=2)

                        # make sure all negative candidates areas==100000000,thus .min() operation wouldn't choose negative candidates
                        INF = 100000000
                        inf_tensor = torch.ones_like(gts_area) * INF
                        gts_area = torch.where(
                            torch.eq(positive_candidates_value, 0.),
                            inf_tensor, gts_area)

                        # get the smallest object candidate index
                        _, min_index = gts_area.min(axis=1)
                        candidate_indexes = (
                            torch.linspace(1, positive_candidates.shape[0],
                                           positive_candidates.shape[0]) -
                            1).long()
                        final_candidate_reg_gts = positive_candidates[
                            candidate_indexes, min_index, :]
                        final_candidate_mask_gts = sample_mask_gts[
                            candidate_indexes, min_index]
                        final_candidate_cls_gts = sample_class_gts[
                            candidate_indexes, min_index]

                        # assign l,t,r,b,center_ness_gt,class_index ground truth
                        per_image_detection_targets[
                            positive_index, 0:4] = final_candidate_reg_gts
                        l, t, r, b = per_image_detection_targets[
                            positive_index, 0:1], per_image_detection_targets[
                                positive_index,
                                1:2], per_image_detection_targets[
                                    positive_index,
                                    2:3], per_image_detection_targets[
                                        positive_index, 3:4]
                        per_image_detection_targets[
                            positive_index, 4:5] = torch.sqrt(
                                (torch.min(l, r) / torch.max(l, r)) *
                                (torch.min(t, b) / torch.max(t, b)))
                        per_image_detection_targets[
                            positive_index, 5:6] = final_candidate_cls_gts + 1
                        per_image_mask_targets = final_candidate_mask_gts

            per_image_detection_targets = per_image_detection_targets.unsqueeze(
                0)

            batch_detection_targets.append(per_image_detection_targets)
            batch_mask_targets.append(per_image_mask_targets)
            batch_positive_idxs.append(per_image_positive_idxs)

        del all_points_mi, all_points_stride, annot_boxes, annot_masks, annot_classes

        batch_detection_targets = torch.cat(batch_detection_targets, dim=0)
        batch_detection_targets = torch.cat(
            [batch_detection_targets, all_points_position], dim=2)

        # batch_detection_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        return cls_preds, reg_preds, center_preds, controllers_preds, batch_detection_targets, batch_mask_targets, batch_positive_idxs


if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    import random
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from simpleAICV.segmentation.models.solov2 import SOLOV2
    net = SOLOV2(resnet_type='resnet50')
    image_h, image_w = 480, 640
    cate_outs, kernel_outs, mask_out = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    import pickle
    with open('sample.pkl', 'rb') as file:
        annotations = pickle.load(file)

    annotations['box'] = torch.tensor(annotations['box'], dtype=torch.float32)
    annotations['mask'] = torch.tensor(annotations['mask'],
                                       dtype=torch.float32)
    annotations['class'] = torch.tensor(annotations['class'],
                                        dtype=torch.float32)
    loss = Solov2Loss()
    loss_dict = loss(annotations, cate_outs, kernel_outs, mask_out)
    print('1111', loss_dict)

    from simpleAICV.segmentation.models.condinst import CondInst
    net = CondInst(resnet_type='resnet50')
    image_h, image_w = 480, 640
    cls_heads, reg_heads, center_heads, controllers_heads, mask_out, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    import pickle
    with open('sample.pkl', 'rb') as file:
        annotations = pickle.load(file)

    annotations['box'] = torch.tensor(annotations['box'], dtype=torch.float32)
    annotations['mask'] = torch.tensor(annotations['mask'],
                                       dtype=torch.float32)
    annotations['class'] = torch.tensor(annotations['class'],
                                        dtype=torch.float32)
    loss = CondInstLoss()
    loss_dict = loss(annotations, cls_heads, reg_heads, center_heads,
                     controllers_heads, mask_out, batch_positions)
    print('2222', loss_dict)