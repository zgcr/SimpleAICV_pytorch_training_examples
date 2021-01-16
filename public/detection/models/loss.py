import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaLoss(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 epsilon=1e-4):
        super(RetinaLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.image_w = image_w
        self.image_h = image_h

    def forward(self, cls_heads, reg_heads, batch_anchors, annotations):
        """
        compute cls loss and reg loss in one batch
        """
        device = annotations.device
        cls_heads = torch.cat(cls_heads, axis=1)
        reg_heads = torch.cat(reg_heads, axis=1)
        batch_anchors = torch.cat(batch_anchors, axis=1)

        cls_heads, reg_heads, batch_anchors = self.drop_out_border_anchors_and_heads(
            cls_heads, reg_heads, batch_anchors, self.image_w, self.image_h)
        batch_anchors_annotations = self.get_batch_anchors_annotations(
            batch_anchors, annotations)

        reg_heads = reg_heads.type_as(cls_heads)
        batch_anchors = batch_anchors.type_as(cls_heads)
        batch_anchors_annotations = batch_anchors_annotations.type_as(
            cls_heads)

        cls_heads = cls_heads.view(-1, cls_heads.shape[-1])
        reg_heads = reg_heads.view(-1, reg_heads.shape[-1])
        batch_anchors_annotations = batch_anchors_annotations.view(
            -1, batch_anchors_annotations.shape[-1])

        positive_anchors_num = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0].shape[0]

        if positive_anchors_num > 0:
            cls_loss = self.compute_batch_focal_loss(
                cls_heads, batch_anchors_annotations)
            reg_loss = self.compute_batch_smoothl1_loss(
                reg_heads, batch_anchors_annotations)
        else:
            cls_loss = torch.tensor(0.).to(device)
            reg_loss = torch.tensor(0.).to(device)

        return cls_loss, reg_loss

    def compute_batch_focal_loss(self, cls_heads, batch_anchors_annotations):
        """
        compute batch focal loss(cls loss)
        cls_heads:[batch_size*anchor_num,num_classes]
        batch_anchors_annotations:[batch_size*anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        device = cls_heads.device
        cls_heads = cls_heads[batch_anchors_annotations[:, 4] >= 0]
        batch_anchors_annotations = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] >= 0]
        positive_anchors_num = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0].shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        cls_heads = torch.clamp(cls_heads,
                                min=self.epsilon,
                                max=1. - self.epsilon)
        num_classes = cls_heads.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_anchors_annotations[:, 4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(cls_heads) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), cls_heads,
                         1. - cls_heads)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        batch_bce_loss = -(
            loss_ground_truth * torch.log(cls_heads) +
            (1. - loss_ground_truth) * torch.log(1. - cls_heads))

        batch_focal_loss = focal_weight * batch_bce_loss
        batch_focal_loss = batch_focal_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        batch_focal_loss = batch_focal_loss / positive_anchors_num

        return batch_focal_loss

    def compute_batch_smoothl1_loss(self, reg_heads,
                                    batch_anchors_annotations):
        """
        compute batch smoothl1 loss(reg loss)
        per_image_reg_heads:[batch_size*anchor_num,4]
        per_image_anchors_annotations:[batch_size*anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate smoothl1 loss
        device = reg_heads.device
        reg_heads = reg_heads[batch_anchors_annotations[:, 4] > 0]
        batch_anchors_annotations = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0]
        positive_anchor_num = batch_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        # compute smoothl1 loss
        loss_ground_truth = batch_anchors_annotations[:, 0:4]

        x = torch.abs(reg_heads - loss_ground_truth)
        batch_smoothl1_loss = torch.where(torch.ge(x, self.beta),
                                          x - 0.5 * self.beta,
                                          0.5 * (x**2) / self.beta)
        batch_smoothl1_loss = batch_smoothl1_loss.mean(axis=1).sum()
        # according to the original paper,We divide the smoothl1 loss by the number of positive sample anchors
        batch_smoothl1_loss = batch_smoothl1_loss / positive_anchor_num

        return batch_smoothl1_loss

    def drop_out_border_anchors_and_heads(self, cls_heads, reg_heads,
                                          batch_anchors, image_w, image_h):
        """
        dropout out of border anchors,cls heads and reg heads
        """
        final_cls_heads, final_reg_heads, final_batch_anchors = [], [], []
        for per_image_cls_head, per_image_reg_head, per_image_anchors in zip(
                cls_heads, reg_heads, batch_anchors):
            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                                      0] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                                      0] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                                    0] > 0.0]

            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                                      1] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                                      1] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                                    1] > 0.0]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 2] < image_w]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 2] < image_w]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 2] < image_w]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 3] < image_h]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 3] < image_h]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 3] < image_h]

            per_image_cls_head = per_image_cls_head.unsqueeze(0)
            per_image_reg_head = per_image_reg_head.unsqueeze(0)
            per_image_anchors = per_image_anchors.unsqueeze(0)

            final_cls_heads.append(per_image_cls_head)
            final_reg_heads.append(per_image_reg_head)
            final_batch_anchors.append(per_image_anchors)

        final_cls_heads = torch.cat(final_cls_heads, axis=0)
        final_reg_heads = torch.cat(final_reg_heads, axis=0)
        final_batch_anchors = torch.cat(final_batch_anchors, axis=0)

        # final cls heads shape:[batch_size, anchor_nums, class_num]
        # final reg heads shape:[batch_size, anchor_nums, 4]
        # final batch anchors shape:[batch_size, anchor_nums, 4]
        return final_cls_heads, final_reg_heads, final_batch_anchors

    def get_batch_anchors_annotations(self, batch_anchors, annotations):
        """
        Assign a ground truth box target and a ground truth class target for each anchor
        if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
        if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
        if anchor gt_class index > 0,this anchor is a object class anchor and used in
        calculate cls loss and reg loss
        """
        device = annotations.device
        assert batch_anchors.shape[0] == annotations.shape[0]
        one_image_anchor_nums = batch_anchors.shape[1]

        batch_anchors_annotations = []
        for one_image_anchors, one_image_annotations in zip(
                batch_anchors, annotations):
            # drop all index=-1 class annotations
            one_image_annotations = one_image_annotations[
                one_image_annotations[:, 4] >= 0]

            if one_image_annotations.shape[0] == 0:
                one_image_anchor_annotations = torch.ones(
                    [one_image_anchor_nums, 5], device=device) * (-1)
            else:
                one_image_gt_bboxes = one_image_annotations[:, 0:4]
                one_image_gt_class = one_image_annotations[:, 4]
                one_image_ious = self.compute_ious_for_one_image(
                    one_image_anchors, one_image_gt_bboxes)

                # snap per gt bboxes to the best iou anchor
                overlap, indices = one_image_ious.max(axis=1)
                # assgin each anchor gt bboxes for max iou annotation
                per_image_anchors_gt_bboxes = one_image_gt_bboxes[indices]
                # transform gt bboxes to [tx,ty,tw,th] format for each anchor
                one_image_anchors_snaped_boxes = self.snap_annotations_as_tx_ty_tw_th(
                    per_image_anchors_gt_bboxes, one_image_anchors)

                one_image_anchors_gt_class = (torch.ones_like(overlap) *
                                              -1).to(device)
                # if iou <0.4,assign anchors gt class as 0:background
                one_image_anchors_gt_class[overlap < 0.4] = 0
                # if iou >=0.5,assign anchors gt class as same as the max iou annotation class:80 classes index from 1 to 80
                one_image_anchors_gt_class[
                    overlap >=
                    0.5] = one_image_gt_class[indices][overlap >= 0.5] + 1

                one_image_anchors_gt_class = one_image_anchors_gt_class.unsqueeze(
                    -1)

                one_image_anchor_annotations = torch.cat([
                    one_image_anchors_snaped_boxes, one_image_anchors_gt_class
                ],
                                                         axis=1)
            one_image_anchor_annotations = one_image_anchor_annotations.unsqueeze(
                0)
            batch_anchors_annotations.append(one_image_anchor_annotations)

        batch_anchors_annotations = torch.cat(batch_anchors_annotations,
                                              axis=0)

        # batch anchors annotations shape:[batch_size, anchor_nums, 5]
        return batch_anchors_annotations

    def snap_annotations_as_tx_ty_tw_th(self, anchors_gt_bboxes, anchors):
        """
        snap each anchor ground truth bbox form format:[x_min,y_min,x_max,y_max] to format:[tx,ty,tw,th]
        """
        anchors_w_h = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_w_h

        anchors_gt_bboxes_w_h = anchors_gt_bboxes[:,
                                                  2:] - anchors_gt_bboxes[:, :2]
        anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1.0)
        anchors_gt_bboxes_ctr = anchors_gt_bboxes[:, :
                                                  2] + 0.5 * anchors_gt_bboxes_w_h

        snaped_annotations_for_anchors = torch.cat(
            [(anchors_gt_bboxes_ctr - anchors_ctr) / anchors_w_h,
             torch.log(anchors_gt_bboxes_w_h / anchors_w_h)],
            axis=1)
        device = snaped_annotations_for_anchors.device
        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

        snaped_annotations_for_anchors = snaped_annotations_for_anchors / factor

        # snaped_annotations_for_anchors shape:[anchor_nums, 4]
        return snaped_annotations_for_anchors

    def compute_ious_for_one_image(self, one_image_anchors,
                                   one_image_annotations):
        """
        compute ious between one image anchors and one image annotations
        """
        # make sure anchors format:[anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        # make sure annotations format: [annotation_nums,4],4:[x_min,y_min,x_max,y_max]
        overlap_area_x_min = torch.max(
            one_image_anchors[:, 0].unsqueeze(-1),
            one_image_annotations[:, 0].unsqueeze(0))
        overlap_area_y_min = torch.max(
            one_image_anchors[:, 1].unsqueeze(-1),
            one_image_annotations[:, 1].unsqueeze(0))
        overlap_area_x_max = torch.min(
            one_image_anchors[:, 2].unsqueeze(-1),
            one_image_annotations[:, 2].unsqueeze(0))
        overlap_area_y_max = torch.min(
            one_image_anchors[:, 3].unsqueeze(-1),
            one_image_annotations[:, 3].unsqueeze(0))
        overlap_areas_w = torch.clamp(overlap_area_x_max - overlap_area_x_min,
                                      min=0)
        overlap_areas_h = torch.clamp(overlap_area_y_max - overlap_area_y_min,
                                      min=0)
        overlaps_area = overlap_areas_w * overlap_areas_h

        anchors_w_h = one_image_anchors[:, 2:] - one_image_anchors[:, :2]
        annotations_w_h = one_image_annotations[:,
                                                2:] - one_image_annotations[:, :
                                                                            2]
        anchors_w_h = torch.clamp(anchors_w_h, min=0)
        annotations_w_h = torch.clamp(annotations_w_h, min=0)
        anchors_area = anchors_w_h[:, 0] * anchors_w_h[:, 1]
        annotations_area = annotations_w_h[:, 0] * annotations_w_h[:, 1]

        unions_area = anchors_area.unsqueeze(-1) + annotations_area.unsqueeze(
            0) - overlaps_area
        unions_area = torch.clamp(unions_area, min=1e-4)
        # compute ious between one image anchors and one image annotations
        one_image_ious = (overlaps_area / unions_area)

        return one_image_ious


INF = 100000000


class FCOSLoss(nn.Module):
    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 mi=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]],
                 alpha=0.25,
                 gamma=2.,
                 reg_weight=2.,
                 epsilon=1e-4,
                 center_sample_radius=1.5,
                 use_center_sample=True):
        super(FCOSLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.epsilon = epsilon
        self.strides = strides
        self.mi = mi
        self.use_center_sample = use_center_sample
        self.center_sample_radius = center_sample_radius

    def forward(self, cls_heads, reg_heads, center_heads, batch_positions,
                annotations):
        """
        compute cls loss, reg loss and center-ness loss in one batch
        """
        device = annotations.device
        cls_preds, reg_preds, center_preds, batch_targets = self.get_batch_position_annotations(
            cls_heads, reg_heads, center_heads, batch_positions, annotations)

        cls_preds = torch.sigmoid(cls_preds)
        reg_preds = torch.exp(reg_preds)
        center_preds = torch.sigmoid(center_preds)

        reg_preds = reg_preds.type_as(cls_preds)
        center_preds = center_preds.type_as(cls_preds)
        batch_targets = batch_targets.type_as(cls_preds)

        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
        reg_preds = reg_preds.view(-1, reg_preds.shape[-1])
        center_preds = center_preds.view(-1, center_preds.shape[-1])
        batch_targets = batch_targets.view(-1, batch_targets.shape[-1])

        positive_points_num = batch_targets[batch_targets[:, 4] > 0].shape[0]

        if positive_points_num > 0:
            cls_loss = self.compute_batch_focal_loss(cls_preds, batch_targets)
            reg_loss = self.compute_batch_giou_loss(reg_preds, batch_targets)
            center_ness_loss = self.compute_batch_centerness_loss(
                center_preds, batch_targets)
        else:
            cls_loss = torch.tensor(0.).to(device)
            reg_loss = torch.tensor(0.).to(device)
            center_ness_loss = torch.tensor(0.).to(device)

        return cls_loss, reg_loss, center_ness_loss

    def compute_batch_focal_loss(self, cls_preds, batch_targets):
        """
        compute batch focal loss(cls loss)
        cls_preds:[batch_size*points_num,num_classes]
        batch_targets:[batch_size*points_num,8]
        """
        device = cls_preds.device
        cls_preds = torch.clamp(cls_preds,
                                min=self.epsilon,
                                max=1. - self.epsilon)
        positive_points_num = batch_targets[batch_targets[:, 4] > 0].shape[0]
        num_classes = cls_preds.shape[1]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_targets[:, 4].long(),
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

    def compute_batch_giou_loss(self, reg_preds, batch_targets):
        """
        compute batch giou loss(reg loss)
        reg_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        """
        # only use positive points sample to compute reg loss
        device = reg_preds.device
        reg_preds = reg_preds[batch_targets[:, 4] > 0]
        batch_targets = batch_targets[batch_targets[:, 4] > 0]
        positive_points_num = batch_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        center_ness_targets = batch_targets[:, 5]

        pred_bboxes_xy_min = batch_targets[:, 6:8] - reg_preds[:, 0:2]
        pred_bboxes_xy_max = batch_targets[:, 6:8] + reg_preds[:, 2:4]
        gt_bboxes_xy_min = batch_targets[:, 6:8] - batch_targets[:, 0:2]
        gt_bboxes_xy_max = batch_targets[:, 6:8] + batch_targets[:, 2:4]

        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                axis=1)
        gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max], axis=1)

        overlap_area_top_left = torch.max(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        overlap_area_bot_right = torch.min(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                         overlap_area_top_left,
                                         min=0)
        overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

        # anchors and annotations convert format to [x1,y1,w,h]
        pred_bboxes_w_h = pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1
        gt_bboxes_w_h = gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2] + 1

        # compute anchors_area and annotations_area
        pred_bboxes_area = pred_bboxes_w_h[:, 0] * pred_bboxes_w_h[:, 1]
        gt_bboxes_area = gt_bboxes_w_h[:, 0] * gt_bboxes_w_h[:, 1]

        # compute union_area
        union_area = pred_bboxes_area + gt_bboxes_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        # compute ious between one image anchors and one image annotations
        ious = overlap_area / union_area

        enclose_area_top_left = torch.min(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        enclose_area_bot_right = torch.max(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                         enclose_area_top_left,
                                         min=0)
        enclose_area = enclose_area_sizes[:, 0] * enclose_area_sizes[:, 1]
        enclose_area = torch.clamp(enclose_area, min=1e-4)

        gious_loss = 1. - ious + (enclose_area - union_area) / enclose_area
        gious_loss = torch.clamp(gious_loss, min=0.0, max=2.0)
        # use center_ness_targets as the weight of gious loss
        gious_loss = gious_loss * center_ness_targets
        gious_loss = gious_loss.sum() / positive_points_num
        gious_loss = self.reg_weight * gious_loss

        return gious_loss

    def compute_batch_centerness_loss(self, center_preds, batch_targets):
        """
        compute batch center_ness loss(center ness loss)
        center_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        """
        # only use positive points sample to compute center_ness loss
        device = center_preds.device
        center_preds = center_preds[batch_targets[:, 4] > 0]
        batch_targets = batch_targets[batch_targets[:, 4] > 0]
        positive_points_num = batch_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        center_ness_targets = batch_targets[:, 5:6]

        center_ness_loss = -(
            center_ness_targets * torch.log(center_preds) +
            (1. - center_ness_targets) * torch.log(1. - center_preds))
        center_ness_loss = center_ness_loss.sum() / positive_points_num

        return center_ness_loss

    def get_batch_position_annotations(self, cls_heads, reg_heads,
                                       center_heads, batch_positions,
                                       annotations):
        """
        Assign a ground truth target for each position on feature map
        """
        device = annotations.device
        batch_mi, batch_stride = [], []
        for reg_head, mi, stride in zip(reg_heads, self.mi, self.strides):
            mi = torch.tensor(mi).to(device)
            B, H, W, _ = reg_head.shape
            per_level_mi = torch.zeros(B, H, W, 2).to(device)
            per_level_mi = per_level_mi + mi
            batch_mi.append(per_level_mi)
            per_level_stride = torch.zeros(B, H, W, 1).to(device)
            per_level_stride = per_level_stride + stride
            batch_stride.append(per_level_stride)

        cls_preds,reg_preds,center_preds,all_points_position,all_points_mi,all_points_stride=[],[],[],[],[],[]
        for cls_pred, reg_pred, center_pred, per_level_position, per_level_mi, per_level_stride in zip(
                cls_heads, reg_heads, center_heads, batch_positions, batch_mi,
                batch_stride):
            cls_pred = cls_pred.view(cls_pred.shape[0], -1, cls_pred.shape[-1])
            reg_pred = reg_pred.view(reg_pred.shape[0], -1, reg_pred.shape[-1])
            center_pred = center_pred.view(center_pred.shape[0], -1,
                                           center_pred.shape[-1])
            per_level_position = per_level_position.view(
                per_level_position.shape[0], -1, per_level_position.shape[-1])
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1,
                                             per_level_mi.shape[-1])
            per_level_stride = per_level_stride.view(
                per_level_stride.shape[0], -1, per_level_stride.shape[-1])

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            center_preds.append(center_pred)
            all_points_position.append(per_level_position)
            all_points_mi.append(per_level_mi)
            all_points_stride.append(per_level_stride)

        cls_preds = torch.cat(cls_preds, axis=1)
        reg_preds = torch.cat(reg_preds, axis=1)
        center_preds = torch.cat(center_preds, axis=1)
        all_points_position = torch.cat(all_points_position, axis=1)
        all_points_mi = torch.cat(all_points_mi, axis=1)
        all_points_stride = torch.cat(all_points_stride, axis=1)

        batch_targets = []
        for per_image_position, per_image_mi, per_image_stride, per_image_annotations in zip(
                all_points_position, all_points_mi, all_points_stride,
                annotations):
            per_image_annotations = per_image_annotations[
                per_image_annotations[:, 4] >= 0]
            points_num = per_image_position.shape[0]

            if per_image_annotations.shape[0] == 0:
                # 6:l,t,r,b,class_index,center-ness_gt
                per_image_targets = torch.zeros([points_num, 6], device=device)
            else:
                annotaion_num = per_image_annotations.shape[0]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                candidates = torch.zeros([points_num, annotaion_num, 4],
                                         device=device)
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)

                per_image_position = per_image_position.unsqueeze(1).repeat(
                    1, annotaion_num, 1)

                if self.use_center_sample:
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
                if self.use_center_sample:
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
                positive_index = (final_sample_flag == True).nonzero().squeeze(
                    dim=-1)

                # if no assign positive sample
                if len(positive_index) == 0:
                    del candidates
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
                                                    device=device)
                else:
                    positive_candidates = candidates[positive_index]

                    del candidates

                    sample_box_gts = per_image_annotations[:, 0:4].unsqueeze(0)
                    sample_box_gts = sample_box_gts.repeat(
                        positive_candidates.shape[0], 1, 1)
                    sample_class_gts = per_image_annotations[:, 4].unsqueeze(
                        -1).unsqueeze(0)
                    sample_class_gts = sample_class_gts.repeat(
                        positive_candidates.shape[0], 1, 1)

                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
                                                    device=device)

                    if positive_candidates.shape[1] == 1:
                        # if only one candidate for each positive sample
                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        # class_index value from 1 to 80 represent 80 positive classes
                        # class_index value 0 represenet negative class
                        positive_candidates = positive_candidates.squeeze(1)
                        sample_class_gts = sample_class_gts.squeeze(1)
                        per_image_targets[positive_index,
                                          0:4] = positive_candidates
                        per_image_targets[positive_index,
                                          4:5] = sample_class_gts + 1

                        l, t, r, b = per_image_targets[
                            positive_index, 0:1], per_image_targets[
                                positive_index, 1:2], per_image_targets[
                                    positive_index,
                                    2:3], per_image_targets[positive_index,
                                                            3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))
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
                        final_candidate_cls_gts = sample_class_gts[
                            candidate_indexes, min_index]

                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        per_image_targets[positive_index,
                                          0:4] = final_candidate_reg_gts
                        per_image_targets[positive_index,
                                          4:5] = final_candidate_cls_gts + 1

                        l, t, r, b = per_image_targets[
                            positive_index, 0:1], per_image_targets[
                                positive_index, 1:2], per_image_targets[
                                    positive_index,
                                    2:3], per_image_targets[positive_index,
                                                            3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))

            per_image_targets = per_image_targets.unsqueeze(0)
            batch_targets.append(per_image_targets)

        batch_targets = torch.cat(batch_targets, axis=0)
        batch_targets = torch.cat([batch_targets, all_points_position], axis=2)

        # batch_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        return cls_preds, reg_preds, center_preds, batch_targets


class CenterNetLoss(nn.Module):
    def __init__(self,
                 alpha=2.,
                 beta=4.,
                 wh_weight=0.1,
                 epsilon=1e-4,
                 min_overlap=0.7,
                 max_object_num=100):
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.wh_weight = wh_weight
        self.epsilon = epsilon
        self.min_overlap = min_overlap
        self.max_object_num = max_object_num

    def forward(self, heatmap_heads, offset_heads, wh_heads, annotations):
        """
        compute heatmap loss, offset loss and wh loss in one batch
        """
        batch_heatmap_targets, batch_wh_targets, batch_offset_targets, batch_reg_to_heatmap_index, batch_positive_targets_mask = self.get_batch_targets(
            heatmap_heads, annotations)

        heatmap_heads = torch.sigmoid(heatmap_heads)
        B, num_classes = heatmap_heads.shape[0], heatmap_heads.shape[1]
        heatmap_heads = heatmap_heads.permute(0, 2, 3, 1).contiguous().view(
            B, -1, num_classes)
        batch_heatmap_targets = batch_heatmap_targets.permute(
            0, 2, 3, 1).contiguous().view(B, -1, num_classes)

        wh_heads = wh_heads.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        offset_heads = offset_heads.permute(0, 2, 3,
                                            1).contiguous().view(B, -1, 2)

        wh_heads = wh_heads.type_as(heatmap_heads)
        offset_heads = offset_heads.type_as(heatmap_heads)
        batch_heatmap_targets = batch_heatmap_targets.type_as(heatmap_heads)
        batch_wh_targets = batch_wh_targets.type_as(heatmap_heads)
        batch_offset_targets = batch_offset_targets.type_as(heatmap_heads)
        batch_reg_to_heatmap_index = batch_reg_to_heatmap_index.type_as(
            heatmap_heads)
        batch_positive_targets_mask = batch_positive_targets_mask.type_as(
            heatmap_heads)

        heatmap_loss, offset_loss, wh_loss = [], [], []
        valid_image_num = 0
        device = annotations.device
        for per_heatmap_heads, per_wh_heads, per_offset_heads, per_heatmap_targets, per_wh_targets, per_offset_targets, per_reg_to_heatmap_index, per_positive_targets_mask in zip(
                heatmap_heads, wh_heads, offset_heads, batch_heatmap_targets,
                batch_wh_targets, batch_offset_targets,
                batch_reg_to_heatmap_index, batch_positive_targets_mask):
            # if no centers on heatmap_targets,this image is not valid
            valid_center_num = (
                per_heatmap_targets[per_heatmap_targets == 1.]).shape[0]

            if valid_center_num == 0:
                heatmap_loss.append(torch.tensor(0.).to(device))
                offset_loss.append(torch.tensor(0.).to(device))
                wh_loss.append(torch.tensor(0.).to(device))
            else:
                valid_image_num += 1
                one_image_focal_loss = self.compute_one_image_focal_loss(
                    per_heatmap_heads, per_heatmap_targets)
                one_image_offsetl1_loss = self.compute_one_image_offsetl1_loss(
                    per_offset_heads, per_offset_targets,
                    per_reg_to_heatmap_index, per_positive_targets_mask)
                one_image_whl1_loss = self.compute_one_image_whl1_loss(
                    per_wh_heads, per_wh_targets, per_reg_to_heatmap_index,
                    per_positive_targets_mask)

                heatmap_loss.append(one_image_focal_loss)
                offset_loss.append(one_image_offsetl1_loss)
                wh_loss.append(one_image_whl1_loss)

        if valid_image_num == 0:
            heatmap_loss = sum(heatmap_loss)
            offset_loss = sum(offset_loss)
            wh_loss = sum(wh_loss)
        else:
            heatmap_loss = sum(heatmap_loss) / valid_image_num
            offset_loss = sum(offset_loss) / valid_image_num
            wh_loss = sum(wh_loss) / valid_image_num

        return heatmap_loss, offset_loss, wh_loss

    def compute_one_image_focal_loss(self, per_image_heatmap_heads,
                                     per_image_heatmap_targets):
        device = per_image_heatmap_heads.device
        per_image_heatmap_heads = torch.clamp(per_image_heatmap_heads,
                                              min=self.epsilon,
                                              max=1. - self.epsilon)
        valid_center_num = (per_image_heatmap_targets[per_image_heatmap_targets
                                                      == 1.]).shape[0]
        if valid_center_num == 0:
            return torch.tensor(0.).to(device)

        # all center points
        positive_indexes = (per_image_heatmap_targets == 1.)
        # all non center points
        negative_indexes = (per_image_heatmap_targets < 1.)

        positive_loss = torch.log(per_image_heatmap_heads) * torch.pow(
            1 - per_image_heatmap_heads, self.alpha) * positive_indexes
        negative_loss = torch.log(1 - per_image_heatmap_heads) * torch.pow(
            per_image_heatmap_heads, self.alpha) * torch.pow(
                1 - per_image_heatmap_targets, self.beta) * negative_indexes

        loss = -(positive_loss.sum() + negative_loss.sum()) / valid_center_num

        return loss

    def compute_one_image_offsetl1_loss(self,
                                        per_image_offset_heads,
                                        per_image_offset_targets,
                                        per_image_reg_to_heatmap_index,
                                        per_image_positive_targets_mask,
                                        factor=1.0 / 9.0):
        device = per_image_offset_heads.device
        per_image_reg_to_heatmap_index = per_image_reg_to_heatmap_index.unsqueeze(
            -1).repeat(1, 2)
        per_image_offset_heads = torch.gather(
            per_image_offset_heads, 0, per_image_reg_to_heatmap_index.long())

        valid_object_num = (per_image_positive_targets_mask[
            per_image_positive_targets_mask == 1.]).shape[0]

        if valid_object_num == 0:
            return torch.tensor(0.).to(device)

        per_image_positive_targets_mask = per_image_positive_targets_mask.unsqueeze(
            -1).repeat(1, 2)
        per_image_offset_heads = per_image_offset_heads * per_image_positive_targets_mask
        per_image_offset_targets = per_image_offset_targets * per_image_positive_targets_mask

        x = torch.abs(per_image_offset_heads - per_image_offset_targets)
        loss = torch.where(torch.ge(x, factor), x - 0.5 * factor,
                           0.5 * (x**2) / factor)
        loss = loss.sum() / valid_object_num

        return loss

    def compute_one_image_whl1_loss(self,
                                    per_image_wh_heads,
                                    per_image_wh_targets,
                                    per_image_reg_to_heatmap_index,
                                    per_image_positive_targets_mask,
                                    factor=1.0 / 9.0):
        device = per_image_wh_heads.device
        per_image_reg_to_heatmap_index = per_image_reg_to_heatmap_index.unsqueeze(
            -1).repeat(1, 2)
        per_image_wh_heads = torch.gather(
            per_image_wh_heads, 0, per_image_reg_to_heatmap_index.long())

        valid_object_num = (per_image_positive_targets_mask[
            per_image_positive_targets_mask == 1.]).shape[0]

        if valid_object_num == 0:
            return torch.tensor(0.).to(device)

        per_image_positive_targets_mask = per_image_positive_targets_mask.unsqueeze(
            -1).repeat(1, 2)
        per_image_wh_heads = per_image_wh_heads * per_image_positive_targets_mask
        per_image_wh_targets = per_image_wh_targets * per_image_positive_targets_mask

        x = torch.abs(per_image_wh_heads - per_image_wh_targets)
        loss = torch.where(torch.ge(x, factor), x - 0.5 * factor,
                           0.5 * (x**2) / factor)
        loss = loss.sum() / valid_object_num
        loss = self.wh_weight * loss

        return loss

    def get_batch_targets(self, heatmap_heads, annotations):
        B, num_classes, H, W = heatmap_heads.shape[0], heatmap_heads.shape[
            1], heatmap_heads.shape[2], heatmap_heads.shape[3]
        device = annotations.device

        batch_heatmap_targets, batch_wh_targets, batch_offset_targets, batch_reg_to_heatmap_index, batch_positive_targets_mask=[],[],[],[],[]
        for per_image_annots in annotations:
            # limit max annots num for per image
            per_image_annots = per_image_annots[per_image_annots[:, 4] >= 0]
            # limit max object num
            num_objs = min(per_image_annots.shape[0], self.max_object_num)

            per_image_heatmap_targets = torch.zeros((num_classes, H, W),
                                                    device=device)
            per_image_wh_targets = torch.zeros((self.max_object_num, 2),
                                               device=device)
            per_image_offset_targets = torch.zeros((self.max_object_num, 2),
                                                   device=device)
            per_image_positive_targets_mask = torch.zeros(
                (self.max_object_num, ), device=device)
            per_image_reg_to_heatmap_index = torch.zeros(
                (self.max_object_num, ), device=device)
            gt_bboxes, gt_classes = per_image_annots[:,
                                                     0:4], per_image_annots[:,
                                                                            4]
            # gt_bboxes divided by 4 to get downsample bboxes
            gt_bboxes = gt_bboxes / 4
            # make sure all height and width >0
            all_h, all_w = gt_bboxes[:,
                                     3] - gt_bboxes[:,
                                                    1], gt_bboxes[:,
                                                                  2] - gt_bboxes[:,
                                                                                 0]

            per_image_wh_targets[0:num_objs, 0] = all_w
            per_image_wh_targets[0:num_objs, 1] = all_h

            centers = torch.cat(
                [((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2).unsqueeze(-1),
                 ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2).unsqueeze(-1)],
                axis=1)
            centers_int = torch.trunc(centers)
            centers_decimal = torch.frac(centers)

            per_image_offset_targets[0:num_objs, :] = centers_decimal
            per_image_positive_targets_mask[0:num_objs] = 1

            per_image_reg_to_heatmap_index[
                0:num_objs] = centers_int[:, 1] * W + centers_int[:, 0]

            all_radius = self.compute_objects_gaussian_radius((all_h, all_w))
            per_image_heatmap_targets = self.draw_umich_gaussian(
                per_image_heatmap_targets, gt_classes, centers_int, all_radius)

            batch_heatmap_targets.append(
                per_image_heatmap_targets.unsqueeze(0))
            batch_wh_targets.append(per_image_wh_targets.unsqueeze(0))
            batch_reg_to_heatmap_index.append(
                per_image_reg_to_heatmap_index.unsqueeze(0))
            batch_offset_targets.append(per_image_offset_targets.unsqueeze(0))
            batch_positive_targets_mask.append(
                per_image_positive_targets_mask.unsqueeze(0))

        batch_heatmap_targets = torch.cat(batch_heatmap_targets, axis=0)
        batch_wh_targets = torch.cat(batch_wh_targets, axis=0)
        batch_offset_targets = torch.cat(batch_offset_targets, axis=0)
        batch_reg_to_heatmap_index = torch.cat(batch_reg_to_heatmap_index,
                                               axis=0)
        batch_positive_targets_mask = torch.cat(batch_positive_targets_mask,
                                                axis=0)

        return batch_heatmap_targets, batch_wh_targets, batch_offset_targets, batch_reg_to_heatmap_index, batch_positive_targets_mask

    def compute_objects_gaussian_radius(self, objects_size):
        all_h, all_w = objects_size
        all_h, all_w = torch.ceil(all_h), torch.ceil(all_w)

        a1 = 1
        b1 = (all_h + all_w)
        c1 = all_w * all_h * (1 - self.min_overlap) / (1 + self.min_overlap)
        sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (all_h + all_w)
        c2 = (1 - self.min_overlap) * all_w * all_h
        sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * self.min_overlap
        b3 = -2 * self.min_overlap * (all_h + all_w)
        c3 = (self.min_overlap - 1) * all_w * all_h
        sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        radius = torch.min(r1, r2)
        radius = torch.min(radius, r3)
        radius = torch.max(torch.zeros_like(radius), torch.trunc(radius))

        return radius

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def draw_umich_gaussian(self,
                            per_image_heatmap_targets,
                            gt_classes,
                            all_centers,
                            all_radius,
                            k=1):
        height, width = per_image_heatmap_targets.shape[
            1], per_image_heatmap_targets.shape[2]
        device = per_image_heatmap_targets.device

        for per_class, per_center, per_radius in zip(gt_classes, all_centers,
                                                     all_radius):
            per_diameter = 2 * per_radius + 1
            per_diameter = int(per_diameter.item())
            gaussian = self.gaussian2D((per_diameter, per_diameter),
                                       sigma=per_diameter / 6)
            gaussian = torch.FloatTensor(gaussian).to(device)

            x, y = per_center[0], per_center[1]
            left, right = min(x, per_radius), min(width - x, per_radius + 1)
            top, bottom = min(y, per_radius), min(height - y, per_radius + 1)

            masked_heatmap = per_image_heatmap_targets[per_class.long(), (
                y - top).long():(y +
                                 bottom).long(), (x -
                                                  left).long():(x +
                                                                right).long()]
            masked_gaussian = gaussian[(per_radius -
                                        top).long():(per_radius +
                                                     bottom).long(),
                                       (per_radius -
                                        left).long():(per_radius +
                                                      right).long()]

            if min(masked_gaussian.shape) > 0 and min(
                    masked_heatmap.shape) > 0:
                # 如果高斯图重叠，重叠点取最大值
                masked_heatmap = torch.max(masked_heatmap, masked_gaussian * k)

            per_image_heatmap_targets[per_class.long(),
                                      (y - top).long():(y + bottom).long(),
                                      (x - left).long():(
                                          x + right).long()] = masked_heatmap

        return per_image_heatmap_targets


class YOLOV3Loss(nn.Module):
    def __init__(self,
                 anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]],
                 per_level_num_anchors=3,
                 strides=[8, 16, 32],
                 obj_weight=1.0,
                 noobj_weight=100.0,
                 epsilon=1e-4):
        super(YOLOV3Loss, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.per_level_num_anchors = per_level_num_anchors
        self.strides = strides
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.epsilon = epsilon

    def forward(self, obj_heads, reg_heads, cls_heads, batch_anchors,
                annotations):
        """
        compute obj loss, reg loss and cls loss in one batch
        """
        device = annotations.device
        batch_anchor_targets = self.get_batch_anchors_targets(
            batch_anchors, annotations)

        obj_noobj_loss = torch.tensor(0.).to(device)
        reg_loss = torch.tensor(0.).to(device)
        cls_loss = torch.tensor(0.).to(device)
        for per_level_obj_pred, per_level_reg_pred, per_level_cls_pred, per_level_anchors, per_level_anchor_targets in zip(
                obj_heads, reg_heads, cls_heads, batch_anchors,
                batch_anchor_targets):
            per_level_obj_pred = per_level_obj_pred.view(
                per_level_obj_pred.shape[0], -1, per_level_obj_pred.shape[-1])
            per_level_reg_pred = per_level_reg_pred.view(
                per_level_reg_pred.shape[0], -1, per_level_reg_pred.shape[-1])
            per_level_cls_pred = per_level_cls_pred.view(
                per_level_cls_pred.shape[0], -1, per_level_cls_pred.shape[-1])
            per_level_anchors = per_level_anchors.view(
                per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

            per_level_obj_pred = torch.sigmoid(per_level_obj_pred)
            per_level_cls_pred = torch.sigmoid(per_level_cls_pred)

            # # snap per_level_reg_pred from tx,ty,tw,th -> x_center,y_center,w,h -> x_min,y_min,x_max,y_max
            # per_level_reg_pred[:, :, 0:2] = (
            #     torch.sigmoid(per_level_reg_pred[:, :, 0:2]) +
            #     per_level_anchors[:, :, 0:2]) * per_level_anchors[:, :, 4:5]
            # # pred_bboxes_wh=exp(twh)*anchor_wh/stride
            # per_level_reg_pred[:, :, 2:4] = torch.exp(
            #     per_level_reg_pred[:, :, 2:4]
            # ) * per_level_anchors[:, :, 2:4] / per_level_anchors[:, :, 4:5]

            # per_level_reg_pred[:, :, 0:
            #                    2] = per_level_reg_pred[:, :, 0:
            #                                            2] - 0.5 * per_level_reg_pred[:, :,
            #                                                                          2:
            #                                                                          4]
            # per_level_reg_pred[:, :, 2:
            #                    4] = per_level_reg_pred[:, :, 2:
            #                                            4] + per_level_reg_pred[:, :,
            #                                                                    0:
            #                                                                    2]

            # per_level_anchor_targets[:, :, 0:2] = (
            #     per_level_anchor_targets[:, :, 0:2] +
            #     per_level_anchors[:, :, 0:2]) * per_level_anchors[:, :, 4:5]
            # per_level_anchor_targets[:, :, 2:4] = torch.exp(
            #     per_level_anchor_targets[:, :, 2:4]
            # ) * per_level_anchors[:, :, 2:4] / per_level_anchors[:, :, 4:5]
            # per_level_anchor_targets[:, :, 0:
            #                          2] = per_level_anchor_targets[:, :, 0:
            #                                                        2] - 0.5 * per_level_anchor_targets[:, :,
            #                                                                                            2:
            #                                                                                            4]
            # per_level_anchor_targets[:, :, 2:
            #                          4] = per_level_anchor_targets[:, :, 2:
            #                                                        4] + per_level_anchor_targets[:, :,
            #                                                                                      0:
            #                                                                                      2]

            per_level_reg_pred = per_level_reg_pred.type_as(per_level_obj_pred)
            per_level_cls_pred = per_level_cls_pred.type_as(per_level_obj_pred)
            per_level_anchors = per_level_anchors.type_as(per_level_obj_pred)
            per_level_anchor_targets = per_level_anchor_targets.type_as(
                per_level_obj_pred)

            per_level_obj_pred = per_level_obj_pred.view(
                -1, per_level_obj_pred.shape[-1])
            per_level_reg_pred = per_level_reg_pred.view(
                -1, per_level_reg_pred.shape[-1])
            per_level_cls_pred = per_level_cls_pred.view(
                -1, per_level_cls_pred.shape[-1])
            per_level_anchor_targets = per_level_anchor_targets.view(
                -1, per_level_anchor_targets.shape[-1])

            obj_noobj_loss = obj_noobj_loss + self.compute_per_level_batch_obj_noobj_loss(
                per_level_obj_pred, per_level_anchor_targets)
            reg_loss = reg_loss + self.compute_per_level_batch_reg_loss(
                per_level_reg_pred, per_level_anchor_targets)
            cls_loss = cls_loss + self.compute_per_level_batch_cls_loss(
                per_level_cls_pred, per_level_anchor_targets)

        return obj_noobj_loss, reg_loss, cls_loss

    def compute_per_level_batch_obj_noobj_loss(self, per_level_obj_pred,
                                               per_level_anchor_targets):
        """
        compute per level batch obj noobj loss(bce loss)
        per_level_obj_pred:[batch_size*per_level_anchor_num,1]
        per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
        """
        device = per_level_obj_pred.device
        positive_anchors_num = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0].shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        positive_obj_preds = per_level_obj_pred[
            per_level_anchor_targets[:, 5] > 0].view(-1)
        positive_obj_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0][:, 5:6].view(-1)

        negative_obj_preds = (
            1. -
            per_level_obj_pred[per_level_anchor_targets[:, 6] > 0]).view(-1)
        negative_obj_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 6] > 0][:, 6:7].view(-1)

        obj_loss = -(positive_obj_targets * torch.log(positive_obj_preds))
        noobj_loss = -(negative_obj_targets * torch.log(negative_obj_preds))

        obj_loss = obj_loss.mean()
        noobj_loss = noobj_loss.mean()
        total_loss = self.obj_weight * obj_loss + self.noobj_weight * noobj_loss

        return total_loss

    def compute_per_level_batch_reg_loss(self, per_level_reg_pred,
                                         per_level_anchor_targets):
        """
        compute per level batch reg loss(mse loss)
        per_level_reg_pred:[batch_size*per_level_anchor_num,4]
        per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
        """
        # only use positive anchor sample to compute reg loss
        device = per_level_reg_pred.device
        per_level_reg_pred = per_level_reg_pred[
            per_level_anchor_targets[:, 5] > 0]
        per_level_reg_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0][:, 0:4]

        positive_anchors_num = per_level_reg_targets.shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        reg_loss = (per_level_reg_pred - per_level_reg_targets)**2
        reg_loss = reg_loss.sum(axis=1)

        reg_loss = reg_loss.mean()

        return reg_loss

    # def compute_per_level_batch_reg_loss(self, per_level_reg_pred,
    #                                      per_level_anchor_targets):
    #     """
    #     compute per level batch reg loss(giou loss)
    #     per_level_reg_pred:[batch_size*per_level_anchor_num,4]
    #     per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
    #     """
    #     # only use positive anchor sample to compute reg loss
    #     device = per_level_reg_pred.device
    #     per_level_reg_pred = per_level_reg_pred[
    #         per_level_anchor_targets[:, 5] > 0]
    #     per_level_reg_targets = per_level_anchor_targets[
    #         per_level_anchor_targets[:, 5] > 0][:, 0:4]

    #     positive_anchors_num = per_level_reg_targets.shape[0]

    #     if positive_anchors_num == 0:
    #         return torch.tensor(0.).to(device)

    #     overlap_area_top_left = torch.max(per_level_reg_pred[:, 0:2],
    #                                       per_level_reg_targets[:, 0:2])
    #     overlap_area_bot_right = torch.min(per_level_reg_pred[:, 2:4],
    #                                        per_level_reg_targets[:, 2:4])
    #     overlap_area_sizes = torch.clamp(overlap_area_bot_right -
    #                                      overlap_area_top_left,
    #                                      min=0)
    #     overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

    #     # anchors and annotations convert format to [x1,y1,w,h]
    #     pred_bboxes_w_h = per_level_reg_pred[:, 2:4] - per_level_reg_pred[:,
    #                                                                       0:2]
    #     annotations_w_h = per_level_reg_targets[:, 2:
    #                                             4] - per_level_reg_targets[:,
    #                                                                        0:2]
    #     pred_bboxes_w_h = torch.clamp(pred_bboxes_w_h, min=0)
    #     annotations_w_h = torch.clamp(annotations_w_h, min=0)
    #     # compute anchors_area and annotations_area
    #     pred_bboxes_area = pred_bboxes_w_h[:, 0] * pred_bboxes_w_h[:, 1]
    #     annotations_area = annotations_w_h[:, 0] * annotations_w_h[:, 1]

    #     # compute union_area
    #     union_area = pred_bboxes_area + annotations_area - overlap_area
    #     union_area = torch.clamp(union_area, min=1e-4)
    #     # compute ious between one image anchors and one image annotations
    #     ious = overlap_area / union_area

    #     enclose_area_top_left = torch.min(per_level_reg_pred[:, 0:2],
    #                                       per_level_reg_targets[:, 0:2])
    #     enclose_area_bot_right = torch.max(per_level_reg_pred[:, 2:4],
    #                                        per_level_reg_targets[:, 2:4])
    #     enclose_area_sizes = torch.clamp(enclose_area_bot_right -
    #                                      enclose_area_top_left,
    #                                      min=0)
    #     enclose_area = enclose_area_sizes[:, 0] * enclose_area_sizes[:, 1]
    #     enclose_area = torch.clamp(enclose_area, min=1e-4)

    #     gious_loss = 1. - ious + (enclose_area - union_area) / enclose_area
    #     gious_loss = gious_loss.sum() / positive_anchors_num

    #     return gious_loss

    def compute_per_level_batch_cls_loss(self, per_level_cls_pred,
                                         per_level_anchor_targets):
        """
        compute per level batch cls loss(bce loss)
        per_level_cls_pred:[batch_size*per_level_anchor_num,num_classes]
        per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
        """
        device = per_level_cls_pred.device
        per_level_cls_pred = per_level_cls_pred[
            per_level_anchor_targets[:, 5] > 0]
        per_level_cls_pred = torch.clamp(per_level_cls_pred,
                                         min=self.epsilon,
                                         max=1. - self.epsilon)
        cls_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0][:, 4]

        positive_anchors_num = cls_targets.shape[0]
        num_classes = per_level_cls_pred.shape[1]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(cls_targets.long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        cls_loss = -(
            loss_ground_truth * torch.log(per_level_cls_pred) +
            (1. - loss_ground_truth) * torch.log(1. - per_level_cls_pred))
        cls_loss = cls_loss.sum(axis=1)

        cls_loss = cls_loss.mean()

        return cls_loss

    def get_batch_anchors_targets(self, batch_anchors, annotations):
        """
        Assign a ground truth target for each anchor
        """
        device = annotations.device

        self.anchor_sizes = torch.tensor(self.anchor_sizes,
                                         dtype=torch.float).to(device)
        anchor_sizes = self.anchor_sizes.view(
            self.per_level_num_anchors,
            len(self.anchor_sizes) // self.per_level_num_anchors, 2)

        anchor_level_feature_map_hw = []
        for per_level_anchor in batch_anchors:
            _, H, W, _, _ = per_level_anchor.shape
            anchor_level_feature_map_hw.append([H, W])
        anchor_level_feature_map_hw = torch.tensor(
            anchor_level_feature_map_hw).to(device)

        per_grid_relative_index = []
        for i in range(self.per_level_num_anchors):
            per_grid_relative_index.append(i)
        per_grid_relative_index = torch.tensor(per_grid_relative_index).to(
            device)

        batch_anchor_targets = []
        for per_level_anchor_sizes, stride, per_level_anchors in zip(
                anchor_sizes, self.strides, batch_anchors):
            B, H, W, _, _ = per_level_anchors.shape
            per_level_reg_cls_target = torch.ones(
                [B, H, W, self.per_level_num_anchors, 5], device=device) * (-1)
            # noobj mask init value=0
            per_level_obj_mask = torch.zeros(
                [B, H, W, self.per_level_num_anchors, 1], device=device)
            # noobj mask init value=1
            per_level_noobj_mask = torch.ones(
                [B, H, W, self.per_level_num_anchors, 1], device=device)
            # 7:[x_min,y_min,x_max,y_max,class_label,obj_mask,noobj_mask]
            per_level_anchor_targets = torch.cat([
                per_level_reg_cls_target, per_level_obj_mask,
                per_level_noobj_mask
            ],
                                                 axis=-1)
            per_level_anchor_targets = per_level_anchor_targets.view(
                per_level_anchor_targets.shape[0], -1,
                per_level_anchor_targets.shape[-1])
            per_level_anchors = per_level_anchors.view(
                per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

            for image_index, one_image_annotations in enumerate(annotations):
                # drop all index=-1 class annotations
                one_image_annotations = one_image_annotations[
                    one_image_annotations[:, 4] >= 0]

                if one_image_annotations.shape[0] != 0:
                    one_image_gt_boxes = one_image_annotations[:, 0:4]
                    one_image_gt_classes = one_image_annotations[:, 4]
                    one_image_gt_boxes_ctr = (one_image_gt_boxes[:, 0:2] +
                                              one_image_gt_boxes[:, 2:4]) / 2

                    # compute all annotations center_grid_index
                    grid_y_indexes = one_image_gt_boxes_ctr[:, 1] // stride
                    grid_x_indexes = one_image_gt_boxes_ctr[:, 0] // stride

                    # compute all annotations gird_indexes transform
                    anchor_indexes_transform = (
                        ((grid_y_indexes * W + grid_x_indexes - 1) *
                         self.per_level_num_anchors).unsqueeze(-1) +
                        per_grid_relative_index.unsqueeze(0)).view(-1)

                    one_image_ious = self.compute_ious_for_one_image(
                        per_level_anchor_sizes, one_image_annotations)

                    # negative anchor includes all anchors with iou <0.5, but the max iou anchor of each annot is not included
                    negative_anchor_flags = (torch.ge(
                        one_image_ious.permute(1, 0), 0.5)).view(-1)
                    negative_anchor_indexes_transform = anchor_indexes_transform[
                        negative_anchor_flags].long()
                    # for anchors which ious>=0.5(ignore threshold),assign noobj_mask label to 0(init value=1)
                    per_level_anchor_targets[image_index,
                                             negative_anchor_indexes_transform,
                                             6] = 0

                    # assign positive sample for max iou anchor of each annot
                    _, positive_anchor_indices = one_image_ious.permute(
                        1, 0).max(axis=1)
                    positive_anchor_indexes_mask = F.one_hot(
                        positive_anchor_indices,
                        num_classes=per_level_anchor_sizes.shape[0]).bool()
                    positive_anchor_indexes_mask = positive_anchor_indexes_mask.view(
                        -1)
                    positive_anchor_indexes_transform = anchor_indexes_transform[
                        positive_anchor_indexes_mask].long()

                    # for positive anchor,assign obj_mask label to 1(init value=0)
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             5] = 1
                    # for positive anchor,assign noobj_mask label to 0(init value=1)
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             6] = 0
                    # for positive anchor,assign class_label:range from 1 to 80
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             4] = one_image_gt_classes + 1
                    # for positive anchor,assign regression_label:[tx,ty,tw,th]
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             0:2] = (one_image_gt_boxes_ctr %
                                                     stride) / stride
                    one_image_gt_whs = one_image_gt_boxes[:, 2:
                                                          4] - one_image_gt_boxes[:,
                                                                                  0:
                                                                                  2]
                    per_level_anchor_targets[
                        image_index, positive_anchor_indexes_transform,
                        2:4] = torch.log((one_image_gt_whs.float() / (
                            (per_level_anchors[
                                image_index, positive_anchor_indexes_transform,
                                2:4]).float() / stride)) + self.epsilon)

                    judge_positive_anchors = ((one_image_gt_whs.float() / (
                        (per_level_anchors[image_index,
                                           positive_anchor_indexes_transform,
                                           2:4]).float() / stride)) >= 1.)
                    judge_flags = ((judge_positive_anchors[:, 0].int() +
                                    judge_positive_anchors[:, 1].int()) < 2)
                    illegal_anchor_mask = []
                    for flag in judge_flags:
                        for _ in range(self.per_level_num_anchors):
                            illegal_anchor_mask.append(flag)
                    illegal_anchor_mask = torch.tensor(illegal_anchor_mask).to(
                        device)
                    illegal_positive_anchor_indexes_transform = anchor_indexes_transform[
                        illegal_anchor_mask].long()

                    per_level_anchor_targets[
                        image_index, illegal_positive_anchor_indexes_transform,
                        0:5] = -1
                    per_level_anchor_targets[
                        image_index, illegal_positive_anchor_indexes_transform,
                        5] = 0
                    per_level_anchor_targets[
                        image_index, illegal_positive_anchor_indexes_transform,
                        6] = 1

            batch_anchor_targets.append(per_level_anchor_targets)

        return batch_anchor_targets

    def compute_ious_for_one_image(self, anchor_sizes, one_image_annotations):
        """
        compute ious between one image anchors and one image annotations
        """
        annotations_wh = one_image_annotations[:, 2:
                                               4] - one_image_annotations[:,
                                                                          0:2]

        # anchor_sizes format:[anchor_nums,4],4:[anchor_w,anchor_h]
        # annotations_wh format: [annotation_nums,4],4:[gt_w,gt_h]
        # When calculating iou, the upper left corner of anchor_sizes and annotations_wh are point (0, 0)

        anchor_sizes = torch.clamp(anchor_sizes, min=0)
        annotations_wh = torch.clamp(annotations_wh, min=0)
        anchor_areas = anchor_sizes[:, 0] * anchor_sizes[:, 1]
        annotations_areas = annotations_wh[:, 0] * annotations_wh[:, 1]

        overlap_areas_w = torch.min(anchor_sizes[:, 0].unsqueeze(-1),
                                    annotations_wh[:, 0].unsqueeze(0))
        overlap_areas_h = torch.min(anchor_sizes[:, 1].unsqueeze(-1),
                                    annotations_wh[:, 1].unsqueeze(0))
        overlap_areas_w = torch.clamp(overlap_areas_w, min=0)
        overlap_areas_h = torch.clamp(overlap_areas_h, min=0)
        overlap_areas = overlap_areas_w * overlap_areas_h

        union_areas = anchor_areas.unsqueeze(-1) + annotations_areas.unsqueeze(
            0) - overlap_areas
        union_areas = torch.clamp(union_areas, min=1e-4)
        # compute ious between one image anchors and one image annotations
        one_image_ious = (overlap_areas / union_areas)

        return one_image_ious


# class YOLOV5Loss(nn.Module):
#     def __init__(self,
#                  anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
#                                [62, 45], [59, 119], [116, 90], [156, 198],
#                                [373, 326]],
#                  per_level_num_anchors=3,
#                  strides=[8, 16, 32],
#                  epsilon=1e-4):
#         super(YOLOV5Loss, self).__init__()
#         self.anchor_sizes = anchor_sizes
#         self.per_level_num_anchors = per_level_num_anchors
#         self.strides = strides
#         self.epsilon = epsilon

#     def forward(self, obj_heads, reg_heads, cls_heads, batch_anchors,
#                 annotations):
#         """
#         compute obj loss, reg loss and cls loss in one batch
#         """
#         device = annotations.device

#         obj_preds, reg_preds, cls_preds, all_anchors = [], [], [], []
#         for per_level_obj_pred, per_level_reg_pred, per_level_cls_pred, per_level_anchors in zip(
#                 obj_heads, reg_heads, cls_heads, batch_anchors):
#             per_level_obj_pred = per_level_obj_pred.view(
#                 per_level_obj_pred.shape[0], -1, per_level_obj_pred.shape[-1])
#             per_level_reg_pred = per_level_reg_pred.view(
#                 per_level_reg_pred.shape[0], -1, per_level_reg_pred.shape[-1])
#             per_level_cls_pred = per_level_cls_pred.view(
#                 per_level_cls_pred.shape[0], -1, per_level_cls_pred.shape[-1])
#             per_level_anchors = per_level_anchors.view(
#                 per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

#             obj_preds.append(per_level_obj_pred)
#             reg_preds.append(per_level_reg_pred)
#             cls_preds.append(per_level_cls_pred)
#             all_anchors.append(per_level_anchors)

#         obj_preds = torch.cat(obj_preds, axis=1)
#         reg_preds = torch.cat(reg_preds, axis=1)
#         cls_preds = torch.cat(cls_preds, axis=1)
#         all_anchors = torch.cat(all_anchors, axis=1)

#         obj_preds = torch.sigmoid(obj_preds)
#         cls_preds = torch.sigmoid(cls_preds)
#         # snap  reg_preds from tx,ty,tw,th -> x_center,y_center,w,h -> x_min,y_min,x_max,y_max
#         reg_preds[:, :,
#                   0:2] = (torch.sigmoid(reg_preds[:, :, 0:2]) +
#                           all_anchors[:, :, 0:2]) * all_anchors[:, :, 4:5]
#         reg_preds[:, :, 2:4] = torch.exp(
#             reg_preds[:, :, 2:4]) * all_anchors[:, :, 2:4]
#         reg_preds[:, :,
#                   0:2] = reg_preds[:, :, 0:2] - 0.5 * reg_preds[:, :, 2:4]
#         reg_preds[:, :, 2:4] = reg_preds[:, :, 2:4] + reg_preds[:, :, 0:2]
#         print("1111", obj_preds.shape, reg_preds.shape, cls_preds.shape,
#               all_anchors.shape)
#         # batch_anchor_targets = self.get_batch_anchors_targets(
#         #     batch_anchors, annotations)

if __name__ == '__main__':
    # from retinanet import RetinaNet
    # net = RetinaNet(resnet_type="resnet50")
    # image_h, image_w = 600, 600
    # cls_heads, reg_heads, batch_anchors = net(
    #     torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    # annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
    #                                   [13, 45, 175, 210, 2]],
    #                                  [[11, 18, 223, 225, 1],
    #                                   [-1, -1, -1, -1, -1]],
    #                                  [[-1, -1, -1, -1, -1],
    #                                   [-1, -1, -1, -1, -1]]])
    # loss = RetinaLoss(image_w, image_h)
    # cls_loss, reg_loss = loss(cls_heads, reg_heads, batch_anchors, annotations)
    # print("1111", cls_loss, reg_loss)

    # from fcos import FCOS
    # net = FCOS(resnet_type="resnet50")
    # image_h, image_w = 600, 600
    # cls_heads, reg_heads, center_heads, batch_positions = net(
    #     torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    # annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
    #                                   [13, 45, 175, 210, 2]],
    #                                  [[11, 18, 223, 225, 1],
    #                                   [-1, -1, -1, -1, -1]],
    #                                  [[-1, -1, -1, -1, -1],
    #                                   [-1, -1, -1, -1, -1]]])
    # loss = FCOSLoss()
    # cls_loss, reg_loss, center_loss = loss(cls_heads, reg_heads, center_heads,
    #                                        batch_positions, annotations)
    # print("2222", cls_loss, reg_loss, center_loss)

    # from centernet import CenterNet
    # net = CenterNet(resnet_type="resnet50")
    # image_h, image_w = 640, 640
    # heatmap_output, offset_output, wh_output = net(
    #     torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    # annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
    #                                   [13, 45, 175, 210, 2]],
    #                                  [[11, 18, 223, 225, 1],
    #                                   [-1, -1, -1, -1, -1]],
    #                                  [[-1, -1, -1, -1, -1],
    #                                   [-1, -1, -1, -1, -1]]])
    # loss = CenterNetLoss()
    # heatmap_loss, offset_loss, wh_loss = loss(heatmap_output, offset_output,
    #                                           wh_output, annotations)
    # print("3333", heatmap_loss, offset_loss, wh_loss)

    from yolov3 import YOLOV3
    net = YOLOV3(backbone_type="darknet53")
    image_h, image_w = 416, 416
    obj_heads, reg_heads, cls_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    loss = YOLOV3Loss()
    obj_loss, reg_loss, cls_loss = loss(obj_heads, reg_heads, cls_heads,
                                        batch_anchors, annotations)
    print("4444", obj_loss, reg_loss, cls_loss)

    # from yolov5 import YOLOV5
    # net = YOLOV5(yolov5_type='yolov5s')
    # image_h, image_w = 640, 640
    # obj_heads, reg_heads, cls_heads, batch_anchors = net(
    #     torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    # annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
    #                                   [13, 45, 175, 210, 2]],
    #                                  [[11, 18, 223, 225, 1],
    #                                   [-1, -1, -1, -1, -1]],
    #                                  [[-1, -1, -1, -1, -1],
    #                                   [-1, -1, -1, -1, -1]]])
    # loss = YOLOV5Loss()
    # loss(obj_heads, reg_heads, cls_heads, batch_anchors, annotations)
