import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.detection.models.anchor import RetinaAnchors, FCOSPositions, TTFNetPositions, YoloxAnchors

__all__ = [
    'RetinaLoss',
    'FCOSLoss',
    'CenterNetLoss',
    'TTFNetLoss',
    'YoloxLoss',
]


class IoUMethod:

    def __init__(self):
        pass

    def __call__(self, boxes1, boxes2, iou_type='IoU', box_type='xyxy'):
        '''
        box1 format:[...,4]
        box2 format:[...,4]
        xyxy type:[x_min,y_min,x_max,y_max]
        xywh type:[x_center,y_center,w,h]
        '''
        assert iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU',
                            'EIoU'], 'wrong IoU type!'
        assert box_type in ['xyxy', 'xywh'], 'wrong box_type type!'

        if box_type == 'xywh':
            # transform format from [x_ctr,y_ctr,w,h] to xyxy
            boxes1_x1y1 = boxes1[..., 0:2] - boxes1[..., 2:4] / 2
            boxes1_x2y2 = boxes1[..., 0:2] + boxes1[..., 2:4] / 2
            boxes1 = torch.cat([boxes1_x1y1, boxes1_x2y2], dim=1)

            boxes2_x1y1 = boxes2[..., 0:2] - boxes2[..., 2:4] / 2
            boxes2_x2y2 = boxes2[..., 0:2] + boxes2[..., 2:4] / 2
            boxes2 = torch.cat([boxes2_x1y1, boxes2_x2y2], dim=1)

        overlap_area_xymin = torch.max(boxes1[..., 0:2], boxes2[..., 0:2])
        overlap_area_xymax = torch.min(boxes1[..., 2:4], boxes2[..., 2:4])
        overlap_area_sizes = torch.clamp(overlap_area_xymax -
                                         overlap_area_xymin,
                                         min=0)
        overlap_area = overlap_area_sizes[..., 0] * overlap_area_sizes[..., 1]

        boxes1_wh = torch.clamp(boxes1[..., 2:4] - boxes1[..., 0:2], min=0)
        boxes2_wh = torch.clamp(boxes2[..., 2:4] - boxes2[..., 0:2], min=0)

        boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
        boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

        # compute ious between boxes1 and boxes2
        union_area = boxes1_area + boxes2_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        ious = overlap_area / union_area

        if iou_type == 'IoU':
            return ious
        else:
            if iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
                enclose_area_top_left = torch.min(boxes1[..., 0:2],
                                                  boxes2[..., 0:2])
                enclose_area_bot_right = torch.max(boxes1[..., 2:4],
                                                   boxes2[..., 2:4])
                enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                                 enclose_area_top_left,
                                                 min=0)
                if iou_type in ['DIoU', 'CIoU', 'EIoU']:
                    # https://arxiv.org/abs/1911.08287v1
                    # compute DIoU c2 and p2
                    # c2:convex diagonal squared
                    c2 = enclose_area_sizes[...,
                                            0]**2 + enclose_area_sizes[...,
                                                                       1]**2
                    c2 = torch.clamp(c2, min=1e-4)
                    # p2:center distance squared
                    boxes1_ctr = (boxes1[..., 2:4] + boxes1[..., 0:2]) / 2
                    boxes2_ctr = (boxes2[..., 2:4] + boxes2[..., 0:2]) / 2
                    p2 = (boxes1_ctr[..., 0] - boxes2_ctr[..., 0])**2 + (
                        boxes1_ctr[..., 1] - boxes2_ctr[..., 1])**2
                    if iou_type == 'DIoU':
                        return ious - p2 / c2
                    elif iou_type == 'CIoU':
                        # compute CIoU v and alpha
                        v = (4 / math.pi**2) * torch.pow(
                            torch.atan(boxes2_wh[:, 0] / boxes2_wh[:, 1]) -
                            torch.atan(boxes1_wh[:, 0] / boxes1_wh[:, 1]), 2)

                        with torch.no_grad():
                            alpha = v / torch.clamp(1 - ious + v, min=1e-4)

                        return ious - (p2 / c2 + v * alpha)
                    elif iou_type == 'EIoU':
                        pw2 = (boxes2_wh[..., 0] - boxes1_wh[..., 0])**2
                        ph2 = (boxes2_wh[..., 1] - boxes1_wh[..., 1])**2
                        cw2 = enclose_area_sizes[..., 0]**2
                        ch2 = enclose_area_sizes[..., 1]**2
                        cw2 = torch.clamp(cw2, min=1e-4)
                        ch2 = torch.clamp(ch2, min=1e-4)

                        return ious - (p2 / c2 + pw2 / cw2 + ph2 / ch2)
                else:
                    enclose_area = enclose_area_sizes[:,
                                                      0] * enclose_area_sizes[:,
                                                                              1]
                    enclose_area = torch.clamp(enclose_area, min=1e-4)

                    return ious - (enclose_area - union_area) / enclose_area


class RetinaLoss(nn.Module):

    def __init__(self,
                 areas=[[32, 32], [64, 64], [128, 128], [256, 256], [512,
                                                                     512]],
                 ratios=[0.5, 1, 2],
                 scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
                 strides=[8, 16, 32, 64, 128],
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 focal_eiou_gamma=0.5,
                 cls_loss_weight=1.,
                 box_loss_weight=1.,
                 box_loss_type='CIoU'):
        super(RetinaLoss, self).__init__()
        assert box_loss_type in [
            'SmoothL1', 'IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU', 'Focal_EIoU'
        ], 'wrong IoU type!'
        self.anchors = RetinaAnchors(areas=areas,
                                     ratios=ratios,
                                     scales=scales,
                                     strides=strides)
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.focal_eiou_gamma = focal_eiou_gamma
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight
        self.box_loss_type = box_loss_type
        self.iou_function = IoUMethod()

    def forward(self, preds, annotations):
        '''
        compute cls loss and reg loss in one batch
        '''
        device = annotations.device
        batch_size = annotations.shape[0]
        cls_preds, reg_preds = preds

        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in cls_preds]
        one_image_anchors = self.anchors(feature_size)
        one_image_anchors = torch.cat([
            torch.tensor(per_level_anchor).view(-1, per_level_anchor.shape[-1])
            for per_level_anchor in one_image_anchors
        ],
                                      dim=0)
        batch_anchors = one_image_anchors.unsqueeze(0).repeat(
            batch_size, 1, 1).to(device)
        batch_anchors_annotations = self.get_batch_anchors_annotations(
            batch_anchors, annotations)

        cls_preds = [
            per_cls_pred.view(per_cls_pred.shape[0], -1,
                              per_cls_pred.shape[-1])
            for per_cls_pred in cls_preds
        ]
        reg_preds = [
            per_reg_pred.view(per_reg_pred.shape[0], -1,
                              per_reg_pred.shape[-1])
            for per_reg_pred in reg_preds
        ]
        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)

        cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)

        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
        reg_preds = reg_preds.view(-1, reg_preds.shape[-1])
        batch_anchors = batch_anchors.view(-1, batch_anchors.shape[-1])
        batch_anchors_annotations = batch_anchors_annotations.view(
            -1, batch_anchors_annotations.shape[-1])

        cls_loss = self.compute_batch_focal_loss(cls_preds,
                                                 batch_anchors_annotations)
        reg_loss = self.compute_batch_box_loss(reg_preds,
                                               batch_anchors_annotations,
                                               batch_anchors)

        cls_loss = self.cls_loss_weight * cls_loss
        reg_loss = self.box_loss_weight * reg_loss

        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
        }

        return loss_dict

    def compute_batch_focal_loss(self, cls_preds, batch_anchors_annotations):
        '''
        compute batch focal loss(cls loss)
        cls_preds:[batch_size*anchor_num,num_classes]
        batch_anchors_annotations:[batch_size*anchor_num,5]
        '''
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        device = cls_preds.device
        cls_preds = cls_preds[batch_anchors_annotations[:, 4] >= 0]
        batch_anchors_annotations = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] >= 0]
        positive_anchors_num = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0].shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        num_classes = cls_preds.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_anchors_annotations[:, 4].long(),
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
        batch_focal_loss = batch_focal_loss / positive_anchors_num

        return batch_focal_loss

    def compute_batch_box_loss(self, reg_preds, batch_anchors_annotations,
                               batch_anchors):
        '''
        compute batch smoothl1 loss(reg loss)
        reg_preds:[batch_size*anchor_num,4]
        batch_anchors_annotations:[batch_size*anchor_num,5]
        batch_anchors:[batch_size*anchor_num,4]
        '''
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate smoothl1 loss
        device = reg_preds.device
        reg_preds = reg_preds[batch_anchors_annotations[:, 4] > 0]
        batch_anchors = batch_anchors[batch_anchors_annotations[:, 4] > 0]
        batch_anchors_annotations = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0]
        positive_anchor_num = batch_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        if self.box_loss_type == 'SmoothL1':
            box_loss = self.compute_batch_smoothl1_loss(
                reg_preds, batch_anchors_annotations)
        else:
            box_loss_type = 'EIoU' if self.box_loss_type == 'Focal_EIoU' else self.box_loss_type
            pred_boxes = self.snap_txtytwth_to_xyxy(reg_preds, batch_anchors)
            ious = self.iou_function(pred_boxes,
                                     batch_anchors_annotations[:, 0:4],
                                     iou_type=box_loss_type,
                                     box_type='xyxy')
            box_loss = 1 - ious

            if self.box_loss_type == 'Focal_EIoU':
                gamma_ious = self.iou_function(pred_boxes,
                                               batch_anchors_annotations[:,
                                                                         0:4],
                                               iou_type='IoU',
                                               box_type='xyxy')
                gamma_ious = torch.pow(gamma_ious, self.focal_eiou_gamma)
                box_loss = gamma_ious * box_loss

            box_loss = box_loss.sum() / positive_anchor_num

        return box_loss

    def compute_batch_smoothl1_loss(self, reg_preds,
                                    batch_anchors_annotations):
        '''
        compute batch smoothl1 loss(reg loss)
        reg_preds:[batch_size*anchor_num,4]
        anchors_annotations:[batch_size*anchor_num,5]
        '''
        device = reg_preds.device
        positive_anchor_num = batch_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        # compute smoothl1 loss
        loss_ground_truth = batch_anchors_annotations[:, 0:4]

        x = torch.abs(reg_preds - loss_ground_truth)
        batch_smoothl1_loss = torch.where(torch.ge(x, self.beta),
                                          x - 0.5 * self.beta,
                                          0.5 * (x**2) / self.beta)

        batch_smoothl1_loss = batch_smoothl1_loss.sum() / positive_anchor_num

        return batch_smoothl1_loss

    def get_batch_anchors_annotations(self, batch_anchors, annotations):
        '''
        Assign a ground truth box target and a ground truth class target for each anchor
        if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
        if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
        if anchor gt_class index > 0,this anchor is a object class anchor and used in
        calculate cls loss and reg loss
        '''
        assert batch_anchors.shape[0] == annotations.shape[0]
        device = annotations.device
        one_image_anchor_nums = batch_anchors.shape[1]

        batch_anchors_annotations = []
        for per_image_anchors, one_image_annotations in zip(
                batch_anchors, annotations):
            # drop all index=-1 class annotations
            one_image_annotations = one_image_annotations[
                one_image_annotations[:, 4] >= 0]

            if one_image_annotations.shape[0] == 0:
                one_image_anchor_annotations = torch.ones(
                    [one_image_anchor_nums, 5],
                    dtype=torch.float32,
                    device=device) * (-1)
            else:
                one_image_gt_bboxes = one_image_annotations[:, 0:4]
                one_image_gt_class = one_image_annotations[:, 4]

                one_image_ious = self.iou_function(
                    per_image_anchors.unsqueeze(1),
                    one_image_gt_bboxes.unsqueeze(0),
                    iou_type='IoU',
                    box_type='xyxy')

                # snap per gt bboxes to the best iou anchor
                overlap, indices = one_image_ious.max(axis=1)
                per_image_anchors_gt_class = (torch.ones_like(overlap) *
                                              -1).to(device)
                # if iou <0.4,assign anchors gt class as 0:background
                per_image_anchors_gt_class[overlap < 0.4] = 0
                # if iou >=0.5,assign anchors gt class as same as the max iou annotation class:80 classes index from 1 to 80
                per_image_anchors_gt_class[
                    overlap >=
                    0.5] = one_image_gt_class[indices][overlap >= 0.5] + 1

                per_image_anchors_gt_class = per_image_anchors_gt_class.unsqueeze(
                    -1)

                # assgin each anchor gt bboxes for max iou annotation
                per_image_anchors_gt_bboxes = one_image_gt_bboxes[indices]
                if self.box_loss_type == 'SmoothL1':
                    # transform gt bboxes to [tx,ty,tw,th] format for each anchor
                    per_image_anchors_gt_bboxes = self.snap_annotations_to_txtytwth(
                        per_image_anchors_gt_bboxes, per_image_anchors)

                one_image_anchor_annotations = torch.cat(
                    [per_image_anchors_gt_bboxes, per_image_anchors_gt_class],
                    dim=1)

            one_image_anchor_annotations = one_image_anchor_annotations.unsqueeze(
                0)
            batch_anchors_annotations.append(one_image_anchor_annotations)

        batch_anchors_annotations = torch.cat(batch_anchors_annotations, dim=0)

        # batch anchors annotations shape:[batch_size, anchor_nums, 5]
        return batch_anchors_annotations

    def snap_annotations_to_txtytwth(self, anchors_gt_bboxes, anchors):
        '''
        snap each anchor ground truth bbox form format:[x_min,y_min,x_max,y_max] to format:[tx,ty,tw,th]
        '''
        anchors_w_h = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_w_h

        anchors_gt_bboxes_w_h = anchors_gt_bboxes[:,
                                                  2:] - anchors_gt_bboxes[:, :2]
        anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1e-4)
        anchors_gt_bboxes_ctr = anchors_gt_bboxes[:, :
                                                  2] + 0.5 * anchors_gt_bboxes_w_h

        snaped_annotations_for_anchors = torch.cat(
            [(anchors_gt_bboxes_ctr - anchors_ctr) / anchors_w_h,
             torch.log(anchors_gt_bboxes_w_h / anchors_w_h)],
            dim=1)

        # snaped_annotations_for_anchors shape:[anchor_nums, 4]
        return snaped_annotations_for_anchors

    def snap_txtytwth_to_xyxy(self, snap_boxes, anchors):
        '''
        snap reg heads to pred bboxes
        snap_boxes:[batch_size*anchor_nums,4],4:[tx,ty,tw,th]
        anchors:[batch_size*anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        '''
        anchors_wh = anchors[:, 2:4] - anchors[:, 0:2]
        anchors_ctr = anchors[:, 0:2] + 0.5 * anchors_wh

        boxes_wh = torch.exp(snap_boxes[:, 2:4]) * anchors_wh
        boxes_ctr = snap_boxes[:, :2] * anchors_wh + anchors_ctr

        boxes_x_min_y_min = boxes_ctr - 0.5 * boxes_wh
        boxes_x_max_y_max = boxes_ctr + 0.5 * boxes_wh

        boxes = torch.cat([boxes_x_min_y_min, boxes_x_max_y_max], dim=1)

        # boxes shape:[anchor_nums,4]
        return boxes


class FCOSLoss(nn.Module):

    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 mi=[[-1, 64], [64, 128], [128, 256], [256, 512],
                     [512, 100000000]],
                 alpha=0.25,
                 gamma=2.,
                 cls_loss_weight=1.,
                 box_loss_weight=1.,
                 center_ness_loss_weight=1.,
                 box_loss_iou_type='CIoU',
                 center_sample_radius=1.5,
                 use_center_sample=True):
        super(FCOSLoss, self).__init__()
        assert box_loss_iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU',
                                     'EIoU'], 'wrong IoU type!'
        self.positions = FCOSPositions(strides=strides)
        self.alpha = alpha
        self.gamma = gamma
        self.strides = strides
        self.mi = mi
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight
        self.center_ness_loss_weight = center_ness_loss_weight
        self.box_loss_iou_type = box_loss_iou_type
        self.center_sample_radius = center_sample_radius
        self.use_center_sample = use_center_sample
        self.iou_function = IoUMethod()

    def forward(self, preds, annotations):
        '''
        compute cls loss, reg loss and center-ness loss in one batch
        '''
        device = annotations.device
        batch_size = annotations.shape[0]
        cls_preds, reg_preds, center_preds = preds

        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in cls_preds]
        one_image_positions = self.positions(feature_size)
        batch_positions = [
            torch.tensor(per_level_position).unsqueeze(0).repeat(
                batch_size, 1, 1, 1).to(device)
            for per_level_position in one_image_positions
        ]

        cls_preds, reg_preds, center_preds, batch_targets = self.get_batch_position_annotations(
            cls_preds,
            reg_preds,
            center_preds,
            batch_positions,
            annotations,
            use_center_sample=self.use_center_sample)

        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
        reg_preds = reg_preds.view(-1, reg_preds.shape[-1])
        center_preds = center_preds.view(-1, center_preds.shape[-1])
        batch_targets = batch_targets.view(-1, batch_targets.shape[-1])

        cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)
        center_preds = torch.clamp(center_preds, min=1e-4, max=1. - 1e-4)

        cls_loss = self.compute_batch_focal_loss(cls_preds, batch_targets)
        reg_loss = self.compute_batch_iou_loss(reg_preds, batch_targets)
        center_ness_loss = self.compute_batch_centerness_loss(
            center_preds, batch_targets)

        cls_loss = self.cls_loss_weight * cls_loss
        reg_loss = self.box_loss_weight * reg_loss
        center_ness_loss = self.center_ness_loss_weight * center_ness_loss

        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'center_ness_loss': center_ness_loss,
        }

        return loss_dict

    def compute_batch_focal_loss(self, cls_preds, batch_targets):
        '''
        compute batch focal loss(cls loss)
        cls_preds:[batch_size*points_num,num_classes]
        batch_targets:[batch_size*points_num,8]
        '''
        device = cls_preds.device
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

    def compute_batch_iou_loss(self, reg_preds, batch_targets):
        '''
        compute batch giou loss(reg loss)
        reg_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        '''
        # only use positive points sample to compute reg loss
        device = reg_preds.device
        reg_preds = torch.exp(reg_preds)
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

    def get_batch_position_annotations(self,
                                       cls_heads,
                                       reg_heads,
                                       center_heads,
                                       batch_positions,
                                       annotations,
                                       use_center_sample=True):
        '''
        Assign a ground truth target for each position on feature map
        '''
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

        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)
        center_preds = torch.cat(center_preds, dim=1)
        all_points_position = torch.cat(all_points_position, dim=1)
        all_points_mi = torch.cat(all_points_mi, dim=1)
        all_points_stride = torch.cat(all_points_stride, dim=1)

        batch_targets = []
        for per_image_position, per_image_mi, per_image_stride, per_image_annotations in zip(
                all_points_position, all_points_mi, all_points_stride,
                annotations):
            per_image_annotations = per_image_annotations[
                per_image_annotations[:, 4] >= 0]
            points_num = per_image_position.shape[0]

            if per_image_annotations.shape[0] == 0:
                # 6:l,t,r,b,class_index,center-ness_gt
                per_image_targets = torch.zeros([points_num, 6],
                                                dtype=torch.float32,
                                                device=device)
            else:
                annotaion_num = per_image_annotations.shape[0]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                candidates = torch.zeros([points_num, annotaion_num, 4],
                                         dtype=torch.float32,
                                         device=device)
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)

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
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
                                                    dtype=torch.float32,
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
                                                    dtype=torch.float32,
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

        batch_targets = torch.cat(batch_targets, dim=0)
        batch_targets = torch.cat([batch_targets, all_points_position], dim=2)

        # batch_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        return cls_preds, reg_preds, center_preds, batch_targets


class CenterNetLoss(nn.Module):

    def __init__(self,
                 alpha=2.,
                 beta=4.,
                 heatmap_loss_weight=1.0,
                 offset_loss_weight=1.0,
                 wh_loss_weight=0.1,
                 min_overlap=0.7,
                 max_object_num=100):
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.heatmap_loss_weight = heatmap_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.wh_loss_weight = wh_loss_weight
        self.min_overlap = min_overlap
        self.max_object_num = max_object_num

    def forward(self, preds, annotations):
        '''
        compute heatmap loss, offset loss and wh loss in one batch
        '''
        device = annotations.device
        heatmap_heads, offset_heads, wh_heads = preds

        batch_heatmap_targets, batch_wh_targets, batch_offset_targets, batch_reg_to_heatmap_index, batch_positive_targets_mask = self.get_batch_targets(
            heatmap_heads, annotations)

        heatmap_heads = torch.clamp(heatmap_heads, min=1e-4, max=1. - 1e-4)

        B, num_classes = heatmap_heads.shape[0], heatmap_heads.shape[1]
        heatmap_heads = heatmap_heads.permute(0, 2, 3, 1).contiguous().view(
            B, -1, num_classes)
        batch_heatmap_targets = batch_heatmap_targets.permute(
            0, 2, 3, 1).contiguous().view(B, -1, num_classes)

        wh_heads = wh_heads.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        offset_heads = offset_heads.permute(0, 2, 3,
                                            1).contiguous().view(B, -1, 2)

        heatmap_loss = self.compute_batch_heatmap_loss(heatmap_heads,
                                                       batch_heatmap_targets)
        offset_loss = self.compute_batch_offsetl1_loss(
            offset_heads, batch_offset_targets, batch_reg_to_heatmap_index,
            batch_positive_targets_mask)
        wh_loss = self.compute_batch_whl1_loss(wh_heads, batch_wh_targets,
                                               batch_reg_to_heatmap_index,
                                               batch_positive_targets_mask)

        heatmap_loss = self.heatmap_loss_weight * heatmap_loss
        offset_loss = self.offset_loss_weight * offset_loss
        wh_loss = self.wh_loss_weight * wh_loss

        loss_dict = {
            'heatmap_loss': heatmap_loss,
            'offset_loss': offset_loss,
            'wh_loss': wh_loss,
        }

        return loss_dict

    def compute_batch_heatmap_loss(self, heatmap_heads, batch_heatmap_targets):
        device = heatmap_heads.device
        positive_point_num = (
            batch_heatmap_targets[batch_heatmap_targets == 1.].float()).sum()
        if positive_point_num == 0:
            return torch.tensor(0.).to(device)

        # all center points
        positive_indexes = (batch_heatmap_targets == 1.)
        # all non center points
        negative_indexes = (batch_heatmap_targets < 1.)

        positive_loss = -torch.log(heatmap_heads) * torch.pow(
            1 - heatmap_heads, self.alpha) * positive_indexes
        negative_loss = -torch.log(1 - heatmap_heads) * torch.pow(
            heatmap_heads, self.alpha) * torch.pow(
                1 - batch_heatmap_targets, self.beta) * negative_indexes

        loss = (positive_loss.sum() + negative_loss.sum()) / positive_point_num

        return loss

    def compute_batch_offsetl1_loss(self,
                                    offset_heads,
                                    batch_offset_targets,
                                    batch_reg_to_heatmap_index,
                                    batch_positive_targets_mask,
                                    factor=1.0 / 9.0):
        device = offset_heads.device
        batch_reg_to_heatmap_index = batch_reg_to_heatmap_index.unsqueeze(
            -1).repeat(1, 1, 2)
        offset_heads = torch.gather(offset_heads, 1,
                                    batch_reg_to_heatmap_index.long())

        positive_point_num = (batch_positive_targets_mask[
            batch_positive_targets_mask == 1.].float()).sum()
        if positive_point_num == 0:
            return torch.tensor(0.).to(device)

        batch_positive_targets_mask = batch_positive_targets_mask.unsqueeze(
            -1).repeat(1, 1, 2)

        offset_heads = offset_heads * batch_positive_targets_mask
        batch_offset_targets = batch_offset_targets * batch_positive_targets_mask

        x = torch.abs(offset_heads - batch_offset_targets)
        loss = torch.where(torch.ge(x, factor), x - 0.5 * factor,
                           0.5 * (x**2) / factor)
        loss = loss.sum() / positive_point_num

        return loss

    def compute_batch_whl1_loss(self,
                                wh_heads,
                                batch_wh_targets,
                                batch_reg_to_heatmap_index,
                                batch_positive_targets_mask,
                                factor=1.0 / 9.0):
        device = wh_heads.device
        batch_reg_to_heatmap_index = batch_reg_to_heatmap_index.unsqueeze(
            -1).repeat(1, 1, 2)
        wh_heads = torch.gather(wh_heads, 1, batch_reg_to_heatmap_index.long())

        positive_point_num = (batch_positive_targets_mask[
            batch_positive_targets_mask == 1.].float()).sum()
        if positive_point_num == 0:
            return torch.tensor(0.).to(device)

        batch_positive_targets_mask = batch_positive_targets_mask.unsqueeze(
            -1).repeat(1, 1, 2)

        wh_heads = wh_heads * batch_positive_targets_mask
        batch_wh_targets = batch_wh_targets * batch_positive_targets_mask

        x = torch.abs(wh_heads - batch_wh_targets)
        loss = torch.where(torch.ge(x, factor), x - 0.5 * factor,
                           0.5 * (x**2) / factor)
        loss = loss.sum() / positive_point_num

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
            gt_bboxes = gt_bboxes / 4.

            gt_bboxes[:, [0, 2]] = torch.clamp(gt_bboxes[:, [0, 2]],
                                               min=0,
                                               max=W - 1)
            gt_bboxes[:, [1, 3]] = torch.clamp(gt_bboxes[:, [1, 3]],
                                               min=0,
                                               max=H - 1)

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


class TTFNetLoss(nn.Module):

    def __init__(self,
                 alpha=2.0,
                 beta=4.0,
                 stride=4,
                 heatmap_loss_weight=1.0,
                 box_loss_weight=5.0,
                 box_loss_iou_type='CIoU',
                 gaussian_alpha=0.54,
                 gaussian_beta=0.54):
        super(TTFNetLoss, self).__init__()
        assert box_loss_iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU',
                                     'EIoU'], 'wrong IoU type!'
        self.positions = TTFNetPositions()
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.heatmap_loss_weight = heatmap_loss_weight
        self.box_loss_weight = box_loss_weight
        self.box_loss_iou_type = box_loss_iou_type
        self.gaussian_alpha = gaussian_alpha
        self.gaussian_beta = gaussian_beta
        self.iou_function = IoUMethod()

    def forward(self, preds, annotations):
        '''
        compute heatmap loss, wh loss in one batch
        '''
        device = annotations.device
        heatmap_heads, wh_heads = preds
        batch_size = heatmap_heads.shape[0]

        feature_map_size = [heatmap_heads.shape[3], heatmap_heads.shape[2]]
        one_image_positions = self.positions(feature_map_size)
        batch_positions = torch.tensor(one_image_positions).unsqueeze(
            0).repeat(batch_size, 1, 1, 1).to(device)

        batch_heatmap_targets, batch_reg_targets = self.get_batch_targets(
            heatmap_heads, annotations)

        heatmap_heads = heatmap_heads.permute(0, 2, 3, 1).contiguous()
        wh_heads = wh_heads.permute(0, 2, 3, 1).contiguous()
        batch_heatmap_targets = batch_heatmap_targets.permute(0, 2, 3,
                                                              1).contiguous()
        batch_reg_targets = batch_reg_targets.permute(0, 2, 3, 1).contiguous()

        heatmap_heads = torch.clamp(heatmap_heads, min=1e-4, max=1. - 1e-4)

        heatmap_heads = heatmap_heads.view(-1, heatmap_heads.shape[-1])
        wh_heads = wh_heads.view(-1, wh_heads.shape[-1])
        batch_heatmap_targets = batch_heatmap_targets.view(
            -1, batch_heatmap_targets.shape[-1])
        batch_reg_targets = batch_reg_targets.view(-1,
                                                   batch_reg_targets.shape[-1])
        batch_positions = batch_positions.view(-1, batch_positions.shape[-1])

        heatmap_loss = self.compute_batch_heatmap_loss(heatmap_heads,
                                                       batch_heatmap_targets)
        box_loss = self.compute_batch_iou_loss(wh_heads, batch_positions,
                                               batch_reg_targets)

        heatmap_loss = self.heatmap_loss_weight * heatmap_loss
        box_loss = self.box_loss_weight * box_loss

        loss_dict = {
            'heatmap_loss': heatmap_loss,
            'box_loss': box_loss,
        }

        return loss_dict

    def compute_batch_heatmap_loss(self, heatmap_heads, batch_heatmap_targets):
        device = heatmap_heads.device
        positive_point_num = (
            batch_heatmap_targets[batch_heatmap_targets == 1.].float()).sum()
        if positive_point_num == 0:
            return torch.tensor(0.).to(device)

        positive_points_mask = (batch_heatmap_targets == 1.)
        negative_points_mask = (batch_heatmap_targets < 1.)

        positive_loss = -torch.log(heatmap_heads) * torch.pow(
            1 - heatmap_heads, self.alpha) * positive_points_mask
        negative_loss = -torch.log(1 - heatmap_heads) * torch.pow(
            heatmap_heads, self.alpha) * torch.pow(
                1 - batch_heatmap_targets, self.beta) * negative_points_mask

        heatmap_loss = (positive_loss.sum() +
                        negative_loss.sum()) / positive_point_num

        return heatmap_loss

    def compute_batch_iou_loss(self, wh_heads, batch_positions,
                               batch_reg_targets):
        # only use positive points sample to compute iou loss
        device = wh_heads.device
        wh_heads = torch.exp(wh_heads)
        wh_heads = wh_heads[batch_reg_targets[:, 4] > 0]
        batch_positions = batch_positions[batch_reg_targets[:, 4] > 0]
        batch_reg_targets = batch_reg_targets[batch_reg_targets[:, 4] > 0]
        positive_points_num = batch_reg_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        # snap ltrb to x1y1x2y2
        pred_bboxes_xy_min = (batch_positions - wh_heads[:, 0:2]) * self.stride
        pred_bboxes_xy_max = (batch_positions + wh_heads[:, 2:4]) * self.stride
        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                dim=1)

        gt_bboxes = batch_reg_targets[:, 0:4]
        gt_bboxes_weight = batch_reg_targets[:, 4]

        ious = self.iou_function(pred_bboxes,
                                 gt_bboxes,
                                 iou_type=self.box_loss_iou_type,
                                 box_type='xyxy')
        ious_loss = 1 - ious

        ious_loss = ious_loss * gt_bboxes_weight
        ious_loss = ious_loss.sum() / (gt_bboxes_weight.sum() + 1e-4)

        return ious_loss

    def get_batch_targets(self, heatmap_heads, annotations):
        B, num_classes, H, W = heatmap_heads.shape[0], heatmap_heads.shape[
            1], heatmap_heads.shape[2], heatmap_heads.shape[3]
        device = annotations.device

        batch_heatmap_targets, batch_reg_targets = [], []
        for per_image_annots in annotations:
            per_image_annots = per_image_annots[per_image_annots[:, 4] >= 0]

            per_image_heatmap_targets = torch.zeros((num_classes, H, W),
                                                    dtype=torch.float32,
                                                    device=device)
            per_image_reg_targets = torch.ones(
                (5, H, W), dtype=torch.float32, device=device) * (-1)

            if per_image_annots.shape[0] != 0:
                per_image_annots_box_wh = per_image_annots[:, 2:
                                                           4] - per_image_annots[:,
                                                                                 0:
                                                                                 2]
                per_image_annots_box_area = per_image_annots_box_wh[:,
                                                                    0] * per_image_annots_box_wh[:,
                                                                                                 1]
                per_image_annots_box_area = torch.log(
                    torch.clamp(per_image_annots_box_area, min=1e-4))
                per_image_topk_annots_box_area, per_image_topk_annots_box_idxs = torch.topk(
                    per_image_annots_box_area,
                    per_image_annots_box_area.shape[0])

                per_image_annots = per_image_annots[
                    per_image_topk_annots_box_idxs]
                per_image_gt_boxes = per_image_annots[:, 0:4]
                per_image_gt_classes = per_image_annots[:, 4]

                per_image_gt_boxes = per_image_gt_boxes / 4.
                per_image_gt_boxes[:, [0, 2]] = torch.clamp(
                    per_image_gt_boxes[:, [0, 2]], min=0, max=W - 1)
                per_image_gt_boxes[:, [1, 3]] = torch.clamp(
                    per_image_gt_boxes[:, [1, 3]], min=0, max=H - 1)
                # make sure all height and width >0
                all_h = per_image_gt_boxes[:, 3] - per_image_gt_boxes[:, 1]
                all_w = per_image_gt_boxes[:, 2] - per_image_gt_boxes[:, 0]

                centers = torch.cat(
                    [((per_image_gt_boxes[:, 0] + per_image_gt_boxes[:, 2]) /
                      2).unsqueeze(-1),
                     ((per_image_gt_boxes[:, 1] + per_image_gt_boxes[:, 3]) /
                      2).unsqueeze(-1)],
                    axis=1)
                centers_int = torch.trunc(centers)

                h_radius_alpha = torch.trunc(
                    (all_h / 2. * self.gaussian_alpha))
                w_radius_alpha = torch.trunc(
                    (all_w / 2. * self.gaussian_alpha))
                h_radius_beta = torch.trunc((all_h / 2. * self.gaussian_beta))
                w_radius_beta = torch.trunc((all_w / 2. * self.gaussian_beta))

                # larger boxes have lower priority than small boxes.
                for i, per_annot in enumerate(per_image_annots):
                    per_gt_box, per_gt_class = per_annot[0:4], per_annot[4]
                    fake_heatmap = torch.zeros((H, W),
                                               dtype=torch.float32,
                                               device=device)
                    fake_heatmap = self.draw_truncate_gaussian(
                        fake_heatmap, centers_int[i], h_radius_alpha[i].item(),
                        w_radius_alpha[i].item())
                    per_image_heatmap_targets[per_gt_class.long()] = torch.max(
                        per_image_heatmap_targets[per_gt_class.long()],
                        fake_heatmap)

                    if self.gaussian_alpha != self.gaussian_beta:
                        # reinit fake_heatmap to value 0
                        fake_heatmap = torch.zeros((H, W),
                                                   dtype=torch.float32,
                                                   device=device)
                        fake_heatmap = self.draw_truncate_gaussian(
                            fake_heatmap, centers_int[i],
                            h_radius_beta[i].item(), w_radius_beta[i].item())

                    gt_box_heatmap_idxs = (fake_heatmap > 0)

                    # gt box has been downsampled by stride
                    per_image_reg_targets[
                        0:4, gt_box_heatmap_idxs] = per_gt_box.unsqueeze(-1)

                    local_heatmap = fake_heatmap[gt_box_heatmap_idxs]
                    center_div = local_heatmap.sum()
                    local_heatmap *= per_image_topk_annots_box_area[i]
                    per_image_reg_targets[
                        4, gt_box_heatmap_idxs] = local_heatmap / center_div

            batch_heatmap_targets.append(
                per_image_heatmap_targets.unsqueeze(0))
            batch_reg_targets.append(per_image_reg_targets.unsqueeze(0))

        batch_heatmap_targets = torch.cat(batch_heatmap_targets, axis=0)
        batch_reg_targets = torch.cat(batch_reg_targets, axis=0)

        return batch_heatmap_targets, batch_reg_targets

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y /
                     (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x, sigma_y = w / 6, h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        device = heatmap.device
        gaussian = torch.from_numpy(gaussian).to(device)

        x, y = int(center[0]), int(center[1])
        h_radius, w_radius = int(h_radius), int(w_radius)

        height, width = heatmap.shape[0], heatmap.shape[1]

        left, right = int(min(x, w_radius)), int(min(width - x, w_radius + 1))
        top, bottom = int(min(y, h_radius)), int(min(height - y, h_radius + 1))

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                                   w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap


class YoloxLoss(nn.Module):

    def __init__(self,
                 strides=[8, 16, 32],
                 obj_loss_weight=1.0,
                 box_loss_weight=5.0,
                 cls_loss_weight=1.0,
                 box_loss_iou_type='CIoU',
                 center_sample_radius=2.5):
        super(YoloxLoss, self).__init__()
        assert box_loss_iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU',
                                     'EIoU'], 'wrong IoU type!'
        self.grid_strides = YoloxAnchors(strides=strides)
        self.strides = strides
        self.obj_loss_weight = obj_loss_weight
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_iou_type = box_loss_iou_type
        self.center_sample_radius = center_sample_radius
        self.iou_function = IoUMethod()

    def forward(self, preds, annotations):
        '''
        compute cls loss, box loss and cls loss in one batch
        '''
        device = annotations.device
        batch_size = annotations.shape[0]
        cls_preds, reg_preds, obj_preds = preds

        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in cls_preds]
        one_image_grid_center_strides = self.grid_strides(feature_size)
        batch_grid_center_strides = [
            torch.tensor(per_level_grid_center_strides).unsqueeze(0).repeat(
                batch_size, 1, 1, 1).to(device)
            for per_level_grid_center_strides in one_image_grid_center_strides
        ]

        all_obj_preds, all_cls_preds, all_reg_preds, all_grid_center_strides, batch_targets = self.get_batch_position_annotations(
            obj_preds, cls_preds, reg_preds, batch_grid_center_strides,
            annotations)

        all_obj_preds = torch.clamp(all_obj_preds, min=1e-4, max=1. - 1e-4)
        all_cls_preds = torch.clamp(all_cls_preds, min=1e-4, max=1. - 1e-4)

        all_obj_preds = all_obj_preds.view(-1, all_obj_preds.shape[-1])
        all_cls_preds = all_cls_preds.view(-1, all_cls_preds.shape[-1])
        all_reg_preds = all_reg_preds.view(-1, all_reg_preds.shape[-1])
        all_grid_center_strides = all_grid_center_strides.view(
            -1, all_grid_center_strides.shape[-1])
        batch_targets = batch_targets.view(-1, batch_targets.shape[-1])

        obj_loss, reg_loss, cls_loss = self.compute_per_batch_loss(
            all_obj_preds, all_cls_preds, all_reg_preds,
            all_grid_center_strides, batch_targets)

        obj_loss = self.obj_loss_weight * obj_loss
        reg_loss = self.box_loss_weight * reg_loss
        cls_loss = self.cls_loss_weight * cls_loss

        loss_dict = {
            'obj_loss': obj_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict

    def compute_per_batch_loss(self, all_obj_preds, all_cls_preds,
                               all_reg_preds, all_grid_center_strides,
                               batch_targets):
        '''
        compute per level batch loss,include obj loss(bce loss)、reg loss(CIoU loss)、cls loss(bce loss)
        all_obj_preds:[batch_size*grid_nums,1]
        all_cls_preds:[batch_size*grid_nums,4]
        all_reg_preds:[batch_size*grid_nums,num_classes]
        all_grid_center_strides:[batch_size*grid_nums,3]
        batch_targets:[batch_size*grid_nums,6]
        '''
        device = batch_targets.device
        positive_grid_nums = batch_targets[batch_targets[:, 5] > 0].shape[0]

        # if no positive points
        if positive_grid_nums == 0:
            return torch.tensor(0.).to(device), torch.tensor(0.).to(
                device), torch.tensor(0.).to(device)

        obj_preds = all_obj_preds[:, 0]
        reg_preds = all_reg_preds[batch_targets[:, 5] > 0]
        cls_preds = all_cls_preds[batch_targets[:, 5] > 0]

        obj_targets = batch_targets[:, 0]
        reg_targets = batch_targets[batch_targets[:, 5] > 0][:, 1:5]
        cls_targets = batch_targets[batch_targets[:, 5] > 0][:, 5]
        all_grid_center_strides = all_grid_center_strides[batch_targets[:,
                                                                        5] > 0]

        # compute obj loss
        obj_loss = -(obj_targets * torch.log(obj_preds) +
                     (1. - obj_targets) * torch.log(1. - obj_preds))
        obj_loss = obj_loss.sum() / positive_grid_nums

        # compute reg loss
        reg_preds = self.snap_ltrb_to_x1y1x2y2(reg_preds,
                                               all_grid_center_strides)
        box_loss_iou_type = 'EIoU' if self.box_loss_iou_type == 'Focal_EIoU' else self.box_loss_iou_type
        ious = self.iou_function(reg_preds,
                                 reg_targets,
                                 iou_type=box_loss_iou_type,
                                 box_type='xywh')
        reg_loss = 1 - ious
        if self.box_loss_iou_type == 'Focal_EIoU':
            gamma_ious = self.iou_function(reg_preds,
                                           reg_targets,
                                           iou_type='IoU',
                                           box_type='xyxy')
            gamma_ious = torch.pow(gamma_ious, self.focal_eiou_gamma)
            reg_loss = gamma_ious * reg_loss
        reg_loss = reg_loss.mean()

        # compute cls loss
        cls_ground_truth = F.one_hot(cls_targets.long(),
                                     num_classes=cls_preds.shape[1] + 1)
        cls_ground_truth = (cls_ground_truth[:, 1:]).float()
        cls_loss = -(cls_ground_truth * torch.log(cls_preds) +
                     (1. - cls_ground_truth) * torch.log(1. - cls_preds))
        cls_loss = cls_loss.mean()

        return obj_loss, reg_loss, cls_loss

    def get_batch_position_annotations(self, obj_preds, cls_preds, reg_preds,
                                       batch_grid_center_strides, annotations):
        '''
        Assign a ground truth target for each position on feature map
        '''
        device = annotations.device

        all_obj_preds, all_cls_preds, all_reg_preds,all_grid_center_strides = [], [], [], []
        for obj_pred, cls_pred, reg_pred, per_level_grid_center_strides in zip(
                obj_preds, cls_preds, reg_preds, batch_grid_center_strides):
            obj_pred = obj_pred.view(obj_pred.shape[0], -1, obj_pred.shape[-1])
            cls_pred = cls_pred.view(cls_pred.shape[0], -1, cls_pred.shape[-1])
            reg_pred = reg_pred.view(reg_pred.shape[0], -1, reg_pred.shape[-1])
            per_level_grid_center_strides = per_level_grid_center_strides.view(
                per_level_grid_center_strides.shape[0], -1,
                per_level_grid_center_strides.shape[-1])

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_grid_center_strides.append(per_level_grid_center_strides)

        all_obj_preds = torch.cat(all_obj_preds, dim=1)
        all_cls_preds = torch.cat(all_cls_preds, dim=1)
        all_reg_preds = torch.cat(all_reg_preds, dim=1)
        all_grid_center_strides = torch.cat(all_grid_center_strides, dim=1)

        batch_targets = []
        for per_image_obj_preds, per_image_cls_preds, per_image_reg_preds, per_image_grid_center_strides, per_image_annotations in zip(
                all_obj_preds, all_cls_preds, all_reg_preds,
                all_grid_center_strides, annotations):
            per_image_annotations = per_image_annotations[
                per_image_annotations[:, 4] >= 0]
            grids_num = per_image_grid_center_strides.shape[0]

            # obj target init value=0
            per_image_obj_target = torch.zeros([grids_num, 1],
                                               dtype=torch.float32,
                                               device=device)
            # reg target init value=0
            per_image_reg_target = torch.zeros([grids_num, 4],
                                               dtype=torch.float32,
                                               device=device)
            # cls target init value=-1
            per_image_cls_target = torch.ones(
                [grids_num, 1], dtype=torch.float32, device=device) * (-1)
            # 6:[obj_target,scale_offset_x,scale_offset_y,tw,th,class_target]
            per_image_targets = torch.cat([
                per_image_obj_target, per_image_reg_target,
                per_image_cls_target
            ],
                                          dim=-1)

            if per_image_annotations.shape[0] > 0:
                annotaion_num = per_image_annotations.shape[0]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                per_image_gt_classes = per_image_annotations[:, 4]
                # each grid center,such as 0.5
                per_image_grid_centers = per_image_grid_center_strides[:, 0:2]
                per_image_strides = per_image_grid_center_strides[:, 2]

                per_image_grid_centers = per_image_grid_centers * per_image_strides.unsqueeze(
                    -1)
                per_image_grid_centers = per_image_grid_centers.unsqueeze(
                    1).repeat(1, annotaion_num, 1)
                candidates = torch.zeros([grids_num, annotaion_num, 4],
                                         dtype=torch.float32,
                                         device=device)
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)

                # center sample
                candidates_center = (candidates[:, :, 2:4] +
                                     candidates[:, :, 0:2]) / 2
                judge_distance = per_image_strides * self.center_sample_radius
                judge_distance = judge_distance.unsqueeze(-1).unsqueeze(
                    -1).repeat(1, annotaion_num, 1)

                # compute each point to each gt box ltrb distance
                points_to_gt_box_lt = per_image_grid_centers[:, :, 0:
                                                             2] - candidates[:, :,
                                                                             0:
                                                                             2]
                points_to_gt_box_rb = candidates[:, :, 2:
                                                 4] - per_image_grid_centers[:, :,
                                                                             0:
                                                                             2]
                points_to_gt_box_ltrb = torch.cat(
                    [points_to_gt_box_lt, points_to_gt_box_rb], dim=-1)
                points_to_gt_box_ltrb_min_value, _ = points_to_gt_box_ltrb.min(
                    axis=-1)
                points_in_gt_box_flag = (points_to_gt_box_ltrb_min_value > 0)
                points_in_all_gt_box_flag = (
                    points_to_gt_box_ltrb_min_value.sum(dim=1) > 0)

                # center sample
                compute_distance = torch.sqrt(
                    (per_image_grid_centers[:, :, 0] -
                     candidates_center[:, :, 0])**2 +
                    (per_image_grid_centers[:, :, 1] -
                     candidates_center[:, :, 1])**2)

                points_in_gt_box_center_flag = (
                    (compute_distance < judge_distance.squeeze(-1)) > 0)
                points_in_all_gt_box_center_flag = (
                    (compute_distance < judge_distance.squeeze(-1)).sum(dim=1)
                    > 0)
                points_in_gt_box_or_center_flag = (
                    points_in_all_gt_box_flag
                    | points_in_all_gt_box_center_flag)
                points_in_gt_box_and_center_flag = (
                    points_in_gt_box_flag[points_in_gt_box_or_center_flag, :]
                    & points_in_gt_box_center_flag[
                        points_in_gt_box_or_center_flag, :])

                if points_in_gt_box_or_center_flag.sum() > 0:
                    cost_per_image_reg_preds = per_image_reg_preds[
                        points_in_gt_box_or_center_flag]
                    cost_per_image_grid_center_strides = per_image_grid_center_strides[
                        points_in_gt_box_or_center_flag]

                    cost_per_image_pred_bboxes = self.snap_ltrb_to_x1y1x2y2(
                        cost_per_image_reg_preds,
                        cost_per_image_grid_center_strides)
                    cost_ious = self.iou_function(
                        cost_per_image_pred_bboxes.unsqueeze(1),
                        per_image_gt_bboxes.unsqueeze(0),
                        iou_type='IoU',
                        box_type='xyxy')
                    cost_ious = -torch.log(cost_ious + 1e-4)

                    cost_per_image_cls_preds = per_image_cls_preds[
                        points_in_gt_box_or_center_flag]
                    cost_per_image_obj_preds = per_image_obj_preds[
                        points_in_gt_box_or_center_flag]
                    cost_per_image_cls_preds = torch.sqrt(
                        cost_per_image_cls_preds * cost_per_image_obj_preds)
                    cost_per_image_cls_preds = cost_per_image_cls_preds.unsqueeze(
                        1).repeat(1, annotaion_num, 1)

                    cost_per_image_gt_classes = F.one_hot(
                        per_image_gt_classes.to(torch.int64),
                        cost_per_image_cls_preds.shape[-1]).float().unsqueeze(
                            0).repeat(cost_per_image_cls_preds.shape[0], 1, 1)
                    cost_cls = F.binary_cross_entropy(
                        cost_per_image_cls_preds,
                        cost_per_image_gt_classes,
                        reduction="none").sum(dim=-1)

                    total_costs = 1.0 * cost_cls + 3.0 * cost_ious + 100000.0 * (
                        ~points_in_gt_box_and_center_flag).float()

                    matching_matrix, match_gt_box_idxs = self.dynamic_k_matching(
                        cost_ious, total_costs)

                    if matching_matrix.sum() > 0:
                        cost_per_image_targets = per_image_targets[
                            points_in_gt_box_or_center_flag]
                        # 0 or 1
                        cost_per_image_targets[matching_matrix, 0] = 1
                        # 1 to 80 for coco dataset
                        cost_per_image_targets[
                            matching_matrix,
                            5] = per_image_gt_classes[match_gt_box_idxs] + 1
                        # [x_min,y_min,x_max,y_max]
                        cost_per_image_targets[
                            matching_matrix,
                            1:5] = per_image_gt_bboxes[match_gt_box_idxs]
                        per_image_targets[
                            points_in_gt_box_or_center_flag] = cost_per_image_targets

            per_image_targets = per_image_targets.unsqueeze(0)
            batch_targets.append(per_image_targets)

        batch_targets = torch.cat(batch_targets, dim=0)

        # batch_targets shape:[batch_size, grids_num, 6],6:[obj_target,scale_offset_x,scale_offset_y,tw,th,class_target]
        return all_obj_preds, all_cls_preds, all_reg_preds, all_grid_center_strides, batch_targets

    def dynamic_k_matching(self, ious, total_costs):
        ious, total_costs = ious.permute(1, 0), total_costs.permute(1, 0)

        device = ious.device
        annotation_nums, point_nums = ious.shape[0], ious.shape[1]
        matching_matrix = torch.zeros([annotation_nums, point_nums],
                                      dtype=torch.uint8,
                                      device=device)

        # 选取iou最大的max_candidate_k个点,然后求和，判断应该有多少点用于该框预测
        if point_nums <= 10:
            max_candidate_k = point_nums
        else:
            max_candidate_k = min(10, point_nums)
        per_image_topk_ious_for_all_gt, _ = torch.topk(ious,
                                                       max_candidate_k,
                                                       dim=1)
        per_image_dynamic_k_num_for_all_gt = (torch.clamp(
            per_image_topk_ious_for_all_gt.sum(dim=1).int(), min=1)).tolist()

        for gt_idx in range(total_costs.shape[0]):
            # 给每个真实框选取最小的动态k个点
            k = per_image_dynamic_k_num_for_all_gt[gt_idx]
            if total_costs[gt_idx].shape[
                    -1] < per_image_dynamic_k_num_for_all_gt[gt_idx]:
                k = total_costs[gt_idx].shape[-1]

            _, pos_idx = torch.topk(total_costs[gt_idx], k=k, largest=False)

            matching_matrix[gt_idx][pos_idx] = 1

        anchor_matching_gt = matching_matrix.sum(dim=0)

        if (anchor_matching_gt > 1).sum() > 0:
            # 当某一个特征点指向多个真实框的时候,选取cost最小的真实框。
            _, cost_argmin = torch.min(total_costs[:, anchor_matching_gt > 1],
                                       dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        match_gt_box_idxs = matching_matrix[:, (
            matching_matrix.sum(dim=0) > 0)].argmax(dim=0)
        matching_matrix = (matching_matrix.sum(dim=0) > 0)

        return matching_matrix, match_gt_box_idxs

    def snap_ltrb_to_x1y1x2y2(self, per_image_reg_preds,
                              per_image_grid_center_strides):
        '''
        snap per image reg preds to per image pred bboxes
        per_image_reg_preds:[point_nums,4],4:[l,t,r,b]
        per_image_grid_center_strides:[point_nums,3],3:[scale_grid_x_center,scale_grid_y_center,stride]
        '''
        per_image_reg_preds = torch.exp(per_image_reg_preds)
        per_image_grid_centers = per_image_grid_center_strides[:, 0:2]
        per_image_strides = per_image_grid_center_strides[:, 2].unsqueeze(-1)

        per_image_pred_bboxes_xy_min = (
            per_image_grid_centers -
            per_image_reg_preds[:, 0:2]) * per_image_strides
        per_image_pred_bboxes_xy_max = (
            per_image_grid_centers +
            per_image_reg_preds[:, 2:4]) * per_image_strides
        per_image_pred_bboxes = torch.cat(
            [per_image_pred_bboxes_xy_min, per_image_pred_bboxes_xy_max],
            dim=1)

        # per_image_pred_bboxes shape:[points_num,4]
        return per_image_pred_bboxes


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

    from tools.path import COCO2017_path

    from simpleAICV.detection.datasets.cocodataset import CocoDetection
    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize, DetectionCollater

    cocodataset = CocoDetection(
        COCO2017_path,
        set_name='train2017',
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            # RandomCrop(prob=0.5),
            # RandomTranslate(prob=0.5),
            YoloStyleResize(resize=640,
                            divisor=32,
                            stride=32,
                            multi_scale=False,
                            multi_scale_range=[0.5, 1.0]),
            # RetinaStyleResize(resize=400,
            #                   divisor=32,
            #                   stride=32,
            #                   multi_scale=False,
            #                   multi_scale_range=[0.8, 1.0]),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = DetectionCollater()
    train_loader = DataLoader(cocodataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.detection.models.retinanet import resnet50_retinanet
    net = resnet50_retinanet()
    loss = RetinaLoss(areas=[[32, 32], [64, 64], [128, 128], [256, 256],
                             [512, 512]],
                      ratios=[0.5, 1, 2],
                      scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
                      strides=[8, 16, 32, 64, 128],
                      alpha=0.25,
                      gamma=2,
                      beta=1.0 / 9.0,
                      focal_eiou_gamma=0.5,
                      cls_loss_weight=1.,
                      box_loss_weight=1.,
                      box_loss_type='CIoU')

    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
        preds = net(images)
        for pred in preds:
            for per_level_pred in pred:
                print('2222', per_level_pred.shape)
        loss_dict = loss(preds, annots)
        print('3333', loss_dict)
        break

    from simpleAICV.detection.models.fcos import resnet50_fcos
    net = resnet50_fcos()
    loss = FCOSLoss(strides=[8, 16, 32, 64, 128],
                    mi=[[-1, 64], [64, 128], [128, 256], [256, 512],
                        [512, 100000000]],
                    alpha=0.25,
                    gamma=2.,
                    cls_loss_weight=1.,
                    box_loss_weight=1.,
                    center_ness_loss_weight=1.,
                    box_loss_iou_type='CIoU',
                    center_sample_radius=1.5,
                    use_center_sample=True)

    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
        preds = net(images)
        for pred in preds:
            for per_level_pred in pred:
                print('2222', per_level_pred.shape)
        loss_dict = loss(preds, annots)
        print('3333', loss_dict)
        break

    from simpleAICV.detection.models.centernet import resnet18_centernet
    net = resnet18_centernet()
    loss = CenterNetLoss(alpha=2.,
                         beta=4.,
                         heatmap_loss_weight=1.0,
                         offset_loss_weight=1.0,
                         wh_loss_weight=0.1,
                         min_overlap=0.7,
                         max_object_num=100)

    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
        preds = net(images)
        print('2222', preds[0].shape, preds[1].shape, preds[2].shape)
        loss_dict = loss(preds, annots)
        print('3333', loss_dict)
        break

    from simpleAICV.detection.models.ttfnet import resnet18_ttfnet
    net = resnet18_ttfnet()
    loss = TTFNetLoss(alpha=2.0,
                      beta=4.0,
                      stride=4,
                      heatmap_loss_weight=1.0,
                      box_loss_weight=5.0,
                      box_loss_iou_type='CIoU',
                      gaussian_alpha=0.54,
                      gaussian_beta=0.54)

    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
        preds = net(images)
        print('2222', preds[0].shape, preds[1].shape)
        loss_dict = loss(preds, annots)
        print('3333', loss_dict)
        break

    from simpleAICV.detection.models.yolox import yoloxm
    net = yoloxm()
    loss = YoloxLoss(strides=[8, 16, 32],
                     obj_loss_weight=1.0,
                     box_loss_weight=5.0,
                     cls_loss_weight=1.0,
                     box_loss_iou_type='CIoU',
                     center_sample_radius=2.5)

    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
        preds = net(images)
        for pred in preds:
            for per_level_pred in pred:
                print('2222', per_level_pred.shape)
        loss_dict = loss(preds, annots)
        print('3333', loss_dict)
        break