import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import math
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.detection.models.anchor import RetinaAnchors, FCOSPositions, TTFNetPositions

__all__ = [
    'RetinaLoss',
    'FCOSLoss',
    'CenterNetLoss',
    'TTFNetLoss',
    'DETRLoss',
    'DINODETRLoss',
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


class DETRLoss(nn.Module):

    def __init__(self,
                 cls_match_cost=1.0,
                 box_match_cost=5.0,
                 giou_match_cost=2.0,
                 cls_loss_weight=1.0,
                 box_l1_loss_weight=5.0,
                 iou_loss_weight=2.0,
                 no_object_cls_weight=0.1,
                 num_classes=80):
        super(DETRLoss, self).__init__()
        self.cls_match_cost = cls_match_cost
        self.box_match_cost = box_match_cost
        self.giou_match_cost = giou_match_cost

        # cls loss
        self.cls_loss_weight = cls_loss_weight
        # box l1 loss
        self.box_l1_loss_weight = box_l1_loss_weight
        # iou box loss
        self.iou_loss_weight = iou_loss_weight
        self.no_object_cls_weight = no_object_cls_weight
        self.num_classes = num_classes

        assert self.cls_match_cost != 0 or self.box_match_cost != 0 or self.giou_match_cost != 0, "all costs cant be 0"

    def forward(self, preds, annotations):
        cls_preds, reg_preds = preds
        reg_preds = torch.clamp(reg_preds, min=1e-4, max=1. - 1e-4)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.get_matched_pred_target_idxs(cls_preds[-1, :, :, :],
                                                    reg_preds[-1, :, :, :],
                                                    annotations)
        loss_dict = {}
        for idx, (per_level_cls_preds,
                  per_level_reg_preds) in enumerate(zip(cls_preds, reg_preds)):
            per_level_cls_loss = self.compute_batch_cls_loss(
                per_level_cls_preds, annotations, indices)
            per_level_box_l1_loss, per_level_box_iou_loss = self.compute_batch_l1_iou_loss(
                per_level_reg_preds, annotations, indices)

            per_level_cls_loss = self.cls_loss_weight * per_level_cls_loss
            per_level_box_l1_loss = self.box_l1_loss_weight * per_level_box_l1_loss
            per_level_box_iou_loss = self.iou_loss_weight * per_level_box_iou_loss

            loss_dict[f'layer_{idx}_cls_loss'] = per_level_cls_loss
            loss_dict[f'layer_{idx}_box_l1_loss'] = per_level_box_l1_loss
            loss_dict[f'layer_{idx}_box_iou_loss'] = per_level_box_iou_loss

        return loss_dict

    def compute_batch_cls_loss(self, cls_preds, annotations, indices):
        batch_size, query_nums = cls_preds.shape[0], cls_preds.shape[1]
        device = cls_preds.device

        idx = self.get_src_permutation_idx(indices)
        filter_targets = [
            per_image_targets[per_image_targets[:, 4] >= 0][:, 4]
            for per_image_targets in annotations
        ]
        target_classes = torch.cat([
            per_image_targets[j]
            for per_image_targets, (_, j) in zip(filter_targets, indices)
        ])

        loss_ground_truth = torch.full((batch_size, query_nums),
                                       self.num_classes).long().to(device)
        loss_ground_truth[idx] = target_classes.long()

        empty_weight = torch.ones(self.num_classes + 1).to(device)
        empty_weight[-1] = self.no_object_cls_weight

        # src_logits:[4, 100, 81] target_classes:[4, 100] empty_weight:81
        batch_cls_loss = F.cross_entropy(cls_preds.transpose(1, 2),
                                         loss_ground_truth, empty_weight)

        return batch_cls_loss

    def compute_batch_l1_iou_loss(self, reg_preds, annotations, indices):
        """
           The target boxes are expected in format [batch_target_boxes_nums, 4] (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self.get_src_permutation_idx(indices)
        reg_preds = reg_preds[idx]

        filter_targets = [
            per_image_targets[per_image_targets[:, 4] >= 0][:, 0:4]
            for per_image_targets in annotations
        ]
        filter_targets_num = sum([
            per_image_targets.shape[0] for per_image_targets in filter_targets
        ])
        target_boxes = torch.cat([
            per_image_targets[j]
            for per_image_targets, (_, j) in zip(filter_targets, indices)
        ],
                                 dim=0)

        box_l1_loss = F.l1_loss(reg_preds, target_boxes, reduction='none')
        box_l1_loss = box_l1_loss.sum() / filter_targets_num

        reg_preds = self.transform_cxcywh_box_to_xyxy_box(reg_preds)
        target_boxes = self.transform_cxcywh_box_to_xyxy_box(target_boxes)
        box_iou_loss = 1 - torch.diag(
            self.compute_box_giou(reg_preds, target_boxes))
        box_iou_loss = box_iou_loss.sum() / filter_targets_num

        return box_l1_loss, box_iou_loss

    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def transform_cxcywh_box_to_xyxy_box(self, boxes):
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:,
                                                                   2], boxes[:,
                                                                             3]
        boxes = torch.stack([(x_center - 0.5 * w), (y_center - 0.5 * h),
                             (x_center + 0.5 * w), (y_center + 0.5 * h)],
                            dim=1)

        return boxes

    def compute_box_giou(self, boxes1, boxes2):
        """
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        area1 = torch.clamp(area1, min=0)
        area2 = torch.clamp(area2, min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        inter = torch.clamp(inter, min=0)

        union = area1[:, None] + area2 - inter
        union = torch.clamp(union, min=1e-4)

        iou = inter / union

        enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)  # [N,M,2]
        enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]
        enclose_area = torch.clamp(enclose_area, min=1e-4)

        return iou - (enclose_area - union) / enclose_area

    @torch.no_grad()
    def get_matched_pred_target_idxs(self, cls_preds, reg_preds, annotations):
        batch_size, query_nums = cls_preds.shape[0], cls_preds.shape[1]

        # [b*query_nums,num_classes]
        cls_preds = cls_preds.flatten(0, 1)
        cls_preds = F.softmax(cls_preds, dim=-1)
        cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)

        # [b*query_nums,4]
        reg_preds = reg_preds.flatten(0, 1)

        batch_gt_boxes_annot = []
        per_image_gt_boxes_num = []
        for per_image_boxes_annot in annotations:
            per_image_boxes_annot = per_image_boxes_annot[
                per_image_boxes_annot[:, 4] >= 0]
            batch_gt_boxes_annot.append(per_image_boxes_annot)
            per_image_gt_boxes_num.append(per_image_boxes_annot.shape[0])
        batch_gt_boxes_annot = torch.cat(batch_gt_boxes_annot, dim=0)
        batch_gt_boxes = batch_gt_boxes_annot[:, 0:4]
        batch_gt_boxes_label = batch_gt_boxes_annot[:, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cls_cost = -cls_preds[:, batch_gt_boxes_label.long()]

        # Compute the L1 cost between boxes
        box_cost = torch.cdist(reg_preds, batch_gt_boxes, p=1)

        reg_preds = self.transform_cxcywh_box_to_xyxy_box(reg_preds)
        batch_gt_boxes = self.transform_cxcywh_box_to_xyxy_box(batch_gt_boxes)

        # Compute the giou cost betwen boxes
        giou_cost = -self.compute_box_giou(reg_preds, batch_gt_boxes)
        # Final cost matrix
        total_cost = self.cls_match_cost * cls_cost + self.box_match_cost * box_cost + self.giou_match_cost * giou_cost
        total_cost = total_cost.view(batch_size, query_nums, -1)

        # for per image,assign one pred box to one GT box
        indices = []
        for idx, per_image_cost in enumerate(
                total_cost.split(per_image_gt_boxes_num, -1)):
            indices.append(
                self.linear_sum_assignment_with_inf(
                    per_image_cost[idx].cpu().numpy()))

        indices = [(torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return indices

    def linear_sum_assignment_with_inf(self, cost_matrix):
        cost_matrix = np.asarray(cost_matrix)

        nan = np.isnan(cost_matrix).any()
        if nan:
            cost_matrix[np.isnan(cost_matrix)] = 1e5

        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")

        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            min_values = values.min()
            max_values = values.max()
            m = min(cost_matrix.shape)

            positive = m * (max_values - min_values + np.abs(max_values) +
                            np.abs(min_values) + 1)
            if max_inf:
                place_holder = (max_values + (m - 1) *
                                (max_values - min_values)) + positive
            elif min_inf:
                place_holder = (min_values + (m - 1) *
                                (min_values - max_values)) - positive

            cost_matrix[np.isinf(cost_matrix)] = place_holder

        results = scipy.optimize.linear_sum_assignment(cost_matrix)

        return results


class DINODETRLoss(nn.Module):

    def __init__(self,
                 cls_match_cost=2.0,
                 box_match_cost=5.0,
                 giou_match_cost=2.0,
                 cls_loss_weight=1.0,
                 box_l1_loss_weight=5.0,
                 iou_loss_weight=2.0,
                 alpha=0.25,
                 gamma=2.0,
                 num_classes=80):
        super(DINODETRLoss, self).__init__()
        self.cls_match_cost = cls_match_cost
        self.box_match_cost = box_match_cost
        self.giou_match_cost = giou_match_cost

        # cls loss
        self.cls_loss_weight = cls_loss_weight
        # box l1 loss
        self.box_l1_loss_weight = box_l1_loss_weight
        # iou box loss
        self.iou_loss_weight = iou_loss_weight
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

        assert self.cls_match_cost != 0 or self.box_match_cost != 0 or self.giou_match_cost != 0, "all costs cant be 0"

    def forward(self, preds, annotations):
        last_cls_preds, last_reg_preds = preds['pred_logits'], preds[
            'pred_boxes']

        # Retrieve the matching between the outputs of the last layer and the targets
        last_indices = self.get_matched_pred_target_idxs(
            last_cls_preds, last_reg_preds, annotations)

        # Compute all the requested loss_dict
        loss_dict = {}

        last_cls_loss = self.compute_batch_cls_loss(last_cls_preds,
                                                    annotations, last_indices)
        last_box_l1_loss, last_box_iou_loss = self.compute_batch_l1_iou_loss(
            last_reg_preds, annotations, last_indices)

        last_cls_loss = self.cls_loss_weight * last_cls_loss
        last_box_l1_loss = self.box_l1_loss_weight * last_box_l1_loss
        last_box_iou_loss = self.iou_loss_weight * last_box_iou_loss

        loss_dict.update({
            'cls_loss': last_cls_loss,
            'box_l1_loss': last_box_l1_loss,
            'box_iou_loss': last_box_iou_loss,
        })

        # prepare for dn loss
        dn_meta = preds['dn_meta']

        if dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(
                dn_meta)

            filter_targets = [
                per_image_targets[per_image_targets[:, 4] >= 0][:, 4]
                for per_image_targets in annotations
            ]

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(filter_targets)):
                if len(filter_targets[i]) > 0:
                    t = torch.arange(0, len(filter_targets[i])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) *
                                  single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            last_dn_cls_preds, last_dn_reg_preds = output_known_lbs_bboxes[
                'pred_logits'], output_known_lbs_bboxes['pred_boxes']

            last_dn_cls_loss = self.compute_batch_cls_loss(
                last_dn_cls_preds, annotations, dn_pos_idx)
            last_dn_box_l1_loss, last_dn_box_iou_loss = self.compute_batch_l1_iou_loss(
                last_dn_reg_preds, annotations, dn_pos_idx)

            last_dn_cls_loss = last_dn_cls_loss / scalar
            last_dn_box_l1_loss = last_dn_box_l1_loss / scalar
            last_dn_box_iou_loss = last_dn_box_iou_loss / scalar

            last_dn_cls_loss = self.cls_loss_weight * last_dn_cls_loss
            last_dn_box_l1_loss = self.box_l1_loss_weight * last_dn_box_l1_loss
            last_dn_box_iou_loss = self.iou_loss_weight * last_dn_box_iou_loss

            loss_dict.update({
                'cls_loss_dn': last_dn_cls_loss,
                'box_l1_loss_dn': last_dn_box_l1_loss,
                'box_iou_loss_dn': last_dn_box_iou_loss,
            })

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in preds:
            for idx, per_level_aux_outputs in enumerate(preds['aux_outputs']):
                per_level_aux_cls_preds, per_level_aux_reg_preds = per_level_aux_outputs[
                    'pred_logits'], per_level_aux_outputs['pred_boxes']

                per_level_indices = self.get_matched_pred_target_idxs(
                    per_level_aux_cls_preds, per_level_aux_reg_preds,
                    annotations)

                per_level_cls_loss = self.compute_batch_cls_loss(
                    per_level_aux_cls_preds, annotations, per_level_indices)
                per_level_box_l1_loss, per_level_box_iou_loss = self.compute_batch_l1_iou_loss(
                    per_level_aux_reg_preds, annotations, per_level_indices)

                per_level_cls_loss = self.cls_loss_weight * per_level_cls_loss
                per_level_box_l1_loss = self.box_l1_loss_weight * per_level_box_l1_loss
                per_level_box_iou_loss = self.iou_loss_weight * per_level_box_iou_loss

                loss_dict.update({
                    f'cls_loss_aux_layer_{idx}':
                    per_level_cls_loss,
                    f'box_l1_loss_aux_layer_{idx}':
                    per_level_box_l1_loss,
                    f'box_iou_loss_aux_layer_{idx}':
                    per_level_box_iou_loss,
                })

                if dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][
                        idx]
                    per_level_dn_cls_preds, per_level_dn_reg_preds = aux_outputs_known[
                        'pred_logits'], aux_outputs_known['pred_boxes']

                    per_level_dn_cls_loss = self.compute_batch_cls_loss(
                        per_level_dn_cls_preds, annotations, dn_pos_idx)
                    per_level_dn_box_l1_loss, per_level_dn_box_iou_loss = self.compute_batch_l1_iou_loss(
                        per_level_dn_reg_preds, annotations, dn_pos_idx)

                    per_level_dn_cls_loss = per_level_dn_cls_loss / scalar
                    per_level_dn_box_l1_loss = per_level_dn_box_l1_loss / scalar
                    per_level_dn_box_iou_loss = per_level_dn_box_iou_loss / scalar

                    per_level_dn_cls_loss = self.cls_loss_weight * per_level_dn_cls_loss
                    per_level_dn_box_l1_loss = self.box_l1_loss_weight * per_level_dn_box_l1_loss
                    per_level_dn_box_iou_loss = self.iou_loss_weight * per_level_dn_box_iou_loss

                    loss_dict.update({
                        f'cls_loss_dn_aux_layer_{idx}':
                        per_level_dn_cls_loss,
                        f'box_l1_loss_dn_aux_layer_{idx}':
                        per_level_dn_box_l1_loss,
                        f'box_iou_loss_dn_aux_layer_{idx}':
                        per_level_dn_box_iou_loss,
                    })

        # interm_outputs loss
        if 'interm_outputs' in preds:
            interm_outputs = preds['interm_outputs']
            interm_cls_preds, interm_reg_preds = interm_outputs[
                'pred_logits'], interm_outputs['pred_boxes']

            interm_indices = self.get_matched_pred_target_idxs(
                interm_cls_preds, interm_reg_preds, annotations)

            interm_cls_loss = self.compute_batch_cls_loss(
                interm_cls_preds, annotations, interm_indices)
            interm_box_l1_loss, interm_box_iou_loss = self.compute_batch_l1_iou_loss(
                interm_reg_preds, annotations, interm_indices)

            interm_cls_loss = self.cls_loss_weight * interm_cls_loss
            interm_box_l1_loss = self.box_l1_loss_weight * interm_box_l1_loss
            interm_box_iou_loss = self.iou_loss_weight * interm_box_iou_loss

            loss_dict.update({
                'cls_loss_interm': interm_cls_loss,
                'box_l1_loss_interm': interm_box_l1_loss,
                'box_iou_loss_interm': interm_box_iou_loss,
            })

        return loss_dict

    def compute_batch_cls_loss(self, cls_preds, annotations, indices):
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)
        batch_size, query_nums = cls_preds.shape[0], cls_preds.shape[1]
        device = cls_preds.device

        idx = self.get_src_permutation_idx(indices)
        filter_targets = [
            per_image_targets[per_image_targets[:, 4] >= 0][:, 4]
            for per_image_targets in annotations
        ]
        filter_targets_num = sum([
            per_image_targets.shape[0] for per_image_targets in filter_targets
        ])
        if filter_targets_num == 0:
            return torch.tensor(0.).to(device)

        target_classes = torch.cat([
            per_image_targets[j]
            for per_image_targets, (_, j) in zip(filter_targets, indices)
        ])
        loss_ground_truth = torch.full((batch_size, query_nums),
                                       self.num_classes).long().to(device)
        loss_ground_truth[idx] = target_classes.long()

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(loss_ground_truth.long(),
                                      num_classes=self.num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, :, :-1]
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
        batch_focal_loss = batch_focal_loss / filter_targets_num

        return batch_focal_loss

    def compute_batch_l1_iou_loss(self, reg_preds, annotations, indices):
        """
           The target boxes are expected in format [batch_target_boxes_nums, 4] (center_x, center_y, w, h), normalized by the image size.
        """
        reg_preds = torch.clamp(reg_preds, min=1e-4, max=1. - 1e-4)

        idx = self.get_src_permutation_idx(indices)
        reg_preds = reg_preds[idx]

        filter_targets = [
            per_image_targets[per_image_targets[:, 4] >= 0][:, 0:4]
            for per_image_targets in annotations
        ]
        filter_targets_num = sum([
            per_image_targets.shape[0] for per_image_targets in filter_targets
        ])
        target_boxes = torch.cat([
            per_image_targets[j]
            for per_image_targets, (_, j) in zip(filter_targets, indices)
        ],
                                 dim=0)

        box_l1_loss = F.l1_loss(reg_preds, target_boxes, reduction='none')
        box_l1_loss = box_l1_loss.sum() / filter_targets_num

        reg_preds = self.transform_cxcywh_box_to_xyxy_box(reg_preds)
        target_boxes = self.transform_cxcywh_box_to_xyxy_box(target_boxes)
        box_iou_loss = 1 - torch.diag(
            self.compute_box_giou(reg_preds, target_boxes))
        box_iou_loss = box_iou_loss.sum() / filter_targets_num

        return box_l1_loss, box_iou_loss

    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def transform_cxcywh_box_to_xyxy_box(self, boxes):
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:,
                                                                   2], boxes[:,
                                                                             3]
        boxes = torch.stack([(x_center - 0.5 * w), (y_center - 0.5 * h),
                             (x_center + 0.5 * w), (y_center + 0.5 * h)],
                            dim=1)

        return boxes

    def compute_box_giou(self, boxes1, boxes2):
        """
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        area1 = torch.clamp(area1, min=0)
        area2 = torch.clamp(area2, min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        inter = torch.clamp(inter, min=0)

        union = area1[:, None] + area2 - inter
        union = torch.clamp(union, min=1e-4)

        iou = inter / union

        enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)  # [N,M,2]
        enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]
        enclose_area = torch.clamp(enclose_area, min=1e-4)

        return iou - (enclose_area - union) / enclose_area

    @torch.no_grad()
    def get_matched_pred_target_idxs(self, cls_preds, reg_preds, annotations):
        batch_size, query_nums = cls_preds.shape[0], cls_preds.shape[1]

        # [b*query_nums,num_classes]
        cls_preds = cls_preds.flatten(0, 1)
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)

        # [b*query_nums,4]
        reg_preds = reg_preds.flatten(0, 1)
        reg_preds = torch.clamp(reg_preds, min=1e-4, max=1. - 1e-4)

        batch_gt_boxes_annot = []
        per_image_gt_boxes_num = []
        for per_image_boxes_annot in annotations:
            per_image_boxes_annot = per_image_boxes_annot[
                per_image_boxes_annot[:, 4] >= 0]
            batch_gt_boxes_annot.append(per_image_boxes_annot)
            per_image_gt_boxes_num.append(per_image_boxes_annot.shape[0])
        batch_gt_boxes_annot = torch.cat(batch_gt_boxes_annot, dim=0)
        batch_gt_boxes = batch_gt_boxes_annot[:, 0:4]
        batch_gt_boxes_label = batch_gt_boxes_annot[:, 4]

        # Compute the classification cost.
        neg_cls_cost = (1 - self.alpha) * (cls_preds**self.gamma) * (
            -torch.log(1 - cls_preds + 1e-4))
        pos_cls_cost = self.alpha * (
            (1 - cls_preds)**self.gamma) * (-torch.log(cls_preds + 1e-4))
        cls_cost = pos_cls_cost[:, batch_gt_boxes_label.long(
        )] - neg_cls_cost[:, batch_gt_boxes_label.long()]

        # Compute the L1 cost between boxes
        box_cost = torch.cdist(reg_preds, batch_gt_boxes, p=1)

        reg_preds = self.transform_cxcywh_box_to_xyxy_box(reg_preds)
        batch_gt_boxes = self.transform_cxcywh_box_to_xyxy_box(batch_gt_boxes)

        # Compute the giou cost betwen boxes
        giou_cost = -self.compute_box_giou(reg_preds, batch_gt_boxes)
        # Final cost matrix
        total_cost = self.cls_match_cost * cls_cost + self.box_match_cost * box_cost + self.giou_match_cost * giou_cost
        total_cost = total_cost.view(batch_size, query_nums, -1)

        # for per image,assign one pred box to one GT box
        indices = []
        for idx, per_image_cost in enumerate(
                total_cost.split(per_image_gt_boxes_num, -1)):
            indices.append(
                self.linear_sum_assignment_with_inf(
                    per_image_cost[idx].cpu().numpy()))

        indices = [(torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return indices

    def linear_sum_assignment_with_inf(self, cost_matrix):
        cost_matrix = np.asarray(cost_matrix)

        nan = np.isnan(cost_matrix).any()
        if nan:
            cost_matrix[np.isnan(cost_matrix)] = 1e5

        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")

        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            min_values = values.min()
            max_values = values.max()
            m = min(cost_matrix.shape)

            positive = m * (max_values - min_values + np.abs(max_values) +
                            np.abs(min_values) + 1)
            if max_inf:
                place_holder = (max_values + (m - 1) *
                                (max_values - min_values)) + positive
            elif min_inf:
                place_holder = (min_values + (m - 1) *
                                (min_values - max_values)) - positive

            cost_matrix[np.isinf(cost_matrix)] = place_holder

        results = scipy.optimize.linear_sum_assignment(cost_matrix)

        return results

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


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
    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionResize, DetectionCollater, DETRDetectionCollater

    cocodataset = CocoDetection(COCO2017_path,
                                set_name='train2017',
                                transform=transforms.Compose([
                                    RandomHorizontalFlip(prob=0.5),
                                    RandomCrop(prob=0.5),
                                    RandomTranslate(prob=0.5),
                                    DetectionResize(
                                        resize=640,
                                        stride=32,
                                        resize_type='yolo_style',
                                        multi_scale=True,
                                        multi_scale_range=[0.8, 1.0]),
                                    Normalize(),
                                ]))

    from torch.utils.data import DataLoader
    collater = DetectionCollater(resize=640,
                                 resize_type='yolo_style',
                                 max_annots_num=100)
    train_loader = DataLoader(cocodataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    # from simpleAICV.detection.models.retinanet import resnet50_retinanet
    # net = resnet50_retinanet()
    # loss = RetinaLoss(areas=[[32, 32], [64, 64], [128, 128], [256, 256],
    #                          [512, 512]],
    #                   ratios=[0.5, 1, 2],
    #                   scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
    #                   strides=[8, 16, 32, 64, 128],
    #                   alpha=0.25,
    #                   gamma=2,
    #                   beta=1.0 / 9.0,
    #                   focal_eiou_gamma=0.5,
    #                   cls_loss_weight=1.,
    #                   box_loss_weight=1.,
    #                   box_loss_type='CIoU')

    # for data in tqdm(train_loader):
    #     images, annots, scales, sizes = data['image'], data['annots'], data[
    #         'scale'], data['size']
    #     print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
    #     preds = net(images)
    #     for pred in preds:
    #         for per_level_pred in pred:
    #             print('2222', per_level_pred.shape)
    #     loss_dict = loss(preds, annots)
    #     print('3333', loss_dict)
    #     break

    # from simpleAICV.detection.models.fcos import resnet50_fcos
    # net = resnet50_fcos()
    # loss = FCOSLoss(strides=[8, 16, 32, 64, 128],
    #                 mi=[[-1, 64], [64, 128], [128, 256], [256, 512],
    #                     [512, 100000000]],
    #                 alpha=0.25,
    #                 gamma=2.,
    #                 cls_loss_weight=1.,
    #                 box_loss_weight=1.,
    #                 center_ness_loss_weight=1.,
    #                 box_loss_iou_type='CIoU',
    #                 center_sample_radius=1.5,
    #                 use_center_sample=True)

    # for data in tqdm(train_loader):
    #     images, annots, scales, sizes = data['image'], data['annots'], data[
    #         'scale'], data['size']
    #     print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
    #     preds = net(images)
    #     for pred in preds:
    #         for per_level_pred in pred:
    #             print('2222', per_level_pred.shape)
    #     loss_dict = loss(preds, annots)
    #     print('3333', loss_dict)
    #     break

    # from simpleAICV.detection.models.centernet import resnet18_centernet
    # net = resnet18_centernet()
    # loss = CenterNetLoss(alpha=2.,
    #                      beta=4.,
    #                      heatmap_loss_weight=1.0,
    #                      offset_loss_weight=1.0,
    #                      wh_loss_weight=0.1,
    #                      min_overlap=0.7,
    #                      max_object_num=100)

    # for data in tqdm(train_loader):
    #     images, annots, scales, sizes = data['image'], data['annots'], data[
    #         'scale'], data['size']
    #     print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
    #     preds = net(images)
    #     print('2222', preds[0].shape, preds[1].shape, preds[2].shape)
    #     loss_dict = loss(preds, annots)
    #     print('3333', loss_dict)
    #     break

    # from simpleAICV.detection.models.ttfnet import resnet18_ttfnet
    # net = resnet18_ttfnet()
    # loss = TTFNetLoss(alpha=2.0,
    #                   beta=4.0,
    #                   stride=4,
    #                   heatmap_loss_weight=1.0,
    #                   box_loss_weight=5.0,
    #                   box_loss_iou_type='CIoU',
    #                   gaussian_alpha=0.54,
    #                   gaussian_beta=0.54)

    # for data in tqdm(train_loader):
    #     images, annots, scales, sizes = data['image'], data['annots'], data[
    #         'scale'], data['size']
    #     print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
    #     preds = net(images)
    #     print('2222', preds[0].shape, preds[1].shape)
    #     loss_dict = loss(preds, annots)
    #     print('3333', loss_dict)
    #     break

    from torch.utils.data import DataLoader
    detr_collater = DETRDetectionCollater(resize=800,
                                          resize_type='yolo_style',
                                          max_annots_num=100)
    detr_train_loader = DataLoader(cocodataset,
                                   batch_size=4,
                                   shuffle=True,
                                   num_workers=2,
                                   collate_fn=detr_collater)

    from simpleAICV.detection.models.detr import resnet50_detr
    net = resnet50_detr()
    loss = DETRLoss(cls_match_cost=1.0,
                    box_match_cost=5.0,
                    giou_match_cost=2.0,
                    cls_loss_weight=1.0,
                    box_l1_loss_weight=5.0,
                    iou_loss_weight=2.0,
                    no_object_cls_weight=0.1,
                    num_classes=80)
    for data in tqdm(detr_train_loader):
        images, annots, masks = data['image'], data['scaled_annots'], data[
            'mask']
        print('1111', images.shape, annots.shape)
        preds = net(images, masks)
        for pred in preds:
            print('2222', pred.shape)
        loss_dict = loss(preds, annots)
        print('3333', loss_dict)
        break

    from torch.utils.data import DataLoader
    detr_collater = DETRDetectionCollater(resize=800,
                                          resize_type='yolo_style',
                                          max_annots_num=100)
    detr_train_loader = DataLoader(cocodataset,
                                   batch_size=2,
                                   shuffle=True,
                                   num_workers=2,
                                   collate_fn=detr_collater)

    from simpleAICV.detection.models.dinodetr import resnet50_dinodetr
    net = resnet50_dinodetr()
    loss = DINODETRLoss(cls_match_cost=2.0,
                        box_match_cost=5.0,
                        giou_match_cost=2.0,
                        cls_loss_weight=1.0,
                        box_l1_loss_weight=5.0,
                        iou_loss_weight=2.0,
                        alpha=0.25,
                        gamma=2.0,
                        num_classes=80)
    for data in tqdm(detr_train_loader):
        images, annots, masks = data['image'], data['scaled_annots'], data[
            'mask']
        net = net.cuda()
        images = images.cuda()
        annots = annots.cuda()
        masks = masks.cuda()
        print('1111', images.shape, masks.shape, annots.shape)
        preds = net(images, masks, annots)
        print('2222', preds.keys())
        loss_dict = loss(preds, annots)
        print('3333', loss_dict)
        break