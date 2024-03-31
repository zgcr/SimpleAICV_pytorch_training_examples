import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.instance_segmentation.models.anchor import YOLACTAnchors

__all__ = [
    'YOLACTLoss',
    'SOLOV2Loss',
]


class YOLACTLoss(nn.Module):

    def __init__(self,
                 resize=544,
                 resize_type='yolo_style',
                 scales=[24, 48, 96, 192, 384],
                 ratios=[1, 1 / 2, 2],
                 strides=[8, 16, 32, 64, 128],
                 cls_loss_weight=1.,
                 box_loss_weight=1.5,
                 mask_loss_weight=6.125,
                 semantic_seg_loss_weight=1.):
        super(YOLACTLoss, self).__init__()
        self.resize = resize
        self.resize_type = resize_type
        if self.resize_type == 'retina_style':
            self.resize = int(round(self.resize * 1333. / 800))

        self.scales = self.resize / 544. * np.array(scales, dtype=np.float32)
        self.ratios = np.array(ratios, dtype=np.float32)
        self.strides = np.array(strides, dtype=np.float32)
        self.anchors = YOLACTAnchors(scales=scales,
                                     ratios=ratios,
                                     strides=strides)

        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.semantic_seg_loss_weight = semantic_seg_loss_weight

    def forward(self, preds, gt_bboxes, gt_masks):
        '''
        compute cls loss and reg loss in one batch
        '''
        class_preds, box_preds, coef_preds, proto_outs, seg_preds = preds
        device = proto_outs.device
        batch_size = proto_outs.shape[0]

        gt_bboxes = [
            per_image_gt_bboxes.to(device) for per_image_gt_bboxes in gt_bboxes
        ]
        gt_masks = [
            per_image_gt_masks.to(device) for per_image_gt_masks in gt_masks
        ]

        # feature map w and h
        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in class_preds]
        one_image_anchors = self.anchors(feature_size)
        one_image_anchors = torch.cat([
            torch.tensor(per_level_anchor).view(-1, per_level_anchor.shape[-1])
            for per_level_anchor in one_image_anchors
        ],
                                      dim=0)
        batch_anchors = one_image_anchors.unsqueeze(0).repeat(
            batch_size, 1, 1).to(device)
        # torch.Size([16, 18525, 4])

        batch_anchor_cls_labels, batch_anchor_box_labels, batch_anchor_max_gt_boxes, batch_anchor_max_gt_indexes = self.get_batch_anchors_annotations(
            batch_anchors, gt_bboxes)

        class_preds = [
            per_class_pred.view(per_class_pred.shape[0], -1,
                                per_class_pred.shape[-1]).contiguous()
            for per_class_pred in class_preds
        ]
        box_preds = [
            per_box_pred.view(per_box_pred.shape[0], -1,
                              per_box_pred.shape[-1]).contiguous()
            for per_box_pred in box_preds
        ]
        coef_preds = [
            per_coef_pred.view(per_coef_pred.shape[0], -1,
                               per_coef_pred.shape[-1]).contiguous()
            for per_coef_pred in coef_preds
        ]

        class_preds = torch.cat(class_preds, dim=1)
        box_preds = torch.cat(box_preds, dim=1)
        coef_preds = torch.cat(coef_preds, dim=1)

        cls_loss = self.compute_batch_cls_loss(class_preds,
                                               batch_anchor_cls_labels)
        box_loss = self.compute_batch_box_loss(box_preds,
                                               batch_anchor_box_labels,
                                               batch_anchor_cls_labels)
        mask_loss = self.compute_batch_mask_loss(coef_preds, proto_outs,
                                                 gt_masks,
                                                 batch_anchor_max_gt_boxes,
                                                 batch_anchor_max_gt_indexes,
                                                 batch_anchor_cls_labels)
        segmantic_seg_loss = self.compute_batch_semantic_seg_loss(
            seg_preds, gt_masks, gt_bboxes)

        cls_loss = cls_loss * self.cls_loss_weight
        box_loss = box_loss * self.box_loss_weight
        mask_loss = mask_loss * self.mask_loss_weight
        segmantic_seg_loss = segmantic_seg_loss * self.semantic_seg_loss_weight

        loss_dict = {
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'mask_loss': mask_loss,
            'segmantic_seg_loss': segmantic_seg_loss,
        }

        return loss_dict

    def compute_batch_cls_loss(self,
                               class_preds,
                               batch_anchor_cls_labels,
                               negative_sample_ratio=3.):
        num_classes = class_preds.shape[-1]
        # Compute max conf across batch for hard negative mining
        batch_conf = class_preds.reshape(-1, num_classes)
        batch_conf_max_value = batch_conf.max()
        mark = torch.log(
            torch.sum(torch.exp(batch_conf - batch_conf_max_value),
                      1)) + batch_conf_max_value - batch_conf[:, 0]

        # [batch_size, anchor_num]
        positive_sample_flag = batch_anchor_cls_labels > 0

        # Hard Negative Mining
        mark = mark.reshape(class_preds.shape[0], -1)
        # filter out positive sample boxes
        mark[positive_sample_flag] = 0
        # filter out ignore sample boxes
        mark[batch_anchor_cls_labels < 0] = 0

        _, idx = mark.sort(dim=1, descending=True)
        # [batch_size, anchor_num]
        _, idx_rank = idx.sort(dim=1)

        # [batch_size, 1]
        positive_sample_num = positive_sample_flag.long().sum(1, keepdim=True)
        # [batch_size, 1]
        negative_sample_num = torch.clamp(
            negative_sample_ratio * positive_sample_num,
            max=positive_sample_flag.shape[1] - 1)

        # [batch_size, anchor_num]
        neg_bool = idx_rank < negative_sample_num.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[positive_sample_flag] = 0
        # Filter out ignore sample boxes
        neg_bool[batch_anchor_cls_labels < 0] = 0

        # Confidence Loss Including Positive and Negative Examples
        # [Positive and Negative anchor_nums,class_num]
        filter_class_preds = class_preds[(positive_sample_flag +
                                          neg_bool)].reshape(-1, num_classes)
        # [Positive and Negative anchor_nums]
        filter_class_gts = batch_anchor_cls_labels[(positive_sample_flag +
                                                    neg_bool)]

        # compute cross entropy loss
        filter_class_preds = filter_class_preds - torch.max(filter_class_preds)
        filter_class_preds = torch.exp(filter_class_preds)

        softmax_molecule = filter_class_preds.gather(
            dim=1, index=filter_class_gts.unsqueeze(-1)).squeeze()
        softmax_denominator = filter_class_preds.sum(1)
        # compute softmax
        softmax = softmax_molecule / softmax_denominator
        softmax = torch.clamp(softmax, min=1e-4, max=1. - 1e-4)

        cls_loss = -torch.log(softmax)
        cls_loss = cls_loss.sum() / positive_sample_num.sum()

        return cls_loss

    def compute_batch_box_loss(self,
                               box_preds,
                               batch_anchor_box_labels,
                               batch_anchor_cls_labels,
                               beta=1.0):
        # [batch_size, anchor_num]
        positive_sample_flag = batch_anchor_cls_labels > 0
        positive_sample_num = positive_sample_flag.sum()

        positive_box_preds = box_preds[positive_sample_flag, :]
        positive_box_labels = batch_anchor_box_labels[positive_sample_flag, :]

        box_loss = torch.abs(positive_box_preds - positive_box_labels)
        box_loss = torch.where(torch.ge(box_loss, beta), box_loss - 0.5 * beta,
                               0.5 * (box_loss**2) / beta)

        box_loss = box_loss.sum() / positive_sample_num

        return box_loss

    def compute_batch_mask_loss(self,
                                coef_preds,
                                proto_outs,
                                gt_masks,
                                batch_anchor_max_gt_boxes,
                                batch_anchor_max_gt_indexes,
                                batch_anchor_cls_labels,
                                binary_threshold=0.5,
                                choose_max_mask_num=100):
        batch_size = coef_preds.shape[0]
        positive_sample_flag = batch_anchor_cls_labels > 0
        proto_feature_h, proto_feature_w = proto_outs.shape[
            1], proto_outs.shape[2]

        positive_sample_num = positive_sample_flag.sum()

        mask_loss = 0.
        for batch_idx in range(batch_size):
            #  gt_masks[batch_idx]: [batch_size,resize,resize]
            #  downsample the gt mask to the size of 'proto_outs'
            #  downsampled_masks:[batch_size,proto_feature_h, proto_feature_w]
            downsampled_masks = F.interpolate(
                gt_masks[batch_idx].unsqueeze(0),
                (proto_feature_h, proto_feature_w),
                mode='bilinear',
                align_corners=False).squeeze(0)

            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
            # binarize the gt mask because of the downsample operation
            downsampled_masks = downsampled_masks.gt(binary_threshold).float()

            positive_anchor_indexes = batch_anchor_max_gt_indexes[batch_idx][
                positive_sample_flag[batch_idx]]
            positive_anchor_box = batch_anchor_max_gt_boxes[batch_idx][
                positive_sample_flag[batch_idx]]
            positive_coef_preds = coef_preds[batch_idx][
                positive_sample_flag[batch_idx]]

            if positive_anchor_indexes.shape[0] == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            per_image_positive_sample_num = positive_coef_preds.shape[0]
            if per_image_positive_sample_num > choose_max_mask_num:
                perm = torch.randperm(positive_coef_preds.shape[0])
                select = perm[:choose_max_mask_num]
                positive_coef_preds = positive_coef_preds[select]
                positive_anchor_indexes = positive_anchor_indexes[select]
                positive_anchor_box = positive_anchor_box[select]

            per_image_positive_sample_num = positive_coef_preds.shape[0]

            positive_gt_masks = downsampled_masks[:, :,
                                                  positive_anchor_indexes]

            # mask assembly by linear combination
            # @ means dot product
            per_image_mask_preds = proto_outs[
                batch_idx] @ positive_coef_preds.t()
            per_image_mask_preds = torch.sigmoid(per_image_mask_preds)
            # pos_anchor_box.shape: (num_pos, 4)
            per_image_mask_preds = self.crop_predict_mask(
                per_image_mask_preds, positive_anchor_box)

            per_image_mask_preds = torch.clamp(per_image_mask_preds,
                                               min=1e-4,
                                               max=1. - 1e-4)

            # per_image_mask_preds:[proto_feature_h, proto_feature_w,select_proto_nums]
            # positive_gt_masks:[proto_feature_h, proto_feature_w,select_proto_nums]
            # compute bce cross entropy loss
            per_image_mask_loss = -(
                positive_gt_masks * torch.log(per_image_mask_preds) +
                (1. - positive_gt_masks) *
                torch.log(1. - per_image_mask_preds))

            # Normalize the mask loss to emulate roi pooling's effect on loss.
            anchor_area = (
                positive_anchor_box[:, 2] - positive_anchor_box[:, 0]) * (
                    positive_anchor_box[:, 3] - positive_anchor_box[:, 1])
            per_image_mask_loss = per_image_mask_loss.sum(
                dim=(0, 1)) / anchor_area

            mask_loss = mask_loss + torch.sum(per_image_mask_loss)

        mask_loss = mask_loss / (proto_feature_h * proto_feature_w *
                                 positive_sample_num)

        return mask_loss

    def compute_batch_semantic_seg_loss(self,
                                        seg_preds,
                                        gt_masks,
                                        gt_bboxes,
                                        binary_threshold=0.5):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
        batch_size = seg_preds.shape[0]
        mask_h = seg_preds.shape[2]
        mask_w = seg_preds.shape[3]

        segmantic_seg_loss = 0.
        for batch_idx in range(batch_size):
            # cur_segment:[num_classes,mask_h,mask_w]
            cur_segment = seg_preds[batch_idx].float()
            # cur_class_gt:[gt_bboxes_num]
            cur_class_gt = gt_bboxes[batch_idx][:, 4].long()

            downsampled_masks = F.interpolate(gt_masks[batch_idx].unsqueeze(0),
                                              (mask_h, mask_w),
                                              mode='bilinear',
                                              align_corners=False).squeeze(0)
            # downsampled_masks:[gt_bboxes_num,mask_h,mask_w]
            downsampled_masks = downsampled_masks.gt(binary_threshold).float()

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment).float()
            for gt_idx in range(downsampled_masks.shape[0]):
                segment_gt[cur_class_gt[gt_idx]] = torch.max(
                    segment_gt[cur_class_gt[gt_idx]],
                    downsampled_masks[gt_idx])

            cur_segment = torch.sigmoid(cur_segment)
            cur_segment = torch.clamp(cur_segment, min=1e-4, max=1. - 1e-4)

            per_image_segmantic_seg_loss = -(
                segment_gt * torch.log(cur_segment) +
                (1. - segment_gt) * torch.log(1. - cur_segment))

            per_image_segmantic_seg_loss = per_image_segmantic_seg_loss.sum()

            segmantic_seg_loss = segmantic_seg_loss + per_image_segmantic_seg_loss

        segmantic_seg_loss = segmantic_seg_loss / (mask_h * mask_w *
                                                   batch_size)

        return segmantic_seg_loss

    def get_batch_anchors_annotations(self, batch_anchors, gt_bboxes):
        assert batch_anchors.shape[0] == len(gt_bboxes)
        device = batch_anchors.device
        one_image_anchor_nums = batch_anchors.shape[1]

        batch_anchor_cls_labels, batch_anchor_box_labels = [], []
        batch_anchor_max_gt_boxes, batch_anchor_max_gt_indexes = [], []
        for one_image_anchors, one_image_gt_bboxes in zip(
                batch_anchors, gt_bboxes):
            if one_image_gt_bboxes.shape[0] == 0:
                one_image_anchor_cls_labels = torch.zeros(
                    [one_image_anchor_nums],
                    dtype=torch.float32,
                    device=device)
                one_image_anchor_box_labels = torch.zeros(
                    [one_image_anchor_nums, 4],
                    dtype=torch.float32,
                    device=device)
                one_image_anchor_max_gt_boxes = torch.zeros(
                    [one_image_anchor_nums, 4],
                    dtype=torch.float32,
                    device=device)
                one_image_anchor_max_gt_indexes = torch.zeros(
                    [one_image_anchor_nums],
                    dtype=torch.float32,
                    device=device)
            else:
                one_image_gt_box_coords = one_image_gt_bboxes[:, 0:4]
                one_image_gt_box_classes = one_image_gt_bboxes[:, 4]

                one_image_anchor_cls_labels, one_image_anchor_box_labels, one_image_anchor_max_gt_boxes, one_image_anchor_max_gt_indexes = self.one_image_anchor_match_gt(
                    one_image_gt_box_coords, one_image_anchors,
                    one_image_gt_box_classes)

                one_image_anchor_cls_labels = one_image_anchor_cls_labels.unsqueeze(
                    0)
                one_image_anchor_box_labels = one_image_anchor_box_labels.unsqueeze(
                    0)
                one_image_anchor_max_gt_boxes = one_image_anchor_max_gt_boxes.unsqueeze(
                    0)
                one_image_anchor_max_gt_indexes = one_image_anchor_max_gt_indexes.unsqueeze(
                    0)

            batch_anchor_cls_labels.append(one_image_anchor_cls_labels)
            batch_anchor_box_labels.append(one_image_anchor_box_labels)
            batch_anchor_max_gt_boxes.append(one_image_anchor_max_gt_boxes)
            batch_anchor_max_gt_indexes.append(one_image_anchor_max_gt_indexes)

        batch_anchor_cls_labels = torch.cat(batch_anchor_cls_labels, dim=0)
        batch_anchor_box_labels = torch.cat(batch_anchor_box_labels, dim=0)
        batch_anchor_max_gt_boxes = torch.cat(batch_anchor_max_gt_boxes, dim=0)
        batch_anchor_max_gt_indexes = torch.cat(batch_anchor_max_gt_indexes,
                                                dim=0)

        return batch_anchor_cls_labels, batch_anchor_box_labels, batch_anchor_max_gt_boxes, batch_anchor_max_gt_indexes

    def one_image_anchor_match_gt(self, one_image_gt_box_coords,
                                  one_image_anchors, one_image_gt_box_classes):
        # Convert prior boxes to the form of [xmin, ymin, xmax, ymax].
        decoded_priors = torch.cat(
            (one_image_anchors[:, :2] - one_image_anchors[:, 2:] / 2,
             one_image_anchors[:, :2] + one_image_anchors[:, 2:] / 2),
            dim=1)

        # (num_gts, num_achors)
        gt_box_to_anchor_overlaps = self.compute_box_iou(
            one_image_gt_box_coords, decoded_priors)

        # [8, 18525]

        # (num_gts, ) the max IoU for each gt box
        _, one_image_gt_max_indexes = gt_box_to_anchor_overlaps.max(1)

        # [8]

        # (num_achors, ) the max IoU for each anchor
        one_image_anchor_max_iou, one_image_anchor_max_gt_indexes = gt_box_to_anchor_overlaps.max(
            0)

        # [18525] [18525]

        # For the max IoU anchor for each gt box, set its IoU to 2. This ensures that it won't be filtered
        # in the threshold step even if the IoU is under the negative threshold. This is because that we want
        # at least one anchor to match with each gt box or else we'd be wasting training data.
        one_image_anchor_max_iou.index_fill_(0, one_image_gt_max_indexes, 2)

        # Set the index of the pair (anchor, gt) we set the overlap for above.
        for j in range(one_image_gt_max_indexes.shape[0]):
            one_image_anchor_max_gt_indexes[one_image_gt_max_indexes[j]] = j

        # (num_achors, 4)
        one_image_anchor_max_gt_boxes = one_image_gt_box_coords[
            one_image_anchor_max_gt_indexes]

        # torch.Size([18525, 4]) torch.Size([18525]) torch.Size([8, 4])

        # the class of the max IoU gt box for each anchor
        one_image_anchor_cls_labels = one_image_gt_box_classes[
            one_image_anchor_max_gt_indexes] + 1
        # label as ignore
        one_image_anchor_cls_labels[one_image_anchor_max_iou < 0.5] = -1
        # label as background
        one_image_anchor_cls_labels[one_image_anchor_max_iou < 0.4] = 0

        one_image_anchor_box_labels = self.compute_box_offsets(
            one_image_anchor_max_gt_boxes, one_image_anchors)

        one_image_anchor_cls_labels = one_image_anchor_cls_labels.long()

        #  torch.Size([18525, 4]) torch.Size([18525]) torch.Size([18525, 4]) torch.Size([18525])

        return one_image_anchor_cls_labels, one_image_anchor_box_labels, one_image_anchor_max_gt_boxes, one_image_anchor_max_gt_indexes

    def compute_box_iou(self, box_a, box_b):
        """
        Compute the IoU of two sets of boxes.
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """

        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

        (n, A), B = box_a.shape[:2], box_b.shape[1]
        # add a dimension
        box_a = box_a[:, :, None, :].expand(n, A, B, 4)
        box_b = box_b[:, None, :, :].expand(n, A, B, 4)

        max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
        min_xy = torch.max(box_a[..., :2], box_b[..., :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter_area = inter[..., 0] * inter[..., 1]

        area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] -
                                                    box_a[..., 1])
        area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] -
                                                    box_b[..., 1])

        out = inter_area / (area_a + area_b - inter_area)
        out = out.squeeze(0)

        return out

    def compute_box_offsets(self, matched, priors):
        variances = [0.1, 0.2]

        # 10 * (Xg - Xa) / Wa
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # 10 * (Yg - Ya) / Ha
        g_cxcy /= (variances[0] * priors[:, 2:])
        # 5 * log(Wg / Wa)
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        # 5 * log(Hg / Ha)
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        # [num_priors, 4]
        box_offsets = torch.cat([g_cxcy, g_wh], 1)

        return box_offsets

    def sanitize_coordinates(self, _x1, _x2, img_size, padding=0):
        """
        Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        Also converts from relative to absolute coordinates and casts the results to long tensors.

        Warning: this does things in-place behind the scenes so copy if necessary.
        """
        _x1 = _x1 * img_size
        _x2 = _x2 * img_size

        x1 = torch.min(_x1, _x2)
        x2 = torch.max(_x1, _x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)

        return x1, x2

    def crop_predict_mask(self, masks, boxes, padding=1):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """
        h, w, n = masks.shape
        x1, x2 = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], w,
                                           padding)
        y1, y2 = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], h,
                                           padding)

        rows = torch.arange(w, device=masks.device,
                            dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
        cols = torch.arange(h, device=masks.device,
                            dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask.float()


class SOLOV2Loss(nn.Module):

    def __init__(self,
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768),
                               (384, 2048)),
                 grid_nums=(40, 36, 24, 16, 12),
                 mask_feature_upsample_scale=4,
                 sigma=0.2,
                 alpha=0.25,
                 gamma=2.0,
                 cls_loss_weight=1.0,
                 dice_loss_weight=3.0):
        super(SOLOV2Loss, self).__init__()
        self.scale_ranges = scale_ranges
        self.grid_nums = grid_nums
        self.mask_feature_upsample_scale = mask_feature_upsample_scale
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.cls_loss_weight = cls_loss_weight
        self.dice_loss_weight = dice_loss_weight

    def forward(self, preds, gt_bboxes, gt_masks):
        mask_feat_pred, kernel_preds, cate_preds = preds

        batch_instance_label_list, batch_cate_label_list, batch_instance_index_label_list, batch_grid_order_list = self.get_batch_assigned_annotations(
            mask_feat_pred, gt_bboxes, gt_masks)

        cls_loss = self.compute_batch_focal_loss(
            cate_preds, batch_cate_label_list, batch_instance_index_label_list)
        dice_loss = self.compute_batch_dice_loss(mask_feat_pred, kernel_preds,
                                                 batch_instance_label_list,
                                                 batch_grid_order_list)

        cls_loss = cls_loss * self.cls_loss_weight
        dice_loss = dice_loss * self.dice_loss_weight

        loss_dict = {
            'cls_loss': cls_loss,
            'dice_loss': dice_loss,
        }

        return loss_dict

    def compute_batch_focal_loss(self, cate_preds, batch_cate_label_list,
                                 batch_instance_index_label_list):
        batch_instance_index_labels = []
        for per_level_instance_index_label_list in zip(
                *batch_instance_index_label_list):
            per_level_instance_index_labels = []
            for per_level_per_image_instance_index_labels in per_level_instance_index_label_list:
                per_level_instance_index_labels.append(
                    per_level_per_image_instance_index_labels.flatten())
            per_level_instance_index_labels = torch.cat(
                per_level_instance_index_labels, dim=0)
            batch_instance_index_labels.append(per_level_instance_index_labels)
        batch_instance_index_labels = torch.cat(batch_instance_index_labels,
                                                dim=0)
        instance_nums = batch_instance_index_labels.sum()
        device = batch_instance_index_labels.device

        if instance_nums == 0:
            return torch.tensor(0.).to(device)

        batch_cate_labels = []
        for per_level_cate_label_list in zip(*batch_cate_label_list):
            per_level_cate_labels = []
            for per_level_per_image_cate_labels in per_level_cate_label_list:
                per_level_cate_labels.append(
                    per_level_per_image_cate_labels.flatten())
            per_level_cate_labels = torch.cat(per_level_cate_labels, dim=0)
            batch_cate_labels.append(per_level_cate_labels)
        batch_cate_labels = torch.cat(batch_cate_labels, dim=0)

        batch_cate_preds = []
        for per_level_cate_preds in cate_preds:
            num_classes = per_level_cate_preds.shape[1]
            batch_cate_preds.append(
                per_level_cate_preds.permute(0, 2, 3,
                                             1).reshape(-1, num_classes))
        batch_cate_preds = torch.cat(batch_cate_preds, dim=0)
        batch_cate_preds = torch.sigmoid(batch_cate_preds)
        batch_cate_preds = torch.clamp(batch_cate_preds,
                                       min=1e-4,
                                       max=1. - 1e-4)
        num_classes = batch_cate_preds.shape[1]

        # class_index value from 1 to 80 represent 80 positive classes
        # generate 80 binary ground truth classes for each point
        loss_ground_truth = F.one_hot(batch_cate_labels.long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(batch_cate_preds) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), batch_cate_preds,
                         1. - batch_cate_preds)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        batch_bce_loss = -(
            loss_ground_truth * torch.log(batch_cate_preds) +
            (1. - loss_ground_truth) * torch.log(1. - batch_cate_preds))

        batch_focal_loss = focal_weight * batch_bce_loss
        batch_focal_loss = batch_focal_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        batch_focal_loss = batch_focal_loss / instance_nums

        return batch_focal_loss

    def compute_batch_dice_loss(self, mask_feat_pred, kernel_preds,
                                batch_instance_label_list,
                                batch_grid_order_list):
        batch_instance_labels = []
        for per_level_instance_labels in zip(*batch_instance_label_list):
            per_level_instance_labels = torch.cat([
                per_image_per_level_instance_labels
                for per_image_per_level_instance_labels in
                per_level_instance_labels
            ],
                                                  dim=0)
            batch_instance_labels.append(per_level_instance_labels)

        batch_kernel_preds = []
        for per_level_kernel_preds, per_level_grid_orders in zip(
                kernel_preds, zip(*batch_grid_order_list)):
            per_level_filter_kernel_preds = []
            for per_level_per_image_kernel_preds, per_level_per_image_grid_orders in zip(
                    per_level_kernel_preds, per_level_grid_orders):
                per_level_filter_kernel_preds.append(
                    per_level_per_image_kernel_preds.view(
                        per_level_per_image_kernel_preds.shape[0],
                        -1)[:, per_level_per_image_grid_orders])
            batch_kernel_preds.append(per_level_filter_kernel_preds)

        batch_filter_instance_preds, batch_filter_instance_labels = [], []
        for level_idx, per_level_kernel_pred in enumerate(batch_kernel_preds):
            per_level_mask_preds = []
            for idx, per_level_per_image_kernel_pred in enumerate(
                    per_level_kernel_pred):
                if per_level_per_image_kernel_pred.shape[-1] == 0:
                    continue

                current_instance_pred = mask_feat_pred[idx].unsqueeze(0)
                H, W = current_instance_pred.shape[
                    2], current_instance_pred.shape[3]
                N, I = per_level_per_image_kernel_pred.shape
                per_level_per_image_kernel_pred = per_level_per_image_kernel_pred.permute(
                    1, 0).view(I, -1, 1, 1)
                current_instance_pred = F.conv2d(
                    current_instance_pred,
                    per_level_per_image_kernel_pred,
                    stride=1)
                current_instance_pred = current_instance_pred.view(-1, H, W)
                per_level_mask_preds.append(current_instance_pred)

            if len(per_level_mask_preds) != 0:
                per_level_mask_preds = torch.cat(per_level_mask_preds, dim=0)
                per_level_mask_preds = torch.sigmoid(per_level_mask_preds)
                per_level_mask_preds = torch.clamp(per_level_mask_preds,
                                                   min=1e-4,
                                                   max=1. - 1e-4)

                batch_filter_instance_preds.append(per_level_mask_preds)
                batch_filter_instance_labels.append(
                    batch_instance_labels[level_idx])

        batch_filter_instance_preds = torch.cat(batch_filter_instance_preds,
                                                dim=0)
        batch_filter_instance_labels = torch.cat(batch_filter_instance_labels,
                                                 dim=0)

        instance_nums = batch_filter_instance_preds.shape[0]
        batch_filter_instance_preds = batch_filter_instance_preds.contiguous(
        ).view(batch_filter_instance_preds.shape[0], -1)
        batch_filter_instance_labels = batch_filter_instance_labels.contiguous(
        ).view(batch_filter_instance_labels.shape[0], -1).float()

        a = torch.sum(batch_filter_instance_preds *
                      batch_filter_instance_labels,
                      dim=1)
        b = torch.sum(batch_filter_instance_preds *
                      batch_filter_instance_preds,
                      dim=1)
        c = torch.sum(batch_filter_instance_labels *
                      batch_filter_instance_labels,
                      dim=1)
        dice_loss = 1 - ((2 * a) / (b + c + 1e-4))
        dice_loss = dice_loss.sum() / instance_nums

        return dice_loss

    def center_of_mass(self, bitmasks):
        device = bitmasks.device
        _, h, w = bitmasks.shape

        ys = torch.arange(0, h, dtype=torch.float32).to(device)
        xs = torch.arange(0, w, dtype=torch.float32).to(device)

        m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-4)
        m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
        m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)

        center_x = m10 / m00
        center_y = m01 / m00

        return center_x, center_y

    def get_batch_assigned_annotations(self, mask_feat_pred, gt_bboxes,
                                       gt_masks):
        device = mask_feat_pred.device
        mask_feat_pred_h, mask_feat_pred_w = mask_feat_pred.shape[
            2], mask_feat_pred.shape[3]

        batch_instance_label_list, batch_cate_label_list = [], []
        batch_instance_index_label_list, batch_grid_order_list = [], []
        for per_image_gt_bboxes, per_image_gt_masks in zip(
                gt_bboxes, gt_masks):
            per_image_gt_bboxes = per_image_gt_bboxes.to(device)
            per_image_gt_masks = per_image_gt_masks.to(device)

            per_image_gt_labels = per_image_gt_bboxes[:, 4]
            per_image_gt_bboxes = per_image_gt_bboxes[:, 0:4]

            per_image_gt_areas = torch.sqrt(
                (per_image_gt_bboxes[:, 2] - per_image_gt_bboxes[:, 0]) *
                (per_image_gt_bboxes[:, 3] - per_image_gt_bboxes[:, 1]))

            per_image_instance_label_list, per_image_cate_label_list = [], []
            per_image_instance_index_label_list, per_image_grid_order_list = [], []
            for (lower_bound,
                 upper_bound), grid_num in zip(self.scale_ranges,
                                               self.grid_nums):
                hit_indices = (
                    (per_image_gt_areas >= lower_bound) &
                    (per_image_gt_areas <= upper_bound)).nonzero().flatten()
                positive_instance_nums = len(hit_indices)

                per_stride_instance_label = []
                per_stride_grid_order = []
                per_stride_cate_label = torch.zeros(
                    [grid_num, grid_num], dtype=torch.int64).to(device)
                per_stride_instance_index_label = torch.zeros(
                    [grid_num**2], dtype=torch.bool).to(device)

                if positive_instance_nums == 0:
                    per_stride_instance_label = torch.zeros(
                        [0, mask_feat_pred_h, mask_feat_pred_w]).to(device)
                    per_image_instance_label_list.append(
                        per_stride_instance_label)
                    per_image_cate_label_list.append(per_stride_cate_label)
                    per_image_instance_index_label_list.append(
                        per_stride_instance_index_label)
                    per_image_grid_order_list.append(per_stride_grid_order)
                    continue

                per_image_positive_gt_bboxes = per_image_gt_bboxes[hit_indices]
                per_image_positive_gt_labels = per_image_gt_labels[hit_indices]
                per_image_positive_gt_masks = per_image_gt_masks[
                    hit_indices, :, :]

                per_image_positive_gt_half_ws = 0.5 * (
                    per_image_positive_gt_bboxes[:, 2] -
                    per_image_positive_gt_bboxes[:, 0]) * self.sigma
                per_image_positive_gt_half_hs = 0.5 * (
                    per_image_positive_gt_bboxes[:, 3] -
                    per_image_positive_gt_bboxes[:, 1]) * self.sigma

                # mass center
                per_image_positive_gt_center_ws, per_image_positive_gt_center_hs = self.center_of_mass(
                    per_image_positive_gt_masks)
                per_image_positive_valid_mask_flags = per_image_positive_gt_masks.sum(
                    dim=-1).sum(dim=-1) > 0

                for per_gt_mask, per_gt_label, per_half_h, per_half_w, per_center_h, per_center_w, per_valid_mask_flag in zip(
                        per_image_positive_gt_masks,
                        per_image_positive_gt_labels,
                        per_image_positive_gt_half_hs,
                        per_image_positive_gt_half_ws,
                        per_image_positive_gt_center_hs,
                        per_image_positive_gt_center_ws,
                        per_image_positive_valid_mask_flags):

                    if not per_valid_mask_flag:
                        continue

                    input_image_size = (mask_feat_pred_h *
                                        self.mask_feature_upsample_scale,
                                        mask_feat_pred_w *
                                        self.mask_feature_upsample_scale)
                    coord_grid_w = int((per_center_w / input_image_size[1]) //
                                       (1. / grid_num))
                    coord_grid_h = int((per_center_h / input_image_size[0]) //
                                       (1. / grid_num))

                    # left, top, right, down
                    box_grid_top = max(
                        0,
                        int(((per_center_h - per_half_h) / input_image_size[0])
                            // (1. / grid_num)))
                    box_grid_down = min(
                        grid_num - 1,
                        int(((per_center_h + per_half_h) / input_image_size[0])
                            // (1. / grid_num)))
                    box_grid_left = max(
                        0,
                        int(((per_center_w - per_half_w) / input_image_size[1])
                            // (1. / grid_num)))
                    box_grid_right = min(
                        grid_num - 1,
                        int(((per_center_w + per_half_w) / input_image_size[1])
                            // (1. / grid_num)))

                    top = max(box_grid_top, coord_grid_h - 1)
                    down = min(box_grid_down, coord_grid_h + 1)
                    left = max(coord_grid_w - 1, box_grid_left)
                    right = min(box_grid_right, coord_grid_w + 1)

                    per_stride_cate_label[top:(down + 1),
                                          left:(right + 1)] = per_gt_label + 1

                    per_gt_mask = torch.unsqueeze(per_gt_mask, dim=0)
                    per_gt_mask = torch.unsqueeze(per_gt_mask, dim=0)

                    per_gt_mask = F.interpolate(per_gt_mask,
                                                size=(mask_feat_pred_h,
                                                      mask_feat_pred_w),
                                                mode='bilinear',
                                                align_corners=True)

                    per_gt_mask = torch.squeeze(per_gt_mask, dim=0)
                    per_gt_mask = torch.squeeze(per_gt_mask, dim=0)

                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            index = int(i * grid_num + j)

                            per_stride_instance_label.append(per_gt_mask)
                            per_stride_instance_index_label[index] = True
                            per_stride_grid_order.append(index)

                if len(per_stride_instance_label) == 0:
                    per_stride_instance_label = torch.zeros(
                        [0, mask_feat_pred_h, mask_feat_pred_w]).to(device)
                else:
                    per_stride_instance_label = torch.stack(
                        per_stride_instance_label, dim=0)

                per_image_instance_label_list.append(per_stride_instance_label)
                per_image_cate_label_list.append(per_stride_cate_label)
                per_image_instance_index_label_list.append(
                    per_stride_instance_index_label)
                per_image_grid_order_list.append(per_stride_grid_order)

            batch_instance_label_list.append(per_image_instance_label_list)
            batch_cate_label_list.append(per_image_cate_label_list)
            batch_instance_index_label_list.append(
                per_image_instance_index_label_list)
            batch_grid_order_list.append(per_image_grid_order_list)

        return batch_instance_label_list, batch_cate_label_list, batch_instance_index_label_list, batch_grid_order_list


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

    from simpleAICV.instance_segmentation.datasets.cocodataset import CocoInstanceSegmentation
    from simpleAICV.instance_segmentation.common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, InstanceSegmentationCollater, YOLACTInstanceSegmentationCollater

    cocodataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='val2017',
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=800,
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = YOLACTInstanceSegmentationCollater(resize=800,
                                                  resize_type='yolo_style')
    solov2_train_loader = DataLoader(cocodataset,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=1,
                                     collate_fn=collater)

    from simpleAICV.instance_segmentation.models import resnet50_yolact
    net = resnet50_yolact()
    loss = YOLACTLoss(resize=800,
                      resize_type='yolo_style',
                      scales=[24, 48, 96, 192, 384],
                      ratios=[1, 1 / 2, 2],
                      strides=[8, 16, 32, 64, 128],
                      cls_loss_weight=1.,
                      box_loss_weight=1.5,
                      mask_loss_weight=6.125,
                      semantic_seg_loss_weight=1.)

    for data in tqdm(solov2_train_loader):
        images = data['image']
        gt_bboxes = data['box']
        gt_masks = data['mask']
        print('1111', images.shape, len(gt_bboxes), gt_bboxes[0].shape,
              len(gt_masks), gt_masks[0].shape)

        preds = net(images)
        for per_out in preds[0]:
            print('3333', per_out.shape)

        for per_out in preds[1]:
            print('4444', per_out.shape)

        for per_out in preds[2]:
            print('5555', per_out.shape)

        print('6666', preds[3].shape)
        print('7777', preds[4].shape)

        loss_dict = loss(preds, gt_bboxes, gt_masks)
        print('8888', loss_dict)
        break

    cocodataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='val2017',
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=800,
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = InstanceSegmentationCollater(resize=800,
                                            resize_type='yolo_style')
    solov2_train_loader = DataLoader(cocodataset,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=1,
                                     collate_fn=collater)

    from simpleAICV.instance_segmentation.models import resnet50_solov2
    net = resnet50_solov2()
    loss = SOLOV2Loss(scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768),
                                    (384, 2048)),
                      grid_nums=(40, 36, 24, 16, 12),
                      mask_feature_upsample_scale=4,
                      sigma=0.2,
                      alpha=0.25,
                      gamma=2.0,
                      cls_loss_weight=1.0,
                      dice_loss_weight=3.0)
    for data in tqdm(solov2_train_loader):
        images = data['image']
        gt_bboxes = data['box']
        gt_masks = data['mask']
        print('1111', images.shape, len(gt_bboxes), gt_bboxes[0].shape,
              len(gt_masks), gt_masks[0].shape)

        preds = net(images)
        print('2222', preds[0].shape)
        for per_out in preds[1]:
            print('3333', per_out.shape)
        for per_out in preds[2]:
            print('4444', per_out.shape)

        loss_dict = loss(preds, gt_bboxes, gt_masks)
        print('5555', loss_dict)
        break
