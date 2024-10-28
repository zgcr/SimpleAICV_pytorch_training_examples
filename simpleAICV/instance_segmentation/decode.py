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
    'YOLACTDecoder',
    'SOLOV2Decoder',
]


class YOLACTDecoder(nn.Module):

    def __init__(self,
                 resize=544,
                 resize_type='yolo_style',
                 scales=[24, 48, 96, 192, 384],
                 ratios=[1, 1 / 2, 2],
                 strides=[8, 16, 32, 64, 128],
                 max_feature_upsample_scale=8,
                 topn=200,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 nms_threshold=0.5):
        super(YOLACTDecoder, self).__init__()
        self.resize = resize
        self.resize_type = resize_type
        if self.resize_type == 'retina_style':
            self.resize = int(round(self.resize * 1333. / 800))

        self.scales = np.array(scales, dtype=np.float32)
        self.ratios = np.array(ratios, dtype=np.float32)
        self.strides = np.array(strides, dtype=np.float32)
        self.anchors = YOLACTAnchors(resize=resize,
                                     scales=scales,
                                     ratios=ratios,
                                     strides=strides)

        self.max_feature_upsample_scale = max_feature_upsample_scale
        self.topn = topn
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.nms_threshold = nms_threshold

    def forward(self, preds, scaled_sizes, origin_sizes):
        with torch.no_grad():
            class_preds, box_preds, coef_preds, proto_outs, _ = preds
            device = proto_outs.device
            batch_size = proto_outs.shape[0]

            # feature map w and h
            feature_size = [[
                per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
            ] for per_level_cls_pred in class_preds]
            one_image_anchors = self.anchors(feature_size)

            class_preds = [
                F.softmax(per_class_pred,
                          dim=-1).view(per_class_pred.shape[0], -1,
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

            one_image_anchors = torch.cat([
                torch.tensor(per_level_anchor).view(-1,
                                                    per_level_anchor.shape[-1])
                for per_level_anchor in one_image_anchors
            ],
                                          dim=0)
            batch_anchors = one_image_anchors.unsqueeze(0).repeat(
                batch_size, 1, 1).to(device)
            # (2, 18525, 81) (2, 18525, 4) (2, 18525, 32) (18525, 4) (2, 18525, 4)

            input_size_w = feature_size[0][0] * self.max_feature_upsample_scale
            input_size_h = feature_size[0][1] * self.max_feature_upsample_scale
            batch_masks, batch_labels, batch_scores = [], [], []
            for batch_idx in range(batch_size):
                per_image_class_preds = class_preds[batch_idx]
                per_image_box_preds = box_preds[batch_idx]
                per_image_coef_preds = coef_preds[batch_idx]
                per_image_proto_outs = proto_outs[batch_idx]
                per_image_anchors = batch_anchors[batch_idx]
                per_image_scaled_size = scaled_sizes[batch_idx]
                per_image_origin_size = origin_sizes[batch_idx]
                # (18525, 81) (18525, 4) (18525, 32)
                # torch.Size([136, 136, 32]) (18525, 4) [480, 544] [564, 640]

                # [81, 18525]
                per_image_class_preds = per_image_class_preds.transpose(
                    1, 0).contiguous()
                # [80, 18525] exclude the background class
                per_image_class_preds = per_image_class_preds[1:, :]
                # get the max score class of 18525 predicted boxes
                # [18525,]
                per_image_box_predict_max_score, _ = torch.max(
                    per_image_class_preds, axis=0)

                # filter < min_score_threshold boxes
                keep_indexes = (per_image_box_predict_max_score
                                > self.min_score_threshold)
                per_image_class_preds = per_image_class_preds[:, keep_indexes]
                per_image_box_preds = per_image_box_preds[keep_indexes, :]
                per_image_coef_preds = per_image_coef_preds[keep_indexes, :]
                per_image_anchors = per_image_anchors[keep_indexes, :]
                # (18525,) (80, 18170) (18170, 4) (18170, 4) (18170, 32)

                empty_per_image_masks = np.zeros(
                    (0, input_size_h, input_size_w), dtype=np.float32)
                empty_per_image_labels = np.zeros((0), dtype=np.float32)
                empty_per_image_scores = np.zeros((0), dtype=np.float32)

                if per_image_class_preds.shape[
                        1] == 0 or per_image_box_preds.shape[
                            0] == 0 or per_image_coef_preds.shape[0] == 0:
                    batch_masks.append(empty_per_image_masks)
                    batch_labels.append(empty_per_image_labels)
                    batch_scores.append(empty_per_image_scores)
                    continue

                # decode boxes
                per_image_box_preds = torch.cat(
                    (per_image_anchors[:, 0:2] + per_image_box_preds[:, 0:2] *
                     0.1 * per_image_anchors[:, 2:4],
                     per_image_anchors[:, 2:4] *
                     torch.exp(per_image_box_preds[:, 2:4] * 0.2)),
                    dim=1)
                # transform to x_min,y_min,x_max,y_max,range[0,1]
                per_image_box_preds[:, 0:2] -= per_image_box_preds[:, 2:4] / 2
                per_image_box_preds[:, 2:4] += per_image_box_preds[:, 0:2]
                per_image_box_preds = torch.clip(per_image_box_preds,
                                                 min=0.,
                                                 max=1.)
                # (18170, 4)

                per_image_class_preds, per_image_box_preds, per_image_coef_preds, per_image_class_ids = self.fast_nms(
                    per_image_class_preds, per_image_box_preds,
                    per_image_coef_preds)
                # torch.Size([100]) torch.Size([100, 4]) torch.Size([100, 32]) torch.Size([100])

                if per_image_class_preds.shape[
                        0] == 0 or per_image_box_preds.shape[
                            0] == 0 or per_image_coef_preds.shape[0] == 0:
                    batch_masks.append(empty_per_image_masks)
                    batch_labels.append(empty_per_image_labels)
                    batch_scores.append(empty_per_image_scores)
                    continue

                per_image_masks = torch.sigmoid(
                    torch.matmul(per_image_proto_outs,
                                 per_image_coef_preds.t()))
                per_image_masks = self.crop_predict_mask(
                    per_image_masks, per_image_box_preds)
                per_image_masks = per_image_masks.permute(2, 0, 1).contiguous()
                # torch.Size([100, 136, 136])

                per_image_origin_size_h, per_image_origin_size_w = int(
                    per_image_origin_size[0]), int(per_image_origin_size[1])
                max_origin_size = max(per_image_origin_size_h,
                                      per_image_origin_size_w)

                # in OpenCV, cv2.resize is `align_corners=False`.
                per_image_masks = F.interpolate(
                    per_image_masks.unsqueeze(0),
                    (max_origin_size, max_origin_size),
                    mode='bilinear',
                    align_corners=False).squeeze(0)

                # Binarize the masks because of interpolation.
                binary_threshold = 0.5
                per_image_masks.gt_(binary_threshold)
                per_image_masks = per_image_masks[:, 0:per_image_origin_size_h,
                                                  0:per_image_origin_size_w]

                per_image_box_preds *= max_origin_size
                per_image_box_preds = per_image_box_preds.int()

                per_image_class_ids = per_image_class_ids.int()

                per_image_class_preds = per_image_class_preds.cpu().numpy()
                per_image_box_preds = per_image_box_preds.cpu().numpy()
                per_image_masks = per_image_masks.cpu().numpy().astype(
                    np.uint8)
                per_image_class_ids = per_image_class_ids.cpu().numpy()
                # (100,) (100, 4) (100, 640, 564) (100,)

                per_image_masks = per_image_masks
                per_image_labels = per_image_class_ids
                per_image_scores = per_image_class_preds

                batch_masks.append(per_image_masks)
                batch_labels.append(per_image_labels)
                batch_scores.append(per_image_scores)

            return batch_masks, batch_labels, batch_scores

    def fast_nms(self, per_image_class_preds, per_image_box_preds,
                 per_image_coef_preds):
        # torch.Size([18169, 4]) torch.Size([18169, 32]) torch.Size([80, 18169])
        per_image_class_preds, sorted_indexes = per_image_class_preds.sort(
            dim=1, descending=True)
        # torch.Size([80, 18169]) torch.Size([80, 18169])

        # filter topn
        sorted_indexes = sorted_indexes[:, :self.topn]
        per_image_class_preds = per_image_class_preds[:, :self.topn]
        # torch.Size([80, 200]) torch.Size([80, 200])

        num_classes, num_dets = sorted_indexes.shape
        per_image_box_preds = per_image_box_preds[
            sorted_indexes.reshape(-1), :].reshape(num_classes, num_dets, 4)
        per_image_coef_preds = per_image_coef_preds[
            sorted_indexes.reshape(-1), :].reshape(num_classes, num_dets, -1)
        # torch.Size([80, 200, 4]) torch.Size([80, 200, 32])

        iou = self.compute_box_iou(per_image_box_preds, per_image_box_preds)
        # torch.Size([80, 200, 200])
        iou.triu_(diagonal=1)
        # torch.Size([80, 200, 200])
        iou_max, _ = iou.max(dim=1)
        # torch.Size([80, 200])

        # filter out the ones higher than the threshold
        keep_indexes = (iou_max <= self.nms_threshold)

        # Assign each kept detection to its corresponding class
        per_image_class_ids = torch.arange(
            num_classes,
            device=per_image_box_preds.device)[:, None].expand_as(keep_indexes)

        per_image_class_ids = per_image_class_ids[keep_indexes]
        per_image_class_preds = per_image_class_preds[keep_indexes]
        per_image_box_preds = per_image_box_preds[keep_indexes]
        per_image_coef_preds = per_image_coef_preds[keep_indexes]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        per_image_class_preds, sorted_indexes = per_image_class_preds.sort(
            dim=0, descending=True)
        # torch.Size([14623]) torch.Size([14623])

        sorted_indexes = sorted_indexes[:self.max_object_num]
        per_image_class_preds = per_image_class_preds[:self.max_object_num]

        per_image_class_ids = per_image_class_ids[sorted_indexes]
        per_image_box_preds = per_image_box_preds[sorted_indexes]
        per_image_coef_preds = per_image_coef_preds[sorted_indexes]
        # torch.Size([100]) torch.Size([100]) torch.Size([100]) torch.Size([100, 4]) torch.Size([100, 32])

        return per_image_class_preds, per_image_box_preds, per_image_coef_preds, per_image_class_ids

    def compute_box_iou(self, box_a, box_b):
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

        return out

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


class SOLOV2Decoder(nn.Module):

    def __init__(self,
                 strides=(8, 8, 16, 32, 32),
                 grid_nums=(40, 36, 24, 16, 12),
                 mask_feature_upsample_scale=4,
                 max_mask_num=100,
                 topn=500,
                 min_score_threshold=0.1,
                 keep_score_threshold=0.1,
                 mask_threshold=0.5,
                 update_threshold=0.05):
        super(SOLOV2Decoder, self).__init__()
        self.strides = strides
        self.grid_nums = grid_nums
        self.mask_feature_upsample_scale = mask_feature_upsample_scale
        self.max_mask_num = max_mask_num
        self.topn = topn
        self.min_score_threshold = min_score_threshold
        self.keep_score_threshold = keep_score_threshold
        self.mask_threshold = mask_threshold
        self.update_threshold = update_threshold

    def forward(self, preds, scaled_sizes, origin_sizes):
        with torch.no_grad():
            mask_feat_pred, kernel_preds, cate_preds = preds
            device = mask_feat_pred.device
            batch_size = mask_feat_pred.shape[0]
            level_nums = len(cate_preds)
            assert len(self.grid_nums) == level_nums
            mask_feat_pred_h, mask_feat_pred_w = mask_feat_pred.shape[
                2], mask_feat_pred.shape[3]

            cate_preds = [
                self.points_nms(torch.sigmoid(per_level_cate_preds)).permute(
                    0, 2, 3, 1) for per_level_cate_preds in cate_preds
            ]
            kernel_preds = [
                per_level_kernel_preds.permute(0, 2, 3, 1)
                for per_level_kernel_preds in kernel_preds
            ]

            input_size_h = mask_feat_pred_h * self.mask_feature_upsample_scale
            input_size_w = mask_feat_pred_w * self.mask_feature_upsample_scale
            batch_masks, batch_labels, batch_scores = [], [], []
            for batch_idx in range(batch_size):
                per_image_cate_preds, per_image_kernel_preds = [], []
                for level_idx in range(level_nums):
                    num_classes = cate_preds[level_idx][batch_idx].shape[-1]
                    per_image_cate_preds.append(
                        cate_preds[level_idx][batch_idx].reshape(
                            -1, num_classes))
                    kernel_planes = kernel_preds[level_idx][batch_idx].shape[
                        -1]
                    per_image_kernel_preds.append(
                        kernel_preds[level_idx][batch_idx].reshape(
                            -1, kernel_planes))
                per_image_mask_feat_pred = mask_feat_pred[batch_idx].unsqueeze(
                    0)
                per_image_cate_preds = torch.cat(per_image_cate_preds, dim=0)
                per_image_kernel_preds = torch.cat(per_image_kernel_preds,
                                                   dim=0)

                min_score_keep_indexs = (per_image_cate_preds
                                         > self.min_score_threshold)

                per_image_scores = per_image_cate_preds[min_score_keep_indexs]

                empty_per_image_masks = np.zeros(
                    (0, input_size_h, input_size_w), dtype=np.float32)
                empty_per_image_labels = np.zeros((0), dtype=np.float32)
                empty_per_image_scores = np.zeros((0), dtype=np.float32)

                if per_image_scores.shape[0] == 0:
                    batch_masks.append(empty_per_image_masks)
                    batch_labels.append(empty_per_image_labels)
                    batch_scores.append(empty_per_image_scores)
                    continue

                # cate_labels & kernel_preds
                min_score_keep_indexs = min_score_keep_indexs.nonzero()
                per_image_labels = min_score_keep_indexs[:, 1]
                per_image_kernel_preds = per_image_kernel_preds[
                    min_score_keep_indexs[:, 0]]

                # trans vector.
                per_image_size_trans = per_image_labels.new_tensor(
                    self.grid_nums).pow(2).cumsum(0)
                strides = per_image_kernel_preds.new_ones(
                    per_image_size_trans[-1])

                strides[:per_image_size_trans[0]] *= self.strides[0]
                for ind in range(1, level_nums):
                    strides[per_image_size_trans[ind - 1]:
                            per_image_size_trans[ind]] *= self.strides[ind]
                strides = strides[min_score_keep_indexs[:, 0]]

                # mask encoding.
                I, N = per_image_kernel_preds.shape
                per_image_kernel_preds = per_image_kernel_preds.view(
                    I, N, 1, 1)

                per_image_mask_feat_pred = F.conv2d(per_image_mask_feat_pred,
                                                    per_image_kernel_preds,
                                                    stride=1).squeeze(0)
                per_image_mask_feat_pred = torch.sigmoid(
                    per_image_mask_feat_pred)

                # mask.
                per_image_masks = per_image_mask_feat_pred > self.mask_threshold
                sum_masks = per_image_masks.sum((1, 2)).float()

                keep_flag = sum_masks > strides
                if keep_flag.sum() == 0:
                    batch_masks.append(empty_per_image_masks)
                    batch_labels.append(empty_per_image_labels)
                    batch_scores.append(empty_per_image_scores)
                    continue

                per_image_masks = per_image_masks[keep_flag]
                per_image_mask_feat_pred = per_image_mask_feat_pred[keep_flag]
                sum_masks = sum_masks[keep_flag]
                per_image_scores = per_image_scores[keep_flag]
                per_image_labels = per_image_labels[keep_flag]

                # maskness.
                per_image_seg_scores = (per_image_mask_feat_pred *
                                        per_image_masks.float()).sum(
                                            (1, 2)) / sum_masks
                per_image_scores *= per_image_seg_scores

                # sort and keep topn
                sort_indexs = torch.argsort(per_image_scores, descending=True)
                if sort_indexs.shape[0] > self.topn:
                    sort_indexs = sort_indexs[:self.topn]

                per_image_masks = per_image_masks[sort_indexs]
                per_image_mask_feat_pred = per_image_mask_feat_pred[
                    sort_indexs]
                sum_masks = sum_masks[sort_indexs]
                per_image_scores = per_image_scores[sort_indexs]
                per_image_labels = per_image_labels[sort_indexs]

                # Matrix NMS
                per_image_scores = self.matrix_nms(per_image_masks,
                                                   per_image_labels,
                                                   per_image_scores,
                                                   sum_masks=sum_masks)

                # filter.
                keep_indexes = per_image_scores >= self.update_threshold
                if keep_indexes.sum() == 0:
                    batch_masks.append(empty_per_image_masks)
                    batch_labels.append(empty_per_image_labels)
                    batch_scores.append(empty_per_image_scores)
                    continue

                per_image_mask_feat_pred = per_image_mask_feat_pred[
                    keep_indexes]
                per_image_scores = per_image_scores[keep_indexes]
                per_image_labels = per_image_labels[keep_indexes]

                keep_score_indexes = per_image_scores >= self.keep_score_threshold
                per_image_mask_feat_pred = per_image_mask_feat_pred[
                    keep_score_indexes]
                per_image_labels = per_image_labels[keep_score_indexes]
                per_image_scores = per_image_scores[keep_score_indexes]

                if per_image_mask_feat_pred.shape[0] == 0:
                    batch_masks.append(empty_per_image_masks)
                    batch_labels.append(empty_per_image_labels)
                    batch_scores.append(empty_per_image_scores)
                    continue

                # sort and keep top_k
                sort_indexs = torch.argsort(per_image_scores, descending=True)
                if sort_indexs.shape[0] > self.max_mask_num:
                    sort_indexs = sort_indexs[:self.max_mask_num]

                per_image_mask_feat_pred = per_image_mask_feat_pred[
                    sort_indexs]
                per_image_scores = per_image_scores[sort_indexs]
                per_image_labels = per_image_labels[sort_indexs]

                if per_image_mask_feat_pred.shape[0] == 0:
                    batch_masks.append(empty_per_image_masks)
                    batch_labels.append(empty_per_image_labels)
                    batch_scores.append(empty_per_image_scores)
                    continue

                upsampled_size_out = (mask_feat_pred_h *
                                      self.mask_feature_upsample_scale,
                                      mask_feat_pred_w *
                                      self.mask_feature_upsample_scale)
                per_image_masks = F.interpolate(
                    per_image_mask_feat_pred.unsqueeze(0),
                    size=upsampled_size_out,
                    mode='bilinear',
                    align_corners=True)

                per_image_scaled_sizes = scaled_sizes[batch_idx]
                per_image_masks = per_image_masks[:, :, :int(
                    per_image_scaled_sizes[0]), :int(per_image_scaled_sizes[1]
                                                     )]

                per_image_origin_sizes = origin_sizes[batch_idx]
                per_image_origin_h, per_image_origin_w = int(
                    per_image_origin_sizes[0]), int(per_image_origin_sizes[1])
                per_image_masks = F.interpolate(
                    per_image_masks,
                    size=[per_image_origin_h, per_image_origin_w],
                    mode='bilinear',
                    align_corners=True)

                per_image_masks = per_image_masks.squeeze(0)
                per_image_masks = (per_image_masks
                                   > self.mask_threshold).to(torch.uint8)

                per_image_masks = per_image_masks.cpu().numpy()
                per_image_labels = per_image_labels.cpu().numpy()
                per_image_scores = per_image_scores.cpu().numpy()

                batch_masks.append(per_image_masks)
                batch_labels.append(per_image_labels)
                batch_scores.append(per_image_scores)

            return batch_masks, batch_labels, batch_scores

    def points_nms(self, heatmap, kernel=2):
        # kernel must be 2
        heatmap_max = F.max_pool2d(heatmap, (kernel, kernel),
                                   stride=1,
                                   padding=1)
        keep = (heatmap_max[:, :, :-1, :-1] == heatmap).float()

        return heatmap * keep

    def matrix_nms(self,
                   seg_masks,
                   cate_labels,
                   cate_scores,
                   sigma=2.0,
                   sum_masks=None):
        """Matrix NMS for multi-class masks.

        Args:
            seg_masks (Tensor): shape (n, h, w)
            cate_labels (Tensor): shape (n), mask labels in descending order
            cate_scores (Tensor): shape (n), mask scores in descending order
            sigma (float): std in gaussian method
            sum_masks (Tensor): The sum of seg_masks

        Returns:
            Tensor: cate_scores_update, tensors of shape (n)
        """

        n_samples = cate_labels.shape[0]
        if sum_masks is None:
            sum_masks = seg_masks.sum((1, 2)).float()
        seg_masks = seg_masks.reshape(n_samples, -1).float()
        # inter.
        inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (
            inter_matrix /
            (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(
                diagonal=1)
        # label_specific matrix.
        cate_labels_x = cate_labels.expand(n_samples, n_samples)
        label_matrix = (cate_labels_x == cate_labels_x.transpose(
            1, 0)).float().triu(diagonal=1)

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(n_samples,
                                               n_samples).transpose(1, 0)

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms, kernel == 'gaussian'
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

        # update the score.
        cate_scores_update = cate_scores * decay_coefficient

        return cate_scores_update


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

    from simpleAICV.instance_segmentation.datasets.cocodataset import CocoInstanceSegmentation
    from simpleAICV.instance_segmentation.common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, SOLOV2InstanceSegmentationCollater, YOLACTInstanceSegmentationCollater

    cocodataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='train2017',
        filter_no_object_image=False,
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=1024,
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=True,
                                       multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = YOLACTInstanceSegmentationCollater(resize=1024,
                                                  resize_type='yolo_style')
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.instance_segmentation.models.yolact import resnet50_yolact
    net = resnet50_yolact()
    decode = YOLACTDecoder(resize=1024,
                           resize_type='yolo_style',
                           scales=[24, 48, 96, 192, 384],
                           ratios=[1, 1 / 2, 2],
                           strides=[8, 16, 32, 64, 128],
                           max_feature_upsample_scale=8,
                           topn=200,
                           max_object_num=100,
                           min_score_threshold=0.05,
                           nms_threshold=0.5)
    for data in tqdm(train_loader):
        images, boxes, masks, scales, sizes, origin_sizes = data[
            'image'], data['box'], data['mask'], data['scale'], data[
                'size'], data['origin_size']
        print('1111', images.shape, len(boxes), len(masks), scales.shape,
              sizes.shape, origin_sizes.shape)
        preds = net(images)
        batch_masks, batch_labels, batch_scores = decode(
            preds, scales, origin_sizes)

        for per_image_masks, per_image_labels, per_image_scores in zip(
                batch_masks, batch_labels, batch_scores):
            print('2222', per_image_masks.shape, per_image_labels.shape,
                  per_image_scores.shape)
        break

    from torch.utils.data import DataLoader
    collater = SOLOV2InstanceSegmentationCollater(resize=1024,
                                                  resize_type='yolo_style')
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.instance_segmentation.models.solov2 import resnet50_solov2
    net = resnet50_solov2()
    decode = SOLOV2Decoder(strides=(8, 8, 16, 32, 32),
                           grid_nums=(40, 36, 24, 16, 12),
                           mask_feature_upsample_scale=4,
                           max_mask_num=100,
                           topn=500,
                           min_score_threshold=0.1,
                           keep_score_threshold=0.1,
                           mask_threshold=0.5,
                           update_threshold=0.05)

    for data in tqdm(train_loader):
        images, boxes, masks, scales, sizes, origin_sizes = data[
            'image'], data['box'], data['mask'], data['scale'], data[
                'size'], data['origin_size']
        print('1111', images.shape, len(boxes), len(masks), scales.shape,
              sizes.shape, origin_sizes.shape)
        preds = net(images)
        batch_masks, batch_labels, batch_scores = decode(
            preds, sizes, origin_sizes)

        for per_image_masks, per_image_labels, per_image_scores in zip(
                batch_masks, batch_labels, batch_scores):
            print('2222', per_image_masks.shape, per_image_labels.shape,
                  per_image_scores.shape)
        break
