import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

__all__ = [
    'RetinaDecoder',
    'FCOSDecoder',
    'Yolov4Decoder',
    'Yolov5Decoder',
]


class DetNMSMethod:
    def __init__(self, nms_type='python_nms', nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'DIoU_python_nms'], 'wrong nms type!'
        self.nms_type = nms_type
        self.nms_threshold = nms_threshold

    def __call__(self, sorted_bboxes, sorted_scores):
        """
        sorted_bboxes:[anchor_nums,4],4:x_min,y_min,x_max,y_max
        sorted_scores:[anchor_nums],classification predict scores
        """
        if self.nms_type == 'torch_nms':
            keep = nms(sorted_bboxes, sorted_scores, self.nms_threshold)
        else:
            sorted_bboxes_wh = sorted_bboxes[:, 2:4] - sorted_bboxes[:, 0:2]
            sorted_bboxes_areas = sorted_bboxes_wh[:, 0] * sorted_bboxes_wh[:,
                                                                            1]
            sorted_bboxes_areas = torch.clamp(sorted_bboxes_areas, min=1e-4)

            device = sorted_scores.device
            indexes = torch.tensor([i for i in range(sorted_scores.shape[0])
                                    ]).to(device)

            keep = []
            while indexes.shape[0] > 0:
                keep_idx = indexes[0]
                keep.append(keep_idx)
                indexes = indexes[1:]
                if len(indexes) == 0:
                    break

                keep_box_area = sorted_bboxes_areas[keep_idx]

                overlap_area_top_left = torch.max(sorted_bboxes[keep_idx, 0:2],
                                                  sorted_bboxes[indexes, 0:2])
                overlap_area_bot_right = torch.min(
                    sorted_bboxes[keep_idx, 2:4], sorted_bboxes[indexes, 2:4])
                overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                                 overlap_area_top_left,
                                                 min=0)
                overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:,
                                                                             1]

                # compute ious for top1 pred_bbox and the other pred_bboxes
                union_area = keep_box_area + sorted_bboxes_areas[
                    indexes] - overlap_area
                union_area = torch.clamp(union_area, min=1e-4)
                ious = overlap_area / union_area

                if self.nms_type == 'DIoU_python_nms':
                    enclose_area_top_left = torch.min(
                        sorted_bboxes[keep_idx, 0:2], sorted_bboxes[indexes,
                                                                    0:2])
                    enclose_area_bot_right = torch.max(
                        sorted_bboxes[keep_idx, 2:4], sorted_bboxes[indexes,
                                                                    2:4])
                    enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                                     enclose_area_top_left,
                                                     min=1e-4)
                    # c2:convex diagonal squared
                    c2 = ((enclose_area_sizes)**2).sum(axis=1)
                    c2 = torch.clamp(c2, min=1e-4)
                    # p2:center distance squared
                    keep_box_ctr = (sorted_bboxes[keep_idx, 2:4] +
                                    sorted_bboxes[keep_idx, 0:2]) / 2
                    other_boxes_ctr = (sorted_bboxes[indexes, 2:4] +
                                       sorted_bboxes[indexes, 0:2]) / 2
                    p2 = (keep_box_ctr - other_boxes_ctr)**2
                    p2 = p2.sum(axis=1)
                    ious = ious - p2 / c2

                candidate_indexes = torch.where(ious < self.nms_threshold)[0]
                indexes = indexes[candidate_indexes]

            keep = torch.tensor(keep).to(device)

        return keep


class RetinaDecoder(nn.Module):
    def __init__(self,
                 topn=1000,
                 min_score_threshold=0.05,
                 nms_type='torch_nms',
                 nms_threshold=0.5,
                 max_object_num=100):
        super(RetinaDecoder, self).__init__()
        assert nms_type in ['torch_nms', 'python_nms',
                            'DIoU_python_nms'], 'wrong nms type!'
        self.topn = topn
        self.min_score_threshold = min_score_threshold
        self.max_object_num = max_object_num
        self.nms_function = DetNMSMethod(nms_type=nms_type,
                                         nms_threshold=nms_threshold)

    def forward(self, cls_heads, reg_heads, batch_anchors):
        with torch.no_grad():
            device = cls_heads[0].device
            cls_heads = torch.cat(cls_heads, axis=1)
            reg_heads = torch.cat(reg_heads, axis=1)
            batch_anchors = torch.cat(batch_anchors, axis=1)
            cls_scores, cls_classes = torch.max(cls_heads, dim=2)
            pred_bboxes = self.snap_txtytwth_to_x1y1x2y2(
                reg_heads, batch_anchors)

            batch_size = cls_scores.shape[0]
            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            for i, (per_image_scores, per_image_score_classes,
                    per_image_pred_bboxes) in enumerate(
                        zip(cls_scores, cls_classes, pred_bboxes)):
                score_classes = per_image_score_classes[
                    per_image_scores > self.min_score_threshold].float()
                bboxes = per_image_pred_bboxes[
                    per_image_scores > self.min_score_threshold].float()
                scores = per_image_scores[
                    per_image_scores > self.min_score_threshold].float()

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_bboxes = bboxes[sorted_indexes]

                    if self.topn < sorted_scores.shape[0]:
                        sorted_scores = sorted_scores[0:self.topn]
                        sorted_score_classes = sorted_score_classes[0:self.
                                                                    topn]
                        sorted_bboxes = sorted_bboxes[0:self.topn]

                    keep = self.nms_function(sorted_bboxes, sorted_scores)

                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_bboxes = sorted_bboxes[keep]

                    final_detection_num = min(self.max_object_num,
                                              keep_scores.shape[0])

                    batch_scores[i, 0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    batch_classes[i, 0:final_detection_num] = keep_classes[
                        0:final_detection_num]
                    batch_bboxes[i, 0:final_detection_num, :] = keep_bboxes[
                        0:final_detection_num, :]

            # batch_scores shape:[batch_size,max_object_num]
            # batch_classes shape:[batch_size,max_object_num]
            # batch_bboxes shape[batch_size,max_object_num,4]
            return batch_scores, batch_classes, batch_bboxes

    def snap_txtytwth_to_x1y1x2y2(self, reg_heads, anchors):
        '''
        snap reg heads to pred bboxes
        reg_heads:[batch_size,anchor_nums,4],4:[tx,ty,tw,th]
        anchors:[batch_size,anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        '''
        anchors_wh = anchors[:, :, 2:4] - anchors[:, :, 0:2]
        anchors_ctr = anchors[:, :, 0:2] + 0.5 * anchors_wh

        pred_bboxes_wh = torch.exp(reg_heads[:, :, 2:4]) * anchors_wh
        pred_bboxes_ctr = reg_heads[:, :, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_min_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_max_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = torch.cat(
            [pred_bboxes_x_min_y_min, pred_bboxes_x_max_y_max], dim=2)
        pred_bboxes = pred_bboxes.int()

        # pred bboxes shape:[anchor_nums,4]
        return pred_bboxes


class FCOSDecoder(nn.Module):
    def __init__(self,
                 topn=1000,
                 min_score_threshold=0.05,
                 nms_type='torch_nms',
                 nms_threshold=0.6,
                 max_object_num=100):
        super(FCOSDecoder, self).__init__()
        assert nms_type in ['torch_nms', 'python_nms',
                            'DIoU_python_nms'], 'wrong nms type!'
        self.topn = topn
        self.min_score_threshold = min_score_threshold
        self.max_object_num = max_object_num
        self.nms_function = DetNMSMethod(nms_type=nms_type,
                                         nms_threshold=nms_threshold)

    def forward(self, cls_heads, reg_heads, center_heads, batch_positions):
        with torch.no_grad():
            device = cls_heads[0].device
            cls_scores, cls_classes, pred_bboxes = [], [], []
            for per_level_cls_head, per_level_reg_head, per_level_center_head, per_level_position in zip(
                    cls_heads, reg_heads, center_heads, batch_positions):
                per_level_cls_head = per_level_cls_head.view(
                    per_level_cls_head.shape[0], -1,
                    per_level_cls_head.shape[-1])
                per_level_reg_head = per_level_reg_head.view(
                    per_level_reg_head.shape[0], -1,
                    per_level_reg_head.shape[-1])
                per_level_center_head = per_level_center_head.view(
                    per_level_center_head.shape[0], -1,
                    per_level_center_head.shape[-1])
                per_level_position = per_level_position.view(
                    per_level_position.shape[0], -1,
                    per_level_position.shape[-1])

                per_level_cls_scores, per_level_cls_classes = torch.max(
                    per_level_cls_head, dim=2)
                per_level_cls_scores = torch.sqrt(
                    per_level_cls_scores * per_level_center_head.squeeze(-1))
                per_level_pred_bboxes = self.snap_ltrb_to_x1y1x2y2(
                    per_level_reg_head, per_level_position)

                cls_scores.append(per_level_cls_scores)
                cls_classes.append(per_level_cls_classes)
                pred_bboxes.append(per_level_pred_bboxes)

            cls_scores = torch.cat(cls_scores, axis=1)
            cls_classes = torch.cat(cls_classes, axis=1)
            pred_bboxes = torch.cat(pred_bboxes, axis=1)

            batch_size = cls_scores.shape[0]
            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            for i, (per_image_scores, per_image_score_classes,
                    per_image_pred_bboxes) in enumerate(
                        zip(cls_scores, cls_classes, pred_bboxes)):
                score_classes = per_image_score_classes[
                    per_image_scores > self.min_score_threshold].float()
                bboxes = per_image_pred_bboxes[
                    per_image_scores > self.min_score_threshold].float()
                scores = per_image_scores[
                    per_image_scores > self.min_score_threshold].float()

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_bboxes = bboxes[sorted_indexes]

                    if self.topn < sorted_scores.shape[0]:
                        sorted_scores = sorted_scores[0:self.topn]
                        sorted_score_classes = sorted_score_classes[0:self.
                                                                    topn]
                        sorted_bboxes = sorted_bboxes[0:self.topn]

                    keep = self.nms_function(sorted_bboxes, sorted_scores)

                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_bboxes = sorted_bboxes[keep]

                    final_detection_num = min(self.max_object_num,
                                              keep_scores.shape[0])

                    batch_scores[i, 0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    batch_classes[i, 0:final_detection_num] = keep_classes[
                        0:final_detection_num]
                    batch_bboxes[i, 0:final_detection_num, :] = keep_bboxes[
                        0:final_detection_num, :]

            # batch_scores shape:[batch_size,max_object_num]
            # batch_classes shape:[batch_size,max_object_num]
            # batch_bboxes shape[batch_size,max_object_num,4]
            return batch_scores, batch_classes, batch_bboxes

    def snap_ltrb_to_x1y1x2y2(self, reg_preds, points_position):
        '''
        snap reg preds to pred bboxes
        reg_preds:[batch_size,point_nums,4],4:[l,t,r,b]
        points_position:[batch_size,point_nums,2],2:[point_ctr_x,point_ctr_y]
        '''
        pred_bboxes_xy_min = points_position - reg_preds[:, :, 0:2]
        pred_bboxes_xy_max = points_position + reg_preds[:, :, 2:4]
        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                dim=2)
        pred_bboxes = pred_bboxes.int()

        # pred bboxes shape:[points_num,4]
        return pred_bboxes


class Yolov4Decoder(nn.Module):
    def __init__(self,
                 topn=1000,
                 min_score_threshold=0.05,
                 nms_type='torch_nms',
                 nms_threshold=0.5,
                 max_object_num=100):
        super(Yolov4Decoder, self).__init__()
        assert nms_type in ['torch_nms', 'python_nms',
                            'DIoU_python_nms'], 'wrong nms type!'
        self.topn = topn
        self.min_score_threshold = min_score_threshold
        self.max_object_num = max_object_num
        self.nms_function = DetNMSMethod(nms_type=nms_type,
                                         nms_threshold=nms_threshold)

    def forward(self, obj_reg_cls_heads, batch_anchors):
        with torch.no_grad():
            device = obj_reg_cls_heads[0].device
            cls_scores, cls_classes, pred_bboxes = [], [], []
            for per_level_obj_reg_cls_head, per_level_anchor in zip(
                    obj_reg_cls_heads, batch_anchors):
                per_level_obj_reg_cls_head = per_level_obj_reg_cls_head.view(
                    per_level_obj_reg_cls_head.shape[0], -1,
                    per_level_obj_reg_cls_head.shape[-1])
                per_level_anchor = per_level_anchor.view(
                    per_level_anchor.shape[0], -1, per_level_anchor.shape[-1])

                per_level_cls_scores, per_level_cls_classes = torch.max(
                    per_level_obj_reg_cls_head[:, :, 5:], dim=2)
                per_level_cls_scores = per_level_cls_scores * per_level_obj_reg_cls_head[:, :,
                                                                                         0]
                per_level_reg_heads = per_level_obj_reg_cls_head[:, :, 1:5]

                per_level_pred_bboxes = self.snap_txtytwth_to_x1y1x2y2(
                    per_level_reg_heads, per_level_anchor)

                cls_scores.append(per_level_cls_scores)
                cls_classes.append(per_level_cls_classes)
                pred_bboxes.append(per_level_pred_bboxes)

            cls_scores = torch.cat(cls_scores, axis=1)
            cls_classes = torch.cat(cls_classes, axis=1)
            pred_bboxes = torch.cat(pred_bboxes, axis=1)

            batch_size = cls_scores.shape[0]
            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            for i, (per_image_scores, per_image_score_classes,
                    per_image_pred_bboxes) in enumerate(
                        zip(cls_scores, cls_classes, pred_bboxes)):
                score_classes = per_image_score_classes[
                    per_image_scores > self.min_score_threshold].float()
                bboxes = per_image_pred_bboxes[
                    per_image_scores > self.min_score_threshold].float()
                scores = per_image_scores[
                    per_image_scores > self.min_score_threshold].float()

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_bboxes = bboxes[sorted_indexes]

                    if self.topn < sorted_scores.shape[0]:
                        sorted_scores = sorted_scores[0:self.topn]
                        sorted_score_classes = sorted_score_classes[0:self.
                                                                    topn]
                        sorted_bboxes = sorted_bboxes[0:self.topn]

                    keep = self.nms_function(sorted_bboxes, sorted_scores)

                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_bboxes = sorted_bboxes[keep]

                    final_detection_num = min(self.max_object_num,
                                              keep_scores.shape[0])

                    batch_scores[i, 0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    batch_classes[i, 0:final_detection_num] = keep_classes[
                        0:final_detection_num]
                    batch_bboxes[i, 0:final_detection_num, :] = keep_bboxes[
                        0:final_detection_num, :]

            # batch_scores shape:[batch_size,max_object_num]
            # batch_classes shape:[batch_size,max_object_num]
            # batch_bboxes shape[batch_size,max_object_num,4]
            return batch_scores, batch_classes, batch_bboxes

    def snap_txtytwth_to_x1y1x2y2(self, reg_heads, batch_anchors):
        '''
        snap reg heads to pred bboxes
        reg_heads:[batch_size,anchor_nums,4],4:[tx,ty,tw,th]
        batch_anchors:[batch_size,anchor_nums,5],2:[grids_x_index,grids_y_index,relative_anchor_w,relative_anchor_h,stride]
        '''
        pred_bboxes_xy_ctr = (reg_heads[:, :, 0:2] + batch_anchors[:, :, 0:2]
                              ) * batch_anchors[:, :, 4:5]
        pred_bboxes_wh = reg_heads[:, :, 2:
                                   4] * batch_anchors[:, :, 2:
                                                      4] * batch_anchors[:, :,
                                                                         4:5]

        pred_bboxes_xy_min = pred_bboxes_xy_ctr - pred_bboxes_wh / 2
        pred_bboxes_xy_max = pred_bboxes_xy_ctr + pred_bboxes_wh / 2
        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                dim=2)
        pred_bboxes = pred_bboxes.int()

        # pred bboxes shape:[points_num,4]
        return pred_bboxes


class Yolov5Decoder(nn.Module):
    def __init__(self,
                 topn=1000,
                 min_score_threshold=0.05,
                 nms_type='torch_nms',
                 nms_threshold=0.5,
                 max_object_num=100):
        super(Yolov5Decoder, self).__init__()
        assert nms_type in ['torch_nms', 'python_nms',
                            'DIoU_python_nms'], 'wrong nms type!'
        self.topn = topn
        self.min_score_threshold = min_score_threshold
        self.max_object_num = max_object_num
        self.nms_function = DetNMSMethod(nms_type=nms_type,
                                         nms_threshold=nms_threshold)

    def forward(self, obj_reg_cls_heads, batch_anchors):
        with torch.no_grad():
            device = obj_reg_cls_heads[0].device
            cls_scores, cls_classes, pred_bboxes = [], [], []
            for per_level_obj_reg_cls_head, per_level_anchor in zip(
                    obj_reg_cls_heads, batch_anchors):
                per_level_obj_reg_cls_head = per_level_obj_reg_cls_head.view(
                    per_level_obj_reg_cls_head.shape[0], -1,
                    per_level_obj_reg_cls_head.shape[-1])
                per_level_anchor = per_level_anchor.view(
                    per_level_anchor.shape[0], -1, per_level_anchor.shape[-1])

                per_level_cls_scores, per_level_cls_classes = torch.max(
                    per_level_obj_reg_cls_head[:, :, 5:], dim=2)
                per_level_cls_scores = per_level_cls_scores * per_level_obj_reg_cls_head[:, :,
                                                                                         0]
                per_level_reg_heads = per_level_obj_reg_cls_head[:, :, 1:5]

                per_level_pred_bboxes = self.snap_txtytwth_to_x1y1x2y2(
                    per_level_reg_heads, per_level_anchor)

                cls_scores.append(per_level_cls_scores)
                cls_classes.append(per_level_cls_classes)
                pred_bboxes.append(per_level_pred_bboxes)

            cls_scores = torch.cat(cls_scores, axis=1)
            cls_classes = torch.cat(cls_classes, axis=1)
            pred_bboxes = torch.cat(pred_bboxes, axis=1)

            batch_size = cls_scores.shape[0]
            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            for i, (per_image_scores, per_image_score_classes,
                    per_image_pred_bboxes) in enumerate(
                        zip(cls_scores, cls_classes, pred_bboxes)):
                score_classes = per_image_score_classes[
                    per_image_scores > self.min_score_threshold].float()
                bboxes = per_image_pred_bboxes[
                    per_image_scores > self.min_score_threshold].float()
                scores = per_image_scores[
                    per_image_scores > self.min_score_threshold].float()

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_bboxes = bboxes[sorted_indexes]

                    if self.topn < sorted_scores.shape[0]:
                        sorted_scores = sorted_scores[0:self.topn]
                        sorted_score_classes = sorted_score_classes[0:self.
                                                                    topn]
                        sorted_bboxes = sorted_bboxes[0:self.topn]

                    keep = self.nms_function(sorted_bboxes, sorted_scores)

                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_bboxes = sorted_bboxes[keep]

                    final_detection_num = min(self.max_object_num,
                                              keep_scores.shape[0])

                    batch_scores[i, 0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    batch_classes[i, 0:final_detection_num] = keep_classes[
                        0:final_detection_num]
                    batch_bboxes[i, 0:final_detection_num, :] = keep_bboxes[
                        0:final_detection_num, :]

            # batch_scores shape:[batch_size,max_object_num]
            # batch_classes shape:[batch_size,max_object_num]
            # batch_bboxes shape[batch_size,max_object_num,4]
            return batch_scores, batch_classes, batch_bboxes

    def snap_txtytwth_to_x1y1x2y2(self, reg_heads, batch_anchors):
        '''
        snap reg heads to pred bboxes
        reg_heads:[batch_size,anchor_nums,4],4:[tx,ty,tw,th]
        batch_anchors:[batch_size,anchor_nums,5],2:[grids_x_index,grids_y_index,relative_anchor_w,relative_anchor_h,stride]
        '''
        pred_bboxes_xy_ctr = (reg_heads[:, :, 0:2] * 2. - 0.5 +
                              batch_anchors[:, :, 0:2]) * batch_anchors[:, :,
                                                                        4:5]
        pred_bboxes_wh = (
            (reg_heads[:, :, 2:4] * 2)**
            2) * batch_anchors[:, :, 2:4] * batch_anchors[:, :, 4:5]

        pred_bboxes_xy_min = pred_bboxes_xy_ctr - pred_bboxes_wh / 2
        pred_bboxes_xy_max = pred_bboxes_xy_ctr + pred_bboxes_wh / 2
        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                dim=2)
        pred_bboxes = pred_bboxes.int()

        # pred bboxes shape:[points_num,4]
        return pred_bboxes


class CenterNetDecoder(nn.Module):
    def __init__(self,
                 topk=100,
                 stride=4,
                 min_score_threshold=0.05,
                 max_object_num=100):
        super(CenterNetDecoder, self).__init__()
        self.topk = topk
        self.stride = stride
        self.min_score_threshold = min_score_threshold
        self.max_object_num = max_object_num

    def forward(self, heatmap_heads, offset_heads, wh_heads):
        with torch.no_grad():
            device = heatmap_heads.device
            heatmap_heads = torch.sigmoid(heatmap_heads)

            batch_size = heatmap_heads.shape[0]
            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            for i, (per_image_heatmap_heads, per_image_offset_heads,
                    per_image_wh_heads) in enumerate(
                        zip(heatmap_heads, offset_heads, wh_heads)):
                #filter and keep points which value large than the surrounding 8 points
                per_image_heatmap_heads = self.nms(per_image_heatmap_heads)
                topk_score, topk_indexes, topk_classes, topk_ys, topk_xs = self.get_topk(
                    per_image_heatmap_heads, K=self.topk)

                per_image_offset_heads = per_image_offset_heads.permute(
                    1, 2, 0).contiguous().view(-1, 2)
                per_image_offset_heads = torch.gather(
                    per_image_offset_heads, 0, topk_indexes.repeat(1, 2))
                topk_xs = topk_xs + per_image_offset_heads[:, 0:1]
                topk_ys = topk_ys + per_image_offset_heads[:, 1:2]

                per_image_wh_heads = per_image_wh_heads.permute(
                    1, 2, 0).contiguous().view(-1, 2)
                per_image_wh_heads = torch.gather(per_image_wh_heads, 0,
                                                  topk_indexes.repeat(1, 2))

                topk_bboxes = torch.cat([
                    topk_xs - per_image_wh_heads[:, 0:1] / 2,
                    topk_ys - per_image_wh_heads[:, 1:2] / 2,
                    topk_xs + per_image_wh_heads[:, 0:1] / 2,
                    topk_ys + per_image_wh_heads[:, 1:2] / 2
                ],
                                        dim=1)
                topk_bboxes = topk_bboxes * self.stride

                topk_classes = topk_classes[
                    topk_score > self.min_score_threshold].float()
                topk_bboxes = topk_bboxes[
                    topk_score > self.min_score_threshold].float()
                topk_score = topk_score[
                    topk_score > self.min_score_threshold].float()

                final_detection_num = min(self.max_object_num,
                                          topk_score.shape[0])

                batch_scores[
                    i,
                    0:final_detection_num] = topk_score[0:final_detection_num]
                batch_classes[i, 0:final_detection_num] = topk_classes[
                    0:final_detection_num]
                batch_bboxes[i, 0:final_detection_num, :] = topk_bboxes[
                    0:final_detection_num, :]

            # batch_scores shape:[batch_size,topk]
            # batch_classes shape:[batch_size,topk]
            # batch_bboxes shape[batch_size,topk,4]
            return batch_scores, batch_classes, batch_bboxes

    def nms(self, per_image_heatmap_heads, kernel=3):
        per_image_heatmap_max = F.max_pool2d(per_image_heatmap_heads,
                                             kernel,
                                             stride=1,
                                             padding=(kernel - 1) // 2)
        keep = (per_image_heatmap_max == per_image_heatmap_heads).float()

        return per_image_heatmap_heads * keep

    def get_topk(self, per_image_heatmap_heads, K):
        num_classes, H, W = per_image_heatmap_heads.shape[
            0], per_image_heatmap_heads.shape[
                1], per_image_heatmap_heads.shape[2]

        per_image_heatmap_heads = per_image_heatmap_heads.view(num_classes, -1)
        # 先取每个类别的heatmap上前k个最大激活点
        topk_scores, topk_indexes = torch.topk(per_image_heatmap_heads.view(
            num_classes, -1),
                                               K,
                                               dim=-1)

        # 取余，计算topk项在feature map上的y和x index(位置)
        topk_indexes = topk_indexes % (H * W)
        topk_ys = (topk_indexes / W).int().float()
        topk_xs = (topk_indexes % W).int().float()

        # 在topk_scores中取前k个最大分数(所有类别混合在一起再取)
        topk_score, topk_score_indexes = torch.topk(topk_scores.view(-1),
                                                    K,
                                                    dim=-1)

        # 整除K得到预测的类编号，因为heatmap view前第一个维度是类别数
        topk_classes = (topk_score_indexes / K).int()

        topk_score_indexes = topk_score_indexes.unsqueeze(-1)
        topk_indexes = torch.gather(topk_indexes.view(-1, 1), 0,
                                    topk_score_indexes)
        topk_ys = torch.gather(topk_ys.view(-1, 1), 0, topk_score_indexes)
        topk_xs = torch.gather(topk_xs.view(-1, 1), 0, topk_score_indexes)

        return topk_score, topk_indexes, topk_classes, topk_ys, topk_xs


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

    from simpleAICV.detection.models.retinanet import RetinaNet
    net = RetinaNet(resnet_type='resnet50')
    image_h, image_w = 640, 640
    cls_heads, reg_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = RetinaDecoder()
    batch_scores, batch_classes, batch_pred_bboxes = decode(
        cls_heads, reg_heads, batch_anchors)
    print('1111', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)

    from simpleAICV.detection.models.fcos import FCOS
    net = FCOS(resnet_type='resnet50')
    image_h, image_w = 600, 600
    cls_heads, reg_heads, center_heads, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = FCOSDecoder()
    batch_scores2, batch_classes2, batch_pred_bboxes2 = decode(
        cls_heads, reg_heads, center_heads, batch_positions)
    print('2222', batch_scores2.shape, batch_classes2.shape,
          batch_pred_bboxes2.shape)

    from simpleAICV.detection.models.yolov4 import YOLOV4
    net = YOLOV4(yolo_type='yolov4')
    image_h, image_w = 608, 608
    obj_reg_cls_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = Yolov4Decoder()
    batch_scores3, batch_classes3, batch_pred_bboxes3 = decode(
        obj_reg_cls_heads, batch_anchors)
    print('3333', batch_scores3.shape, batch_classes3.shape,
          batch_pred_bboxes3.shape)

    from simpleAICV.detection.models.yolov5 import YOLOV5
    net = YOLOV5(yolo_type='yolov5l')
    image_h, image_w = 640, 640
    obj_reg_cls_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = Yolov5Decoder()
    batch_scores4, batch_classes4, batch_pred_bboxes4 = decode(
        obj_reg_cls_heads, batch_anchors)
    print('4444', batch_scores4.shape, batch_classes4.shape,
          batch_pred_bboxes4.shape)

    from simpleAICV.detection.models.centernet import CenterNet
    net = CenterNet(resnet_type='resnet18')
    image_h, image_w = 512, 512
    heatmap_output, offset_output, wh_output = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = CenterNetDecoder()
    batch_scores5, batch_classes5, batch_pred_bboxes5 = decode(
        heatmap_output, offset_output, wh_output)
    print('5555', batch_scores5.shape, batch_classes5.shape,
          batch_pred_bboxes5.shape)
