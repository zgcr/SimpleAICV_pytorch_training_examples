import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms

from simpleAICV.detection.models.anchor import RetinaAnchors, FCOSPositions

__all__ = [
    'RetinaDecoder',
    'FCOSDecoder',
    'DETRDecoder',
    'DINODETRDecoder',
]


class DetNMSMethod:

    def __init__(self, nms_type='python_nms', nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        self.nms_type = nms_type
        self.nms_threshold = nms_threshold

    def __call__(self, sorted_bboxes, sorted_scores):
        '''
        sorted_bboxes:[anchor_nums,4],4:x_min,y_min,x_max,y_max
        sorted_scores:[anchor_nums],classification predict scores
        '''
        if self.nms_type == 'torch_nms':
            sorted_bboxes, sorted_scores = torch.tensor(sorted_bboxes).cpu(
            ).detach(), torch.tensor(sorted_scores).cpu().detach()
            keep = nms(sorted_bboxes, sorted_scores, self.nms_threshold)
            keep = keep.cpu().detach().numpy()
        else:
            sorted_bboxes_wh = sorted_bboxes[:, 2:4] - sorted_bboxes[:, 0:2]
            sorted_bboxes_areas = sorted_bboxes_wh[:, 0] * sorted_bboxes_wh[:,
                                                                            1]
            sorted_bboxes_areas = np.maximum(sorted_bboxes_areas, 0)

            indexes = np.array([i for i in range(sorted_scores.shape[0])],
                               dtype=np.int32)

            keep = []
            while indexes.shape[0] > 0:
                keep_idx = indexes[0]
                keep.append(keep_idx)
                indexes = indexes[1:]
                if len(indexes) == 0:
                    break

                keep_box_area = sorted_bboxes_areas[keep_idx]

                overlap_area_top_left = np.maximum(
                    sorted_bboxes[keep_idx, 0:2], sorted_bboxes[indexes, 0:2])
                overlap_area_bot_right = np.minimum(
                    sorted_bboxes[keep_idx, 2:4], sorted_bboxes[indexes, 2:4])
                overlap_area_sizes = np.maximum(
                    overlap_area_bot_right - overlap_area_top_left, 0)
                overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:,
                                                                             1]

                # compute ious for top1 pred_bbox and the other pred_bboxes
                union_area = keep_box_area + sorted_bboxes_areas[
                    indexes] - overlap_area
                union_area = np.maximum(union_area, 1e-4)
                ious = overlap_area / union_area

                if self.nms_type == 'diou_python_nms':
                    enclose_area_top_left = np.minimum(
                        sorted_bboxes[keep_idx, 0:2], sorted_bboxes[indexes,
                                                                    0:2])
                    enclose_area_bot_right = np.maximum(
                        sorted_bboxes[keep_idx, 2:4], sorted_bboxes[indexes,
                                                                    2:4])
                    enclose_area_sizes = np.maximum(
                        enclose_area_bot_right - enclose_area_top_left, 0)
                    # c2:convex diagonal squared
                    c2 = ((enclose_area_sizes)**2).sum(axis=1)
                    c2 = np.maximum(c2, 1e-4)
                    # p2:center distance squared
                    keep_box_ctr = (sorted_bboxes[keep_idx, 2:4] +
                                    sorted_bboxes[keep_idx, 0:2]) / 2
                    other_boxes_ctr = (sorted_bboxes[indexes, 2:4] +
                                       sorted_bboxes[indexes, 0:2]) / 2
                    p2 = (keep_box_ctr - other_boxes_ctr)**2
                    p2 = p2.sum(axis=1)
                    ious = ious - p2 / c2

                candidate_indexes = np.where(ious < self.nms_threshold)[0]
                indexes = indexes[candidate_indexes]

            keep = np.array(keep)

        return keep


class DecodeMethod:

    def __init__(self,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_function = DetNMSMethod(nms_type=nms_type,
                                         nms_threshold=nms_threshold)

    def __call__(self, cls_scores, cls_classes, pred_bboxes):
        batch_size = cls_scores.shape[0]
        batch_scores = np.ones(
            (batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_classes = np.ones(
            (batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_bboxes = np.zeros((batch_size, self.max_object_num, 4),
                                dtype=np.float32)

        for i, (per_image_scores, per_image_score_classes,
                per_image_pred_bboxes) in enumerate(
                    zip(cls_scores, cls_classes, pred_bboxes)):
            score_classes = per_image_score_classes[
                per_image_scores > self.min_score_threshold].astype(np.float32)
            bboxes = per_image_pred_bboxes[
                per_image_scores > self.min_score_threshold].astype(np.float32)
            scores = per_image_scores[
                per_image_scores > self.min_score_threshold].astype(np.float32)

            if scores.shape[0] != 0:
                # descending sort
                sorted_indexes = np.argsort(-scores)
                sorted_scores = scores[sorted_indexes]
                sorted_score_classes = score_classes[sorted_indexes]
                sorted_bboxes = bboxes[sorted_indexes]

                if self.topn < sorted_scores.shape[0]:
                    sorted_scores = sorted_scores[0:self.topn]
                    sorted_score_classes = sorted_score_classes[0:self.topn]
                    sorted_bboxes = sorted_bboxes[0:self.topn]

                # nms
                keep = self.nms_function(sorted_bboxes, sorted_scores)
                keep_scores = sorted_scores[keep]
                keep_classes = sorted_score_classes[keep]
                keep_bboxes = sorted_bboxes[keep]

                final_detection_num = min(self.max_object_num,
                                          keep_scores.shape[0])

                batch_scores[
                    i,
                    0:final_detection_num] = keep_scores[0:final_detection_num]
                batch_classes[i, 0:final_detection_num] = keep_classes[
                    0:final_detection_num]
                batch_bboxes[i, 0:final_detection_num, :] = keep_bboxes[
                    0:final_detection_num, :]

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]


class RetinaDecoder:

    def __init__(self,
                 areas=[[32, 32], [64, 64], [128, 128], [256, 256], [512,
                                                                     512]],
                 ratios=[0.5, 1, 2],
                 scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
                 strides=[8, 16, 32, 64, 128],
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        self.anchors = RetinaAnchors(areas=areas,
                                     ratios=ratios,
                                     scales=scales,
                                     strides=strides)
        self.decode_function = DecodeMethod(
            max_object_num=max_object_num,
            min_score_threshold=min_score_threshold,
            topn=topn,
            nms_type=nms_type,
            nms_threshold=nms_threshold)

    def __call__(self, preds):
        cls_preds, reg_preds = preds
        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in cls_preds]
        one_image_anchors = self.anchors(feature_size)

        cls_preds = np.concatenate([
            per_cls_pred.cpu().detach().numpy().reshape(
                per_cls_pred.shape[0], -1, per_cls_pred.shape[-1])
            for per_cls_pred in cls_preds
        ],
                                   axis=1)
        reg_preds = np.concatenate([
            per_reg_pred.cpu().detach().numpy().reshape(
                per_reg_pred.shape[0], -1, per_reg_pred.shape[-1])
            for per_reg_pred in reg_preds
        ],
                                   axis=1)

        one_image_anchors = np.concatenate([
            per_level_anchor.reshape(-1, per_level_anchor.shape[-1])
            for per_level_anchor in one_image_anchors
        ],
                                           axis=0)
        batch_anchors = np.repeat(np.expand_dims(one_image_anchors, axis=0),
                                  cls_preds.shape[0],
                                  axis=0)

        cls_classes = np.argmax(cls_preds, axis=2)
        cls_scores = np.concatenate([
            np.expand_dims(per_image_preds[np.arange(per_image_preds.shape[0]),
                                           per_image_cls_classes],
                           axis=0)
            for per_image_preds, per_image_cls_classes in zip(
                cls_preds, cls_classes)
        ],
                                    axis=0)

        pred_bboxes = self.snap_txtytwth_to_x1y1x2y2(reg_preds, batch_anchors)

        [batch_scores, batch_classes,
         batch_bboxes] = self.decode_function(cls_scores, cls_classes,
                                              pred_bboxes)

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]

    def snap_txtytwth_to_x1y1x2y2(self, reg_preds, anchors):
        '''
        snap reg heads to pred bboxes
        reg_preds:[batch_size,anchor_nums,4],4:[tx,ty,tw,th]
        anchors:[batch_size,anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        '''
        anchors_wh = anchors[:, :, 2:4] - anchors[:, :, 0:2]
        anchors_ctr = anchors[:, :, 0:2] + 0.5 * anchors_wh

        pred_bboxes_wh = np.exp(reg_preds[:, :, 2:4]) * anchors_wh
        pred_bboxes_ctr = reg_preds[:, :, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_min_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_max_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = np.concatenate(
            [pred_bboxes_x_min_y_min, pred_bboxes_x_max_y_max], axis=2)
        pred_bboxes = pred_bboxes.astype(np.int32)

        # pred bboxes shape:[batch,anchor_nums,4]
        return pred_bboxes


class FCOSDecoder:

    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.6):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        self.positions = FCOSPositions(strides=strides)
        self.decode_function = DecodeMethod(
            max_object_num=max_object_num,
            min_score_threshold=min_score_threshold,
            topn=topn,
            nms_type=nms_type,
            nms_threshold=nms_threshold)

    def __call__(self, preds):
        cls_preds, reg_preds, center_preds = preds
        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in cls_preds]
        one_image_positions = self.positions(feature_size)

        cls_preds = [
            per_cls_pred.cpu().detach().numpy().reshape(
                per_cls_pred.shape[0], -1, per_cls_pred.shape[-1])
            for per_cls_pred in cls_preds
        ]
        reg_preds = [
            per_reg_pred.cpu().detach().numpy().reshape(
                per_reg_pred.shape[0], -1, per_reg_pred.shape[-1])
            for per_reg_pred in reg_preds
        ]
        center_preds = [
            per_center_pred.cpu().detach().numpy().reshape(
                per_center_pred.shape[0], -1, per_center_pred.shape[-1])
            for per_center_pred in center_preds
        ]

        cls_preds = np.concatenate(cls_preds, axis=1)
        reg_preds = np.concatenate(reg_preds, axis=1)
        center_preds = np.concatenate(center_preds, axis=1)
        one_image_positions = np.concatenate([
            per_level_position.reshape(-1, per_level_position.shape[-1])
            for per_level_position in one_image_positions
        ],
                                             axis=0)
        batch_positions = np.repeat(np.expand_dims(one_image_positions,
                                                   axis=0),
                                    cls_preds.shape[0],
                                    axis=0)

        cls_classes = np.argmax(cls_preds, axis=2)
        cls_scores = np.concatenate([
            np.expand_dims(per_image_preds[np.arange(per_image_preds.shape[0]),
                                           per_image_cls_classes],
                           axis=0)
            for per_image_preds, per_image_cls_classes in zip(
                cls_preds, cls_classes)
        ],
                                    axis=0)
        cls_scores = np.sqrt(cls_scores * center_preds.squeeze(-1))
        pred_bboxes = self.snap_ltrb_to_x1y1x2y2(reg_preds, batch_positions)

        [batch_scores, batch_classes,
         batch_bboxes] = self.decode_function(cls_scores, cls_classes,
                                              pred_bboxes)

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]

    def snap_ltrb_to_x1y1x2y2(self, reg_preds, points_position):
        '''
        snap reg preds to pred bboxes
        reg_preds:[batch_size,point_nums,4],4:[l,t,r,b]
        points_position:[batch_size,point_nums,2],2:[point_ctr_x,point_ctr_y]
        '''
        reg_preds = np.exp(reg_preds)
        pred_bboxes_xy_min = points_position - reg_preds[:, :, 0:2]
        pred_bboxes_xy_max = points_position + reg_preds[:, :, 2:4]
        pred_bboxes = np.concatenate([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                     axis=2)
        pred_bboxes = pred_bboxes.astype(np.int32)

        # pred bboxes shape:[batch,points_num,4]
        return pred_bboxes


class DETRDecoder:

    def __init__(self,
                 num_classes=80,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=100,
                 nms_type=None,
                 nms_threshold=0.5):
        self.num_classes = num_classes
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_type = nms_type

        if self.nms_type:
            assert nms_type in ['torch_nms', 'python_nms',
                                'diou_python_nms'], 'wrong nms type!'
            self.nms_function = DetNMSMethod(nms_type=nms_type,
                                             nms_threshold=nms_threshold)

    def __call__(self, preds, scaled_sizes):
        cls_preds, reg_preds = preds[0][-1, :, :, :], preds[1][-1, :, :, :]

        cls_preds = F.softmax(cls_preds, dim=2)
        cls_preds = cls_preds.cpu().detach().numpy()
        reg_preds = reg_preds.cpu().detach().numpy()

        cls_classes = np.argmax(cls_preds, axis=2)
        cls_scores = np.concatenate([
            np.expand_dims(per_image_preds[np.arange(per_image_preds.shape[0]),
                                           per_image_cls_classes],
                           axis=0)
            for per_image_preds, per_image_cls_classes in zip(
                cls_preds, cls_classes)
        ],
                                    axis=0)

        pred_bboxes = []
        for idx, per_image_reg_preds in enumerate(reg_preds):
            # x_center,y_center,w,h -> x_min,y_min,x_max,y_max
            per_image_reg_preds = self.transform_cxcywh_box_to_xyxy_box(
                per_image_reg_preds)
            h, w = scaled_sizes[idx][0], scaled_sizes[idx][1]
            per_image_size = np.array([[w, h, w, h]], dtype=np.float32)
            per_image_reg_preds = per_image_reg_preds * per_image_size
            per_image_reg_preds = np.expand_dims(per_image_reg_preds, axis=0)
            pred_bboxes.append(per_image_reg_preds)
        pred_bboxes = np.concatenate(pred_bboxes, axis=0)

        batch_size = cls_scores.shape[0]
        batch_scores = np.ones(
            (batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_classes = np.ones(
            (batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_bboxes = np.zeros((batch_size, self.max_object_num, 4),
                                dtype=np.float32)

        for i, (per_image_scores, per_image_score_classes,
                per_image_pred_bboxes) in enumerate(
                    zip(cls_scores, cls_classes, pred_bboxes)):
            per_image_pred_bboxes = per_image_pred_bboxes[
                per_image_score_classes < self.num_classes].astype(np.float32)
            per_image_scores = per_image_scores[
                per_image_score_classes < self.num_classes].astype(np.float32)
            per_image_score_classes = per_image_score_classes[
                per_image_score_classes < self.num_classes].astype(np.float32)

            per_image_score_classes = per_image_score_classes[
                per_image_scores > self.min_score_threshold].astype(np.float32)
            per_image_pred_bboxes = per_image_pred_bboxes[
                per_image_scores > self.min_score_threshold].astype(np.float32)
            per_image_scores = per_image_scores[
                per_image_scores > self.min_score_threshold].astype(np.float32)

            if per_image_scores.shape[0] != 0:
                # descending sort
                sorted_indexes = np.argsort(-per_image_scores)
                scores = per_image_scores[sorted_indexes]
                score_classes = per_image_score_classes[sorted_indexes]
                bboxes = per_image_pred_bboxes[sorted_indexes]

                if self.topn < scores.shape[0]:
                    scores = scores[0:self.topn]
                    score_classes = score_classes[0:self.topn]
                    bboxes = bboxes[0:self.topn]

                if self.nms_type:
                    # nms
                    keep = self.nms_function(bboxes, scores)
                    scores = scores[keep]
                    score_classes = score_classes[keep]
                    bboxes = bboxes[keep]

                final_detection_num = min(self.max_object_num, scores.shape[0])

                batch_scores[
                    i, 0:final_detection_num] = scores[0:final_detection_num]
                batch_classes[i, 0:final_detection_num] = score_classes[
                    0:final_detection_num]
                batch_bboxes[i, 0:final_detection_num, :] = bboxes[
                    0:final_detection_num, :]

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]

    def transform_cxcywh_box_to_xyxy_box(self, boxes):
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:,
                                                                   2], boxes[:,
                                                                             3]
        boxes = np.stack([(x_center - 0.5 * w), (y_center - 0.5 * h),
                          (x_center + 0.5 * w), (y_center + 0.5 * h)],
                         axis=1)

        return boxes


class DINODETRDecoder:

    def __init__(self,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=300,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_type = nms_type

        if self.nms_type:
            assert nms_type in ['torch_nms', 'python_nms',
                                'diou_python_nms'], 'wrong nms type!'
            self.nms_function = DetNMSMethod(nms_type=nms_type,
                                             nms_threshold=nms_threshold)

    def __call__(self, preds, scaled_sizes):
        cls_preds, reg_preds = preds['pred_logits'], preds['pred_boxes']

        cls_preds = cls_preds.float()
        cls_preds = F.sigmoid(cls_preds)
        cls_preds = cls_preds.cpu().detach().numpy()
        reg_preds = reg_preds.cpu().detach().numpy()

        cls_classes = np.argmax(cls_preds, axis=2)
        cls_scores = np.concatenate([
            np.expand_dims(per_image_preds[np.arange(per_image_preds.shape[0]),
                                           per_image_cls_classes],
                           axis=0)
            for per_image_preds, per_image_cls_classes in zip(
                cls_preds, cls_classes)
        ],
                                    axis=0)

        pred_bboxes = []
        for idx, per_image_reg_preds in enumerate(reg_preds):
            # x_center,y_center,w,h -> x_min,y_min,x_max,y_max
            per_image_reg_preds = self.transform_cxcywh_box_to_xyxy_box(
                per_image_reg_preds)
            h, w = scaled_sizes[idx][0], scaled_sizes[idx][1]
            per_image_size = np.array([[w, h, w, h]], dtype=np.float32)
            per_image_reg_preds = per_image_reg_preds * per_image_size
            per_image_reg_preds = np.expand_dims(per_image_reg_preds, axis=0)
            pred_bboxes.append(per_image_reg_preds)
        pred_bboxes = np.concatenate(pred_bboxes, axis=0)

        batch_size = cls_scores.shape[0]
        batch_scores = np.ones(
            (batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_classes = np.ones(
            (batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_bboxes = np.zeros((batch_size, self.max_object_num, 4),
                                dtype=np.float32)

        for i, (per_image_scores, per_image_score_classes,
                per_image_pred_bboxes) in enumerate(
                    zip(cls_scores, cls_classes, pred_bboxes)):
            per_image_score_classes = per_image_score_classes[
                per_image_scores > self.min_score_threshold].astype(np.float32)
            per_image_pred_bboxes = per_image_pred_bboxes[
                per_image_scores > self.min_score_threshold].astype(np.float32)
            per_image_scores = per_image_scores[
                per_image_scores > self.min_score_threshold].astype(np.float32)

            if per_image_scores.shape[0] != 0:
                # descending sort
                sorted_indexes = np.argsort(-per_image_scores)
                scores = per_image_scores[sorted_indexes]
                score_classes = per_image_score_classes[sorted_indexes]
                bboxes = per_image_pred_bboxes[sorted_indexes]

                if self.topn < scores.shape[0]:
                    scores = scores[0:self.topn]
                    score_classes = score_classes[0:self.topn]
                    bboxes = bboxes[0:self.topn]

                if self.nms_type:
                    # nms
                    keep = self.nms_function(bboxes, scores)
                    scores = scores[keep]
                    score_classes = score_classes[keep]
                    bboxes = bboxes[keep]

                final_detection_num = min(self.max_object_num, scores.shape[0])

                batch_scores[
                    i, 0:final_detection_num] = scores[0:final_detection_num]
                batch_classes[i, 0:final_detection_num] = score_classes[
                    0:final_detection_num]
                batch_bboxes[i, 0:final_detection_num, :] = bboxes[
                    0:final_detection_num, :]

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]

    def transform_cxcywh_box_to_xyxy_box(self, boxes):
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:,
                                                                   2], boxes[:,
                                                                             3]
        boxes = np.stack([(x_center - 0.5 * w), (y_center - 0.5 * h),
                          (x_center + 0.5 * w), (y_center + 0.5 * h)],
                         axis=1)

        return boxes


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
                                        multi_scale=False,
                                        multi_scale_range=[0.8, 1.0]),
                                    Normalize(),
                                ]))

    from torch.utils.data import DataLoader
    collater = DetectionCollater(resize=640,
                                 resize_type='yolo_style',
                                 max_annots_num=100)
    train_loader = DataLoader(cocodataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.detection.models.retinanet import resnet50_retinanet
    net = resnet50_retinanet()
    # 'torch_nms', 'python_nms', 'diou_python_nms'
    decode = RetinaDecoder(areas=[[32, 32], [64, 64], [128, 128], [256, 256],
                                  [512, 512]],
                           ratios=[0.5, 1, 2],
                           scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
                           strides=[8, 16, 32, 64, 128],
                           topn=1000,
                           min_score_threshold=0.01,
                           nms_type='python_nms',
                           nms_threshold=0.5,
                           max_object_num=100)
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
        preds = net(images)
        batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
        print('2222', batch_scores.shape, batch_classes.shape,
              batch_pred_bboxes.shape)
        break

    from simpleAICV.detection.models.fcos import resnet50_fcos
    net = resnet50_fcos()
    # 'torch_nms', 'python_nms', 'diou_python_nms'
    decode = FCOSDecoder(strides=[8, 16, 32, 64, 128],
                         topn=1000,
                         min_score_threshold=0.01,
                         nms_type='python_nms',
                         nms_threshold=0.6,
                         max_object_num=100)
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('1111', images.shape, annots.shape, scales.shape, sizes.shape)
        preds = net(images)
        batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
        print('2222', batch_scores.shape, batch_classes.shape,
              batch_pred_bboxes.shape)
        break

    ############################################################################
    from torch.utils.data import DataLoader
    detr_collater = DETRDetectionCollater(resize=640,
                                          resize_type='yolo_style',
                                          max_annots_num=100)
    detr_train_loader = DataLoader(cocodataset,
                                   batch_size=2,
                                   shuffle=True,
                                   num_workers=1,
                                   collate_fn=detr_collater)

    from simpleAICV.detection.models.detr import resnet50_detr
    net = resnet50_detr()
    decode = DETRDecoder(num_classes=80,
                         max_object_num=100,
                         min_score_threshold=0.05,
                         topn=100,
                         nms_type=None,
                         nms_threshold=0.5)
    for data in tqdm(detr_train_loader):
        images, annots, masks, scaled_sizes = data['image'], data[
            'scaled_annots'], data['mask'], data['scaled_size']
        print('1111', images.shape, annots.shape)
        preds = net(images, masks)
        for pred in preds:
            print('2222', pred.shape)
        batch_scores, batch_classes, batch_pred_bboxes = decode(
            preds, scaled_sizes)
        print('3333', batch_scores.shape, batch_classes.shape,
              batch_pred_bboxes.shape)
        break

    from torch.utils.data import DataLoader
    detr_collater = DETRDetectionCollater(resize=640,
                                          resize_type='yolo_style',
                                          max_annots_num=100)
    detr_train_loader = DataLoader(cocodataset,
                                   batch_size=2,
                                   shuffle=True,
                                   num_workers=1,
                                   collate_fn=detr_collater)

    from simpleAICV.detection.models.dinodetr import resnet50_dinodetr
    net = resnet50_dinodetr().cuda()
    decode = DINODETRDecoder(max_object_num=100,
                             min_score_threshold=0.05,
                             topn=100,
                             nms_type='python_nms',
                             nms_threshold=0.5)
    for data in tqdm(detr_train_loader):
        images, annots, masks, scaled_sizes = data['image'], data[
            'scaled_annots'], data['mask'], data['scaled_size']
        net = net.cuda()
        images = images.cuda()
        annots = annots.cuda()
        masks = masks.cuda()
        print('1111', images.shape, masks.shape, annots.shape)
        preds = net(images, masks, annots)
        print('2222', preds.keys())
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                print('2222', key, value.shape)
            else:
                print('2222', key)

        batch_scores, batch_classes, batch_pred_bboxes = decode(
            preds, scaled_sizes)
        print('3333', batch_scores.shape, batch_classes.shape,
              batch_pred_bboxes.shape)
        break
