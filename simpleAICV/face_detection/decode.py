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

from simpleAICV.face_detection.models.anchor import RetinaFaceAnchors

__all__ = [
    'RetinaFaceDecoder',
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


class RetinaFaceDecoder:

    def __init__(self,
                 anchor_sizes=[[16, 32], [64, 128], [256, 512]],
                 strides=[8, 16, 32],
                 max_object_num=100,
                 min_score_threshold=0.3,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.3):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        self.anchors = RetinaFaceAnchors(anchor_sizes=anchor_sizes,
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


if __name__ == '__main__':
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

    from simpleAICV.face_detection.models.retinaface import resnet50_retinaface
    net = resnet50_retinaface(anchor_num=2)
    image_h, image_w = 960, 960
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    decode = RetinaFaceDecoder(anchor_sizes=[[16, 32], [64, 128], [256, 512]],
                               strides=[8, 16, 32],
                               max_object_num=100,
                               min_score_threshold=0.01,
                               topn=1000,
                               nms_type='python_nms',
                               nms_threshold=0.3)
    batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
    print('1111', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)
