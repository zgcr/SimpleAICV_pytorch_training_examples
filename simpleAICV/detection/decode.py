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

from simpleAICV.detection.models.anchor import RetinaAnchors, FCOSPositions, TTFNetPositions, YoloxAnchors

__all__ = [
    'RetinaDecoder',
    'FCOSDecoder',
    'CenterNetDecoder',
    'TTFNetDecoder',
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

    def forward(self, preds):
        with torch.no_grad():
            heatmap_heads, offset_heads, wh_heads = preds
            device = heatmap_heads.device
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

            batch_scores, batch_classes, batch_bboxes = batch_scores.cpu(
            ).numpy(), batch_classes.cpu().numpy(), batch_bboxes.cpu().numpy()

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


class TTFNetDecoder(nn.Module):

    def __init__(self,
                 topk=100,
                 stride=4,
                 min_score_threshold=0.05,
                 max_object_num=100):
        super(TTFNetDecoder, self).__init__()
        self.positions = TTFNetPositions()
        self.topk = topk
        self.stride = stride
        self.min_score_threshold = min_score_threshold
        self.max_object_num = max_object_num

    def forward(self, preds):
        with torch.no_grad():
            heatmap_heads, wh_heads = preds
            device = heatmap_heads.device
            batch_size = heatmap_heads.shape[0]

            feature_map_size = [heatmap_heads.shape[3], heatmap_heads.shape[2]]
            one_image_positions = self.positions(feature_map_size)
            batch_positions = torch.tensor(one_image_positions).unsqueeze(
                0).repeat(batch_size, 1, 1, 1).to(device)

            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            heatmap_heads = self.nms(heatmap_heads)
            topk_scores, topk_idxs, topk_classes, topk_ys, topk_xs = self.get_topk(
                heatmap_heads, K=self.topk)

            # batch_positions shape:[b,h,w,2]
            # wh_heads shape:[b,4,h,w]->[b,h,w,4]->[b,h*w,4]
            wh_heads = wh_heads.permute(0, 2, 3, 1).contiguous()
            pred_bboxes = self.snap_ltrb_to_x1y1x2y2(wh_heads, batch_positions)
            pred_bboxes = pred_bboxes.view(pred_bboxes.shape[0], -1,
                                           pred_bboxes.shape[3])
            pred_bboxes = torch.gather(pred_bboxes, 1,
                                       topk_idxs.unsqueeze(-1).repeat(1, 1, 4))

            for i, (per_image_scores, per_image_classes,
                    per_image_bboxes) in enumerate(
                        zip(topk_scores, topk_classes, pred_bboxes)):
                per_image_classes = per_image_classes[
                    per_image_scores > self.min_score_threshold].float()
                per_image_bboxes = per_image_bboxes[
                    per_image_scores > self.min_score_threshold].float()
                per_image_scores = per_image_scores[
                    per_image_scores > self.min_score_threshold].float()

                final_detection_num = min(self.max_object_num,
                                          per_image_scores.shape[0])

                batch_scores[i, 0:final_detection_num] = per_image_scores[
                    0:final_detection_num]
                batch_classes[i, 0:final_detection_num] = per_image_classes[
                    0:final_detection_num]
                batch_bboxes[i, 0:final_detection_num, :] = per_image_bboxes[
                    0:final_detection_num, :]

            batch_scores, batch_classes, batch_bboxes = batch_scores.cpu(
            ).numpy(), batch_classes.cpu().numpy(), batch_bboxes.cpu().numpy()

            # batch_scores shape:[batch_size,topk]
            # batch_classes shape:[batch_size,topk]
            # batch_bboxes shape[batch_size,topk,4]
            return batch_scores, batch_classes, batch_bboxes

    def nms(self, heatmap_heads, kernel=3):
        #filter and keep points which value large than the surrounding 8 points
        heatmap_max = F.max_pool2d(heatmap_heads,
                                   kernel,
                                   stride=1,
                                   padding=(kernel - 1) // 2)
        keep = (heatmap_max == heatmap_heads).float()

        return heatmap_heads * keep

    def get_topk(self, heatmap_heads, K):
        B, num_classes, H, W = heatmap_heads.shape[0], heatmap_heads.shape[
            1], heatmap_heads.shape[2], heatmap_heads.shape[3]

        # 先取每个类别的heatmap上前k个最大激活点
        topk_scores, topk_idxs = torch.topk(
            heatmap_heads.view(B, num_classes, -1), K)

        # 取余，计算topk项在feature map上的y和x index(位置)
        topk_idxs = topk_idxs % (H * W)
        topk_ys = (topk_idxs / W).int().float()
        topk_xs = (topk_idxs % W).int().float()

        # 在topk_scores中取前k个最大分数(所有类别混合在一起再取)
        topk_score, topk_score_indexes = torch.topk(topk_scores.view(B, -1), K)

        # 整除K得到预测的类编号，因为heatmap view前第一个维度是类别数
        topk_classes = (topk_score_indexes / K).int()

        topk_score_indexes = topk_score_indexes.unsqueeze(2)
        topk_idxs = topk_idxs.view(B, -1,
                                   1).gather(1, topk_score_indexes).view(B, K)
        topk_ys = topk_ys.view(B, -1, 1).gather(1,
                                                topk_score_indexes).view(B, K)
        topk_xs = topk_xs.view(B, -1, 1).gather(1,
                                                topk_score_indexes).view(B, K)

        return topk_score, topk_idxs, topk_classes, topk_ys, topk_xs

    def snap_ltrb_to_x1y1x2y2(self, wh_heads, batch_positions):
        '''
        snap wh_heads to pred bboxes
        wh_heads:[B,H,W,4],4:[l,t,r,b]
        batch_positions:[B,H,W,2],2:[point_ctr_x,point_ctr_y]
        '''
        wh_heads = torch.exp(wh_heads)
        pred_bboxes_xy_min = (batch_positions -
                              wh_heads[:, :, :, 0:2]) * self.stride
        pred_bboxes_xy_max = (batch_positions +
                              wh_heads[:, :, :, 2:4]) * self.stride
        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                dim=3)
        pred_bboxes = pred_bboxes.int()

        # pred bboxes shape:[B,H,W,4]
        return pred_bboxes


class YoloxDecoder:

    def __init__(self,
                 strides=[8, 16, 32],
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'

        self.grid_strides = YoloxAnchors(strides=strides)
        self.decode_function = DecodeMethod(
            max_object_num=max_object_num,
            min_score_threshold=min_score_threshold,
            topn=topn,
            nms_type=nms_type,
            nms_threshold=nms_threshold)

    def __call__(self, preds):
        obj_preds, cls_preds, reg_preds = preds

        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in cls_preds]
        one_image_grid_center_strides = self.grid_strides(feature_size)

        obj_preds = [
            per_obj_pred.cpu().detach().numpy().reshape(
                per_obj_pred.shape[0], -1, per_obj_pred.shape[-1])
            for per_obj_pred in obj_preds
        ]
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

        obj_preds = np.concatenate(obj_preds, axis=1)
        cls_preds = np.concatenate(cls_preds, axis=1)
        reg_preds = np.concatenate(reg_preds, axis=1)

        one_image_grid_center_strides = np.concatenate([
            per_level_grid_center_strides.reshape(
                -1, per_level_grid_center_strides.shape[-1])
            for per_level_grid_center_strides in one_image_grid_center_strides
        ],
                                                       axis=0)
        batch_grid_center_strides = np.repeat(np.expand_dims(
            one_image_grid_center_strides, axis=0),
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

        cls_scores = cls_scores * obj_preds[:, :, 0]
        pred_bboxes = self.snap_ltrb_to_x1y1x2y2(reg_preds,
                                                 batch_grid_center_strides)

        [batch_scores, batch_classes,
         batch_bboxes] = self.decode_function(cls_scores, cls_classes,
                                              pred_bboxes)

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]

    def snap_ltrb_to_x1y1x2y2(self, reg_preds, grid_center_strides):
        '''
        snap per image reg preds to per image pred bboxes
        reg_preds:[batch_size,point_nums,4],4:[l,t,r,b]
        grid_center_strides:[batch_size,point_nums,3],3:[scale_grid_x_center,scale_grid_y_center,stride]
        '''
        reg_preds = np.exp(reg_preds)
        grid_centers = grid_center_strides[:, :, 0:2]
        strides = np.expand_dims(grid_center_strides[:, :, 2], axis=-1)

        pred_bboxes_xy_min = (grid_centers - reg_preds[:, :, 0:2]) * strides
        pred_bboxes_xy_max = (grid_centers + reg_preds[:, :, 2:4]) * strides
        pred_bboxes = np.concatenate([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                     axis=2)

        pred_bboxes = pred_bboxes.astype(np.int32)

        # pred bboxes shape:[batch,points_num,4]
        return pred_bboxes


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

    from simpleAICV.detection.models.retinanet import resnet50_retinanet
    net = resnet50_retinanet()
    image_h, image_w = 640, 640
    preds = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
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
    batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
    print('1111', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)

    from simpleAICV.detection.models.fcos import resnet50_fcos
    net = resnet50_fcos()
    image_h, image_w = 640, 640
    preds = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    decode = FCOSDecoder(strides=[8, 16, 32, 64, 128],
                         topn=1000,
                         min_score_threshold=0.01,
                         nms_type='torch_nms',
                         nms_threshold=0.6,
                         max_object_num=100)
    batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
    print('2222', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)

    from simpleAICV.detection.models.centernet import resnet18_centernet
    net = resnet18_centernet()
    image_h, image_w = 512, 512
    preds = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = CenterNetDecoder(topk=100,
                              stride=4,
                              min_score_threshold=0.05,
                              max_object_num=100)
    batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
    print('3333', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)

    from simpleAICV.detection.models.ttfnet import resnet18_ttfnet
    net = resnet18_ttfnet()
    image_h, image_w = 512, 512
    preds = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = TTFNetDecoder(topk=100,
                           stride=4,
                           min_score_threshold=0.05,
                           max_object_num=100)
    batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
    print('4444', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)

    from simpleAICV.detection.models.yolox import yoloxl
    net = yoloxl()
    image_h, image_w = 640, 640
    preds = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    decode = YoloxDecoder(strides=[8, 16, 32],
                          max_object_num=100,
                          min_score_threshold=0.05,
                          topn=1000,
                          nms_type='python_nms',
                          nms_threshold=0.6)
    batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
    print('5555', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)
