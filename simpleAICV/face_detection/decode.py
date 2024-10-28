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

from simpleAICV.detection.decode import DecodeMethod
from simpleAICV.face_detection.models.anchor import RetinaFaceAnchors

__all__ = [
    'RetinaFaceDecoder',
]


class RetinaFaceDecoder:

    def __init__(self,
                 anchor_sizes=[[8, 16, 32], [32, 64, 128], [128, 256, 512]],
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
    net = resnet50_retinaface()
    image_h, image_w = 1024, 1024
    preds = net(torch.autograd.Variable(torch.randn(4, 3, image_h, image_w)))
    decode = RetinaFaceDecoder(anchor_sizes=[[8, 16, 32], [32, 64, 128],
                                             [128, 256, 512]],
                               strides=[8, 16, 32],
                               max_object_num=100,
                               min_score_threshold=0.01,
                               topn=1000,
                               nms_type='python_nms',
                               nms_threshold=0.3)
    batch_scores, batch_classes, batch_pred_bboxes = decode(preds)
    print('1111', batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)
