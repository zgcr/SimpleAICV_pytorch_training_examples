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

from simpleAICV.face_detection.models.anchor import RetinaFaceAnchors

__all__ = [
    'RetinaFaceLoss',
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


class RetinaFaceLoss(nn.Module):

    def __init__(self,
                 anchor_sizes=[[16, 32], [64, 128], [256, 512]],
                 strides=[8, 16, 32],
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 focal_eiou_gamma=0.5,
                 cls_loss_weight=1.,
                 box_loss_weight=1.,
                 box_loss_type='CIoU'):
        super(RetinaFaceLoss, self).__init__()
        assert box_loss_type in [
            'SmoothL1', 'IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU', 'Focal_EIoU'
        ], 'wrong IoU type!'
        self.anchors = RetinaFaceAnchors(anchor_sizes=anchor_sizes,
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
                # if iou <0.35,assign anchors gt class as 0:background
                per_image_anchors_gt_class[overlap < 0.35] = 0
                # if iou >=0.35,assign anchors gt class as same as the max iou annotation class:80 classes index from 1 to 1
                per_image_anchors_gt_class[
                    overlap >=
                    0.35] = one_image_gt_class[indices][overlap >= 0.35] + 1

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

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from tools.path import face_detection_dataset_path

    from simpleAICV.face_detection.datasets.face_detection_dataset import FaceDetectionDataset
    from simpleAICV.face_detection.common import YoloStyleResize, RandomCrop, RandomTranslate, RandomHorizontalFlip, RandomVerticalFlip, RandomGaussianBlur, MainDirectionRandomRotate, Normalize, FaceDetectionCollater

    vocdataset = FaceDetectionDataset(
        face_detection_dataset_path,
        set_name_list=[
            'wider_face',
        ],
        set_type='train',
        transform=transforms.Compose([
            RandomGaussianBlur(sigma=[0.5, 1.5], prob=0.3),
            MainDirectionRandomRotate(angle=[0, 90, 180, 270],
                                      prob=[0.55, 0.15, 0.15, 0.15]),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            YoloStyleResize(resize=960,
                            divisor=32,
                            stride=32,
                            multi_scale=False,
                            multi_scale_range=[0.8, 1.0]),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = FaceDetectionCollater(resize=960)
    train_loader = DataLoader(vocdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.face_detection.models.retinaface import resnet50_retinaface
    net = resnet50_retinaface(anchor_num=2)
    loss = RetinaFaceLoss(anchor_sizes=[[16, 32], [64, 128], [256, 512]],
                          strides=[8, 16, 32],
                          alpha=0.25,
                          gamma=2,
                          beta=1.0 / 9.0,
                          focal_eiou_gamma=0.5,
                          cls_loss_weight=1.,
                          box_loss_weight=1.,
                          box_loss_type='SmoothL1')

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
