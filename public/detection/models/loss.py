import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaLoss(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 epsilon=1e-4):
        super(RetinaLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.image_w = image_w
        self.image_h = image_h

    def forward(self, cls_heads, reg_heads, batch_anchors, annotations):
        """
        compute cls loss and reg loss in one batch
        """
        device = annotations.device
        cls_heads = torch.cat(cls_heads, axis=1)
        reg_heads = torch.cat(reg_heads, axis=1)
        batch_anchors = torch.cat(batch_anchors, axis=1)

        cls_heads, reg_heads, batch_anchors = self.drop_out_border_anchors_and_heads(
            cls_heads, reg_heads, batch_anchors, self.image_w, self.image_h)
        batch_anchors_annotations = self.get_batch_anchors_annotations(
            batch_anchors, annotations)

        cls_loss, reg_loss = [], []
        valid_image_num = 0
        for per_image_cls_heads, per_image_reg_heads, per_image_anchors_annotations in zip(
                cls_heads, reg_heads, batch_anchors_annotations):
            # valid anchors contain all positive anchors
            valid_anchors_num = (per_image_anchors_annotations[
                per_image_anchors_annotations[:, 4] > 0]).shape[0]

            if valid_anchors_num == 0:
                cls_loss.append(torch.tensor(0.).to(device))
                reg_loss.append(torch.tensor(0.).to(device))
            else:
                valid_image_num += 1
                one_image_cls_loss = self.compute_one_image_focal_loss(
                    per_image_cls_heads, per_image_anchors_annotations)
                one_image_reg_loss = self.compute_one_image_smoothl1_loss(
                    per_image_reg_heads, per_image_anchors_annotations)
                cls_loss.append(one_image_cls_loss)
                reg_loss.append(one_image_reg_loss)

        cls_loss = sum(cls_loss) / valid_image_num
        reg_loss = sum(reg_loss) / valid_image_num

        return cls_loss, reg_loss

    def compute_one_image_focal_loss(self, per_image_cls_heads,
                                     per_image_anchors_annotations):
        """
        compute one image focal loss(cls loss)
        per_image_cls_heads:[anchor_num,num_classes]
        per_image_anchors_annotations:[anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        per_image_cls_heads = per_image_cls_heads[
            per_image_anchors_annotations[:, 4] >= 0]
        per_image_anchors_annotations = per_image_anchors_annotations[
            per_image_anchors_annotations[:, 4] >= 0]

        per_image_cls_heads = torch.clamp(per_image_cls_heads,
                                          min=self.epsilon,
                                          max=1. - self.epsilon)
        num_classes = per_image_cls_heads.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(per_image_anchors_annotations[:,
                                                                    4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(per_image_cls_heads) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), per_image_cls_heads,
                         1. - per_image_cls_heads)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        bce_loss = -(
            loss_ground_truth * torch.log(per_image_cls_heads) +
            (1. - loss_ground_truth) * torch.log(1. - per_image_cls_heads))

        one_image_focal_loss = focal_weight * bce_loss

        one_image_focal_loss = one_image_focal_loss.sum()
        positive_anchors_num = per_image_anchors_annotations[
            per_image_anchors_annotations[:, 4] > 0].shape[0]
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        one_image_focal_loss = one_image_focal_loss / positive_anchors_num

        return one_image_focal_loss

    def compute_one_image_smoothl1_loss(self, per_image_reg_heads,
                                        per_image_anchors_annotations):
        """
        compute one image smoothl1 loss(reg loss)
        per_image_reg_heads:[anchor_num,4]
        per_image_anchors_annotations:[anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate smoothl1 loss
        device = per_image_reg_heads.device
        per_image_reg_heads = per_image_reg_heads[
            per_image_anchors_annotations[:, 4] > 0]
        per_image_anchors_annotations = per_image_anchors_annotations[
            per_image_anchors_annotations[:, 4] > 0]
        positive_anchor_num = per_image_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        # compute smoothl1 loss
        loss_ground_truth = per_image_anchors_annotations[:, 0:4]
        x = torch.abs(per_image_reg_heads - loss_ground_truth)
        one_image_smoothl1_loss = torch.where(torch.ge(x, self.beta),
                                              x - 0.5 * self.beta,
                                              0.5 * (x**2) / self.beta)
        one_image_smoothl1_loss = one_image_smoothl1_loss.mean(axis=1).sum()
        # according to the original paper,We divide the smoothl1 loss by the number of positive sample anchors
        one_image_smoothl1_loss = one_image_smoothl1_loss / positive_anchor_num

        return one_image_smoothl1_loss

    def drop_out_border_anchors_and_heads(self, cls_heads, reg_heads,
                                          batch_anchors, image_w, image_h):
        """
        dropout out of border anchors,cls heads and reg heads
        """
        final_cls_heads, final_reg_heads, final_batch_anchors = [], [], []
        for per_image_cls_head, per_image_reg_head, per_image_anchors in zip(
                cls_heads, reg_heads, batch_anchors):
            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                                      0] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                                      0] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                                    0] > 0.0]

            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                                      1] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                                      1] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                                    1] > 0.0]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 2] < image_w]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 2] < image_w]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 2] < image_w]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 3] < image_h]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 3] < image_h]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 3] < image_h]

            per_image_cls_head = per_image_cls_head.unsqueeze(0)
            per_image_reg_head = per_image_reg_head.unsqueeze(0)
            per_image_anchors = per_image_anchors.unsqueeze(0)

            final_cls_heads.append(per_image_cls_head)
            final_reg_heads.append(per_image_reg_head)
            final_batch_anchors.append(per_image_anchors)

        final_cls_heads = torch.cat(final_cls_heads, axis=0)
        final_reg_heads = torch.cat(final_reg_heads, axis=0)
        final_batch_anchors = torch.cat(final_batch_anchors, axis=0)

        # final cls heads shape:[batch_size, anchor_nums, class_num]
        # final reg heads shape:[batch_size, anchor_nums, 4]
        # final batch anchors shape:[batch_size, anchor_nums, 4]
        return final_cls_heads, final_reg_heads, final_batch_anchors

    def get_batch_anchors_annotations(self, batch_anchors, annotations):
        """
        Assign a ground truth box target and a ground truth class target for each anchor
        if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
        if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
        if anchor gt_class index > 0,this anchor is a object class anchor and used in
        calculate cls loss and reg loss
        """
        device = annotations.device
        assert batch_anchors.shape[0] == annotations.shape[0]
        one_image_anchor_nums = batch_anchors.shape[1]

        batch_anchors_annotations = []
        for one_image_anchors, one_image_annotations in zip(
                batch_anchors, annotations):
            # drop all index=-1 class annotations
            one_image_annotations = one_image_annotations[
                one_image_annotations[:, 4] >= 0]

            if one_image_annotations.shape[0] == 0:
                one_image_anchor_annotations = torch.ones(
                    [one_image_anchor_nums, 5], device=device) * (-1)
            else:
                one_image_gt_bboxes = one_image_annotations[:, 0:4]
                one_image_gt_class = one_image_annotations[:, 4]
                one_image_ious = self.compute_ious_for_one_image(
                    one_image_anchors, one_image_gt_bboxes)

                # snap per gt bboxes to the best iou anchor
                overlap, indices = one_image_ious.max(axis=1)
                # assgin each anchor gt bboxes for max iou annotation
                per_image_anchors_gt_bboxes = one_image_gt_bboxes[indices]
                # transform gt bboxes to [tx,ty,tw,th] format for each anchor
                one_image_anchors_snaped_boxes = self.snap_annotations_as_tx_ty_tw_th(
                    per_image_anchors_gt_bboxes, one_image_anchors)

                one_image_anchors_gt_class = (torch.ones_like(overlap) *
                                              -1).to(device)
                # if iou <0.4,assign anchors gt class as 0:background
                one_image_anchors_gt_class[overlap < 0.4] = 0
                # if iou >=0.5,assign anchors gt class as same as the max iou annotation class:80 classes index from 1 to 80
                one_image_anchors_gt_class[
                    overlap >=
                    0.5] = one_image_gt_class[indices][overlap >= 0.5] + 1

                one_image_anchors_gt_class = one_image_anchors_gt_class.unsqueeze(
                    -1)

                one_image_anchor_annotations = torch.cat([
                    one_image_anchors_snaped_boxes, one_image_anchors_gt_class
                ],
                                                         axis=1)
            one_image_anchor_annotations = one_image_anchor_annotations.unsqueeze(
                0)
            batch_anchors_annotations.append(one_image_anchor_annotations)

        batch_anchors_annotations = torch.cat(batch_anchors_annotations,
                                              axis=0)

        # batch anchors annotations shape:[batch_size, anchor_nums, 5]
        return batch_anchors_annotations

    def snap_annotations_as_tx_ty_tw_th(self, anchors_gt_bboxes, anchors):
        """
        snap each anchor ground truth bbox form format:[x_min,y_min,x_max,y_max] to format:[tx,ty,tw,th]
        """
        anchors_w_h = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_w_h

        anchors_gt_bboxes_w_h = anchors_gt_bboxes[:,
                                                  2:] - anchors_gt_bboxes[:, :2]
        anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1.0)
        anchors_gt_bboxes_ctr = anchors_gt_bboxes[:, :
                                                  2] + 0.5 * anchors_gt_bboxes_w_h

        snaped_annotations_for_anchors = torch.cat(
            [(anchors_gt_bboxes_ctr - anchors_ctr) / anchors_w_h,
             torch.log(anchors_gt_bboxes_w_h / anchors_w_h)],
            axis=1)
        device = snaped_annotations_for_anchors.device
        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

        snaped_annotations_for_anchors = snaped_annotations_for_anchors / factor

        # snaped_annotations_for_anchors shape:[batch_size, anchor_nums, 4]
        return snaped_annotations_for_anchors

    def compute_ious_for_one_image(self, one_image_anchors,
                                   one_image_annotations):
        """
        compute ious between one image anchors and one image annotations
        """
        # make sure anchors format:[anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        # make sure annotations format: [annotation_nums,4],4:[x_min,y_min,x_max,y_max]
        annotation_num = one_image_annotations.shape[0]

        one_image_ious = []
        for annotation_index in range(annotation_num):
            annotation = one_image_annotations[
                annotation_index:annotation_index + 1, :]
            overlap_area_top_left = torch.max(one_image_anchors[:, :2],
                                              annotation[:, :2])
            overlap_area_bot_right = torch.min(one_image_anchors[:, 2:],
                                               annotation[:, 2:])
            overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                             overlap_area_top_left,
                                             min=0)
            overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]
            # anchors and annotations convert format to [x1,y1,w,h]
            anchors_w_h = one_image_anchors[:,
                                            2:] - one_image_anchors[:, :2] + 1
            annotations_w_h = annotation[:, 2:] - annotation[:, :2] + 1
            # compute anchors_area and annotations_area
            anchors_area = anchors_w_h[:, 0] * anchors_w_h[:, 1]
            annotations_area = annotations_w_h[:, 0] * annotations_w_h[:, 1]

            # compute union_area
            union_area = anchors_area + annotations_area - overlap_area
            union_area = torch.clamp(union_area, min=1e-4)
            # compute ious between one image anchors and one image annotations
            ious = (overlap_area / union_area).unsqueeze(-1)

            one_image_ious.append(ious)

        one_image_ious = torch.cat(one_image_ious, axis=1)

        # one image ious shape:[anchors_num,annotation_num]
        return one_image_ious


INF = 100000000


class FCOSLoss(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 strides=[8, 16, 32, 64, 128],
                 mi=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]],
                 alpha=0.25,
                 gamma=2.,
                 epsilon=1e-4):
        super(FCOSLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.image_w = image_w
        self.image_h = image_h
        self.strides = strides
        self.mi = mi

    def forward(self, cls_heads, reg_heads, center_heads, annotations):
        """
        compute cls loss, reg loss and center-ness loss in one batch
        """
        cls_preds, reg_preds, center_preds, batch_targets = self.get_batch_position_annotations(
            cls_heads, reg_heads, center_heads, annotations)

        cls_preds = torch.sigmoid(cls_preds)
        reg_preds = torch.exp(reg_preds)
        center_preds = torch.sigmoid(center_preds)
        batch_targets[:, :, 5:6] = torch.sigmoid(batch_targets[:, :, 5:6])

        device = annotations.device
        cls_loss, reg_loss, center_ness_loss = [], [], []
        valid_image_num = 0
        for per_image_cls_preds, per_image_reg_preds, per_image_center_preds, per_image_targets in zip(
                cls_preds, reg_preds, center_preds, batch_targets):
            positive_points_num = (
                per_image_targets[per_image_targets[:, 4] > 0]).shape[0]
            if positive_points_num == 0:
                cls_loss.append(torch.tensor(0.).to(device))
                reg_loss.append(torch.tensor(0.).to(device))
                center_ness_loss.append(torch.tensor(0.).to(device))
            else:
                valid_image_num += 1
                one_image_cls_loss = self.compute_one_image_focal_loss(
                    per_image_cls_preds, per_image_targets)
                one_image_reg_loss = self.compute_one_image_giou_loss(
                    per_image_reg_preds, per_image_targets)
                one_image_center_ness_loss = self.compute_one_image_center_ness_loss(
                    per_image_center_preds, per_image_targets)

                cls_loss.append(one_image_cls_loss)
                reg_loss.append(one_image_reg_loss)
                center_ness_loss.append(one_image_center_ness_loss)

        cls_loss = sum(cls_loss) / valid_image_num
        reg_loss = sum(reg_loss) / valid_image_num
        center_ness_loss = sum(center_ness_loss) / valid_image_num

        return cls_loss, reg_loss, center_ness_loss

    def compute_one_image_focal_loss(self, per_image_cls_preds,
                                     per_image_targets):
        """
        compute one image focal loss(cls loss)
        per_image_cls_preds:[points_num,num_classes]
        per_image_targets:[points_num,8]
        """
        per_image_cls_preds = torch.clamp(per_image_cls_preds,
                                          min=self.epsilon,
                                          max=1. - self.epsilon)
        num_classes = per_image_cls_preds.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(per_image_targets[:, 4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(per_image_cls_preds) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), per_image_cls_preds,
                         1. - per_image_cls_preds)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        bce_loss = -(
            loss_ground_truth * torch.log(per_image_cls_preds) +
            (1. - loss_ground_truth) * torch.log(1. - per_image_cls_preds))

        one_image_focal_loss = focal_weight * bce_loss

        one_image_focal_loss = one_image_focal_loss.sum()
        positive_points_num = per_image_targets[
            per_image_targets[:, 4] > 0].shape[0]
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        one_image_focal_loss = one_image_focal_loss / positive_points_num

        return one_image_focal_loss

    def compute_one_image_giou_loss(self, per_image_reg_preds,
                                    per_image_targets):
        """
        compute one image giou loss(reg loss)
        per_image_reg_preds:[points_num,4]
        per_image_targets:[anchor_num,8]
        """
        # only use positive points sample to compute reg loss
        device = per_image_reg_preds.device
        per_image_reg_preds = per_image_reg_preds[per_image_targets[:, 4] > 0]
        per_image_targets = per_image_targets[per_image_targets[:, 4] > 0]
        positive_points_num = per_image_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        center_ness_targets = per_image_targets[:, 5]

        pred_bboxes_xy_min = per_image_targets[:,
                                               6:8] - per_image_reg_preds[:,
                                                                          0:2]
        pred_bboxes_xy_max = per_image_targets[:,
                                               6:8] + per_image_reg_preds[:,
                                                                          2:4]
        gt_bboxes_xy_min = per_image_targets[:, 6:8] - per_image_targets[:,
                                                                         0:2]
        gt_bboxes_xy_max = per_image_targets[:, 6:8] + per_image_targets[:,
                                                                         2:4]

        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                axis=1)
        gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max], axis=1)

        overlap_area_top_left = torch.max(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        overlap_area_bot_right = torch.min(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                         overlap_area_top_left,
                                         min=0)
        overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

        # anchors and annotations convert format to [x1,y1,w,h]
        pred_bboxes_w_h = pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1
        gt_bboxes_w_h = gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2] + 1

        # compute anchors_area and annotations_area
        pred_bboxes_area = pred_bboxes_w_h[:, 0] * pred_bboxes_w_h[:, 1]
        gt_bboxes_area = gt_bboxes_w_h[:, 0] * gt_bboxes_w_h[:, 1]

        # compute union_area
        union_area = pred_bboxes_area + gt_bboxes_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        # compute ious between one image anchors and one image annotations
        ious = overlap_area / union_area

        enclose_area_top_left = torch.min(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        enclose_area_bot_right = torch.max(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                         enclose_area_top_left,
                                         min=0)
        enclose_area = enclose_area_sizes[:, 0] * enclose_area_sizes[:, 1]
        enclose_area = torch.clamp(enclose_area, min=1e-4)

        gious_loss = 1. - ious + (enclose_area - union_area) / enclose_area
        gious_loss = torch.clamp(gious_loss, min=-1.0, max=1.0)
        # use center_ness_targets as the weight of gious loss
        gious_loss = gious_loss * center_ness_targets
        gious_loss = gious_loss.sum() / positive_points_num
        gious_loss = 2. * gious_loss

        return gious_loss

    def compute_one_image_center_ness_loss(self, per_image_center_preds,
                                           per_image_targets):
        """
        compute one image center_ness loss(center ness loss)
        per_image_center_preds:[points_num,4]
        per_image_targets:[anchor_num,8]
        """
        # only use positive points sample to compute center_ness loss
        device = per_image_center_preds.device
        per_image_center_preds = per_image_center_preds[
            per_image_targets[:, 4] > 0]
        per_image_targets = per_image_targets[per_image_targets[:, 4] > 0]
        positive_points_num = per_image_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        center_ness_targets = per_image_targets[:, 5:6]

        center_ness_loss = -(
            center_ness_targets * torch.log(per_image_center_preds) +
            (1. - center_ness_targets) *
            torch.log(1. - per_image_center_preds))
        center_ness_loss = center_ness_loss.sum() / positive_points_num

        return center_ness_loss

    def get_batch_position_annotations(self, cls_heads, reg_heads,
                                       center_heads, annotations):
        """
        Assign a ground truth target for each position on feature map
        """
        device = annotations.device
        batch_positions, batch_mi = [], []
        for reg_head, stride, mi in zip(reg_heads, self.strides, self.mi):
            mi = torch.tensor(mi).to(device)
            B, H, W, _ = reg_head.shape
            per_level_position = torch.zeros(B, H, W, 2).to(device)
            per_level_mi = torch.zeros(B, H, W, 2).to(device)
            per_level_mi = per_level_mi + mi
            for h_index in range(H):
                for w_index in range(W):
                    w_ctr = (w_index + 0.5) * stride
                    h_ctr = (h_index + 0.5) * stride
                    per_level_position[:, h_index, w_index, 0] = w_ctr
                    per_level_position[:, h_index, w_index, 1] = h_ctr
            batch_positions.append(per_level_position)
            batch_mi.append(per_level_mi)

        cls_preds,reg_preds,center_preds,all_points_position,all_points_mi=[],[],[],[],[]
        for cls_head, reg_head, center_head, per_level_position, per_level_mi in zip(
                cls_heads, reg_heads, center_heads, batch_positions, batch_mi):
            cls_pred = cls_head.view(cls_head.shape[0], -1, cls_head.shape[-1])
            reg_pred = reg_head.view(reg_head.shape[0], -1, reg_head.shape[-1])
            center_pred = center_head.view(center_head.shape[0], -1,
                                           center_head.shape[-1])
            per_level_position = per_level_position.view(
                per_level_position.shape[0], -1, per_level_position.shape[-1])
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1,
                                             per_level_mi.shape[-1])

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            center_preds.append(center_pred)
            all_points_position.append(per_level_position)
            all_points_mi.append(per_level_mi)

        cls_preds = torch.cat(cls_preds, axis=1)
        reg_preds = torch.cat(reg_preds, axis=1)
        center_preds = torch.cat(center_preds, axis=1)
        all_points_position = torch.cat(all_points_position, axis=1)
        all_points_mi = torch.cat(all_points_mi, axis=1)

        batch_targets = []
        for per_image_position, per_image_mi, per_image_annotations in zip(
                all_points_position, all_points_mi, annotations):
            per_image_annotations = per_image_annotations[
                per_image_annotations[:, 4] >= 0]
            points_num = per_image_position.shape[0]

            if per_image_annotations.shape[0] == 0:
                # 6:l,t,r,b,class_index,center-ness_gt
                per_image_targets = torch.zeros([points_num, 6], device=device)
            else:
                annotaion_num = per_image_annotations.shape[0]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                candidates = torch.zeros([points_num, annotaion_num, 4],
                                         device=device)
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)
                per_image_position = per_image_position.unsqueeze(1).repeat(
                    1, annotaion_num, 2)
                candidates[:, :,
                           0:2] = per_image_position[:, :,
                                                     0:2] - candidates[:, :,
                                                                       0:2]
                candidates[:, :,
                           2:4] = candidates[:, :,
                                             2:4] - per_image_position[:, :,
                                                                       2:4]

                candidates_min_value, _ = candidates.min(axis=-1, keepdim=True)
                sample_flag = (candidates_min_value[:, :, 0] >
                               0).int().unsqueeze(-1)
                # get all negative reg targets which points ctr out of gt box
                candidates = candidates * sample_flag

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
                positive_index = (final_sample_flag == True).nonzero().squeeze(
                    dim=-1)

                # if no assign positive sample
                if len(positive_index) == 0:
                    del candidates
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
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

        batch_targets = torch.cat(batch_targets, axis=0)
        batch_targets = torch.cat([batch_targets, all_points_position], axis=2)

        # batch_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        return cls_preds, reg_preds, center_preds, batch_targets


if __name__ == '__main__':
    from retinanet import RetinaNet
    net = RetinaNet(resnet_type="resnet50")
    image_h, image_w = 600, 600
    cls_heads, reg_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    loss = RetinaLoss(image_w, image_h)
    cls_loss, reg_loss = loss(cls_heads, reg_heads, batch_anchors, annotations)
    print("1111", cls_loss, reg_loss)

    from fcos import FCOS
    net = FCOS(resnet_type="resnet50")
    image_h, image_w = 600, 600
    cls_heads, reg_heads, center_heads = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    loss = FCOSLoss(image_w, image_h)
    cls_loss, reg_loss, center_loss = loss(cls_heads, reg_heads, center_heads,
                                           annotations)
    print("2222", cls_loss, reg_loss, center_loss)
