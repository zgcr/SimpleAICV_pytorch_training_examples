import torch
import torch.nn as nn


class RetinaDecoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 top_n=1000,
                 min_score_threshold=0.05,
                 nms_threshold=0.5,
                 max_detection_num=100):
        super(RetinaDecoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.top_n = top_n
        self.min_score_threshold = min_score_threshold
        self.nms_threshold = nms_threshold
        self.max_detection_num = max_detection_num

    def forward(self, cls_heads, reg_heads, batch_anchors):
        with torch.no_grad():
            filter_scores,filter_score_classes,filter_reg_heads,filter_batch_anchors=[],[],[],[]
            for per_level_cls_head, per_level_reg_head, per_level_anchor in zip(
                    cls_heads, reg_heads, batch_anchors):
                scores, score_classes = torch.max(per_level_cls_head, dim=2)
                if scores.shape[1] >= self.top_n:
                    scores, indexes = torch.topk(scores,
                                                 self.top_n,
                                                 dim=1,
                                                 largest=True,
                                                 sorted=True)
                    score_classes = torch.gather(score_classes, 1, indexes)
                    per_level_reg_head = torch.gather(
                        per_level_reg_head, 1,
                        indexes.unsqueeze(-1).repeat(1, 1, 4))
                    per_level_anchor = torch.gather(
                        per_level_anchor, 1,
                        indexes.unsqueeze(-1).repeat(1, 1, 4))

                filter_scores.append(scores)
                filter_score_classes.append(score_classes)
                filter_reg_heads.append(per_level_reg_head)
                filter_batch_anchors.append(per_level_anchor)

            filter_scores = torch.cat(filter_scores, axis=1)
            filter_score_classes = torch.cat(filter_score_classes, axis=1)
            filter_reg_heads = torch.cat(filter_reg_heads, axis=1)
            filter_batch_anchors = torch.cat(filter_batch_anchors, axis=1)

            batch_scores, batch_classes, batch_pred_bboxes = [], [], []
            for per_image_scores, per_image_score_classes, per_image_reg_heads, per_image_anchors in zip(
                    filter_scores, filter_score_classes, filter_reg_heads,
                    filter_batch_anchors):
                pred_bboxes = self.snap_tx_ty_tw_th_reg_heads_to_x1_y1_x2_y2_bboxes(
                    per_image_reg_heads, per_image_anchors)
                score_classes = per_image_score_classes[
                    per_image_scores > self.min_score_threshold].float()
                pred_bboxes = pred_bboxes[
                    per_image_scores > self.min_score_threshold].float()
                scores = per_image_scores[
                    per_image_scores > self.min_score_threshold].float()

                sorted_keep_scores, sorted_keep_classes, sorted_keep_pred_bboxes = self.nms(
                    scores, score_classes, pred_bboxes)

                sorted_keep_scores = sorted_keep_scores.unsqueeze(0)
                sorted_keep_classes = sorted_keep_classes.unsqueeze(0)
                sorted_keep_pred_bboxes = sorted_keep_pred_bboxes.unsqueeze(0)

                batch_scores.append(sorted_keep_scores)
                batch_classes.append(sorted_keep_classes)
                batch_pred_bboxes.append(sorted_keep_pred_bboxes)

            batch_scores = torch.cat(batch_scores, axis=0)
            batch_classes = torch.cat(batch_classes, axis=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

            # batch_scores shape:[batch_size,max_detection_num]
            # batch_classes shape:[batch_size,max_detection_num]
            # batch_pred_bboxes shape[batch_size,max_detection_num,4]
            return batch_scores, batch_classes, batch_pred_bboxes

    def nms(self, one_image_scores, one_image_classes, one_image_pred_bboxes):
        """
        one_image_scores:[anchor_nums],4:classification predict scores
        one_image_classes:[anchor_nums],class indexes for predict scores
        one_image_pred_bboxes:[anchor_nums,4],4:x_min,y_min,x_max,y_max
        """
        device = one_image_scores.device
        final_scores = (-1) * torch.ones(
            (self.max_detection_num, ), device=device)
        final_classes = (-1) * torch.ones(
            (self.max_detection_num, ), device=device)
        final_pred_bboxes = (-1) * torch.ones(
            (self.max_detection_num, 4), device=device)

        if one_image_scores.shape[0] == 0:
            return final_scores, final_classes, final_pred_bboxes

        # Sort boxes
        sorted_one_image_scores, sorted_one_image_scores_indexes = torch.sort(
            one_image_scores, descending=True)
        sorted_one_image_classes = one_image_classes[
            sorted_one_image_scores_indexes]
        sorted_one_image_pred_bboxes = one_image_pred_bboxes[
            sorted_one_image_scores_indexes]
        sorted_pred_bboxes_w_h = sorted_one_image_pred_bboxes[:,
                                                              2:] - sorted_one_image_pred_bboxes[:, :
                                                                                                 2]

        sorted_pred_bboxes_areas = sorted_pred_bboxes_w_h[:,
                                                          0] * sorted_pred_bboxes_w_h[:,
                                                                                      1]

        keep_scores, keep_classes, keep_pred_bboxes = [], [], []
        while sorted_one_image_scores.numel() > 0:
            top1_score, top1_class, top1_pred_bbox = sorted_one_image_scores[
                0:1], sorted_one_image_classes[
                    0:1], sorted_one_image_pred_bboxes[0:1]

            keep_scores.append(top1_score)
            keep_classes.append(top1_class)
            keep_pred_bboxes.append(top1_pred_bbox)

            top1_areas = sorted_pred_bboxes_areas[0]

            if len(keep_scores) >= self.max_detection_num:
                break

            if sorted_one_image_scores.numel() == 1:
                break

            sorted_one_image_scores = sorted_one_image_scores[1:]
            sorted_one_image_classes = sorted_one_image_classes[1:]
            sorted_one_image_pred_bboxes = sorted_one_image_pred_bboxes[1:]
            sorted_pred_bboxes_areas = sorted_pred_bboxes_areas[1:]

            overlap_area_top_left = torch.max(
                sorted_one_image_pred_bboxes[:, :2], top1_pred_bbox[:, :2])
            overlap_area_bot_right = torch.min(
                sorted_one_image_pred_bboxes[:, 2:], top1_pred_bbox[:, 2:])
            overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                             overlap_area_top_left,
                                             min=0)
            overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

            # compute union_area
            union_area = top1_areas + sorted_pred_bboxes_areas - overlap_area
            union_area = torch.clamp(union_area, min=1e-4)
            # compute ious for top1 pred_bbox and the other pred_bboxes
            ious = overlap_area / union_area

            sorted_one_image_scores = sorted_one_image_scores[
                ious < self.nms_threshold]
            sorted_one_image_classes = sorted_one_image_classes[
                ious < self.nms_threshold]
            sorted_one_image_pred_bboxes = sorted_one_image_pred_bboxes[
                ious < self.nms_threshold]
            sorted_pred_bboxes_areas = sorted_pred_bboxes_areas[
                ious < self.nms_threshold]

        keep_scores = torch.cat(keep_scores, axis=0)
        keep_classes = torch.cat(keep_classes, axis=0)
        keep_pred_bboxes = torch.cat(keep_pred_bboxes, axis=0)

        final_detection_num = min(self.max_detection_num, keep_scores.shape[0])

        final_scores[0:final_detection_num] = keep_scores[
            0:final_detection_num]
        final_classes[0:final_detection_num] = keep_classes[
            0:final_detection_num]
        final_pred_bboxes[0:final_detection_num, :] = keep_pred_bboxes[
            0:final_detection_num, :]

        return final_scores, final_classes, final_pred_bboxes

    def snap_tx_ty_tw_th_reg_heads_to_x1_y1_x2_y2_bboxes(
            self, reg_heads, anchors):
        """
        snap reg heads to pred bboxes
        reg_heads:[anchor_nums,4],4:[tx,ty,tw,th]
        anchors:[anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        """
        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

        device = anchors.device
        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

        reg_heads = reg_heads * factor

        pred_bboxes_wh = torch.exp(reg_heads[:, 2:]) * anchors_wh
        pred_bboxes_ctr = reg_heads[:, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_min_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_max_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = torch.cat(
            [pred_bboxes_x_min_y_min, pred_bboxes_x_max_y_max], axis=1)
        pred_bboxes = pred_bboxes.int()

        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2],
                                        max=self.image_w - 1)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3],
                                        max=self.image_h - 1)

        # pred bboxes shape:[anchor_nums,4]
        return pred_bboxes


class FCOSDecoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 strides=[8, 16, 32, 64, 128],
                 top_n=1000,
                 min_score_threshold=0.05,
                 nms_threshold=0.6,
                 max_detection_num=100):
        super(FCOSDecoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.strides = strides
        self.top_n = top_n
        self.min_score_threshold = min_score_threshold
        self.nms_threshold = nms_threshold
        self.max_detection_num = max_detection_num

    def forward(self, cls_heads, reg_heads, center_heads):
        with torch.no_grad():
            device = cls_heads[0].device
            batch_positions = []
            for reg_head, stride in zip(reg_heads, self.strides):
                B, H, W, _ = reg_head.shape
                per_level_position = torch.zeros(B, H, W, 2).to(device)
                for h_index in range(H):
                    for w_index in range(W):
                        w_ctr = (w_index + 0.5) * stride
                        h_ctr = (h_index + 0.5) * stride
                        per_level_position[:, h_index, w_index, 0] = w_ctr
                        per_level_position[:, h_index, w_index, 1] = h_ctr
                batch_positions.append(per_level_position)

            filter_scores,filter_score_classes,filter_reg_heads,filter_batch_positions=[],[],[],[]
            for per_level_cls_head, per_level_reg_head, per_level_center_head, per_level_position in zip(
                    cls_heads, reg_heads, center_heads, batch_positions):
                per_level_cls_head = torch.sigmoid(per_level_cls_head)
                per_level_reg_head = torch.exp(per_level_reg_head)
                per_level_center_head = torch.sigmoid(per_level_center_head)

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

                scores, score_classes = torch.max(per_level_cls_head, dim=2)
                scores = torch.sqrt(scores * per_level_center_head.squeeze(-1))
                if scores.shape[1] >= self.top_n:
                    scores, indexes = torch.topk(scores,
                                                 self.top_n,
                                                 dim=1,
                                                 largest=True,
                                                 sorted=True)
                    score_classes = torch.gather(score_classes, 1, indexes)
                    per_level_reg_head = torch.gather(
                        per_level_reg_head, 1,
                        indexes.unsqueeze(-1).repeat(1, 1, 4))
                    per_level_center_head = torch.gather(
                        per_level_center_head, 1,
                        indexes.unsqueeze(-1).repeat(1, 1, 1))
                    per_level_position = torch.gather(
                        per_level_position, 1,
                        indexes.unsqueeze(-1).repeat(1, 1, 2))
                filter_scores.append(scores)
                filter_score_classes.append(score_classes)
                filter_reg_heads.append(per_level_reg_head)
                filter_batch_positions.append(per_level_position)

            filter_scores = torch.cat(filter_scores, axis=1)
            filter_score_classes = torch.cat(filter_score_classes, axis=1)
            filter_reg_heads = torch.cat(filter_reg_heads, axis=1)
            filter_batch_positions = torch.cat(filter_batch_positions, axis=1)

            batch_scores, batch_classes, batch_pred_bboxes = [], [], []
            for scores, score_classes, per_image_reg_preds, per_image_points_position in zip(
                    filter_scores, filter_score_classes, filter_reg_heads,
                    filter_batch_positions):
                pred_bboxes = self.snap_ltrb_reg_heads_to_x1_y1_x2_y2_bboxes(
                    per_image_reg_preds, per_image_points_position)

                score_classes = score_classes[
                    scores > self.min_score_threshold].float()
                pred_bboxes = pred_bboxes[
                    scores > self.min_score_threshold].float()
                scores = scores[scores > self.min_score_threshold].float()

                sorted_keep_scores, sorted_keep_classes, sorted_keep_pred_bboxes = self.nms(
                    scores, score_classes, pred_bboxes)

                sorted_keep_scores = sorted_keep_scores.unsqueeze(0)
                sorted_keep_classes = sorted_keep_classes.unsqueeze(0)
                sorted_keep_pred_bboxes = sorted_keep_pred_bboxes.unsqueeze(0)

                batch_scores.append(sorted_keep_scores)
                batch_classes.append(sorted_keep_classes)
                batch_pred_bboxes.append(sorted_keep_pred_bboxes)

            batch_scores = torch.cat(batch_scores, axis=0)
            batch_classes = torch.cat(batch_classes, axis=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

            # batch_scores shape:[batch_size,max_detection_num]
            # batch_classes shape:[batch_size,max_detection_num]
            # batch_pred_bboxes shape[batch_size,max_detection_num,4]
            return batch_scores, batch_classes, batch_pred_bboxes

    def nms(self, one_image_scores, one_image_classes, one_image_pred_bboxes):
        """
        one_image_scores:[anchor_nums],4:classification predict scores
        one_image_classes:[anchor_nums],class indexes for predict scores
        one_image_pred_bboxes:[anchor_nums,4],4:x_min,y_min,x_max,y_max
        """
        device = one_image_scores.device
        final_scores = (-1) * torch.ones(
            (self.max_detection_num, ), device=device)
        final_classes = (-1) * torch.ones(
            (self.max_detection_num, ), device=device)
        final_pred_bboxes = (-1) * torch.ones(
            (self.max_detection_num, 4), device=device)

        if one_image_scores.shape[0] == 0:
            return final_scores, final_classes, final_pred_bboxes

        # Sort boxes
        sorted_one_image_scores, sorted_one_image_scores_indexes = torch.sort(
            one_image_scores, descending=True)
        sorted_one_image_classes = one_image_classes[
            sorted_one_image_scores_indexes]
        sorted_one_image_pred_bboxes = one_image_pred_bboxes[
            sorted_one_image_scores_indexes]
        sorted_pred_bboxes_w_h = sorted_one_image_pred_bboxes[:,
                                                              2:] - sorted_one_image_pred_bboxes[:, :
                                                                                                 2]

        sorted_pred_bboxes_areas = sorted_pred_bboxes_w_h[:,
                                                          0] * sorted_pred_bboxes_w_h[:,
                                                                                      1]

        keep_scores, keep_classes, keep_pred_bboxes = [], [], []
        while sorted_one_image_scores.numel() > 0:
            top1_score, top1_class, top1_pred_bbox = sorted_one_image_scores[
                0:1], sorted_one_image_classes[
                    0:1], sorted_one_image_pred_bboxes[0:1]

            keep_scores.append(top1_score)
            keep_classes.append(top1_class)
            keep_pred_bboxes.append(top1_pred_bbox)

            top1_areas = sorted_pred_bboxes_areas[0]

            if len(keep_scores) >= self.max_detection_num:
                break

            if sorted_one_image_scores.numel() == 1:
                break

            sorted_one_image_scores = sorted_one_image_scores[1:]
            sorted_one_image_classes = sorted_one_image_classes[1:]
            sorted_one_image_pred_bboxes = sorted_one_image_pred_bboxes[1:]
            sorted_pred_bboxes_areas = sorted_pred_bboxes_areas[1:]

            overlap_area_top_left = torch.max(
                sorted_one_image_pred_bboxes[:, :2], top1_pred_bbox[:, :2])
            overlap_area_bot_right = torch.min(
                sorted_one_image_pred_bboxes[:, 2:], top1_pred_bbox[:, 2:])
            overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                             overlap_area_top_left,
                                             min=0)
            overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

            # compute union_area
            union_area = top1_areas + sorted_pred_bboxes_areas - overlap_area
            union_area = torch.clamp(union_area, min=1e-4)
            # compute ious for top1 pred_bbox and the other pred_bboxes
            ious = overlap_area / union_area

            sorted_one_image_scores = sorted_one_image_scores[
                ious < self.nms_threshold]
            sorted_one_image_classes = sorted_one_image_classes[
                ious < self.nms_threshold]
            sorted_one_image_pred_bboxes = sorted_one_image_pred_bboxes[
                ious < self.nms_threshold]
            sorted_pred_bboxes_areas = sorted_pred_bboxes_areas[
                ious < self.nms_threshold]

        keep_scores = torch.cat(keep_scores, axis=0)
        keep_classes = torch.cat(keep_classes, axis=0)
        keep_pred_bboxes = torch.cat(keep_pred_bboxes, axis=0)

        final_detection_num = min(self.max_detection_num, keep_scores.shape[0])

        final_scores[0:final_detection_num] = keep_scores[
            0:final_detection_num]
        final_classes[0:final_detection_num] = keep_classes[
            0:final_detection_num]
        final_pred_bboxes[0:final_detection_num, :] = keep_pred_bboxes[
            0:final_detection_num, :]

        return final_scores, final_classes, final_pred_bboxes

    def snap_ltrb_reg_heads_to_x1_y1_x2_y2_bboxes(self, reg_preds,
                                                  points_position):
        """
        snap reg preds to pred bboxes
        reg_preds:[points_num,4],4:[l,t,r,b]
        points_position:[points_num,2],2:[point_ctr_x,point_ctr_y]
        """
        pred_bboxes_xy_min = points_position - reg_preds[:, 0:2]
        pred_bboxes_xy_max = points_position + reg_preds[:, 2:4]
        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                axis=1)
        pred_bboxes = pred_bboxes.int()

        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2],
                                        max=self.image_w - 1)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3],
                                        max=self.image_h - 1)

        # pred bboxes shape:[points_num,4]
        return pred_bboxes


if __name__ == '__main__':
    from retinanet import RetinaNet
    net = RetinaNet(resnet_type="resnet50")
    image_h, image_w = 640, 640
    cls_heads, reg_heads, batch_anchors = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = RetinaDecoder(image_w, image_h)
    batch_scores, batch_classes, batch_pred_bboxes = decode(
        cls_heads, reg_heads, batch_anchors)
    print("1111", batch_scores.shape, batch_classes.shape,
          batch_pred_bboxes.shape)

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
    decode = FCOSDecoder(image_w, image_h)
    batch_scores2, batch_classes2, batch_pred_bboxes2 = decode(
        cls_heads, reg_heads, center_heads)
    print("2222", batch_scores2.shape, batch_classes2.shape,
          batch_pred_bboxes2.shape)
