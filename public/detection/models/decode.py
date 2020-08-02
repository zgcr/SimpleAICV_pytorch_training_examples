import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


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
        device = cls_heads[0].device
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

                one_image_scores = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_classes = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_pred_bboxes = (-1) * torch.ones(
                    (self.max_detection_num, 4), device=device)

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_pred_bboxes = pred_bboxes[sorted_indexes]

                    keep = nms(sorted_pred_bboxes, sorted_scores,
                               self.nms_threshold)
                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_pred_bboxes = sorted_pred_bboxes[keep]

                    final_detection_num = min(self.max_detection_num,
                                              keep_scores.shape[0])

                    one_image_scores[0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    one_image_classes[0:final_detection_num] = keep_classes[
                        0:final_detection_num]
                    one_image_pred_bboxes[
                        0:final_detection_num, :] = keep_pred_bboxes[
                            0:final_detection_num, :]

                one_image_scores = one_image_scores.unsqueeze(0)
                one_image_classes = one_image_classes.unsqueeze(0)
                one_image_pred_bboxes = one_image_pred_bboxes.unsqueeze(0)

                batch_scores.append(one_image_scores)
                batch_classes.append(one_image_classes)
                batch_pred_bboxes.append(one_image_pred_bboxes)

            batch_scores = torch.cat(batch_scores, axis=0)
            batch_classes = torch.cat(batch_classes, axis=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

            # batch_scores shape:[batch_size,max_detection_num]
            # batch_classes shape:[batch_size,max_detection_num]
            # batch_pred_bboxes shape[batch_size,max_detection_num,4]
            return batch_scores, batch_classes, batch_pred_bboxes

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
                 min_score_threshold=0.01,
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

    def forward(self, cls_heads, reg_heads, center_heads, batch_positions):
        with torch.no_grad():
            device = cls_heads[0].device

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

                one_image_scores = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_classes = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_pred_bboxes = (-1) * torch.ones(
                    (self.max_detection_num, 4), device=device)

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_pred_bboxes = pred_bboxes[sorted_indexes]

                    keep = nms(sorted_pred_bboxes, sorted_scores,
                               self.nms_threshold)
                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_pred_bboxes = sorted_pred_bboxes[keep]

                    final_detection_num = min(self.max_detection_num,
                                              keep_scores.shape[0])

                    one_image_scores[0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    one_image_classes[0:final_detection_num] = keep_classes[
                        0:final_detection_num]
                    one_image_pred_bboxes[
                        0:final_detection_num, :] = keep_pred_bboxes[
                            0:final_detection_num, :]

                one_image_scores = one_image_scores.unsqueeze(0)
                one_image_classes = one_image_classes.unsqueeze(0)
                one_image_pred_bboxes = one_image_pred_bboxes.unsqueeze(0)

                batch_scores.append(one_image_scores)
                batch_classes.append(one_image_classes)
                batch_pred_bboxes.append(one_image_pred_bboxes)

            batch_scores = torch.cat(batch_scores, axis=0)
            batch_classes = torch.cat(batch_classes, axis=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

            # batch_scores shape:[batch_size,max_detection_num]
            # batch_classes shape:[batch_size,max_detection_num]
            # batch_pred_bboxes shape[batch_size,max_detection_num,4]
            return batch_scores, batch_classes, batch_pred_bboxes

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


class CenterNetDecoder(nn.Module):
    def __init__(self, image_w, image_h, topk=100, stride=4):
        super(CenterNetDecoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.topk = topk
        self.stride = stride

    def forward(self, heatmap_heads, offset_heads, wh_heads):
        with torch.no_grad():
            heatmap_heads = torch.sigmoid(heatmap_heads)

            batch_scores, batch_classes, batch_pred_bboxes = [], [], []
            for per_image_heatmap_heads, per_image_offset_heads, per_image_wh_heads in zip(
                    heatmap_heads, offset_heads, wh_heads):
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

                topk_bboxes[:, 0] = torch.clamp(topk_bboxes[:, 0], min=0)
                topk_bboxes[:, 1] = torch.clamp(topk_bboxes[:, 1], min=0)
                topk_bboxes[:, 2] = torch.clamp(topk_bboxes[:, 2],
                                                max=self.image_w - 1)
                topk_bboxes[:, 3] = torch.clamp(topk_bboxes[:, 3],
                                                max=self.image_h - 1)

                batch_scores.append(topk_score.unsqueeze(0))
                batch_classes.append(topk_classes.unsqueeze(0))
                batch_pred_bboxes.append(topk_bboxes.unsqueeze(0))

            batch_scores = torch.cat(batch_scores, axis=0)
            batch_classes = torch.cat(batch_classes, axis=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

            # batch_scores shape:[batch_size,topk]
            # batch_classes shape:[batch_size,topk]
            # batch_pred_bboxes shape[batch_size,topk,4]
            return batch_scores, batch_classes, batch_pred_bboxes

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
    cls_heads, reg_heads, center_heads, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = FCOSDecoder(image_w, image_h)
    batch_scores2, batch_classes2, batch_pred_bboxes2 = decode(
        cls_heads, reg_heads, center_heads, batch_positions)
    print("2222", batch_scores2.shape, batch_classes2.shape,
          batch_pred_bboxes2.shape)

    from centernet import CenterNet
    net = CenterNet(resnet_type="resnet50")
    image_h, image_w = 640, 640
    heatmap_output, offset_output, wh_output = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])
    decode = CenterNetDecoder()
    batch_scores3, batch_classes3, batch_pred_bboxes3 = decode(
        heatmap_output, offset_output, wh_output)
    print("3333", batch_scores3.shape, batch_classes3.shape,
          batch_pred_bboxes3.shape)