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

from simpleAICV.detection.decode import DetNMSMethod


class MaskNMSMethod:
    def __init__(self, nms_threshold=0.5):
        self.nms_threshold = nms_threshold

    def __call__(self, cate_scores, cate_classes, masks, masks_sum):
        device = cate_scores.device
        masks = masks.float()
        object_nums = cate_scores.shape[0]

        keep = torch.ones(cate_scores.shape, dtype=torch.bool, device=device)

        for i in range(object_nums - 1):
            if not keep[i]:
                continue

            mask_i, class_i = masks[i], cate_classes[i]
            for j in range(i + 1, object_nums):
                if not keep[j]:
                    continue

                mask_j, class_j = masks[j], cate_classes[j]
                if class_i != class_j:
                    continue

                inter = (mask_i * mask_j).sum()
                union = masks_sum[i] + masks_sum[j] - inter

                if union > 0:
                    if (inter / union) > self.nms_threshold:
                        keep[j] = False
                else:
                    keep[j] = False

        return keep


class MatrixNMSMethod:
    def __init__(self, update_threshold=0.05, sigma=2.0, kernel='gaussian'):
        assert kernel in ['linear', 'gaussian'], 'wrong kernel type!'
        self.update_threshold = update_threshold
        self.sigma = sigma
        self.kernel = kernel

    def __call__(self, cate_scores, cate_classes, masks, masks_sum):
        object_nums = cate_scores.shape[0]
        masks = masks.view(object_nums, -1).float()

        inter_matrix = torch.mm(masks, masks.transpose(1, 0))
        masks_sum_x = masks_sum.expand(object_nums, object_nums)
        iou_matrix = (
            inter_matrix /
            (masks_sum_x + masks_sum_x.transpose(1, 0) - inter_matrix)).triu(
                diagonal=1)

        # label_specific matrix.
        cate_classes_x = cate_classes.expand(object_nums, object_nums)
        label_matrix = (cate_classes_x == cate_classes_x.transpose(
            1, 0)).float().triu(diagonal=1)

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(object_nums,
                                               object_nums).transpose(1, 0)

        # IoU decay / soft nms
        delay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'linear':
            delay_matrix = (1 - delay_iou) / (1 - compensate_iou)
            delay_coefficient, _ = delay_matrix.min(0)
        elif self.kernel == 'gaussian':
            delay_matrix = torch.exp(-1 * self.sigma * (delay_iou**2))
            compensate_matrix = torch.exp(-1 * self.sigma *
                                          (compensate_iou**2))
            delay_coefficient, _ = (delay_matrix / compensate_matrix).min(0)

        cate_scores_update = cate_scores * delay_coefficient
        keep = (cate_scores_update >= self.update_threshold)

        return keep


class Solov2Decoder(nn.Module):
    def __init__(self,
                 topn=500,
                 min_score_threshold=0.1,
                 mask_stride=4,
                 num_grids=[40, 36, 24, 16, 12],
                 strides=[8, 8, 16, 32, 32],
                 mask_threshold=0.5,
                 nms_type='mask_nms',
                 max_object_num=100):
        super(Solov2Decoder, self).__init__()
        assert nms_type in ['mask_nms', 'matrix_nms'], 'wrong nms type!'

        self.topn = topn
        self.min_score_threshold = min_score_threshold
        self.mask_stride = mask_stride
        self.num_grids = num_grids
        self.strides = strides
        self.mask_threshold = mask_threshold
        self.max_object_num = max_object_num
        if nms_type == 'mask_nms':
            self.nms_function = MaskNMSMethod(
                nms_threshold=self.mask_threshold)
        elif nms_type == 'matrix_nms':
            self.nms_function = MatrixNMSMethod(update_threshold=0.05,
                                                sigma=2.0,
                                                kernel='gaussian')

    def forward(self, cate_outs, kernel_outs, mask_out):
        with torch.no_grad():
            device = mask_out.device

            for i in range(len(cate_outs)):
                per_level_cate_outs_max = F.max_pool2d(cate_outs[i], (2, 2),
                                                       stride=1,
                                                       padding=1)
                per_level_cate_outs_keep = (
                    per_level_cate_outs_max[:, :, :-1, :-1] == cate_outs[i]
                ).float()
                cate_outs[i] = cate_outs[i] * per_level_cate_outs_keep

            cate_outs = torch.cat([
                per_level_cate_outs.view(per_level_cate_outs.shape[0], -1,
                                         per_level_cate_outs.shape[-1])
                for per_level_cate_outs in cate_outs
            ],
                                  dim=1)

            kernel_outs = torch.cat([
                per_level_kernel_outs.view(per_level_kernel_outs.shape[0], -1,
                                           per_level_kernel_outs.shape[-1])
                for per_level_kernel_outs in kernel_outs
            ],
                                    dim=1)
            cls_scores, cls_classes = torch.max(cate_outs, dim=2)

            batch_size = cls_scores.shape[0]
            input_h, input_w = mask_out.shape[
                -2] * self.mask_stride, mask_out.shape[-1] * self.mask_stride
            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_masks = torch.zeros(
                (batch_size, self.max_object_num, input_h, input_w),
                dtype=torch.bool,
                device=device)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            for img_idx, (per_image_scores, per_image_classes,
                          per_image_kernel_outs,
                          per_image_mask_out) in enumerate(
                              zip(cls_scores, cls_classes, kernel_outs,
                                  mask_out)):
                valid_idxs = (per_image_scores > self.min_score_threshold)
                kernel_preds = per_image_kernel_outs[valid_idxs]
                cate_classes = per_image_classes[valid_idxs]
                cate_scores = per_image_scores[valid_idxs]

                if cate_scores.shape[0] == 0:
                    continue

                all_grid_strides = []
                for num, stride in zip(self.num_grids, self.strides):
                    per_level_strides = np.ones([num * num],
                                                dtype=np.int32) * stride
                    all_grid_strides.extend(per_level_strides)
                all_grid_strides = torch.tensor(all_grid_strides,
                                                dtype=torch.float32,
                                                device=device)
                valid_idxs_stride = all_grid_strides[valid_idxs]

                kernel_preds = kernel_preds.unsqueeze(-1).unsqueeze(-1)
                per_image_mask_out = per_image_mask_out.unsqueeze(0)
                mask_preds = F.conv2d(per_image_mask_out,
                                      kernel_preds,
                                      stride=1)
                mask_preds = torch.sigmoid(mask_preds.squeeze(0))
                masks = (mask_preds > self.mask_threshold)
                masks_sum = masks.sum(dim=-1).sum(dim=-1).float()

                # filter mask
                keep = (masks_sum > valid_idxs_stride)
                masks = masks[keep]
                mask_preds = mask_preds[keep]
                masks_sum = masks_sum[keep]
                cate_scores = cate_scores[keep]
                cate_classes = cate_classes[keep]

                if cate_scores.shape[0] == 0:
                    continue

                mask_scores = (mask_preds * masks.float()).sum(dim=-1).sum(
                    dim=-1) / masks_sum
                cate_scores = cate_scores * mask_scores
                _, sorted_cate_indexes = torch.sort(cate_scores,
                                                    descending=True)
                sorted_cate_indexes = sorted_cate_indexes[
                    0:min(self.topn, sorted_cate_indexes.shape[0])]
                masks = masks[sorted_cate_indexes]
                mask_preds = mask_preds[sorted_cate_indexes]
                masks_sum = masks_sum[sorted_cate_indexes]
                cate_scores = cate_scores[sorted_cate_indexes]
                cate_classes = cate_classes[sorted_cate_indexes]

                keep = self.nms_function(cate_scores, cate_classes, masks,
                                         masks_sum)
                mask_preds = mask_preds[keep]
                cate_scores = cate_scores[keep]
                cate_classes = cate_classes[keep]

                if cate_scores.shape[0] == 0:
                    continue

                _, sorted_cate_indexes = torch.sort(cate_scores,
                                                    descending=True)
                final_detection_num = min(self.max_object_num,
                                          sorted_cate_indexes.shape[0])
                sorted_cate_indexes = sorted_cate_indexes[
                    0:final_detection_num]
                mask_preds = mask_preds[sorted_cate_indexes]
                cate_scores = cate_scores[sorted_cate_indexes]
                cate_classes = cate_classes[sorted_cate_indexes]
                masks = F.interpolate(mask_preds.unsqueeze(0),
                                      size=(input_h, input_w),
                                      mode='nearest').squeeze(0)
                masks = (masks > self.mask_threshold)
                boxes = torch.zeros((cate_scores.shape[0], 4),
                                    dtype=torch.float32,
                                    device=device)
                for obj_idx, per_object_mask in enumerate(masks):
                    ys, xs = torch.where(per_object_mask)
                    if ys.shape[0] != 0 and xs.shape[0] != 0:
                        boxes[obj_idx, 0] = xs.min().float()
                        boxes[obj_idx, 1] = ys.min().float()
                        boxes[obj_idx, 2] = xs.max().float()
                        boxes[obj_idx, 3] = ys.max().float()

                batch_scores[img_idx, 0:final_detection_num] = cate_scores
                batch_classes[img_idx, 0:final_detection_num] = cate_classes
                batch_masks[img_idx, 0:final_detection_num] = masks
                batch_bboxes[img_idx, 0:final_detection_num] = boxes

        return batch_scores, batch_classes, batch_masks, batch_bboxes


class CondInstDecoder(nn.Module):
    def __init__(
            self,
            mask_stride=8,
            num_masks=8,
            topn=1000,
            min_score_threshold=0.1,  #0.05
            nms_type='torch_nms',
            nms_threshold=0.6,
            mask_threshold=0.5,
            max_object_num=100):
        super(CondInstDecoder, self).__init__()
        assert nms_type in ['torch_nms', 'python_nms',
                            'DIoU_python_nms'], 'wrong nms type!'

        # condinst use pred boxes to do nms
        self.mask_stride = mask_stride
        self.num_masks = num_masks
        self.topn = topn
        self.min_score_threshold = min_score_threshold
        self.mask_threshold = mask_threshold
        self.max_object_num = max_object_num
        self.nms_function = DetNMSMethod(nms_type=nms_type,
                                         nms_threshold=nms_threshold)

    def forward(self, cls_heads, reg_heads, center_heads, controllers_heads,
                mask_out, batch_positions):
        with torch.no_grad():
            device = cls_heads[0].device
            cls_scores, cls_classes, pred_bboxes,controllers_preds= [], [], [],[]
            for per_level_cls_head, per_level_reg_head, per_level_center_head, per_level_controllers_head, per_level_position in zip(
                    cls_heads, reg_heads, center_heads, controllers_heads,
                    batch_positions):
                per_level_cls_head = per_level_cls_head.view(
                    per_level_cls_head.shape[0], -1,
                    per_level_cls_head.shape[-1])
                per_level_reg_head = per_level_reg_head.view(
                    per_level_reg_head.shape[0], -1,
                    per_level_reg_head.shape[-1])
                per_level_center_head = per_level_center_head.view(
                    per_level_center_head.shape[0], -1,
                    per_level_center_head.shape[-1])
                per_level_controllers_head = per_level_controllers_head.view(
                    per_level_controllers_head.shape[0], -1,
                    per_level_controllers_head.shape[-1])
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
                controllers_preds.append(per_level_controllers_head)

            cls_scores = torch.cat(cls_scores, axis=1)
            cls_classes = torch.cat(cls_classes, axis=1)
            pred_bboxes = torch.cat(pred_bboxes, axis=1)
            controllers_preds = torch.cat(controllers_preds, axis=1)

            batch_size = cls_scores.shape[0]
            final_mask_h, final_mask_w = mask_out.shape[
                -3] * self.mask_stride, mask_out.shape[-2] * self.mask_stride
            batch_scores = torch.ones((batch_size, self.max_object_num),
                                      dtype=torch.float32,
                                      device=device) * (-1)
            batch_classes = torch.ones((batch_size, self.max_object_num),
                                       dtype=torch.float32,
                                       device=device) * (-1)
            batch_masks = torch.zeros(
                (batch_size, self.max_object_num, final_mask_h, final_mask_w),
                dtype=torch.bool,
                device=device)
            batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4),
                                       dtype=torch.float32,
                                       device=device)

            for img_idx, (per_image_scores, per_image_score_classes,
                          per_image_pred_bboxes, per_image_controllers_pred,
                          per_image_mask_out) in enumerate(
                              zip(cls_scores, cls_classes, pred_bboxes,
                                  controllers_preds, mask_out)):
                score_classes = per_image_score_classes[
                    per_image_scores > self.min_score_threshold].float()
                bboxes = per_image_pred_bboxes[
                    per_image_scores > self.min_score_threshold].float()
                per_image_controllers_pred = per_image_controllers_pred[
                    per_image_scores > self.min_score_threshold]
                scores = per_image_scores[
                    per_image_scores > self.min_score_threshold].float()

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_bboxes = bboxes[sorted_indexes]
                    sorted_controllers = per_image_controllers_pred[
                        sorted_indexes]

                    if self.topn < sorted_scores.shape[0]:
                        sorted_scores = sorted_scores[0:self.topn]
                        sorted_score_classes = sorted_score_classes[0:self.
                                                                    topn]
                        sorted_bboxes = sorted_bboxes[0:self.topn]
                        sorted_controllers = sorted_controllers[0:self.topn]

                    keep = self.nms_function(sorted_bboxes, sorted_scores)

                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_bboxes = sorted_bboxes[keep]
                    keep_controllers = sorted_controllers[keep]

                    final_detection_num = min(self.max_object_num,
                                              keep_scores.shape[0])
                    keep_scores = sorted_scores[0:final_detection_num]
                    keep_classes = sorted_score_classes[0:final_detection_num]
                    keep_bboxes = sorted_bboxes[0:final_detection_num]
                    keep_controllers = sorted_controllers[
                        0:final_detection_num]
                    keep_masks = self.get_pred_masks(keep_controllers,
                                                     per_image_mask_out)
                    keep_masks = (keep_masks > self.mask_threshold)

                    batch_scores[img_idx, 0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    batch_classes[img_idx,
                                  0:final_detection_num] = keep_classes[
                                      0:final_detection_num]
                    batch_masks[img_idx, 0:final_detection_num] = keep_masks
                    batch_bboxes[img_idx,
                                 0:final_detection_num, :] = keep_bboxes[
                                     0:final_detection_num, :]

            return batch_scores, batch_classes, batch_masks, batch_bboxes

    def get_pred_masks(self, keep_controllers, per_image_mask_out):
        device = per_image_mask_out.device
        mask_h, mask_w, _ = per_image_mask_out.shape
        relative_coord_x = torch.arange(mask_w).view(1, -1).float().repeat(
            mask_h, 1) / (mask_w - 1) * 2 - 1
        relative_coord_y = torch.arange(mask_h).view(-1, 1).float().repeat(
            1, mask_w) / (mask_h - 1) * 2 - 1
        relative_coord_x = relative_coord_x.view(1, mask_h, mask_w)
        relative_coord_y = relative_coord_y.view(1, mask_h, mask_w)
        relative_coord_x = relative_coord_x.permute(1, 2, 0).to(device)
        relative_coord_y = relative_coord_y.permute(1, 2, 0).to(device)
        per_image_mask_out = torch.cat(
            [per_image_mask_out, relative_coord_x, relative_coord_y], dim=-1)

        # 3 fcn conv layers weight and bias params idx
        conv1_w_end = int((self.num_masks + 2) * self.num_masks)
        conv1_bias_end = int(conv1_w_end + self.num_masks)
        conv2_w_end = int(conv1_bias_end + self.num_masks * self.num_masks)
        conv2_bias_end = int(conv2_w_end + self.num_masks)
        conv3_w_end = int(conv2_bias_end + self.num_masks)
        conv3_bias_end = int(conv3_w_end + 1)
        final_mask_h, final_mask_w = int(mask_h * self.mask_stride), int(
            mask_w * self.mask_stride)

        # get mask fcn head 3 conv layers weight and bias
        sample_nums = keep_controllers.shape[0]
        conv1_weights = keep_controllers[:, 0:conv1_w_end].reshape(
            -1, int(self.num_masks), int(self.num_masks + 2)).reshape(
                -1, int(self.num_masks + 2)).unsqueeze(-1).unsqueeze(-1)
        conv1_bias = keep_controllers[:, conv1_w_end:conv1_bias_end].flatten()
        conv2_weights = keep_controllers[:,
                                         conv1_bias_end:conv2_w_end].reshape(
                                             -1, int(self.num_masks),
                                             int(self.num_masks)).reshape(
                                                 -1, int(self.num_masks)
                                             ).unsqueeze(-1).unsqueeze(-1)
        conv2_bias = keep_controllers[:, conv2_w_end:conv2_bias_end].flatten()
        conv3_weights = keep_controllers[:,
                                         conv2_bias_end:conv3_w_end].unsqueeze(
                                             -1).unsqueeze(-1)
        conv3_bias = keep_controllers[:, conv3_w_end:conv3_bias_end].flatten()

        # get mask preds through mask fcn head(3 conv layers)
        per_image_mask_out = per_image_mask_out.permute(2, 0, 1).unsqueeze(0)
        per_image_pred_masks = F.conv2d(per_image_mask_out, conv1_weights,
                                        conv1_bias)
        per_image_pred_masks = F.relu(per_image_pred_masks)
        per_image_pred_masks = F.conv2d(per_image_pred_masks,
                                        conv2_weights,
                                        conv2_bias,
                                        groups=sample_nums)
        per_image_pred_masks = F.relu(per_image_pred_masks)
        per_image_pred_masks = F.conv2d(per_image_pred_masks,
                                        conv3_weights,
                                        conv3_bias,
                                        groups=sample_nums)

        # resize mask preds to upsample 2x
        per_image_pred_masks = F.pad(per_image_pred_masks,
                                     pad=(0, 1, 0, 1),
                                     mode="replicate")
        oh, ow = final_mask_h + 1, final_mask_w + 1
        per_image_pred_masks = F.interpolate(per_image_pred_masks,
                                             size=(oh, ow),
                                             mode='nearest')
        pad_factor = int(self.mask_stride / 2.)
        per_image_pred_masks = F.pad(per_image_pred_masks,
                                     pad=(pad_factor, 0, pad_factor, 0),
                                     mode="replicate")
        per_image_pred_masks = per_image_pred_masks[:, :, :oh - 1, :ow - 1]
        per_image_pred_masks = torch.sigmoid(per_image_pred_masks)
        per_image_pred_masks = per_image_pred_masks.squeeze(0)

        return per_image_pred_masks

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

    from simpleAICV.segmentation.models.solov2 import SOLOV2
    net = SOLOV2(resnet_type='resnet50')
    image_h, image_w = 480, 640
    cate_outs, kernel_outs, mask_out = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    import pickle
    with open('sample.pkl', 'rb') as file:
        annotations = pickle.load(file)

    annotations['box'] = torch.tensor(annotations['box'], dtype=torch.float32)
    annotations['mask'] = torch.tensor(annotations['mask'],
                                       dtype=torch.float32)
    annotations['class'] = torch.tensor(annotations['class'],
                                        dtype=torch.float32)
    decode = Solov2Decoder()
    batch_scores, batch_classes, batch_masks, batch_bboxes = decode(
        cate_outs, kernel_outs, mask_out)
    print('1111', batch_scores.shape, batch_classes.shape, batch_masks.shape,
          batch_bboxes.shape)

    from simpleAICV.segmentation.models.condinst import CondInst
    net = CondInst(resnet_type='resnet50')
    image_h, image_w = 480, 640
    cls_heads, reg_heads, center_heads, controllers_heads, mask_out, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    import pickle
    with open('sample.pkl', 'rb') as file:
        annotations = pickle.load(file)

    annotations['box'] = torch.tensor(annotations['box'], dtype=torch.float32)
    annotations['mask'] = torch.tensor(annotations['mask'],
                                       dtype=torch.float32)
    annotations['class'] = torch.tensor(annotations['class'],
                                        dtype=torch.float32)
    decode = CondInstDecoder()
    batch_scores, batch_classes, batch_masks, batch_bboxes = decode(
        cls_heads, reg_heads, center_heads, controllers_heads, mask_out,
        batch_positions)
    print('2222', batch_scores.shape, batch_classes.shape, batch_masks.shape,
          batch_bboxes.shape)
