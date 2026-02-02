"""
https://github.com/huggingface/transformers/blob/main/src/transformers/models/mask2former/modeling_mask2former.py
https://github.com/tue-mps/eomt/blob/985630d27fc5adc05afd1a674c0fc0fb307ef928/training/mask_classification_loss.py
"""
from scipy.optimize import linear_sum_assignment

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = [
    'UniversalMattingLoss',
]


class Mask2FormerHungarianMatcher(nn.Module):

    def __init__(self,
                 global_trimap_ce_cost=1.0,
                 global_trimap_iou_cost=1.0,
                 local_alpha_cost=1.0,
                 fusion_alpha_cost=1.0,
                 class_cost=1.0):
        super(Mask2FormerHungarianMatcher, self).__init__()

        self.global_trimap_ce_cost = global_trimap_ce_cost
        self.global_trimap_iou_cost = global_trimap_iou_cost
        self.local_alpha_cost = local_alpha_cost
        self.fusion_alpha_cost = fusion_alpha_cost
        self.class_cost = class_cost

    @torch.no_grad()
    def forward(self, global_preds, local_preds, fused_preds, class_preds,
                trimap_gts, alpha_gts, class_gts):
        """
        Args:
            global_preds: [B, query_num, 3, H, W]
            local_preds: [B, query_num, 1, H, W]
            fused_preds: [B, query_num, 1, H, W]
            class_preds: [B, query_num, num_classes]
            trimap_gts: list of [num_objects, H, W]
            alpha_gts: list of [num_objects, H, W]
            class_gts: list of [num_objects]
        """
        batch_size = fused_preds.shape[0]
        indices = []

        for i in range(batch_size):
            # 分类cost
            pred_probs = class_preds[i].softmax(dim=-1)
            class_cost = -pred_probs[:,
                                     class_gts[i]]  # [query_num, num_objects]

            # 全局trimap预测和目标
            pred_global = global_preds[i]  # [query_num, 3, H, W]
            target_trimap = trimap_gts[i].to(
                pred_global)  # [num_objects, H, W]

            # 局部alpha预测和目标
            pred_local = local_preds[i].squeeze(1)  # [query_num, H, W]
            pred_fused = fused_preds[i].squeeze(1)  # [query_num, H, W]
            target_alpha = alpha_gts[i].to(pred_fused)  # [num_objects, H, W]

            # 1. 全局trimap CE cost
            global_trimap_ce_cost = self.compute_pair_wise_trimap_ce_cost(
                pred_global, target_trimap)

            # 2. 全局trimap IOU cost
            global_trimap_iou_cost = self.compute_pair_wise_trimap_iou_cost(
                pred_global, target_trimap)

            # 3. 局部alpha L1 cost（仅在trimap=128的区域）
            local_alpha_cost = self.compute_pair_wise_local_alpha_cost(
                pred_local, target_alpha, target_trimap)

            # 4. 融合alpha L1 cost（全图）
            fusion_alpha_cost = self.compute_pair_wise_fusion_alpha_cost(
                pred_fused, target_alpha)

            # 总cost
            cost_matrix = (
                self.global_trimap_ce_cost * global_trimap_ce_cost +
                self.global_trimap_iou_cost * global_trimap_iou_cost +
                self.local_alpha_cost * local_alpha_cost +
                self.fusion_alpha_cost * fusion_alpha_cost +
                self.class_cost * class_cost)

            # 数值稳定性处理
            cost_matrix = torch.clamp(cost_matrix, min=-1e10, max=1e10)
            cost_matrix = torch.nan_to_num(cost_matrix, 0)

            # 匈牙利算法匹配
            assigned_indices = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        matched_indices = [(torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64))
                           for i, j in indices]

        return matched_indices

    def compute_pair_wise_trimap_ce_cost(self, pred_global, target_trimap):
        """
        计算全局trimap的CE cost
        Args:
            pred_global: [query_num, 3, H, W]
            target_trimap: [num_objects, H, W]
        Returns:
            cost: [query_num, num_objects]
        """
        query_num = pred_global.shape[0]
        num_objects = target_trimap.shape[0]

        # 转换trimap到类别索引
        convert_trimap = target_trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        # [query_num, 3, H, W] -> [query_num, H, W, 3]
        pred_global = pred_global.permute(0, 2, 3, 1)
        pred_global = torch.clamp(pred_global, min=1e-4, max=1. - 1e-4)

        cost_list = []
        for obj_idx in range(num_objects):
            target = convert_trimap[obj_idx]  # [H, W]
            target_onehot = F.one_hot(target.long(),
                                      num_classes=3).float()  # [H, W, 3]

            # 计算每个query对这个target的CE
            bce_loss = -(target_onehot * torch.log(pred_global) +
                         (1. - target_onehot) * torch.log(1. - pred_global))
            # [query_num, H, W, 3] -> [query_num]
            bce_loss = bce_loss.mean(dim=(1, 2, 3))
            cost_list.append(bce_loss)

        # [query_num, num_objects]
        cost = torch.stack(cost_list, dim=1)
        return cost

    def compute_pair_wise_trimap_iou_cost(self, pred_global, target_trimap):
        """
        计算全局trimap的IOU cost
        Args:
            pred_global: [query_num, 3, H, W]
            target_trimap: [num_objects, H, W]
        Returns:
            cost: [query_num, num_objects]
        """
        query_num = pred_global.shape[0]
        num_objects = target_trimap.shape[0]

        # 转换trimap到类别索引
        convert_trimap = target_trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        # [query_num, 3, H, W] -> [query_num, H, W, 3]
        pred_global = pred_global.permute(0, 2, 3, 1)
        pred_global = torch.clamp(pred_global, min=1e-4, max=1. - 1e-4)

        cost_list = []
        for obj_idx in range(num_objects):
            target = convert_trimap[obj_idx]  # [H, W]
            target_onehot = F.one_hot(target.long(),
                                      num_classes=3).float()  # [H, W, 3]

            # 计算IOU
            intersection = (pred_global * target_onehot).sum(dim=(1, 2, 3))
            union = (pred_global.sum(dim=(1, 2, 3)) +
                     target_onehot.sum(dim=(0, 1, 2)) - intersection)

            iou_loss = 1. - (intersection + 1e-4) / (union + 1e-4)
            cost_list.append(iou_loss)

        cost = torch.stack(cost_list, dim=1)
        return cost

    def compute_pair_wise_local_alpha_cost(self, pred_local, target_alpha,
                                           target_trimap):
        """
        计算局部alpha的L1 cost（仅在trimap=128区域）
        Args:
            pred_local: [query_num, H, W]
            target_alpha: [num_objects, H, W]
            target_trimap: [num_objects, H, W]
        Returns:
            cost: [query_num, num_objects]
        """
        query_num = pred_local.shape[0]
        num_objects = target_alpha.shape[0]

        pred_local = torch.clamp(pred_local, min=1e-4, max=1. - 1e-4)

        cost_list = []
        for obj_idx in range(num_objects):
            target = target_alpha[obj_idx]  # [H, W]
            trimap = target_trimap[obj_idx]  # [H, W]

            # 只在trimap=128的区域计算
            weighted = (trimap == 128).float()

            # [query_num, H, W]
            diff = torch.abs(pred_local - target)
            diff = diff * weighted

            # 归一化
            alpha_cost = diff.sum(dim=(1, 2)) / (weighted.sum() + 1.)
            cost_list.append(alpha_cost)

        cost = torch.stack(cost_list, dim=1)
        return cost

    def compute_pair_wise_fusion_alpha_cost(self, pred_fused, target_alpha):
        """
        计算融合alpha的L1 cost（全图）
        Args:
            pred_fused: [query_num, H, W]
            target_alpha: [num_objects, H, W]
        Returns:
            cost: [query_num, num_objects]
        """
        query_num = pred_fused.shape[0]
        num_objects = target_alpha.shape[0]

        pred_fused = torch.clamp(pred_fused, min=1e-4, max=1. - 1e-4)

        cost_list = []
        for obj_idx in range(num_objects):
            target = target_alpha[obj_idx]  # [H, W]

            # [query_num, H, W]
            diff = torch.abs(pred_fused - target)
            alpha_cost = diff.mean(dim=(1, 2))
            cost_list.append(alpha_cost)

        cost = torch.stack(cost_list, dim=1)

        return cost


class UniversalMattingLoss(nn.Module):

    def __init__(self,
                 global_trimap_ce_cost=1.0,
                 global_trimap_iou_cost=1.0,
                 local_alpha_cost=1.0,
                 fusion_alpha_cost=1.0,
                 class_cost=1.0,
                 num_classes=2,
                 global_trimap_ce_loss_weight=1.0,
                 global_trimap_iou_loss_weight=1.0,
                 local_alpha_loss_weight=1.0,
                 local_laplacian_loss_weight=1.0,
                 fusion_alpha_loss_weight=1.0,
                 fusion_laplacian_loss_weight=1.0,
                 class_loss_weight=1.0,
                 no_object_class_weight=0.1):
        super(UniversalMattingLoss, self).__init__()
        # num_classes has background class
        self.num_classes = num_classes

        self.global_trimap_ce_loss_weight = global_trimap_ce_loss_weight
        self.global_trimap_iou_loss_weight = global_trimap_iou_loss_weight
        self.local_alpha_loss_weight = local_alpha_loss_weight
        self.local_laplacian_loss_weight = local_laplacian_loss_weight
        self.fusion_alpha_loss_weight = fusion_alpha_loss_weight
        self.fusion_laplacian_loss_weight = fusion_laplacian_loss_weight
        self.class_loss_weight = class_loss_weight

        self.hungarian_matcher = Mask2FormerHungarianMatcher(
            global_trimap_ce_cost=global_trimap_ce_cost,
            global_trimap_iou_cost=global_trimap_iou_cost,
            local_alpha_cost=local_alpha_cost,
            fusion_alpha_cost=fusion_alpha_cost,
            class_cost=class_cost)

        self.register_buffer("ce_loss_weight", torch.ones(self.num_classes))
        self.ce_loss_weight[-1].fill_(no_object_class_weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_loss_weight)

    def get_pred_permutation_indices(self, indices):
        batch_indices = torch.cat([
            torch.full_like(pred_idx, i)
            for i, (pred_idx, _) in enumerate(indices)
        ])
        pred_indices = torch.cat([pred_idx for (pred_idx, _) in indices])

        return batch_indices, pred_indices

    def get_target_permutation_indices(self, indices):
        batch_indices = torch.cat([
            torch.full_like(target_idx, i)
            for i, (_, target_idx) in enumerate(indices)
        ])
        target_indices = torch.cat([target_idx for (_, target_idx) in indices])

        return batch_indices, target_indices

    def get_assigned_preds_and_targets(self, global_preds, local_preds,
                                       fused_preds, trimap_gts, alpha_gts,
                                       indices):
        device = fused_preds.device

        pred_idx = self.get_pred_permutation_indices(indices)
        target_idx = self.get_target_permutation_indices(indices)

        # get matched predictions
        # [Matched_Num, 3, H, W]
        matched_global_preds = global_preds[pred_idx]
        # [Matched_Num, 1, H, W]
        matched_local_preds = local_preds[pred_idx]
        # [Matched_Num, 1, H, W]
        matched_fused_preds = fused_preds[pred_idx]

        batch_size, batch_max_object_num, max_height, max_width = len(
            trimap_gts), 0, 0, 0
        for per_image_trimap_gts in trimap_gts:
            object_num, height, width = per_image_trimap_gts.shape
            batch_max_object_num = max(batch_max_object_num, object_num)
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        target_trimaps = torch.zeros(
            [batch_size, batch_max_object_num, max_height, max_width],
            dtype=torch.float32).to(device)
        for idx, per_image_trimap_gts in enumerate(trimap_gts):
            target_trimaps[
                idx, :per_image_trimap_gts.shape[0], :per_image_trimap_gts.
                shape[1], :per_image_trimap_gts.
                shape[2]] = per_image_trimap_gts
        matched_target_trimaps = target_trimaps[target_idx]

        target_alphas = torch.zeros(
            [batch_size, batch_max_object_num, max_height, max_width],
            dtype=torch.float32).to(device)
        for idx, per_image_alpha_gts in enumerate(alpha_gts):
            target_alphas[
                idx, :per_image_alpha_gts.shape[0], :per_image_alpha_gts.
                shape[1], :per_image_alpha_gts.shape[2]] = per_image_alpha_gts
        matched_target_alphas = target_alphas[target_idx]

        return matched_global_preds, matched_local_preds, matched_fused_preds, matched_target_trimaps, matched_target_alphas

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        gaussian = lambda x: np.exp(-((x - size // 2)**2) / (2 * sigma**2))
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        kernel = np.tile(kernel, (n_channels, 1, 1))
        kernel = torch.FloatTensor(kernel[:, None, :, :])

        return Variable(kernel, requires_grad=False)

    def laplacian_pyramid(self, img, kernel, max_levels=5):
        current = img
        pyr = []
        for _ in range(max_levels):
            filtered = self.conv_gauss(current, kernel)
            diff = current - filtered
            pyr.append(diff)
            current = F.avg_pool2d(filtered, 2)
        pyr.append(current)

        return pyr

    def conv_gauss(self, img, kernel):
        """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2),
                    mode='replicate')
        img = F.conv2d(img, kernel, groups=n_channels)

        return img

    def compute_global_trimap_ce_loss(self, global_pred, trimap):
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.float()
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]

        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        global_pred = global_pred.view(-1, num_classes)
        convert_trimap = convert_trimap.view(-1)
        loss_ground_truth = F.one_hot(convert_trimap.long(),
                                      num_classes=num_classes).float()
        bce_loss = -(loss_ground_truth * torch.log(global_pred) +
                     (1. - loss_ground_truth) * torch.log(1. - global_pred))

        bce_loss = bce_loss.mean()

        return bce_loss

    def compute_global_trimap_iou_loss(self, global_pred, trimap):
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.float()
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]

        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        global_pred = global_pred.view(-1, num_classes)
        convert_trimap = convert_trimap.view(-1)

        label = F.one_hot(convert_trimap.long(),
                          num_classes=num_classes).float()

        intersection = global_pred * label

        iou_loss = 1. - (torch.sum(intersection, dim=1) + 1e-4) / (
            torch.sum(global_pred, dim=1) + torch.sum(label, dim=1) -
            torch.sum(intersection, dim=1) + 1e-4)
        iou_loss = iou_loss.mean()

        return iou_loss

    def compute_local_alpha_loss(self, local_pred, alpha, trimap):
        # local_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,h,w]
        # trimap shape:[b,h,w]
        local_pred = local_pred.float()
        local_pred = local_pred.permute(0, 2, 3, 1).contiguous()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        diff = local_pred - alpha
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum() + 1.)

        return alpha_loss

    def compute_local_laplacian_loss(self, local_pred, alpha, trimap):
        # local_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,h,w]
        # trimap shape:[b,h,w]
        device = local_pred.device
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)

        alpha = torch.unsqueeze(alpha, dim=1)
        trimap = torch.unsqueeze(trimap, dim=1)

        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        local_pred = local_pred * weighted
        alpha = alpha * weighted
        gauss_kernel = self.build_gauss_kernel(size=5, sigma=1.0,
                                               n_channels=1).to(device)
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(local_pred, gauss_kernel, 5)

        laplacian_loss = sum(
            F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        return laplacian_loss

    def compute_fusion_alpha_loss(self, fusion_pred, alpha):
        # fusion_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = fusion_pred.permute(0, 2, 3, 1).contiguous()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        weighted = torch.ones_like(alpha)

        diff = fusion_pred - alpha
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())

        return alpha_loss

    def compute_fusion_laplacian_loss(self, fusion_pred, alpha):
        # fusion_pred shape:[b,1,h,w]
        # alpha shape:[b,h,w]
        device = fusion_pred.device
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)

        alpha = torch.unsqueeze(alpha, dim=1)

        gauss_kernel = self.build_gauss_kernel(size=5, sigma=1.0,
                                               n_channels=1).to(device)
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(fusion_pred, gauss_kernel, 5)

        laplacian_loss = sum(
            F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        return laplacian_loss

    def compute_batch_class_loss(self, class_preds, class_gts, indices):
        """计算分类损失"""
        device = class_preds.device
        batch_size, query_num = class_preds.shape[0], class_preds.shape[1]

        idx = self.get_pred_permutation_indices(indices)
        # [batch_size, query_num]
        # self.num_classes - 1 is background class index
        class_targets = torch.full((batch_size, query_num),
                                   fill_value=self.num_classes - 1,
                                   dtype=torch.int64,
                                   device=device)
        class_target_objects = torch.cat(
            [target[j] for target, (_, j) in zip(class_gts, indices)])
        class_targets[idx] = class_target_objects

        # [batch_size, query_num, num_classes] -> [batch_size, num_classes, query_num]
        class_preds = class_preds.transpose(1, 2)
        class_loss = self.ce_loss(class_preds, class_targets)

        return class_loss

    def forward(self, global_preds, local_preds, fused_preds, class_preds,
                trimap_gts, alpha_gts, class_gts):
        """
        global_preds: [B, query_num, 3, H, W] - global trimap predictions
        local_preds: [B, query_num, 1, H, W] - local alpha predictions
        fused_preds: [B, query_num, 1, H, W] - fused alpha predictions
        class_preds: [B, query_num, num_classes] - classification predictions
        trimap_gts: list of [num_objects, H, W] - trimap ground truths
        alpha_gts: list of [num_objects, H, W] - alpha matte ground truths
        class_gts: list of [num_objects] - class labels ground truths
        """
        global_preds = global_preds.float()
        local_preds = local_preds.float()
        fused_preds = fused_preds.float()
        class_preds = class_preds.float()

        device = fused_preds.device

        trimap_gts = [
            per_image_trimap_gts.float().to(device)
            for per_image_trimap_gts in trimap_gts
        ]
        alpha_gts = [
            per_image_alpha_gts.float().to(device)
            for per_image_alpha_gts in alpha_gts
        ]
        class_gts = [
            per_image_class_gts.long().to(device)
            for per_image_class_gts in class_gts
        ]

        indices = self.hungarian_matcher(global_preds=global_preds,
                                         local_preds=local_preds,
                                         fused_preds=fused_preds,
                                         class_preds=class_preds,
                                         trimap_gts=trimap_gts,
                                         alpha_gts=alpha_gts,
                                         class_gts=class_gts)

        matched_global_preds, matched_local_preds, matched_fused_preds, matched_target_trimaps, matched_target_alphas = self.get_assigned_preds_and_targets(
            global_preds, local_preds, fused_preds, trimap_gts, alpha_gts,
            indices)

        global_trimap_ce_loss = self.compute_global_trimap_ce_loss(
            matched_global_preds, matched_target_trimaps)
        global_trimap_iou_loss = self.compute_global_trimap_iou_loss(
            matched_global_preds, matched_target_trimaps)
        local_alpha_loss = self.compute_local_alpha_loss(
            matched_local_preds, matched_target_alphas, matched_target_trimaps)
        local_laplacian_loss = self.compute_local_laplacian_loss(
            matched_local_preds, matched_target_alphas, matched_target_trimaps)
        fusion_alpha_loss = self.compute_fusion_alpha_loss(
            matched_fused_preds, matched_target_alphas)
        fusion_laplacian_loss = self.compute_fusion_laplacian_loss(
            matched_fused_preds, matched_target_alphas)
        class_loss = self.compute_batch_class_loss(class_preds, class_gts,
                                                   indices)

        global_trimap_ce_loss = self.global_trimap_ce_loss_weight * global_trimap_ce_loss
        global_trimap_iou_loss = self.global_trimap_iou_loss_weight * global_trimap_iou_loss
        local_alpha_loss = self.local_alpha_loss_weight * local_alpha_loss
        local_laplacian_loss = self.local_laplacian_loss_weight * local_laplacian_loss
        fusion_alpha_loss = self.fusion_alpha_loss_weight * fusion_alpha_loss
        fusion_laplacian_loss = self.fusion_laplacian_loss_weight * fusion_laplacian_loss
        class_loss = self.class_loss_weight * class_loss

        loss_dict = {
            'global_trimap_ce_loss': global_trimap_ce_loss,
            'global_trimap_iou_loss': global_trimap_iou_loss,
            'local_alpha_loss': local_alpha_loss,
            'local_laplacian_loss': local_laplacian_loss,
            'fusion_alpha_loss': fusion_alpha_loss,
            'fusion_laplacian_loss': fusion_laplacian_loss,
            'class_loss': class_loss,
        }

        return loss_dict


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

    from tools.path import human_matting_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.datasets.human_matting_dataset import HumanMattingDataset
    from SimpleAICV.universal_segmentation.human_matting_common import RandomHorizontalFlip, Resize, Normalize, HumanMattingTrainCollater

    human_matting_dataset = HumanMattingDataset(
        human_matting_dataset_path,
        set_name_list=[
            'Deep_Automatic_Portrait_Matting',
            'RealWorldPortrait636',
            'P3M10K',
        ],
        set_type='train',
        max_side=2048,
        kernel_size_range=[15, 15],
        transform=transforms.Compose([
            Resize(resize=512),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = HumanMattingTrainCollater(resize=512)
    train_loader = DataLoader(human_matting_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.universal_segmentation.models.dinov3_universal_matting import dinov3_vit_base_patch16_universal_matting
    net = dinov3_vit_base_patch16_universal_matting(
        backbone_pretrained_path='', image_size=512, num_classes=2)

    loss1 = UniversalMattingLoss(global_trimap_ce_cost=1.0,
                                 global_trimap_iou_cost=1.0,
                                 local_alpha_cost=1.0,
                                 fusion_alpha_cost=1.0,
                                 class_cost=1.0,
                                 num_classes=2,
                                 global_trimap_ce_loss_weight=1.0,
                                 global_trimap_iou_loss_weight=1.0,
                                 local_alpha_loss_weight=1.0,
                                 local_laplacian_loss_weight=1.0,
                                 fusion_alpha_loss_weight=1.0,
                                 fusion_laplacian_loss_weight=1.0,
                                 class_loss_weight=1.0,
                                 no_object_class_weight=0.1)
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, labels, sizes = data[
            'image'], data['mask'], data['trimap'], data['fg_map'], data[
                'bg_map'], data['label'], data['size']
        print('1111', images.shape, len(masks), len(trimaps), len(fg_maps),
              len(bg_maps), len(labels), sizes.shape)

        for per_image_masks, per_image_trimaps, per_image_fg_maps, per_image_bg_maps, per_image_labels in zip(
                masks, trimaps, fg_maps, bg_maps, labels):
            print('2222', per_image_masks.shape, per_image_trimaps.shape,
                  per_image_fg_maps.shape, per_image_bg_maps.shape,
                  per_image_labels.shape)
            print('3333', per_image_labels)

        global_preds, local_preds, fused_preds, class_preds = net(images)
        out = loss1(global_preds, local_preds, fused_preds, class_preds,
                    trimaps, masks, labels)
        print('3333', out)

        break
