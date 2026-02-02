import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = [
    'SAMMattingLoss',
    'SAMMattingMultiLevelLoss',
]


class SAMMattingLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 global_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 iou_predict_loss_weight=1,
                 supervise_all_iou=True,
                 mask_threshold=0.5):
        super(SAMMattingLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.global_pred_trimap_iou_loss_weight = global_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.supervise_all_iou = supervise_all_iou
        self.mask_threshold = mask_threshold

    def forward(self, images, inputs, targets):
        # images: torch.Size([2, 3, 1024, 1024])
        # all_iter_global_preds: iter_num, torch.Size([2, 4, 3, 1024, 1024])
        # all_iter_local_preds: iter_num, torch.Size([2, 4, 1, 1024, 1024])
        # all_iter_fused_preds: iter_num, torch.Size([2, 4, 1, 1024, 1024])
        # all_iter_iou_preds: iter_num, torch.Size([2, 4])
        all_iter_global_preds, all_iter_local_preds, all_iter_fused_preds, all_iter_iou_preds = inputs
        # masks: torch.Size([2, 1, 1024, 1024])
        # trimaps: torch.Size([2, 1024, 1024])
        # fg_maps: torch.Size([2, 3, 1024, 1024])
        # bg_maps: torch.Size([2, 3, 1024, 1024])
        masks, trimaps, fg_maps, bg_maps = targets

        assert len(all_iter_global_preds) == len(all_iter_local_preds) == len(
            all_iter_fused_preds) == len(all_iter_iou_preds)

        global_pred_trimap_ce_loss = 0.
        global_pred_trimap_iou_loss = 0.
        local_pred_alpha_loss = 0.
        local_pred_laplacian_loss = 0.
        fusion_pred_alpha_loss = 0.
        fusion_pred_laplacian_loss = 0.
        composition_loss = 0.
        iou_predict_loss = 0.
        iter_num = len(all_iter_global_preds)
        for per_iter_global_preds, per_iter_local_preds, per_iter_fused_preds, per_iter_iou_preds in zip(
                all_iter_global_preds, all_iter_local_preds,
                all_iter_fused_preds, all_iter_iou_preds):
            # per_iter_global_preds: torch.Size([2, 4, 3, 1024, 1024])
            # per_iter_local_preds: torch.Size([2, 4, 1, 1024, 1024])
            # per_iter_fused_preds: torch.Size([2, 4, 1, 1024, 1024])
            # per_iter_iou_preds: torch.Size([2, 4])

            per_iter_global_pred_trimap_ce_loss, per_iter_global_pred_trimap_iou_loss, per_iter_local_pred_alpha_loss, \
            per_iter_local_pred_laplacian_loss, per_iter_fusion_pred_alpha_loss,per_iter_fusion_pred_laplacian_loss, \
            per_iter_composition_loss,per_iter_iou_predict_loss = self.compute_per_iter_matting_loss(
                per_iter_global_preds,
                per_iter_local_preds,
                per_iter_fused_preds,
                per_iter_iou_preds,
                images,
                masks,
                trimaps,
                fg_maps,
                bg_maps,
            )

            global_pred_trimap_ce_loss += per_iter_global_pred_trimap_ce_loss
            global_pred_trimap_iou_loss += per_iter_global_pred_trimap_iou_loss
            local_pred_alpha_loss += per_iter_local_pred_alpha_loss
            local_pred_laplacian_loss += per_iter_local_pred_laplacian_loss
            fusion_pred_alpha_loss += per_iter_fusion_pred_alpha_loss
            fusion_pred_laplacian_loss += per_iter_fusion_pred_laplacian_loss
            composition_loss += per_iter_composition_loss
            iou_predict_loss += per_iter_iou_predict_loss

        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss / float(
            iter_num)
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss / float(
            iter_num)
        local_pred_alpha_loss = local_pred_alpha_loss / float(iter_num)
        local_pred_laplacian_loss = local_pred_laplacian_loss / float(iter_num)
        fusion_pred_alpha_loss = fusion_pred_alpha_loss / float(iter_num)
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss / float(
            iter_num)
        composition_loss = composition_loss / float(iter_num)
        iou_predict_loss = iou_predict_loss / float(iter_num)

        global_pred_trimap_ce_loss = self.global_pred_trimap_ce_loss_weight * global_pred_trimap_ce_loss
        global_pred_trimap_iou_loss = self.global_pred_trimap_iou_loss_weight * global_pred_trimap_iou_loss
        local_pred_alpha_loss = self.local_pred_alpha_loss_weight * local_pred_alpha_loss
        local_pred_laplacian_loss = self.local_pred_laplacian_loss_weight * local_pred_laplacian_loss
        fusion_pred_alpha_loss = self.fusion_pred_alpha_loss_weight * fusion_pred_alpha_loss
        fusion_pred_laplacian_loss = self.fusion_pred_laplacian_loss_weight * fusion_pred_laplacian_loss
        composition_loss = self.composition_loss_weight * composition_loss
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'global_pred_trimap_iou_loss': global_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def compute_per_iter_matting_loss(
        self,
        per_iter_global_preds,
        per_iter_local_preds,
        per_iter_fused_preds,
        per_iter_iou_preds,
        images,
        masks,
        trimaps,
        fg_maps,
        bg_maps,
    ):
        # per_iter_global_preds: torch.Size([2, 4, 3, 1024, 1024])
        # per_iter_local_preds: torch.Size([2, 4, 1, 1024, 1024])
        # per_iter_fused_preds: torch.Size([2, 4, 1, 1024, 1024])
        # per_iter_iou_preds: torch.Size([2, 4])
        # images: torch.Size([2, 3, 1024, 1024])
        # masks: torch.Size([2, 1, 1024, 1024])
        # trimaps: torch.Size([2, 1024, 1024])
        # fg_maps: torch.Size([2, 3, 1024, 1024])
        # bg_maps: torch.Size([2, 3, 1024, 1024])

        # global_pred_trimap_ce_loss: torch.Size([2, 4])
        # global_pred_trimap_iou_loss: torch.Size([2, 4])
        # local_pred_alpha_loss: torch.Size([2, 4])
        # local_pred_laplacian_loss: torch.Size([2, 4])
        # fusion_pred_alpha_loss: torch.Size([2, 4])
        # fusion_pred_laplacian_loss: torch.Size([2, 4])
        # composition_loss: torch.Size([2, 4])
        # iou_predict_loss: torch.Size([2, 4])
        global_pred_trimap_ce_loss = self.compute_global_trimap_ce_loss(
            per_iter_global_preds, trimaps)
        global_pred_trimap_iou_loss = self.compute_global_trimap_iou_loss(
            per_iter_global_preds, trimaps)
        local_pred_alpha_loss = self.compute_local_alpha_loss(
            per_iter_local_preds, masks, trimaps)
        local_pred_laplacian_loss = self.compute_local_laplacian_loss(
            per_iter_local_preds, masks, trimaps)
        fusion_pred_alpha_loss = self.compute_fusion_alpha_loss(
            per_iter_fused_preds, masks)
        fusion_pred_laplacian_loss = self.compute_fusion_laplacian_loss(
            per_iter_fused_preds, masks)
        composition_loss = self.compute_composition_loss(
            images, masks, fg_maps, bg_maps, per_iter_fused_preds)
        iou_predict_loss = self.compute_iou_predict_loss(
            per_iter_fused_preds, masks, per_iter_iou_preds)

        # 如果预测了多个mask,只对最优mask回传focal loss和dice loss；
        # 但对于iou_predict_loss,多mask全部回传
        if global_pred_trimap_ce_loss.shape[1] > 1:
            # [B, M], 组合loss选最优
            combine_loss = global_pred_trimap_ce_loss * self.global_pred_trimap_ce_loss_weight \
            + global_pred_trimap_iou_loss * self.global_pred_trimap_iou_loss_weight \
            + local_pred_alpha_loss * self.local_pred_alpha_loss_weight \
            + local_pred_laplacian_loss * self.local_pred_laplacian_loss_weight \
            + fusion_pred_alpha_loss * self.fusion_pred_alpha_loss_weight \
            + fusion_pred_laplacian_loss * self.fusion_pred_laplacian_loss_weight \
            + composition_loss * self.composition_loss_weight

            best_index = torch.argmin(combine_loss, dim=-1)
            batch_index = torch.arange(combine_loss.shape[0],
                                       device=combine_loss.device)

            # focal loss和dice loss取最优mask的loss
            global_pred_trimap_ce_loss = global_pred_trimap_ce_loss[
                batch_index, best_index].unsqueeze(1)
            global_pred_trimap_iou_loss = global_pred_trimap_iou_loss[
                batch_index, best_index].unsqueeze(1)
            local_pred_alpha_loss = local_pred_alpha_loss[
                batch_index, best_index].unsqueeze(1)
            local_pred_laplacian_loss = local_pred_laplacian_loss[
                batch_index, best_index].unsqueeze(1)
            fusion_pred_alpha_loss = fusion_pred_alpha_loss[
                batch_index, best_index].unsqueeze(1)
            fusion_pred_laplacian_loss = fusion_pred_laplacian_loss[
                batch_index, best_index].unsqueeze(1)
            composition_loss = composition_loss[batch_index,
                                                best_index].unsqueeze(1)

            # supervise_all_iou为True, iou_predict_loss多mask全监督, 否则只监督最优那个
            if self.supervise_all_iou:
                iou_predict_loss = iou_predict_loss.mean(dim=-1, keepdim=True)
            else:
                iou_predict_loss = iou_predict_loss[batch_index,
                                                    best_index].unsqueeze(1)

        # global_pred_trimap_ce_loss: torch.Size([2, 1])
        # global_pred_trimap_iou_loss: torch.Size([2, 1])
        # local_pred_alpha_loss: torch.Size([2, 1])
        # local_pred_laplacian_loss: torch.Size([2, 1])
        # fusion_pred_alpha_loss: torch.Size([2, 1])
        # fusion_pred_laplacian_loss: torch.Size([2, 1])
        # composition_loss: torch.Size([2, 1])
        # iou_predict_loss: torch.Size([2, 1])
        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss.sum()
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss.sum()
        local_pred_alpha_loss = local_pred_alpha_loss.sum()
        local_pred_laplacian_loss = local_pred_laplacian_loss.sum()
        fusion_pred_alpha_loss = fusion_pred_alpha_loss.sum()
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss.sum()
        composition_loss = composition_loss.sum()
        iou_predict_loss = iou_predict_loss.sum()

        return global_pred_trimap_ce_loss, global_pred_trimap_iou_loss, local_pred_alpha_loss, \
        local_pred_laplacian_loss, fusion_pred_alpha_loss, fusion_pred_laplacian_loss, \
        composition_loss, iou_predict_loss

    def compute_global_trimap_ce_loss(self, global_pred, trimap):
        # global_pred: torch.Size([2, 4, 3, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # global_pred: [b,4,3,h,w] -> [b,4,h,w,3]
        # trimap: [b,h,w]->[b,4,h,w]
        batch_size = global_pred.shape[0]

        global_pred = global_pred.float()
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        # trimap: torch.Size([b, 4, h, w])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, global_pred.shape[1], dim=1)

        # convert_trimap: [b,4,h,w]
        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        # loss_ground_truth: torch.Size([2, 4, 1024, 1024, 3])
        # global_pred: torch.Size([2, 4, 1024, 1024, 3])
        loss_ground_truth = F.one_hot(convert_trimap.long(),
                                      num_classes=num_classes).float()
        bce_loss = -(loss_ground_truth * torch.log(global_pred) +
                     (1. - loss_ground_truth) * torch.log(1. - global_pred))
        # h,w,num_classes维度合成一维然后mean
        bce_loss = bce_loss.flatten(2).mean(dim=-1)
        # batch_size维度上平均,但保留维度
        bce_loss = bce_loss / batch_size

        return bce_loss

    def compute_global_trimap_iou_loss(self, global_pred, trimap):
        # global_pred: torch.Size([2, 4, 3, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # global_pred: [b,4,3,h,w] -> [b,4,h,w,3]
        # trimap: [b,h,w]->[b,4,h,w]
        batch_size = global_pred.shape[0]

        global_pred = global_pred.float()
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        # trimap: torch.Size([b, 4, h, w])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, global_pred.shape[1], dim=1)

        # convert_trimap: [b,4,h,w]
        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        # loss_ground_truth: torch.Size([2, 4, 1024, 1024, 3])
        # intersection: torch.Size([2, 4, 1024, 1024, 3])
        loss_ground_truth = F.one_hot(convert_trimap.long(),
                                      num_classes=num_classes).float()
        intersection = global_pred * loss_ground_truth

        # num_classes维度累加
        # global_pred: torch.Size([2, 4, 1024, 1024])
        # intersection: torch.Size([2, 4, 1024, 1024])
        # loss_ground_truth: torch.Size([2, 4, 1024, 1024])
        global_pred = global_pred.sum(dim=-1)
        intersection = intersection.sum(dim=-1)
        loss_ground_truth = loss_ground_truth.sum(dim=-1)

        iou_loss = 1. - (intersection + 1e-4) / (
            global_pred + loss_ground_truth - intersection + 1e-4)
        # h,w维度合成一维然后mean
        iou_loss = iou_loss.flatten(2).mean(dim=-1)
        # batch_size维度上平均,但保留维度
        iou_loss = iou_loss / batch_size

        return iou_loss

    def compute_local_alpha_loss(self, local_pred, alpha, trimap):
        # local_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # local_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        # trimap: [b,h,w]->[b,1,h,w]
        local_pred = local_pred.float()
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        # trimap: torch.Size([2, 4, 1024, 1024])
        # weighted: torch.Size([2, 4, 1024, 1024])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        # local_pred: torch.Size([2, 4, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        diff = local_pred - alpha
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff**2 + 1e-12)

        alpha_loss = alpha_loss.sum(dim=[2, 3])
        weighted = weighted.sum(dim=[2, 3])

        alpha_loss = alpha_loss / (weighted + 1.)

        return alpha_loss

    def compute_local_laplacian_loss(self, local_pred, alpha, trimap):
        # local_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # local_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        # trimap: [b,h,w] -> [b,1,h,w]
        batch_size = local_pred.shape[0]

        device = local_pred.device
        local_pred = local_pred.float()
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        # alpha: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, local_pred.shape[1], dim=1)

        # trimap: torch.Size([2, 4, 1024, 1024])
        # weighted: torch.Size([2, 4, 1024, 1024])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        # local_pred: torch.Size([2, 4, 1024, 1024])
        # alpha: torch.Size([2, 4, 1024, 1024])
        local_pred = local_pred * weighted
        alpha = alpha * weighted

        # gauss_kernel: torch.Size([4, 1, 5, 5])
        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=local_pred.shape[1]).to(device)

        # pyr_alpha, pyr_predict: torch.Size([2, 4, 1024, 1024]) torch.Size([2, 4, 1024, 1024])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 512, 512]) torch.Size([2, 4, 512, 512])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 256, 256]) torch.Size([2, 4, 256, 256])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 128, 128]) torch.Size([2, 4, 128, 128])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 64, 64]) torch.Size([2, 4, 64, 64])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 32, 32]) torch.Size([2, 4, 32, 32])
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(local_pred, gauss_kernel, 5)

        # laplacian_loss: torch.Size([2, 4, 1024, 1024])
        # laplacian_loss: torch.Size([2, 4, 512, 512])
        # laplacian_loss: torch.Size([2, 4, 256, 256])
        # laplacian_loss: torch.Size([2, 4, 128, 128])
        # laplacian_loss: torch.Size([2, 4, 64, 64])
        # laplacian_loss: torch.Size([2, 4, 32, 32])
        laplacian_loss = [
            F.l1_loss(a, b, reduction="none")
            for a, b in zip(pyr_alpha, pyr_predict)
        ]

        total_laplacian_loss = 0.
        for per_level_laplacian_loss in laplacian_loss:
            # h,w维度合成一维然后mean
            per_level_laplacian_loss = per_level_laplacian_loss.flatten(
                2).mean(dim=-1)
            # batch_size维度上平均,但保留维度
            per_level_laplacian_loss = per_level_laplacian_loss / batch_size
            total_laplacian_loss += per_level_laplacian_loss

        return total_laplacian_loss

    def compute_fusion_alpha_loss(self, fusion_pred, alpha):
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # fusion_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        # alpha: torch.Size([2, 4, 1024, 1024])
        # weighted: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)
        weighted = torch.ones_like(alpha)

        diff = fusion_pred - alpha
        alpha_loss = torch.sqrt(diff**2 + 1e-12)

        alpha_loss = alpha_loss.sum(dim=[2, 3])
        weighted = weighted.sum(dim=[2, 3])

        alpha_loss = alpha_loss / (weighted + 1.)

        return alpha_loss

    def compute_fusion_laplacian_loss(self, fusion_pred, alpha):
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # fusion_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        batch_size = fusion_pred.shape[0]

        device = fusion_pred.device
        fusion_pred = fusion_pred.float()
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        # alpha: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)

        # gauss_kernel: torch.Size([4, 1, 5, 5])
        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=fusion_pred.shape[1]).to(device)

        # pyr_alpha, pyr_predict: torch.Size([2, 4, 1024, 1024]) torch.Size([2, 4, 1024, 1024])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 512, 512]) torch.Size([2, 4, 512, 512])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 256, 256]) torch.Size([2, 4, 256, 256])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 128, 128]) torch.Size([2, 4, 128, 128])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 64, 64]) torch.Size([2, 4, 64, 64])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 32, 32]) torch.Size([2, 4, 32, 32])
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(fusion_pred, gauss_kernel, 5)

        # laplacian_loss: torch.Size([2, 4, 1024, 1024])
        # laplacian_loss: torch.Size([2, 4, 512, 512])
        # laplacian_loss: torch.Size([2, 4, 256, 256])
        # laplacian_loss: torch.Size([2, 4, 128, 128])
        # laplacian_loss: torch.Size([2, 4, 64, 64])
        # laplacian_loss: torch.Size([2, 4, 32, 32])
        laplacian_loss = [
            F.l1_loss(a, b, reduction="none")
            for a, b in zip(pyr_alpha, pyr_predict)
        ]

        total_laplacian_loss = 0.
        for per_level_laplacian_loss in laplacian_loss:
            # h,w维度合成一维然后mean
            per_level_laplacian_loss = per_level_laplacian_loss.flatten(
                2).mean(dim=-1)
            # batch_size维度上平均,但保留维度
            per_level_laplacian_loss = per_level_laplacian_loss / batch_size
            total_laplacian_loss += per_level_laplacian_loss

        return total_laplacian_loss

    def compute_composition_loss(self, image, alpha, fg_map, bg_map,
                                 fusion_pred):
        # image: torch.Size([2, 3, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # fg_map: torch.Size([2, 3, 1024, 1024])
        # bg_map: torch.Size([2, 3, 1024, 1024])
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # image:[b,3,h,w]
        # alpha:[b,1,h,w]
        # fg_map:[b,3,h,w]
        # bg_map:[b,3,h,w]
        # fusion_pred:[b,4,1,h,w] -> [b,4,3,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.cat([fusion_pred, fusion_pred, fusion_pred], dim=2)

        # alpha: torch.Size([2, 3, 1024, 1024])
        # weighted: torch.Size([2, 3, 1024, 1024])
        alpha = torch.cat([alpha, alpha, alpha], dim=1)
        weighted = torch.ones_like(alpha)

        # fg_map: torch.Size([2, 1, 3, 1024, 1024])
        # bg_map: torch.Size([2, 1, 3, 1024, 1024])
        # image: torch.Size([2, 1, 3, 1024, 1024])
        # weighted: torch.Size([2, 4, 3, 1024, 1024])
        fg_map = torch.unsqueeze(fg_map, dim=1)
        bg_map = torch.unsqueeze(bg_map, dim=1)
        image = torch.unsqueeze(image, dim=1)
        weighted = torch.unsqueeze(weighted, dim=1)
        weighted = torch.repeat_interleave(weighted,
                                           fusion_pred.shape[1],
                                           dim=1)

        composition = fusion_pred * fg_map + (1. - fusion_pred) * bg_map
        composition_loss = torch.sqrt((composition - image)**2 + 1e-12)

        composition_loss = composition_loss.sum(dim=[2, 3, 4])
        weighted = weighted.sum(dim=[2, 3, 4])

        composition_loss = composition_loss / weighted

        return composition_loss

    def compute_iou_predict_loss(self, fusion_pred, alpha, pred_ious):
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # pred_ious: torch.Size([2, 4])
        # fusion_pred:[b,4,1,h,w] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        batch_size = fusion_pred.shape[0]

        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=2)

        # alpha: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)

        fusion_pred = (fusion_pred > self.mask_threshold)
        alpha = (alpha > self.mask_threshold)

        # h,w维度合成一维
        # fusion_pred: torch.Size([2, 4, h*w])
        # alpha: torch.Size([2, 4, h*w])
        fusion_pred = fusion_pred.flatten(2)
        alpha = alpha.flatten(2)

        # intersection: torch.Size([2, 4])
        # union: torch.Size([2, 4])
        # gt_ious: torch.Size([2, 4])
        intersection = torch.sum(fusion_pred & alpha, dim=-1).float()
        union = torch.sum(fusion_pred | alpha, dim=-1).float()
        gt_ious = intersection / torch.clamp(union, min=1e-6)
        gt_ious = torch.clamp(gt_ious, min=0.0, max=1.0)

        # iou_predict_loss: torch.Size([2, 4])
        iou_predict_loss = F.l1_loss(pred_ious, gt_ious, reduction="none")
        # batch_size维度上平均,但保留维度
        iou_predict_loss = iou_predict_loss / batch_size

        return iou_predict_loss

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


class SAMMattingMultiLevelLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 global_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.5):
        super(SAMMattingMultiLevelLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.global_pred_trimap_iou_loss_weight = global_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.mask_threshold = mask_threshold

    def forward(self, images, inputs, targets):
        # images: torch.Size([2, 3, 1024, 1024])
        # all_iter_global_preds: iter_num, torch.Size([2, 4, 3, 1024, 1024])
        # all_iter_local_preds: iter_num, torch.Size([2, 4, 1, 1024, 1024])
        # all_iter_fused_preds: iter_num, torch.Size([2, 4, 1, 1024, 1024])
        # all_iter_iou_preds: iter_num, torch.Size([2, 4])
        all_iter_global_preds, all_iter_local_preds, all_iter_fused_preds, all_iter_iou_preds = inputs
        # masks: torch.Size([2, 1, 1024, 1024])
        # trimaps: torch.Size([2, 1024, 1024])
        # fg_maps: torch.Size([2, 3, 1024, 1024])
        # bg_maps: torch.Size([2, 3, 1024, 1024])
        masks, trimaps, fg_maps, bg_maps = targets

        assert len(all_iter_global_preds) == len(all_iter_local_preds) == len(
            all_iter_fused_preds) == len(all_iter_iou_preds)

        global_pred_trimap_ce_loss = 0.
        global_pred_trimap_iou_loss = 0.
        local_pred_alpha_loss = 0.
        local_pred_laplacian_loss = 0.
        fusion_pred_alpha_loss = 0.
        fusion_pred_laplacian_loss = 0.
        composition_loss = 0.
        iou_predict_loss = 0.
        iter_num = len(all_iter_global_preds)
        for per_iter_global_preds, per_iter_local_preds, per_iter_fused_preds, per_iter_iou_preds in zip(
                all_iter_global_preds, all_iter_local_preds,
                all_iter_fused_preds, all_iter_iou_preds):
            # per_iter_global_preds: torch.Size([2, 4, 3, 1024, 1024])
            # per_iter_local_preds: torch.Size([2, 4, 1, 1024, 1024])
            # per_iter_fused_preds: torch.Size([2, 4, 1, 1024, 1024])
            # per_iter_iou_preds: torch.Size([2, 4])

            per_iter_global_pred_trimap_ce_loss, per_iter_global_pred_trimap_iou_loss, per_iter_local_pred_alpha_loss, \
            per_iter_local_pred_laplacian_loss, per_iter_fusion_pred_alpha_loss,per_iter_fusion_pred_laplacian_loss, \
            per_iter_composition_loss,per_iter_iou_predict_loss = self.compute_per_iter_matting_loss(
                per_iter_global_preds,
                per_iter_local_preds,
                per_iter_fused_preds,
                per_iter_iou_preds,
                images,
                masks,
                trimaps,
                fg_maps,
                bg_maps,
            )

            global_pred_trimap_ce_loss += per_iter_global_pred_trimap_ce_loss
            global_pred_trimap_iou_loss += per_iter_global_pred_trimap_iou_loss
            local_pred_alpha_loss += per_iter_local_pred_alpha_loss
            local_pred_laplacian_loss += per_iter_local_pred_laplacian_loss
            fusion_pred_alpha_loss += per_iter_fusion_pred_alpha_loss
            fusion_pred_laplacian_loss += per_iter_fusion_pred_laplacian_loss
            composition_loss += per_iter_composition_loss
            iou_predict_loss += per_iter_iou_predict_loss

        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss / float(
            iter_num)
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss / float(
            iter_num)
        local_pred_alpha_loss = local_pred_alpha_loss / float(iter_num)
        local_pred_laplacian_loss = local_pred_laplacian_loss / float(iter_num)
        fusion_pred_alpha_loss = fusion_pred_alpha_loss / float(iter_num)
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss / float(
            iter_num)
        composition_loss = composition_loss / float(iter_num)
        iou_predict_loss = iou_predict_loss / float(iter_num)

        global_pred_trimap_ce_loss = self.global_pred_trimap_ce_loss_weight * global_pred_trimap_ce_loss
        global_pred_trimap_iou_loss = self.global_pred_trimap_iou_loss_weight * global_pred_trimap_iou_loss
        local_pred_alpha_loss = self.local_pred_alpha_loss_weight * local_pred_alpha_loss
        local_pred_laplacian_loss = self.local_pred_laplacian_loss_weight * local_pred_laplacian_loss
        fusion_pred_alpha_loss = self.fusion_pred_alpha_loss_weight * fusion_pred_alpha_loss
        fusion_pred_laplacian_loss = self.fusion_pred_laplacian_loss_weight * fusion_pred_laplacian_loss
        composition_loss = self.composition_loss_weight * composition_loss
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'global_pred_trimap_iou_loss': global_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def compute_per_iter_matting_loss(
        self,
        per_iter_global_preds,
        per_iter_local_preds,
        per_iter_fused_preds,
        per_iter_iou_preds,
        images,
        masks,
        trimaps,
        fg_maps,
        bg_maps,
    ):
        # per_iter_global_preds: torch.Size([2, 4, 3, 1024, 1024])
        # per_iter_local_preds: torch.Size([2, 4, 1, 1024, 1024])
        # per_iter_fused_preds: torch.Size([2, 4, 1, 1024, 1024])
        # per_iter_iou_preds: torch.Size([2, 4])
        # images: torch.Size([2, 3, 1024, 1024])
        # masks: torch.Size([2, 1, 1024, 1024])
        # trimaps: torch.Size([2, 1024, 1024])
        # fg_maps: torch.Size([2, 3, 1024, 1024])
        # bg_maps: torch.Size([2, 3, 1024, 1024])

        # global_pred_trimap_ce_loss: torch.Size([2, 4])
        # global_pred_trimap_iou_loss: torch.Size([2, 4])
        # local_pred_alpha_loss: torch.Size([2, 4])
        # local_pred_laplacian_loss: torch.Size([2, 4])
        # fusion_pred_alpha_loss: torch.Size([2, 4])
        # fusion_pred_laplacian_loss: torch.Size([2, 4])
        # composition_loss: torch.Size([2, 4])
        # iou_predict_loss: torch.Size([2, 4])
        global_pred_trimap_ce_loss = self.compute_global_trimap_ce_loss(
            per_iter_global_preds, trimaps)
        global_pred_trimap_iou_loss = self.compute_global_trimap_iou_loss(
            per_iter_global_preds, trimaps)
        local_pred_alpha_loss = self.compute_local_alpha_loss(
            per_iter_local_preds, masks, trimaps)
        local_pred_laplacian_loss = self.compute_local_laplacian_loss(
            per_iter_local_preds, masks, trimaps)
        fusion_pred_alpha_loss = self.compute_fusion_alpha_loss(
            per_iter_fused_preds, masks)
        fusion_pred_laplacian_loss = self.compute_fusion_laplacian_loss(
            per_iter_fused_preds, masks)
        composition_loss = self.compute_composition_loss(
            images, masks, fg_maps, bg_maps, per_iter_fused_preds)
        iou_predict_loss = self.compute_iou_predict_loss(
            per_iter_fused_preds, masks, per_iter_iou_preds)

        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss.mean(
            dim=-1, keepdim=True)
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss.mean(
            dim=-1, keepdim=True)
        local_pred_alpha_loss = local_pred_alpha_loss.mean(dim=-1,
                                                           keepdim=True)
        local_pred_laplacian_loss = local_pred_laplacian_loss.mean(
            dim=-1, keepdim=True)
        fusion_pred_alpha_loss = fusion_pred_alpha_loss.mean(dim=-1,
                                                             keepdim=True)
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss.mean(
            dim=-1, keepdim=True)
        composition_loss = composition_loss.mean(dim=-1, keepdim=True)
        iou_predict_loss = iou_predict_loss.mean(dim=-1, keepdim=True)

        # global_pred_trimap_ce_loss: torch.Size([2, 1])
        # global_pred_trimap_iou_loss: torch.Size([2, 1])
        # local_pred_alpha_loss: torch.Size([2, 1])
        # local_pred_laplacian_loss: torch.Size([2, 1])
        # fusion_pred_alpha_loss: torch.Size([2, 1])
        # fusion_pred_laplacian_loss: torch.Size([2, 1])
        # composition_loss: torch.Size([2, 1])
        # iou_predict_loss: torch.Size([2, 1])
        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss.sum()
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss.sum()
        local_pred_alpha_loss = local_pred_alpha_loss.sum()
        local_pred_laplacian_loss = local_pred_laplacian_loss.sum()
        fusion_pred_alpha_loss = fusion_pred_alpha_loss.sum()
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss.sum()
        composition_loss = composition_loss.sum()
        iou_predict_loss = iou_predict_loss.sum()

        return global_pred_trimap_ce_loss, global_pred_trimap_iou_loss, local_pred_alpha_loss, \
        local_pred_laplacian_loss, fusion_pred_alpha_loss, fusion_pred_laplacian_loss, \
        composition_loss, iou_predict_loss

    def compute_global_trimap_ce_loss(self, global_pred, trimap):
        # global_pred: torch.Size([2, 4, 3, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # global_pred: [b,4,3,h,w] -> [b,4,h,w,3]
        # trimap: [b,h,w]->[b,4,h,w]
        batch_size = global_pred.shape[0]

        global_pred = global_pred.float()
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        # trimap: torch.Size([b, 4, h, w])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, global_pred.shape[1], dim=1)

        # convert_trimap: [b,4,h,w]
        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        # loss_ground_truth: torch.Size([2, 4, 1024, 1024, 3])
        # global_pred: torch.Size([2, 4, 1024, 1024, 3])
        loss_ground_truth = F.one_hot(convert_trimap.long(),
                                      num_classes=num_classes).float()
        bce_loss = -(loss_ground_truth * torch.log(global_pred) +
                     (1. - loss_ground_truth) * torch.log(1. - global_pred))
        # h,w,num_classes维度合成一维然后mean
        bce_loss = bce_loss.flatten(2).mean(dim=-1)
        # batch_size维度上平均,但保留维度
        bce_loss = bce_loss / batch_size

        return bce_loss

    def compute_global_trimap_iou_loss(self, global_pred, trimap):
        # global_pred: torch.Size([2, 4, 3, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # global_pred: [b,4,3,h,w] -> [b,4,h,w,3]
        # trimap: [b,h,w]->[b,4,h,w]
        batch_size = global_pred.shape[0]

        global_pred = global_pred.float()
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        # trimap: torch.Size([b, 4, h, w])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, global_pred.shape[1], dim=1)

        # convert_trimap: [b,4,h,w]
        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        # loss_ground_truth: torch.Size([2, 4, 1024, 1024, 3])
        # intersection: torch.Size([2, 4, 1024, 1024, 3])
        loss_ground_truth = F.one_hot(convert_trimap.long(),
                                      num_classes=num_classes).float()
        intersection = global_pred * loss_ground_truth

        # num_classes维度累加
        # global_pred: torch.Size([2, 4, 1024, 1024])
        # intersection: torch.Size([2, 4, 1024, 1024])
        # loss_ground_truth: torch.Size([2, 4, 1024, 1024])
        global_pred = global_pred.sum(dim=-1)
        intersection = intersection.sum(dim=-1)
        loss_ground_truth = loss_ground_truth.sum(dim=-1)

        iou_loss = 1. - (intersection + 1e-4) / (
            global_pred + loss_ground_truth - intersection + 1e-4)
        # h,w维度合成一维然后mean
        iou_loss = iou_loss.flatten(2).mean(dim=-1)
        # batch_size维度上平均,但保留维度
        iou_loss = iou_loss / batch_size

        return iou_loss

    def compute_local_alpha_loss(self, local_pred, alpha, trimap):
        # local_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # local_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        # trimap: [b,h,w]->[b,1,h,w]
        local_pred = local_pred.float()
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        # trimap: torch.Size([2, 4, 1024, 1024])
        # weighted: torch.Size([2, 4, 1024, 1024])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        # local_pred: torch.Size([2, 4, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        diff = local_pred - alpha
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff**2 + 1e-12)

        alpha_loss = alpha_loss.sum(dim=[2, 3])
        weighted = weighted.sum(dim=[2, 3])

        alpha_loss = alpha_loss / (weighted + 1.)

        return alpha_loss

    def compute_local_laplacian_loss(self, local_pred, alpha, trimap):
        # local_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # trimap: torch.Size([2, 1024, 1024])
        # local_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        # trimap: [b,h,w] -> [b,1,h,w]
        batch_size = local_pred.shape[0]

        device = local_pred.device
        local_pred = local_pred.float()
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        # alpha: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, local_pred.shape[1], dim=1)

        # trimap: torch.Size([2, 4, 1024, 1024])
        # weighted: torch.Size([2, 4, 1024, 1024])
        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        # local_pred: torch.Size([2, 4, 1024, 1024])
        # alpha: torch.Size([2, 4, 1024, 1024])
        local_pred = local_pred * weighted
        alpha = alpha * weighted

        # gauss_kernel: torch.Size([4, 1, 5, 5])
        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=local_pred.shape[1]).to(device)

        # pyr_alpha, pyr_predict: torch.Size([2, 4, 1024, 1024]) torch.Size([2, 4, 1024, 1024])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 512, 512]) torch.Size([2, 4, 512, 512])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 256, 256]) torch.Size([2, 4, 256, 256])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 128, 128]) torch.Size([2, 4, 128, 128])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 64, 64]) torch.Size([2, 4, 64, 64])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 32, 32]) torch.Size([2, 4, 32, 32])
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(local_pred, gauss_kernel, 5)

        # laplacian_loss: torch.Size([2, 4, 1024, 1024])
        # laplacian_loss: torch.Size([2, 4, 512, 512])
        # laplacian_loss: torch.Size([2, 4, 256, 256])
        # laplacian_loss: torch.Size([2, 4, 128, 128])
        # laplacian_loss: torch.Size([2, 4, 64, 64])
        # laplacian_loss: torch.Size([2, 4, 32, 32])
        laplacian_loss = [
            F.l1_loss(a, b, reduction="none")
            for a, b in zip(pyr_alpha, pyr_predict)
        ]

        total_laplacian_loss = 0.
        for per_level_laplacian_loss in laplacian_loss:
            # h,w维度合成一维然后mean
            per_level_laplacian_loss = per_level_laplacian_loss.flatten(
                2).mean(dim=-1)
            # batch_size维度上平均,但保留维度
            per_level_laplacian_loss = per_level_laplacian_loss / batch_size
            total_laplacian_loss += per_level_laplacian_loss

        return total_laplacian_loss

    def compute_fusion_alpha_loss(self, fusion_pred, alpha):
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # fusion_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        # alpha: torch.Size([2, 4, 1024, 1024])
        # weighted: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)
        weighted = torch.ones_like(alpha)

        diff = fusion_pred - alpha
        alpha_loss = torch.sqrt(diff**2 + 1e-12)

        alpha_loss = alpha_loss.sum(dim=[2, 3])
        weighted = weighted.sum(dim=[2, 3])

        alpha_loss = alpha_loss / (weighted + 1.)

        return alpha_loss

    def compute_fusion_laplacian_loss(self, fusion_pred, alpha):
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # fusion_pred: [b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        batch_size = fusion_pred.shape[0]

        device = fusion_pred.device
        fusion_pred = fusion_pred.float()
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        # alpha: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)

        # gauss_kernel: torch.Size([4, 1, 5, 5])
        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=fusion_pred.shape[1]).to(device)

        # pyr_alpha, pyr_predict: torch.Size([2, 4, 1024, 1024]) torch.Size([2, 4, 1024, 1024])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 512, 512]) torch.Size([2, 4, 512, 512])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 256, 256]) torch.Size([2, 4, 256, 256])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 128, 128]) torch.Size([2, 4, 128, 128])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 64, 64]) torch.Size([2, 4, 64, 64])
        # pyr_alpha, pyr_predict: torch.Size([2, 4, 32, 32]) torch.Size([2, 4, 32, 32])
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(fusion_pred, gauss_kernel, 5)

        # laplacian_loss: torch.Size([2, 4, 1024, 1024])
        # laplacian_loss: torch.Size([2, 4, 512, 512])
        # laplacian_loss: torch.Size([2, 4, 256, 256])
        # laplacian_loss: torch.Size([2, 4, 128, 128])
        # laplacian_loss: torch.Size([2, 4, 64, 64])
        # laplacian_loss: torch.Size([2, 4, 32, 32])
        laplacian_loss = [
            F.l1_loss(a, b, reduction="none")
            for a, b in zip(pyr_alpha, pyr_predict)
        ]

        total_laplacian_loss = 0.
        for per_level_laplacian_loss in laplacian_loss:
            # h,w维度合成一维然后mean
            per_level_laplacian_loss = per_level_laplacian_loss.flatten(
                2).mean(dim=-1)
            # batch_size维度上平均,但保留维度
            per_level_laplacian_loss = per_level_laplacian_loss / batch_size
            total_laplacian_loss += per_level_laplacian_loss

        return total_laplacian_loss

    def compute_composition_loss(self, image, alpha, fg_map, bg_map,
                                 fusion_pred):
        # image: torch.Size([2, 3, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # fg_map: torch.Size([2, 3, 1024, 1024])
        # bg_map: torch.Size([2, 3, 1024, 1024])
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # image:[b,3,h,w]
        # alpha:[b,1,h,w]
        # fg_map:[b,3,h,w]
        # bg_map:[b,3,h,w]
        # fusion_pred:[b,4,1,h,w] -> [b,4,3,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.cat([fusion_pred, fusion_pred, fusion_pred], dim=2)

        # alpha: torch.Size([2, 3, 1024, 1024])
        # weighted: torch.Size([2, 3, 1024, 1024])
        alpha = torch.cat([alpha, alpha, alpha], dim=1)
        weighted = torch.ones_like(alpha)

        # fg_map: torch.Size([2, 1, 3, 1024, 1024])
        # bg_map: torch.Size([2, 1, 3, 1024, 1024])
        # image: torch.Size([2, 1, 3, 1024, 1024])
        # weighted: torch.Size([2, 4, 3, 1024, 1024])
        fg_map = torch.unsqueeze(fg_map, dim=1)
        bg_map = torch.unsqueeze(bg_map, dim=1)
        image = torch.unsqueeze(image, dim=1)
        weighted = torch.unsqueeze(weighted, dim=1)
        weighted = torch.repeat_interleave(weighted,
                                           fusion_pred.shape[1],
                                           dim=1)

        composition = fusion_pred * fg_map + (1. - fusion_pred) * bg_map
        composition_loss = torch.sqrt((composition - image)**2 + 1e-12)

        composition_loss = composition_loss.sum(dim=[2, 3, 4])
        weighted = weighted.sum(dim=[2, 3, 4])

        composition_loss = composition_loss / weighted

        return composition_loss

    def compute_iou_predict_loss(self, fusion_pred, alpha, pred_ious):
        # fusion_pred: torch.Size([2, 4, 1, 1024, 1024])
        # alpha: torch.Size([2, 1, 1024, 1024])
        # pred_ious: torch.Size([2, 4])
        # fusion_pred:[b,4,1,h,w] -> [b,4,h,w]
        # alpha: [b,1,h,w]
        batch_size = fusion_pred.shape[0]

        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=2)

        # alpha: torch.Size([2, 4, 1024, 1024])
        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)

        fusion_pred = (fusion_pred > self.mask_threshold)
        alpha = (alpha > self.mask_threshold)

        # h,w维度合成一维
        # fusion_pred: torch.Size([2, 4, h*w])
        # alpha: torch.Size([2, 4, h*w])
        fusion_pred = fusion_pred.flatten(2)
        alpha = alpha.flatten(2)

        # intersection: torch.Size([2, 4])
        # union: torch.Size([2, 4])
        # gt_ious: torch.Size([2, 4])
        intersection = torch.sum(fusion_pred & alpha, dim=-1).float()
        union = torch.sum(fusion_pred | alpha, dim=-1).float()
        gt_ious = intersection / torch.clamp(union, min=1e-6)
        gt_ious = torch.clamp(gt_ious, min=0.0, max=1.0)

        # iou_predict_loss: torch.Size([2, 4])
        iou_predict_loss = F.l1_loss(pred_ious, gt_ious, reduction="none")
        # batch_size维度上平均,但保留维度
        iou_predict_loss = iou_predict_loss / batch_size

        return iou_predict_loss

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

    from tools.path import interactive_segmentation_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.interactive_segmentation.datasets.sam_matting_dataset import SAMMattingDataset
    from SimpleAICV.interactive_segmentation.common_matting import SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingBatchCollater, load_state_dict

    samdataset = SAMMattingDataset(
        interactive_segmentation_dataset_path,
        set_name=[
            'DIS5K',
            'sa_000000',
        ],
        set_type='train',
        per_set_image_choose_max_num={
            'DIS5K': 1000000,
            'sa_000000': 1000000,
        },
        max_side=2048,
        kernel_size_range=[15, 15],
        per_image_mask_chosse_max_num=16,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            SAMMattingResize(resize=1024),
            SAMMattingRandomHorizontalFlip(prob=0.5),
            SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAMMattingBatchCollater(resize=1024)
    train_loader = DataLoader(samdataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.interactive_segmentation.models.segment_anything_matting.sam_matting import sam_b_matting

    net = sam_b_matting(use_gradient_checkpoint=True)
    load_state_dict(
        '/root/autodl-tmp/pretrained_models/sam_pytorch_official_weights/sam_vit_b_01ec64.pth',
        net)

    loss = SAMMattingLoss(global_pred_trimap_ce_loss_weight=1,
                          global_pred_trimap_iou_loss_weight=1,
                          local_pred_alpha_loss_weight=1,
                          local_pred_laplacian_loss_weight=1,
                          fusion_pred_alpha_loss_weight=1,
                          fusion_pred_laplacian_loss_weight=1,
                          composition_loss_weight=1,
                          iou_predict_loss_weight=1,
                          supervise_all_iou=True,
                          mask_threshold=0.5)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
            'bg_map']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

        trimaps = trimaps.cuda()
        fg_maps = fg_maps.cuda()
        bg_maps = bg_maps.cuda()

        print('2222', trimaps.shape, fg_maps.shape, bg_maps.shape)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('3333', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        batch_image_embeddings = net.forward_image_encoder(input_images)

        print('4444', batch_image_embeddings.shape)

        global_preds, local_preds, fused_preds, iou_preds = net.forward_prompt_encoder_mask_decoder(
            batch_image_embeddings, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('5555', global_preds.shape, local_preds.shape, fused_preds.shape,
              iou_preds.shape)

        all_iter_global_preds, all_iter_local_preds = [global_preds
                                                       ], [local_preds]

        all_iter_fused_preds, all_iter_iou_preds = [fused_preds], [iou_preds]

        print('6666', len(all_iter_global_preds), len(all_iter_local_preds),
              len(all_iter_fused_preds), len(all_iter_iou_preds))

        for per_iter_global_preds, per_iter_local_preds, per_iter_fused_preds, per_iter_iou_preds in zip(
                all_iter_global_preds, all_iter_local_preds,
                all_iter_fused_preds, all_iter_iou_preds):
            print('7777', per_iter_global_preds.shape,
                  per_iter_local_preds.shape, per_iter_fused_preds.shape,
                  per_iter_iou_preds.shape)

        loss_dict = loss(input_images, [
            all_iter_global_preds,
            all_iter_local_preds,
            all_iter_fused_preds,
            all_iter_iou_preds,
        ], [input_masks, trimaps, fg_maps, bg_maps])
        print('8888', loss_dict)

        break

    loss = SAMMattingMultiLevelLoss(global_pred_trimap_ce_loss_weight=1,
                                    global_pred_trimap_iou_loss_weight=1,
                                    local_pred_alpha_loss_weight=1,
                                    local_pred_laplacian_loss_weight=1,
                                    fusion_pred_alpha_loss_weight=1,
                                    fusion_pred_laplacian_loss_weight=1,
                                    composition_loss_weight=1,
                                    iou_predict_loss_weight=1,
                                    mask_threshold=0.5)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
            'bg_map']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

        trimaps = trimaps.cuda()
        fg_maps = fg_maps.cuda()
        bg_maps = bg_maps.cuda()

        print('2222', trimaps.shape, fg_maps.shape, bg_maps.shape)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('3333', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        batch_image_embeddings = net.forward_image_encoder(input_images)

        print('4444', batch_image_embeddings.shape)

        global_preds, local_preds, fused_preds, iou_preds = net.forward_prompt_encoder_mask_decoder(
            batch_image_embeddings, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('5555', global_preds.shape, local_preds.shape, fused_preds.shape,
              iou_preds.shape)

        all_iter_global_preds, all_iter_local_preds = [global_preds
                                                       ], [local_preds]

        all_iter_fused_preds, all_iter_iou_preds = [fused_preds], [iou_preds]

        print('6666', len(all_iter_global_preds), len(all_iter_local_preds),
              len(all_iter_fused_preds), len(all_iter_iou_preds))

        for per_iter_global_preds, per_iter_local_preds, per_iter_fused_preds, per_iter_iou_preds in zip(
                all_iter_global_preds, all_iter_local_preds,
                all_iter_fused_preds, all_iter_iou_preds):
            print('7777', per_iter_global_preds.shape,
                  per_iter_local_preds.shape, per_iter_fused_preds.shape,
                  per_iter_iou_preds.shape)

        loss_dict = loss(input_images, [
            all_iter_global_preds,
            all_iter_local_preds,
            all_iter_fused_preds,
            all_iter_iou_preds,
        ], [input_masks, trimaps, fg_maps, bg_maps])
        print('8888', loss_dict)

        break
