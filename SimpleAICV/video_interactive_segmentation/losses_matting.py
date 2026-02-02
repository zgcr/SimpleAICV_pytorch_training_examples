import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = [
    'SAM2MattingLoss',
    'SAM2MattingMultiLevelLoss',
]


class SAM2MattingLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 global_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 iou_predict_loss_weight=1,
                 class_loss_weight=1,
                 supervise_all_iou=True,
                 mask_threshold=0.5):

        super(SAM2MattingLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.global_pred_trimap_iou_loss_weight = global_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.class_loss_weight = class_loss_weight
        self.supervise_all_iou = supervise_all_iou
        self.mask_threshold = mask_threshold

    def forward(self, images, inputs, targets):
        # T T T T T
        all_frame_global_preds, all_frame_local_preds, all_frame_fused_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = inputs

        # torch.Size([8, 2, 1024, 1024])
        # torch.Size([8, 2, 1024, 1024])
        # torch.Size([8, 2, 1024, 1024, 3])
        # torch.Size([8, 2, 1024, 1024, 3])
        masks, trimaps, fg_maps, bg_maps = targets

        del inputs, targets

        assert len(all_frame_global_preds) == len(
            all_frame_local_preds) == len(all_frame_fused_preds) == len(
                all_frame_iou_preds) == len(
                    all_frame_pred_object_score_logits) == len(masks) == len(
                        trimaps) == len(fg_maps) == len(bg_maps)

        global_pred_trimap_ce_loss = 0.
        global_pred_trimap_iou_loss = 0.
        local_pred_alpha_loss = 0.
        local_pred_laplacian_loss = 0.
        fusion_pred_alpha_loss = 0.
        fusion_pred_laplacian_loss = 0.
        composition_loss = 0.
        iou_predict_loss = 0.
        cls_loss = 0.
        frame_num = len(all_frame_fused_preds)

        for per_frame_global_preds, per_frame_local_preds, per_frame_fused_preds, per_frame_iou_preds, per_frame_pred_object_score_logits, per_frame_images, per_frame_masks, per_frame_trimaps, per_frame_fg_maps, per_frame_bg_maps in zip(
                all_frame_global_preds,
                all_frame_local_preds,
                all_frame_fused_preds,
                all_frame_iou_preds,
                all_frame_pred_object_score_logits,
                images,
                masks,
                trimaps,
                fg_maps,
                bg_maps,
        ):
            assert len(per_frame_global_preds) == len(
                per_frame_local_preds) == len(per_frame_fused_preds) == len(
                    per_frame_iou_preds) == len(
                        per_frame_pred_object_score_logits)

            per_frame_global_pred_trimap_ce_loss = 0.
            per_frame_global_pred_trimap_iou_loss = 0.
            per_frame_local_pred_alpha_loss = 0.
            per_frame_local_pred_laplacian_loss = 0.
            per_frame_fusion_pred_alpha_loss = 0.
            per_frame_fusion_pred_laplacian_loss = 0.
            per_frame_composition_loss = 0.
            per_frame_iou_predict_loss = 0.
            per_frame_cls_loss = 0.
            iter_num = len(per_frame_global_preds)

            for per_frame_iter_global_preds, per_frame_iter_local_preds, per_frame_iter_fused_preds, per_frame_iter_iou_preds, per_frame_iter_pred_object_score_logits in zip(
                    per_frame_global_preds,
                    per_frame_local_preds,
                    per_frame_fused_preds,
                    per_frame_iou_preds,
                    per_frame_pred_object_score_logits,
            ):

                per_frame_iter_global_pred_trimap_ce_loss, per_frame_iter_global_pred_trimap_iou_loss, per_frame_iter_local_pred_alpha_loss, per_frame_iter_local_pred_laplacian_loss, per_frame_iter_fusion_pred_alpha_loss, per_frame_iter_fusion_pred_laplacian_loss, per_frame_iter_composition_loss, per_frame_iter_iou_predict_loss, per_frame_iter_cls_loss = self.compute_per_frame_iter_loss(
                    per_frame_iter_global_preds,
                    per_frame_iter_local_preds,
                    per_frame_iter_fused_preds,
                    per_frame_iter_iou_preds,
                    per_frame_iter_pred_object_score_logits,
                    per_frame_images,
                    per_frame_masks,
                    per_frame_trimaps,
                    per_frame_fg_maps,
                    per_frame_bg_maps,
                )

                per_frame_global_pred_trimap_ce_loss += per_frame_iter_global_pred_trimap_ce_loss
                per_frame_global_pred_trimap_iou_loss += per_frame_iter_global_pred_trimap_iou_loss
                per_frame_local_pred_alpha_loss += per_frame_iter_local_pred_alpha_loss
                per_frame_local_pred_laplacian_loss += per_frame_iter_local_pred_laplacian_loss
                per_frame_fusion_pred_alpha_loss += per_frame_iter_fusion_pred_alpha_loss
                per_frame_fusion_pred_laplacian_loss += per_frame_iter_fusion_pred_laplacian_loss
                per_frame_composition_loss += per_frame_iter_composition_loss
                per_frame_iou_predict_loss += per_frame_iter_iou_predict_loss
                per_frame_cls_loss += per_frame_iter_cls_loss

            per_frame_global_pred_trimap_ce_loss = per_frame_global_pred_trimap_ce_loss / float(
                iter_num)
            per_frame_global_pred_trimap_iou_loss = per_frame_global_pred_trimap_iou_loss / float(
                iter_num)
            per_frame_local_pred_alpha_loss = per_frame_local_pred_alpha_loss / float(
                iter_num)
            per_frame_local_pred_laplacian_loss = per_frame_local_pred_laplacian_loss / float(
                iter_num)
            per_frame_fusion_pred_alpha_loss = per_frame_fusion_pred_alpha_loss / float(
                iter_num)
            per_frame_fusion_pred_laplacian_loss = per_frame_fusion_pred_laplacian_loss / float(
                iter_num)
            per_frame_composition_loss = per_frame_composition_loss / float(
                iter_num)
            per_frame_iou_predict_loss = per_frame_iou_predict_loss / float(
                iter_num)
            per_frame_cls_loss = per_frame_cls_loss / float(iter_num)

            global_pred_trimap_ce_loss += per_frame_global_pred_trimap_ce_loss
            global_pred_trimap_iou_loss += per_frame_global_pred_trimap_iou_loss
            local_pred_alpha_loss += per_frame_local_pred_alpha_loss
            local_pred_laplacian_loss += per_frame_local_pred_laplacian_loss
            fusion_pred_alpha_loss += per_frame_fusion_pred_alpha_loss
            fusion_pred_laplacian_loss += per_frame_fusion_pred_laplacian_loss
            composition_loss += per_frame_composition_loss
            iou_predict_loss += per_frame_iou_predict_loss
            cls_loss += per_frame_cls_loss

        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss / float(
            frame_num)
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss / float(
            frame_num)
        local_pred_alpha_loss = local_pred_alpha_loss / float(frame_num)
        local_pred_laplacian_loss = local_pred_laplacian_loss / float(
            frame_num)
        fusion_pred_alpha_loss = fusion_pred_alpha_loss / float(frame_num)
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss / float(
            frame_num)
        composition_loss = composition_loss / float(frame_num)
        iou_predict_loss = iou_predict_loss / float(frame_num)
        cls_loss = cls_loss / float(frame_num)

        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss * self.global_pred_trimap_ce_loss_weight
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss * self.global_pred_trimap_iou_loss_weight
        local_pred_alpha_loss = local_pred_alpha_loss * self.local_pred_alpha_loss_weight
        local_pred_laplacian_loss = local_pred_laplacian_loss * self.local_pred_laplacian_loss_weight
        fusion_pred_alpha_loss = fusion_pred_alpha_loss * self.fusion_pred_alpha_loss_weight
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss * self.fusion_pred_laplacian_loss_weight
        composition_loss = composition_loss * self.composition_loss_weight
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight
        cls_loss = cls_loss * self.class_loss_weight

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'global_pred_trimap_iou_loss': global_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'iou_predict_loss': iou_predict_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict

    def compute_per_frame_iter_loss(
        self,
        per_frame_iter_global_preds,
        per_frame_iter_local_preds,
        per_frame_iter_fused_preds,
        per_frame_iter_iou_preds,
        per_frame_iter_pred_object_score_logits,
        per_frame_images,
        per_frame_masks,
        per_frame_trimaps,
        per_frame_fg_maps,
        per_frame_bg_maps,
    ):
        # per_frame_iter_global_preds: torch.Size([N_o, mask_out_idxs_num, 3, 1024, 1024])
        # per_frame_iter_local_preds: torch.Size([N_o, mask_out_idxs_num, 1, 1024, 1024])
        # per_frame_iter_fused_preds: torch.Size([N_o, mask_out_idxs_num, 1, 1024, 1024])
        # per_frame_iter_iou_preds: torch.Size([N_o, mask_out_idxs_num])
        # per_frame_iter_pred_object_score_logits: torch.Size([N_o, 1])
        per_frame_iter_global_preds = per_frame_iter_global_preds.float()
        per_frame_iter_local_preds = per_frame_iter_local_preds.float()
        per_frame_iter_fused_preds = per_frame_iter_fused_preds.float()
        per_frame_iter_iou_preds = per_frame_iter_iou_preds.float()
        per_frame_iter_pred_object_score_logits = per_frame_iter_pred_object_score_logits.float(
        )

        per_frame_images = per_frame_images.float()
        per_frame_masks = per_frame_masks.float()
        per_frame_trimaps = per_frame_trimaps.float()
        per_frame_fg_maps = per_frame_fg_maps.float()
        per_frame_bg_maps = per_frame_bg_maps.float()

        # per_frame_images: torch.Size([N_o, 3, 1024, 1024])
        # per_frame_masks: torch.Size([N_o, 1, 1024, 1024])
        # per_frame_trimaps: torch.Size([N_o, 1024, 1024])
        # per_frame_fg_maps: torch.Size([N_o, 3, 1024, 1024])
        # per_frame_bg_maps: torch.Size([N_o, 3, 1024, 1024])
        per_frame_masks = torch.unsqueeze(per_frame_masks, dim=1)
        per_frame_images = per_frame_images.permute(0, 3, 1, 2)
        per_frame_fg_maps = per_frame_fg_maps.permute(0, 3, 1, 2)
        per_frame_bg_maps = per_frame_bg_maps.permute(0, 3, 1, 2)

        # global_pred_trimap_ce_loss: torch.Size([N_o, mask_out_idxs_num])
        # global_pred_trimap_iou_loss: torch.Size([N_o, mask_out_idxs_num])
        # local_pred_alpha_loss: torch.Size([N_o, mask_out_idxs_num])
        # local_pred_laplacian_loss: torch.Size([N_o, mask_out_idxs_num])
        # fusion_pred_alpha_loss: torch.Size([N_o, mask_out_idxs_num])
        # fusion_pred_laplacian_loss: torch.Size([N_o, mask_out_idxs_num])
        # composition_loss: torch.Size([N_o, mask_out_idxs_num])
        # iou_predict_loss: torch.Size([N_o, mask_out_idxs_num])
        # target_object: torch.Size([N_o, 1])
        # cls_loss: torch.Size([])
        global_pred_trimap_ce_loss = self.compute_global_trimap_ce_loss(
            per_frame_iter_global_preds, per_frame_trimaps)
        global_pred_trimap_iou_loss = self.compute_global_trimap_iou_loss(
            per_frame_iter_global_preds, per_frame_trimaps)
        local_pred_alpha_loss = self.compute_local_alpha_loss(
            per_frame_iter_local_preds, per_frame_masks, per_frame_trimaps)
        local_pred_laplacian_loss = self.compute_local_laplacian_loss(
            per_frame_iter_local_preds, per_frame_masks, per_frame_trimaps)
        fusion_pred_alpha_loss = self.compute_fusion_alpha_loss(
            per_frame_iter_fused_preds, per_frame_masks)
        fusion_pred_laplacian_loss = self.compute_fusion_laplacian_loss(
            per_frame_iter_fused_preds, per_frame_masks)
        composition_loss = self.compute_composition_loss(
            per_frame_images, per_frame_masks, per_frame_fg_maps,
            per_frame_bg_maps, per_frame_iter_fused_preds)
        iou_predict_loss = self.compute_iou_predict_loss(
            per_frame_iter_fused_preds, per_frame_masks,
            per_frame_iter_iou_preds)

        # target_obj: 是否出现物体
        # per_frame_target_mask[:, 0],0这个维度是为了对齐pred加的,实际上target_mask每个object只有一个mask
        target_object = torch.unsqueeze(torch.any((per_frame_masks[:, 0]
                                                   > 0).flatten(1),
                                                  dim=-1),
                                        dim=-1).float()
        cls_loss = self.cls_loss(per_frame_iter_pred_object_score_logits,
                                 target_object)

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

        # 只有物体存在时(target_obj=1)时才回传
        # global_pred_trimap_ce_loss: torch.Size([N_o, 1])
        # global_pred_trimap_iou_loss: torch.Size([N_o, 1])
        # local_pred_alpha_loss: torch.Size([N_o, 1])
        # local_pred_laplacian_loss: torch.Size([N_o, 1])
        # fusion_pred_alpha_loss: torch.Size([N_o, 1])
        # fusion_pred_laplacian_loss: torch.Size([N_o, 1])
        # composition_loss: torch.Size([N_o, 1])
        # iou_predict_loss: torch.Size([N_o, 1])
        # target_object: torch.Size([N_o, 1])
        # cls_loss: torch.Size([])
        global_pred_trimap_ce_loss = (global_pred_trimap_ce_loss *
                                      target_object).sum()
        global_pred_trimap_iou_loss = (global_pred_trimap_iou_loss *
                                       target_object).sum()
        local_pred_alpha_loss = (local_pred_alpha_loss * target_object).sum()
        local_pred_laplacian_loss = (local_pred_laplacian_loss *
                                     target_object).sum()
        fusion_pred_alpha_loss = (fusion_pred_alpha_loss * target_object).sum()
        fusion_pred_laplacian_loss = (fusion_pred_laplacian_loss *
                                      target_object).sum()
        composition_loss = (composition_loss * target_object).sum()
        iou_predict_loss = (iou_predict_loss * target_object).sum()
        cls_loss = cls_loss

        return global_pred_trimap_ce_loss, global_pred_trimap_iou_loss, local_pred_alpha_loss, local_pred_laplacian_loss, fusion_pred_alpha_loss, fusion_pred_laplacian_loss, composition_loss, iou_predict_loss, cls_loss

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

    def cls_loss(self, inputs, targets):
        object_nums = inputs.shape[0]
        inputs = inputs.float()
        cls_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction="none")
        cls_loss = cls_loss.mean(dim=1).sum() / object_nums

        return cls_loss

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


class SAM2MattingMultiLevelLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 global_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 iou_predict_loss_weight=1,
                 class_loss_weight=1,
                 mask_threshold=0.5):

        super(SAM2MattingMultiLevelLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.global_pred_trimap_iou_loss_weight = global_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.class_loss_weight = class_loss_weight
        self.mask_threshold = mask_threshold

    def forward(self, images, inputs, targets):
        # T T T T T
        all_frame_global_preds, all_frame_local_preds, all_frame_fused_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = inputs

        # torch.Size([8, 2, 1024, 1024])
        # torch.Size([8, 2, 1024, 1024])
        # torch.Size([8, 2, 1024, 1024, 3])
        # torch.Size([8, 2, 1024, 1024, 3])
        masks, trimaps, fg_maps, bg_maps = targets

        del inputs, targets

        assert len(all_frame_global_preds) == len(
            all_frame_local_preds) == len(all_frame_fused_preds) == len(
                all_frame_iou_preds) == len(
                    all_frame_pred_object_score_logits) == len(masks) == len(
                        trimaps) == len(fg_maps) == len(bg_maps)

        global_pred_trimap_ce_loss = 0.
        global_pred_trimap_iou_loss = 0.
        local_pred_alpha_loss = 0.
        local_pred_laplacian_loss = 0.
        fusion_pred_alpha_loss = 0.
        fusion_pred_laplacian_loss = 0.
        composition_loss = 0.
        iou_predict_loss = 0.
        cls_loss = 0.
        frame_num = len(all_frame_fused_preds)

        for per_frame_global_preds, per_frame_local_preds, per_frame_fused_preds, per_frame_iou_preds, per_frame_pred_object_score_logits, per_frame_images, per_frame_masks, per_frame_trimaps, per_frame_fg_maps, per_frame_bg_maps in zip(
                all_frame_global_preds,
                all_frame_local_preds,
                all_frame_fused_preds,
                all_frame_iou_preds,
                all_frame_pred_object_score_logits,
                images,
                masks,
                trimaps,
                fg_maps,
                bg_maps,
        ):
            assert len(per_frame_global_preds) == len(
                per_frame_local_preds) == len(per_frame_fused_preds) == len(
                    per_frame_iou_preds) == len(
                        per_frame_pred_object_score_logits)

            per_frame_global_pred_trimap_ce_loss = 0.
            per_frame_global_pred_trimap_iou_loss = 0.
            per_frame_local_pred_alpha_loss = 0.
            per_frame_local_pred_laplacian_loss = 0.
            per_frame_fusion_pred_alpha_loss = 0.
            per_frame_fusion_pred_laplacian_loss = 0.
            per_frame_composition_loss = 0.
            per_frame_iou_predict_loss = 0.
            per_frame_cls_loss = 0.
            iter_num = len(per_frame_global_preds)

            for per_frame_iter_global_preds, per_frame_iter_local_preds, per_frame_iter_fused_preds, per_frame_iter_iou_preds, per_frame_iter_pred_object_score_logits in zip(
                    per_frame_global_preds,
                    per_frame_local_preds,
                    per_frame_fused_preds,
                    per_frame_iou_preds,
                    per_frame_pred_object_score_logits,
            ):

                per_frame_iter_global_pred_trimap_ce_loss, per_frame_iter_global_pred_trimap_iou_loss, per_frame_iter_local_pred_alpha_loss, per_frame_iter_local_pred_laplacian_loss, per_frame_iter_fusion_pred_alpha_loss, per_frame_iter_fusion_pred_laplacian_loss, per_frame_iter_composition_loss, per_frame_iter_iou_predict_loss, per_frame_iter_cls_loss = self.compute_per_frame_iter_loss(
                    per_frame_iter_global_preds,
                    per_frame_iter_local_preds,
                    per_frame_iter_fused_preds,
                    per_frame_iter_iou_preds,
                    per_frame_iter_pred_object_score_logits,
                    per_frame_images,
                    per_frame_masks,
                    per_frame_trimaps,
                    per_frame_fg_maps,
                    per_frame_bg_maps,
                )

                per_frame_global_pred_trimap_ce_loss += per_frame_iter_global_pred_trimap_ce_loss
                per_frame_global_pred_trimap_iou_loss += per_frame_iter_global_pred_trimap_iou_loss
                per_frame_local_pred_alpha_loss += per_frame_iter_local_pred_alpha_loss
                per_frame_local_pred_laplacian_loss += per_frame_iter_local_pred_laplacian_loss
                per_frame_fusion_pred_alpha_loss += per_frame_iter_fusion_pred_alpha_loss
                per_frame_fusion_pred_laplacian_loss += per_frame_iter_fusion_pred_laplacian_loss
                per_frame_composition_loss += per_frame_iter_composition_loss
                per_frame_iou_predict_loss += per_frame_iter_iou_predict_loss
                per_frame_cls_loss += per_frame_iter_cls_loss

            per_frame_global_pred_trimap_ce_loss = per_frame_global_pred_trimap_ce_loss / float(
                iter_num)
            per_frame_global_pred_trimap_iou_loss = per_frame_global_pred_trimap_iou_loss / float(
                iter_num)
            per_frame_local_pred_alpha_loss = per_frame_local_pred_alpha_loss / float(
                iter_num)
            per_frame_local_pred_laplacian_loss = per_frame_local_pred_laplacian_loss / float(
                iter_num)
            per_frame_fusion_pred_alpha_loss = per_frame_fusion_pred_alpha_loss / float(
                iter_num)
            per_frame_fusion_pred_laplacian_loss = per_frame_fusion_pred_laplacian_loss / float(
                iter_num)
            per_frame_composition_loss = per_frame_composition_loss / float(
                iter_num)
            per_frame_iou_predict_loss = per_frame_iou_predict_loss / float(
                iter_num)
            per_frame_cls_loss = per_frame_cls_loss / float(iter_num)

            global_pred_trimap_ce_loss += per_frame_global_pred_trimap_ce_loss
            global_pred_trimap_iou_loss += per_frame_global_pred_trimap_iou_loss
            local_pred_alpha_loss += per_frame_local_pred_alpha_loss
            local_pred_laplacian_loss += per_frame_local_pred_laplacian_loss
            fusion_pred_alpha_loss += per_frame_fusion_pred_alpha_loss
            fusion_pred_laplacian_loss += per_frame_fusion_pred_laplacian_loss
            composition_loss += per_frame_composition_loss
            iou_predict_loss += per_frame_iou_predict_loss
            cls_loss += per_frame_cls_loss

        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss / float(
            frame_num)
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss / float(
            frame_num)
        local_pred_alpha_loss = local_pred_alpha_loss / float(frame_num)
        local_pred_laplacian_loss = local_pred_laplacian_loss / float(
            frame_num)
        fusion_pred_alpha_loss = fusion_pred_alpha_loss / float(frame_num)
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss / float(
            frame_num)
        composition_loss = composition_loss / float(frame_num)
        iou_predict_loss = iou_predict_loss / float(frame_num)
        cls_loss = cls_loss / float(frame_num)

        global_pred_trimap_ce_loss = global_pred_trimap_ce_loss * self.global_pred_trimap_ce_loss_weight
        global_pred_trimap_iou_loss = global_pred_trimap_iou_loss * self.global_pred_trimap_iou_loss_weight
        local_pred_alpha_loss = local_pred_alpha_loss * self.local_pred_alpha_loss_weight
        local_pred_laplacian_loss = local_pred_laplacian_loss * self.local_pred_laplacian_loss_weight
        fusion_pred_alpha_loss = fusion_pred_alpha_loss * self.fusion_pred_alpha_loss_weight
        fusion_pred_laplacian_loss = fusion_pred_laplacian_loss * self.fusion_pred_laplacian_loss_weight
        composition_loss = composition_loss * self.composition_loss_weight
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight
        cls_loss = cls_loss * self.class_loss_weight

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'global_pred_trimap_iou_loss': global_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'iou_predict_loss': iou_predict_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict

    def compute_per_frame_iter_loss(
        self,
        per_frame_iter_global_preds,
        per_frame_iter_local_preds,
        per_frame_iter_fused_preds,
        per_frame_iter_iou_preds,
        per_frame_iter_pred_object_score_logits,
        per_frame_images,
        per_frame_masks,
        per_frame_trimaps,
        per_frame_fg_maps,
        per_frame_bg_maps,
    ):
        # per_frame_iter_global_preds: torch.Size([N_o, mask_out_idxs_num, 3, 1024, 1024])
        # per_frame_iter_local_preds: torch.Size([N_o, mask_out_idxs_num, 1, 1024, 1024])
        # per_frame_iter_fused_preds: torch.Size([N_o, mask_out_idxs_num, 1, 1024, 1024])
        # per_frame_iter_iou_preds: torch.Size([N_o, mask_out_idxs_num])
        # per_frame_iter_pred_object_score_logits: torch.Size([N_o, 1])
        per_frame_iter_global_preds = per_frame_iter_global_preds.float()
        per_frame_iter_local_preds = per_frame_iter_local_preds.float()
        per_frame_iter_fused_preds = per_frame_iter_fused_preds.float()
        per_frame_iter_iou_preds = per_frame_iter_iou_preds.float()
        per_frame_iter_pred_object_score_logits = per_frame_iter_pred_object_score_logits.float(
        )

        per_frame_images = per_frame_images.float()
        per_frame_masks = per_frame_masks.float()
        per_frame_trimaps = per_frame_trimaps.float()
        per_frame_fg_maps = per_frame_fg_maps.float()
        per_frame_bg_maps = per_frame_bg_maps.float()

        # per_frame_images: torch.Size([N_o, 3, 1024, 1024])
        # per_frame_masks: torch.Size([N_o, 1, 1024, 1024])
        # per_frame_trimaps: torch.Size([N_o, 1024, 1024])
        # per_frame_fg_maps: torch.Size([N_o, 3, 1024, 1024])
        # per_frame_bg_maps: torch.Size([N_o, 3, 1024, 1024])
        per_frame_masks = torch.unsqueeze(per_frame_masks, dim=1)
        per_frame_images = per_frame_images.permute(0, 3, 1, 2)
        per_frame_fg_maps = per_frame_fg_maps.permute(0, 3, 1, 2)
        per_frame_bg_maps = per_frame_bg_maps.permute(0, 3, 1, 2)

        # global_pred_trimap_ce_loss: torch.Size([N_o, mask_out_idxs_num])
        # global_pred_trimap_iou_loss: torch.Size([N_o, mask_out_idxs_num])
        # local_pred_alpha_loss: torch.Size([N_o, mask_out_idxs_num])
        # local_pred_laplacian_loss: torch.Size([N_o, mask_out_idxs_num])
        # fusion_pred_alpha_loss: torch.Size([N_o, mask_out_idxs_num])
        # fusion_pred_laplacian_loss: torch.Size([N_o, mask_out_idxs_num])
        # composition_loss: torch.Size([N_o, mask_out_idxs_num])
        # iou_predict_loss: torch.Size([N_o, mask_out_idxs_num])
        # target_object: torch.Size([N_o, 1])
        # cls_loss: torch.Size([])
        global_pred_trimap_ce_loss = self.compute_global_trimap_ce_loss(
            per_frame_iter_global_preds, per_frame_trimaps)
        global_pred_trimap_iou_loss = self.compute_global_trimap_iou_loss(
            per_frame_iter_global_preds, per_frame_trimaps)
        local_pred_alpha_loss = self.compute_local_alpha_loss(
            per_frame_iter_local_preds, per_frame_masks, per_frame_trimaps)
        local_pred_laplacian_loss = self.compute_local_laplacian_loss(
            per_frame_iter_local_preds, per_frame_masks, per_frame_trimaps)
        fusion_pred_alpha_loss = self.compute_fusion_alpha_loss(
            per_frame_iter_fused_preds, per_frame_masks)
        fusion_pred_laplacian_loss = self.compute_fusion_laplacian_loss(
            per_frame_iter_fused_preds, per_frame_masks)
        composition_loss = self.compute_composition_loss(
            per_frame_images, per_frame_masks, per_frame_fg_maps,
            per_frame_bg_maps, per_frame_iter_fused_preds)
        iou_predict_loss = self.compute_iou_predict_loss(
            per_frame_iter_fused_preds, per_frame_masks,
            per_frame_iter_iou_preds)

        # target_obj: 是否出现物体
        # per_frame_target_mask[:, 0],0这个维度是为了对齐pred加的,实际上target_mask每个object只有一个mask
        target_object = torch.unsqueeze(torch.any((per_frame_masks[:, 0]
                                                   > 0).flatten(1),
                                                  dim=-1),
                                        dim=-1).float()
        cls_loss = self.cls_loss(per_frame_iter_pred_object_score_logits,
                                 target_object)

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

        # 只有物体存在时(target_obj=1)时才回传
        # global_pred_trimap_ce_loss: torch.Size([N_o, 1])
        # global_pred_trimap_iou_loss: torch.Size([N_o, 1])
        # local_pred_alpha_loss: torch.Size([N_o, 1])
        # local_pred_laplacian_loss: torch.Size([N_o, 1])
        # fusion_pred_alpha_loss: torch.Size([N_o, 1])
        # fusion_pred_laplacian_loss: torch.Size([N_o, 1])
        # composition_loss: torch.Size([N_o, 1])
        # iou_predict_loss: torch.Size([N_o, 1])
        # target_object: torch.Size([N_o, 1])
        # cls_loss: torch.Size([])
        global_pred_trimap_ce_loss = (global_pred_trimap_ce_loss *
                                      target_object).sum()
        global_pred_trimap_iou_loss = (global_pred_trimap_iou_loss *
                                       target_object).sum()
        local_pred_alpha_loss = (local_pred_alpha_loss * target_object).sum()
        local_pred_laplacian_loss = (local_pred_laplacian_loss *
                                     target_object).sum()
        fusion_pred_alpha_loss = (fusion_pred_alpha_loss * target_object).sum()
        fusion_pred_laplacian_loss = (fusion_pred_laplacian_loss *
                                      target_object).sum()
        composition_loss = (composition_loss * target_object).sum()
        iou_predict_loss = (iou_predict_loss * target_object).sum()
        cls_loss = cls_loss

        return global_pred_trimap_ce_loss, global_pred_trimap_iou_loss, local_pred_alpha_loss, local_pred_laplacian_loss, fusion_pred_alpha_loss, fusion_pred_laplacian_loss, composition_loss, iou_predict_loss, cls_loss

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

    def cls_loss(self, inputs, targets):
        object_nums = inputs.shape[0]
        inputs = inputs.float()
        cls_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction="none")
        cls_loss = cls_loss.mean(dim=1).sum() / object_nums

        return cls_loss

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

    from tools.path import interactive_segmentation_dataset_path, video_interactive_segmentation_dataset_path, background_video_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.video_interactive_segmentation.datasets.sam2_video_matting_dataset import SAM2VideoMattingDataset
    from SimpleAICV.video_interactive_segmentation.common_matting import Sam2MattingResize, Sam2MattingRandomHorizontalFlip, Sam2MattingRandomMosaicAug, Sam2MattingRandomRsverseFrameOrder, Sam2MattingNormalize, SAM2MattingVideoBatchCollater, load_state_dict

    sam2_video_dataset = SAM2VideoMattingDataset(
        image_root_dir=interactive_segmentation_dataset_path,
        image_set_name=[
            'P3M10K',
        ],
        image_set_type='train',
        image_per_set_image_choose_max_num={
            'P3M10K': 1000000,
        },
        per_image_mask_chosse_max_num=16,
        video_root_dir=video_interactive_segmentation_dataset_path,
        video_set_name=[
            # 'DAVIS2017',
        ],
        video_set_type='train',
        video_matting_root_dir=video_interactive_segmentation_dataset_path,
        video_matting_set_name_list=[
            'VideoMatte240K',
        ],
        video_matting_use_background_video_prob={
            'VideoMatte240K': 1.0,
        },
        video_matting_set_type='train',
        video_matting_background_dir=background_video_dataset_path,
        video_matting_background_set_type='train',
        per_video_choose_frame_nums=3,
        per_video_choose_object_nums=1,
        max_side=2048,
        kernel_size_range=[15, 15],
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            Sam2MattingResize(resize=1024),
            Sam2MattingNormalize(mean=[123.675, 116.28, 103.53],
                                 std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAM2MattingVideoBatchCollater(resize=1024, use_image_prob=0.0)
    train_loader = DataLoader(sam2_video_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from SimpleAICV.video_interactive_segmentation.models.segment_anything2_matting.sam2videomatting_train import hiera_b_plus_sam2video_matting
    net = hiera_b_plus_sam2video_matting(use_gradient_checkpoint=True)
    load_state_dict(
        '/root/autodl-tmp/pretrained_models/sam2.1_convert_from_pytorch_official_weights/sam2.1_hiera_base_plus_convert_from_pytorch_official_weight.pth',
        net)

    loss = SAM2MattingLoss(global_pred_trimap_ce_loss_weight=1,
                           global_pred_trimap_iou_loss_weight=1,
                           local_pred_alpha_loss_weight=1,
                           local_pred_laplacian_loss_weight=1,
                           fusion_pred_alpha_loss_weight=1,
                           fusion_pred_laplacian_loss_weight=1,
                           composition_loss_weight=1,
                           iou_predict_loss_weight=1,
                           class_loss_weight=1,
                           supervise_all_iou=True,
                           mask_threshold=0.5)

    for data in tqdm(train_loader):
        batch_images, input_masks, input_images = data['batch_image'], data[
            'mask'], data['input_image']

        input_trimaps, input_fg_maps, input_bg_maps = data['trimap'], data[
            'fg_map'], data['bg_map']

        input_prompt_points, input_prompt_boxes, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        object_to_frame_idxs = data['object_to_frame_idx']

        print('1111', batch_images.shape, input_masks.shape,
              input_images.shape, input_trimaps.shape, input_fg_maps.shape,
              input_bg_maps.shape, input_prompt_points.shape,
              input_prompt_boxes.shape, input_prompt_masks.shape,
              object_to_frame_idxs.shape)

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda()

        net = net.cuda()

        input_images = data['input_image']
        input_masks = data['mask']
        input_trimaps = data['trimap']
        input_fg_maps = data['fg_map']
        input_bg_maps = data['bg_map']
        targets = [
            input_masks,
            input_trimaps,
            input_fg_maps,
            input_bg_maps,
        ]

        preds = net(data)
        loss_dict = loss(input_images, preds, targets)
        print('2222', loss_dict)

        break

    # 释放显存
    del net
    del loss
    del preds
    del loss_dict
    del data
    torch.cuda.empty_cache()

    from torch.utils.data import DataLoader

    collater = SAM2MattingVideoBatchCollater(resize=1024, use_image_prob=0.0)
    train_loader = DataLoader(sam2_video_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from SimpleAICV.video_interactive_segmentation.models.segment_anything2_matting.sam2videomatting_train import hiera_b_plus_sam2video_matting
    net = hiera_b_plus_sam2video_matting(use_gradient_checkpoint=True)
    load_state_dict(
        '/root/autodl-tmp/pretrained_models/sam2.1_convert_from_pytorch_official_weights/sam2.1_hiera_base_plus_convert_from_pytorch_official_weight.pth',
        net)

    loss = SAM2MattingMultiLevelLoss(global_pred_trimap_ce_loss_weight=1,
                                     global_pred_trimap_iou_loss_weight=1,
                                     local_pred_alpha_loss_weight=1,
                                     local_pred_laplacian_loss_weight=1,
                                     fusion_pred_alpha_loss_weight=1,
                                     fusion_pred_laplacian_loss_weight=1,
                                     composition_loss_weight=1,
                                     iou_predict_loss_weight=1,
                                     class_loss_weight=1,
                                     mask_threshold=0.5)

    for data in tqdm(train_loader):
        batch_images, input_masks, input_images = data['batch_image'], data[
            'mask'], data['input_image']

        input_trimaps, input_fg_maps, input_bg_maps = data['trimap'], data[
            'fg_map'], data['bg_map']

        input_prompt_points, input_prompt_boxes, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        object_to_frame_idxs = data['object_to_frame_idx']

        print('1111', batch_images.shape, input_masks.shape,
              input_images.shape, input_trimaps.shape, input_fg_maps.shape,
              input_bg_maps.shape, input_prompt_points.shape,
              input_prompt_boxes.shape, input_prompt_masks.shape,
              object_to_frame_idxs.shape)

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda()

        net = net.cuda()

        input_images = data['input_image']
        input_masks = data['mask']
        input_trimaps = data['trimap']
        input_fg_maps = data['fg_map']
        input_bg_maps = data['bg_map']
        targets = [
            input_masks,
            input_trimaps,
            input_fg_maps,
            input_bg_maps,
        ]

        preds = net(data)
        loss_dict = loss(input_images, preds, targets)
        print('2222', loss_dict)

        break
