import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = [
    'SAMMattingOneLevelLoss',
    'SAMMattingMultiLevelLoss',
    'SAMMattingMultiLevelIoUMaxLoss',
    'SAMMattingMultiLevelAssignLoss',
]


class SAMMattingOneLevelLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 gloabel_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 fused_pred_iou_predict_loss_weight=1,
                 mask_threshold=0.5):
        super(SAMMattingOneLevelLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.gloabel_pred_trimap_iou_loss_weight = gloabel_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.fused_pred_iou_predict_loss_weight = fused_pred_iou_predict_loss_weight
        self.mask_threshold = mask_threshold

    def forward(self, images, inputs, targets):
        # images: torch.Size([2, 3, 1024, 1024])
        # torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 4])
        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = inputs
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 3, 1024, 1024])
        batch_masks, trimaps, fg_maps, bg_maps = targets

        global_pred_trimap_ce_loss = self.global_trimap_ce_loss(
            batch_masks_global_preds, trimaps)
        gloabel_pred_trimap_iou_loss = self.gloabel_trimap_iou_loss(
            batch_masks_global_preds, trimaps)
        local_pred_alpha_loss = self.local_alpha_loss(batch_masks_local_preds,
                                                      batch_masks, trimaps)
        local_pred_laplacian_loss = self.local_laplacian_loss(
            batch_masks_local_preds, batch_masks, trimaps)
        fusion_pred_alpha_loss = self.fusion_alpha_loss(
            batch_masks_fused_preds, batch_masks)
        fusion_pred_laplacian_loss = self.fusion_laplacian_loss(
            batch_masks_fused_preds, batch_masks)
        composition_loss = self.composition_loss(images, batch_masks, fg_maps,
                                                 bg_maps,
                                                 batch_masks_fused_preds)

        global_pred_trimap_ce_loss = self.global_pred_trimap_ce_loss_weight * global_pred_trimap_ce_loss
        gloabel_pred_trimap_iou_loss = self.gloabel_pred_trimap_iou_loss_weight * gloabel_pred_trimap_iou_loss
        local_pred_alpha_loss = self.local_pred_alpha_loss_weight * local_pred_alpha_loss
        local_pred_laplacian_loss = self.local_pred_laplacian_loss_weight * local_pred_laplacian_loss
        fusion_pred_alpha_loss = self.fusion_pred_alpha_loss_weight * fusion_pred_alpha_loss
        fusion_pred_laplacian_loss = self.fusion_pred_laplacian_loss_weight * fusion_pred_laplacian_loss
        composition_loss = self.composition_loss_weight * composition_loss

        fused_pred_iou_predict_loss = self.fusion_iou_predict_loss(
            batch_masks_fused_preds, batch_masks, batch_iou_preds)

        fused_pred_iou_predict_loss = self.fused_pred_iou_predict_loss_weight * fused_pred_iou_predict_loss

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'gloabel_pred_trimap_iou_loss': gloabel_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'fused_pred_iou_predict_loss': fused_pred_iou_predict_loss,
        }

        return loss_dict

    def global_trimap_ce_loss(self, global_pred, trimap):
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]
        global_pred = global_pred.float()
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

    def gloabel_trimap_iou_loss(self, global_pred, trimap):
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]
        global_pred = global_pred.float()
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

    def local_alpha_loss(self, local_pred, alpha, trimap):
        # local_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,1,h,w] -> [b,h,w]
        # trimap shape:[b,h,w]
        local_pred = local_pred.permute(0, 2, 3, 1).contiguous()
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        alpha = torch.squeeze(alpha, dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        diff = local_pred - alpha
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum() + 1.)

        return alpha_loss

    def local_laplacian_loss(self, local_pred, alpha, trimap):
        # local_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,1,h,w]
        # trimap shape:[b,h,w]
        device = local_pred.device
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)

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

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        gaussian = lambda x: np.exp((x - size // 2)**2 / (-2 * sigma**2))**2
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

    def fusion_alpha_loss(self, fusion_pred, alpha):
        # fusion_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,1,h,w] -> [b,h,w]
        fusion_pred = fusion_pred.permute(0, 2, 3, 1).contiguous()
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        alpha = torch.squeeze(alpha, dim=1)
        weighted = torch.ones_like(alpha)

        diff = fusion_pred - alpha
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())

        return alpha_loss

    def fusion_laplacian_loss(self, fusion_pred, alpha):
        # fusion_pred shape:[b,1,h,w]
        # alpha shape:[b,1,h,w]
        device = fusion_pred.device
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)

        gauss_kernel = self.build_gauss_kernel(size=5, sigma=1.0,
                                               n_channels=1).to(device)
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(fusion_pred, gauss_kernel, 5)

        laplacian_loss = sum(
            F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        return laplacian_loss

    def composition_loss(self, image, alpha, fg_map, bg_map, fusion_pred):
        # image shape:[b,3,h,w]
        # alpha shape:[b,1,h,w]
        # fg_map shape:[b,3,h,w]
        # bg_map shape:[b,3,h,w]
        # fusion_pred shape:[b,1,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.cat([fusion_pred, fusion_pred, fusion_pred], dim=1)

        alpha = torch.cat([alpha, alpha, alpha], dim=1)
        weighted = torch.ones_like(alpha)

        composition = fusion_pred * fg_map + (1. - fusion_pred) * bg_map
        composition_loss = torch.sqrt((composition - image)**2 + 1e-12)
        composition_loss = composition_loss.sum() / weighted.sum()

        return composition_loss

    def fusion_iou_predict_loss(self, inputs, targets, iou_predictions):
        # torch.Size([3, 1, 1024, 1024]) torch.Size([3, 1, 1024, 1024])
        batch_size = inputs.shape[0]
        inputs = inputs.float()
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

        inputs = (inputs >= self.mask_threshold).float()
        targets = (targets >= self.mask_threshold).float()

        inputs = inputs.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)

        intersection = inputs * targets
        iou_gt = (torch.sum(intersection, dim=1) + 1e-4) / (
            (torch.sum(inputs, dim=1) + torch.sum(targets, dim=1) -
             torch.sum(intersection, dim=1)) + 1e-4)
        iou_predictions = iou_predictions.squeeze(dim=1)
        iou_predict_loss = F.mse_loss(iou_predictions, iou_gt, reduction='sum')
        batch_size = inputs.shape[0]
        iou_predict_loss = iou_predict_loss / batch_size

        return iou_predict_loss


class SAMMattingMultiLevelLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 gloabel_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 fused_pred_iou_predict_loss_weight=1,
                 mask_threshold=0.5):
        super(SAMMattingMultiLevelLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.gloabel_pred_trimap_iou_loss_weight = gloabel_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.fused_pred_iou_predict_loss_weight = fused_pred_iou_predict_loss_weight
        self.mask_threshold = mask_threshold

    def forward(self, images, inputs, targets):
        # images: torch.Size([2, 3, 1024, 1024])
        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 4])
        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = inputs
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 3, 1024, 1024])
        batch_masks, trimaps, fg_maps, bg_maps = targets

        assert batch_masks_global_preds.shape[
            1] == batch_masks_local_preds.shape[
                1] == batch_masks_fused_preds.shape[
                    1] == batch_iou_preds.shape[1]

        global_pred_trimap_ce_loss = self.global_trimap_ce_loss(
            batch_masks_global_preds, trimaps)
        gloabel_pred_trimap_iou_loss = self.gloabel_trimap_iou_loss(
            batch_masks_global_preds, trimaps)
        local_pred_alpha_loss = self.local_alpha_loss(batch_masks_local_preds,
                                                      batch_masks, trimaps)
        local_pred_laplacian_loss = self.local_laplacian_loss(
            batch_masks_local_preds, batch_masks, trimaps)
        fusion_pred_alpha_loss = self.fusion_alpha_loss(
            batch_masks_fused_preds, batch_masks)
        fusion_pred_laplacian_loss = self.fusion_laplacian_loss(
            batch_masks_fused_preds, batch_masks)
        composition_loss = self.composition_loss(images, batch_masks, fg_maps,
                                                 bg_maps,
                                                 batch_masks_fused_preds)

        global_pred_trimap_ce_loss = self.global_pred_trimap_ce_loss_weight * global_pred_trimap_ce_loss
        gloabel_pred_trimap_iou_loss = self.gloabel_pred_trimap_iou_loss_weight * gloabel_pred_trimap_iou_loss
        local_pred_alpha_loss = self.local_pred_alpha_loss_weight * local_pred_alpha_loss
        local_pred_laplacian_loss = self.local_pred_laplacian_loss_weight * local_pred_laplacian_loss
        fusion_pred_alpha_loss = self.fusion_pred_alpha_loss_weight * fusion_pred_alpha_loss
        fusion_pred_laplacian_loss = self.fusion_pred_laplacian_loss_weight * fusion_pred_laplacian_loss
        composition_loss = self.composition_loss_weight * composition_loss

        fused_pred_iou_predict_loss = self.fusion_iou_predict_loss(
            batch_masks_fused_preds, batch_masks, batch_iou_preds)

        fused_pred_iou_predict_loss = self.fused_pred_iou_predict_loss_weight * fused_pred_iou_predict_loss

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'gloabel_pred_trimap_iou_loss': gloabel_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'fused_pred_iou_predict_loss': fused_pred_iou_predict_loss,
        }

        return loss_dict

    def global_trimap_ce_loss(self, global_pred, trimap):
        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 1024, 1024])
        # global_pred shape:[b,4,3,h,w] -> [b,4,h,w,3]
        # trimap shape:[b,h,w]->[b,1,h,w]
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = global_pred.float()
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, global_pred.shape[1], dim=1)
        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        loss_ground_truth = F.one_hot(convert_trimap.long(),
                                      num_classes=num_classes).float()
        bce_loss = -(loss_ground_truth * torch.log(global_pred) +
                     (1. - loss_ground_truth) * torch.log(1. - global_pred))
        bce_loss = bce_loss.mean()

        return bce_loss

    def gloabel_trimap_iou_loss(self, global_pred, trimap):
        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 1024, 1024])
        # global_pred shape:[b,4,3,h,w] -> [b,4,h,w,3]
        # trimap shape:[b,h,w]->[b,1,h,w]
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = global_pred.float()
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, global_pred.shape[1], dim=1)
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

    def local_alpha_loss(self, local_pred, alpha, trimap):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024])
        # local_pred shape:[b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha shape:[b,1,h,w]
        # trimap shape:[b,h,w]->[b,1,h,w]
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        diff = local_pred - alpha
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum() + 1.)

        return alpha_loss

    def local_laplacian_loss(self, local_pred, alpha, trimap):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024])
        # local_pred shape:[b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha shape:[b,1,h,w]
        # trimap shape:[b,h,w] -> [b,1,h,w]
        device = local_pred.device
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        alpha = torch.repeat_interleave(alpha, local_pred.shape[1], dim=1)

        local_pred = local_pred * weighted
        alpha = alpha * weighted

        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=local_pred.shape[1]).to(device)
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(local_pred, gauss_kernel, 5)
        laplacian_loss = sum(
            F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        return laplacian_loss

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        gaussian = lambda x: np.exp((x - size // 2)**2 / (-2 * sigma**2))**2
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

    def fusion_alpha_loss(self, fusion_pred, alpha):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024])
        # fusion_pred shape:[b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha shape:[b,1,h,w]
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)
        weighted = torch.ones_like(alpha)

        diff = fusion_pred - alpha
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())

        return alpha_loss

    def fusion_laplacian_loss(self, fusion_pred, alpha):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024])
        # fusion_pred shape:[b,4,1,h,w]
        # alpha shape:[b,1,h,w]
        device = fusion_pred.device
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)

        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=fusion_pred.shape[1]).to(device)
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(fusion_pred, gauss_kernel, 5)
        laplacian_loss = sum(
            F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        return laplacian_loss

    def composition_loss(self, image, alpha, fg_map, bg_map, fusion_pred):
        # torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024])
        # image shape:[b,3,h,w]
        # alpha shape:[b,1,h,w]
        # fg_map shape:[b,3,h,w]
        # bg_map shape:[b,3,h,w]
        # fusion_pred shape:[b,4,1,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.cat([fusion_pred, fusion_pred, fusion_pred], dim=2)

        alpha = torch.cat([alpha, alpha, alpha], dim=1)
        weighted = torch.ones_like(alpha)

        fg_map = torch.unsqueeze(fg_map, dim=1)
        bg_map = torch.unsqueeze(bg_map, dim=1)
        image = torch.unsqueeze(image, dim=1)
        weighted = torch.unsqueeze(weighted, dim=1)
        weighted = torch.repeat_interleave(weighted,
                                           fusion_pred.shape[1],
                                           dim=1)

        composition = fusion_pred * fg_map + (1. - fusion_pred) * bg_map
        composition_loss = torch.sqrt((composition - image)**2 + 1e-12)
        composition_loss = composition_loss.sum() / weighted.sum()

        return composition_loss

    def fusion_iou_predict_loss(self, inputs, targets, iou_predictions):
        # torch.Size([3, 4, 1, 1024, 1024]) torch.Size([3, 1, 1024, 1024])
        batch_size = inputs.shape[0]
        inputs = torch.squeeze(inputs, dim=2)
        inputs = inputs.float()
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

        targets = torch.repeat_interleave(targets, inputs.shape[1], dim=1)

        inputs = (inputs >= self.mask_threshold).float()
        targets = (targets >= self.mask_threshold).float()

        inputs = inputs.reshape(batch_size, inputs.shape[1], -1)
        targets = targets.reshape(batch_size, inputs.shape[1], -1)

        intersection = inputs * targets
        iou_gt = (torch.sum(intersection, dim=-1) + 1e-4) / (
            (torch.sum(inputs, dim=-1) + torch.sum(targets, dim=-1) -
             torch.sum(intersection, dim=-1)) + 1e-4)
        iou_predict_loss = F.mse_loss(iou_predictions, iou_gt, reduction='sum')
        iou_predict_loss = iou_predict_loss / batch_size
        iou_predict_loss = iou_predict_loss / inputs.shape[1]

        return iou_predict_loss


class SAMMattingMultiLevelIoUMaxLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 gloabel_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 fused_pred_iou_predict_loss_weight=1,
                 mask_threshold=0.5):
        super(SAMMattingMultiLevelIoUMaxLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.gloabel_pred_trimap_iou_loss_weight = gloabel_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.fused_pred_iou_predict_loss_weight = fused_pred_iou_predict_loss_weight
        self.mask_threshold = mask_threshold

    def forward(self, images, inputs, targets):
        # images: torch.Size([2, 3, 1024, 1024])
        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 4])
        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = inputs
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 3, 1024, 1024])
        batch_masks, trimaps, fg_maps, bg_maps = targets

        assert batch_masks_global_preds.shape[
            1] == batch_masks_local_preds.shape[
                1] == batch_masks_fused_preds.shape[
                    1] == batch_iou_preds.shape[1]

        batch_masks_fused_preds_0_1 = (batch_masks_fused_preds
                                       >= self.mask_threshold).float()
        batch_masks_0_1 = (batch_masks.unsqueeze(1)
                           >= self.mask_threshold).float()
        intersection = (batch_masks_fused_preds_0_1 *
                        batch_masks_0_1).sum(dim=[-3, -2, -1])

        union = batch_masks_fused_preds_0_1.sum(
            dim=[-3, -2, -1]) + batch_masks_0_1.sum(
                dim=[-3, -2, -1]) - intersection + 1e-4
        ious = intersection / union
        max_iou_idx = ious.argmax(dim=1)

        batch_range = torch.arange(batch_masks_fused_preds.shape[0],
                                   device=batch_masks_fused_preds.device)
        batch_masks_global_preds = batch_masks_global_preds[batch_range,
                                                            max_iou_idx]
        batch_masks_local_preds = batch_masks_local_preds[batch_range,
                                                          max_iou_idx]
        batch_masks_fused_preds = batch_masks_fused_preds[batch_range,
                                                          max_iou_idx]
        batch_iou_preds = batch_iou_preds[batch_range, max_iou_idx]
        batch_iou_preds = batch_iou_preds.unsqueeze(dim=1)
        # torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1])

        global_pred_trimap_ce_loss = self.global_trimap_ce_loss(
            batch_masks_global_preds, trimaps)
        gloabel_pred_trimap_iou_loss = self.gloabel_trimap_iou_loss(
            batch_masks_global_preds, trimaps)
        local_pred_alpha_loss = self.local_alpha_loss(batch_masks_local_preds,
                                                      batch_masks, trimaps)
        local_pred_laplacian_loss = self.local_laplacian_loss(
            batch_masks_local_preds, batch_masks, trimaps)
        fusion_pred_alpha_loss = self.fusion_alpha_loss(
            batch_masks_fused_preds, batch_masks)
        fusion_pred_laplacian_loss = self.fusion_laplacian_loss(
            batch_masks_fused_preds, batch_masks)
        composition_loss = self.composition_loss(images, batch_masks, fg_maps,
                                                 bg_maps,
                                                 batch_masks_fused_preds)

        global_pred_trimap_ce_loss = self.global_pred_trimap_ce_loss_weight * global_pred_trimap_ce_loss
        gloabel_pred_trimap_iou_loss = self.gloabel_pred_trimap_iou_loss_weight * gloabel_pred_trimap_iou_loss
        local_pred_alpha_loss = self.local_pred_alpha_loss_weight * local_pred_alpha_loss
        local_pred_laplacian_loss = self.local_pred_laplacian_loss_weight * local_pred_laplacian_loss
        fusion_pred_alpha_loss = self.fusion_pred_alpha_loss_weight * fusion_pred_alpha_loss
        fusion_pred_laplacian_loss = self.fusion_pred_laplacian_loss_weight * fusion_pred_laplacian_loss
        composition_loss = self.composition_loss_weight * composition_loss

        fused_pred_iou_predict_loss = self.fusion_iou_predict_loss(
            batch_masks_fused_preds, batch_masks, batch_iou_preds)

        fused_pred_iou_predict_loss = self.fused_pred_iou_predict_loss_weight * fused_pred_iou_predict_loss

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'gloabel_pred_trimap_iou_loss': gloabel_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'fused_pred_iou_predict_loss': fused_pred_iou_predict_loss,
        }

        return loss_dict

    def global_trimap_ce_loss(self, global_pred, trimap):
        # torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1024, 1024])
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]->[b,h,w]
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]
        global_pred = global_pred.float()
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

    def gloabel_trimap_iou_loss(self, global_pred, trimap):
        # torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1024, 1024])
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]->[b,h,w]
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]
        global_pred = global_pred.float()
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

    def local_alpha_loss(self, local_pred, alpha, trimap):
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024])
        # local_pred shape:[b,1,h,w]
        # alpha shape:[b,1,h,w]
        # trimap shape:[b,h,w]->[b,1,h,w]
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)

        trimap = torch.unsqueeze(trimap, dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        diff = local_pred - alpha
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum() + 1.)

        return alpha_loss

    def local_laplacian_loss(self, local_pred, alpha, trimap):
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024])
        # local_pred shape:[b,1,h,w]
        # alpha shape:[b,1,h,w]
        # trimap shape:[b,h,w] -> [b,1,h,w]
        device = local_pred.device
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)

        trimap = torch.unsqueeze(trimap, dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        local_pred = local_pred * weighted
        alpha = alpha * weighted

        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=local_pred.shape[1]).to(device)
        pyr_alpha = self.laplacian_pyramid(
            torch.repeat_interleave(alpha, local_pred.shape[1], dim=1),
            gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(local_pred, gauss_kernel, 5)
        laplacian_loss = sum(
            F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        return laplacian_loss

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        gaussian = lambda x: np.exp((x - size // 2)**2 / (-2 * sigma**2))**2
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

    def fusion_alpha_loss(self, fusion_pred, alpha):
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024])
        # fusion_pred shape:[b,1,h,w]
        # alpha shape:[b,1,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)

        weighted = torch.ones_like(alpha)

        diff = fusion_pred - alpha
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())

        return alpha_loss

    def fusion_laplacian_loss(self, fusion_pred, alpha):
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024])
        # fusion_pred shape:[b,1,h,w]
        # alpha shape:[b,1,h,w]
        device = fusion_pred.device
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)

        gauss_kernel = self.build_gauss_kernel(
            size=5, sigma=1.0, n_channels=fusion_pred.shape[1]).to(device)
        pyr_alpha = self.laplacian_pyramid(
            torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1),
            gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(fusion_pred, gauss_kernel, 5)
        laplacian_loss = sum(
            F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        return laplacian_loss

    def composition_loss(self, image, alpha, fg_map, bg_map, fusion_pred):
        # torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1, 1024, 1024])
        # image shape:[b,3,h,w]
        # alpha shape:[b,1,h,w]
        # fg_map shape:[b,3,h,w]
        # bg_map shape:[b,3,h,w]
        # fusion_pred shape:[b,1,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.cat([fusion_pred, fusion_pred, fusion_pred], dim=1)

        alpha = torch.cat([alpha, alpha, alpha], dim=1)
        weighted = torch.ones_like(alpha)

        composition = fusion_pred * fg_map + (1. - fusion_pred) * bg_map
        composition_loss = torch.sqrt((composition - image)**2 + 1e-12)
        composition_loss = composition_loss.sum() / weighted.sum()

        return composition_loss

    def fusion_iou_predict_loss(self, inputs, targets, iou_predictions):
        # torch.Size([3, 1, 1024, 1024]) torch.Size([3, 1, 1024, 1024])
        batch_size = inputs.shape[0]
        inputs = inputs.float()
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

        inputs = (inputs >= self.mask_threshold).float()
        targets = (targets >= self.mask_threshold).float()

        inputs = inputs.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)

        intersection = inputs * targets
        iou_gt = (torch.sum(intersection, dim=1) + 1e-4) / (
            (torch.sum(inputs, dim=1) + torch.sum(targets, dim=1) -
             torch.sum(intersection, dim=1)) + 1e-4)
        iou_predictions = iou_predictions.squeeze(dim=1)
        iou_predict_loss = F.mse_loss(iou_predictions, iou_gt, reduction='sum')
        batch_size = inputs.shape[0]
        iou_predict_loss = iou_predict_loss / batch_size

        return iou_predict_loss


class SAMMattingMultiLevelAssignLoss(nn.Module):

    def __init__(self,
                 global_pred_trimap_ce_loss_weight=1,
                 gloabel_pred_trimap_iou_loss_weight=1,
                 local_pred_alpha_loss_weight=1,
                 local_pred_laplacian_loss_weight=1,
                 fusion_pred_alpha_loss_weight=1,
                 fusion_pred_laplacian_loss_weight=1,
                 composition_loss_weight=1,
                 fused_pred_iou_predict_loss_weight=1,
                 mask_threshold=0.5,
                 idx_nums=4,
                 area_ranges=[[0.04, 0.64], [0.0, 0.04], [0.01, 0.25],
                              [0.16, 1.0]]):
        super(SAMMattingMultiLevelAssignLoss, self).__init__()
        self.global_pred_trimap_ce_loss_weight = global_pred_trimap_ce_loss_weight
        self.gloabel_pred_trimap_iou_loss_weight = gloabel_pred_trimap_iou_loss_weight
        self.local_pred_alpha_loss_weight = local_pred_alpha_loss_weight
        self.local_pred_laplacian_loss_weight = local_pred_laplacian_loss_weight
        self.fusion_pred_alpha_loss_weight = fusion_pred_alpha_loss_weight
        self.fusion_pred_laplacian_loss_weight = fusion_pred_laplacian_loss_weight
        self.composition_loss_weight = composition_loss_weight
        self.fused_pred_iou_predict_loss_weight = fused_pred_iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.idx_nums = idx_nums
        self.area_ranges = area_ranges
        assert len(self.area_ranges) == self.idx_nums

    def forward(self, images, inputs, targets):
        # images: torch.Size([2, 3, 1024, 1024])
        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 4])
        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = inputs
        # torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 3, 1024, 1024])
        batch_masks, trimaps, fg_maps, bg_maps = targets

        assert batch_masks_global_preds.shape[
            1] == batch_masks_local_preds.shape[
                1] == batch_masks_fused_preds.shape[
                    1] == batch_iou_preds.shape[1]

        global_pred_trimap_ce_loss = self.global_trimap_ce_loss(
            batch_masks_global_preds, batch_masks, trimaps)
        gloabel_pred_trimap_iou_loss = self.gloabel_trimap_iou_loss(
            batch_masks_global_preds, batch_masks, trimaps)
        local_pred_alpha_loss = self.local_alpha_loss(batch_masks_local_preds,
                                                      batch_masks, trimaps)
        local_pred_laplacian_loss = self.local_laplacian_loss(
            batch_masks_local_preds, batch_masks, trimaps)
        fusion_pred_alpha_loss = self.fusion_alpha_loss(
            batch_masks_fused_preds, batch_masks)
        fusion_pred_laplacian_loss = self.fusion_laplacian_loss(
            batch_masks_fused_preds, batch_masks)
        composition_loss = self.composition_loss(images, batch_masks, fg_maps,
                                                 bg_maps,
                                                 batch_masks_fused_preds)

        global_pred_trimap_ce_loss = self.global_pred_trimap_ce_loss_weight * global_pred_trimap_ce_loss
        gloabel_pred_trimap_iou_loss = self.gloabel_pred_trimap_iou_loss_weight * gloabel_pred_trimap_iou_loss
        local_pred_alpha_loss = self.local_pred_alpha_loss_weight * local_pred_alpha_loss
        local_pred_laplacian_loss = self.local_pred_laplacian_loss_weight * local_pred_laplacian_loss
        fusion_pred_alpha_loss = self.fusion_pred_alpha_loss_weight * fusion_pred_alpha_loss
        fusion_pred_laplacian_loss = self.fusion_pred_laplacian_loss_weight * fusion_pred_laplacian_loss
        composition_loss = self.composition_loss_weight * composition_loss

        fused_pred_iou_predict_loss = self.fusion_iou_predict_loss(
            batch_masks_fused_preds, batch_masks, batch_iou_preds)

        fused_pred_iou_predict_loss = self.fused_pred_iou_predict_loss_weight * fused_pred_iou_predict_loss

        loss_dict = {
            'global_pred_trimap_ce_loss': global_pred_trimap_ce_loss,
            'gloabel_pred_trimap_iou_loss': gloabel_pred_trimap_iou_loss,
            'local_pred_alpha_loss': local_pred_alpha_loss,
            'local_pred_laplacian_loss': local_pred_laplacian_loss,
            'fusion_pred_alpha_loss': fusion_pred_alpha_loss,
            'fusion_pred_laplacian_loss': fusion_pred_laplacian_loss,
            'composition_loss': composition_loss,
            'fused_pred_iou_predict_loss': fused_pred_iou_predict_loss,
        }

        return loss_dict

    def global_trimap_ce_loss(self, global_pred, alpha, trimap):
        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 1024, 1024])
        # global_pred shape:[b,4,3,h,w] -> [b,4,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = global_pred.float()
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        trimap = torch.unsqueeze(trimap, dim=1)
        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        alpha = torch.repeat_interleave(alpha, global_pred.shape[1], dim=1)

        batch_size = global_pred.shape[0]
        idx_nums = global_pred.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_bce_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_bce_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_global_pred = global_pred[per_sample_idx]
            per_sample_convert_trimap = convert_trimap[per_sample_idx]
            per_sample_convert_trimap = torch.squeeze(
                per_sample_convert_trimap, dim=0)

            per_sample_alpha = alpha[per_sample_idx]

            per_sample_alpha_h, per_sample_alpha_w = per_sample_alpha.shape[
                -2], per_sample_alpha.shape[-1]

            per_sample_alpha_to_compute_area = per_sample_alpha[0]
            per_sample_alpha_area_ratio = torch.sum(
                torch.where(per_sample_alpha_to_compute_area > 0, 1,
                            0)) / float(
                                per_sample_alpha_h * per_sample_alpha_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_alpha_area_ratio < per_area_range2:
                    per_idx_global_pred = per_sample_global_pred[per_idx, :, :]

                    per_idx_loss_ground_truth = F.one_hot(
                        per_sample_convert_trimap.long(),
                        num_classes=num_classes).float()

                    per_idx_bce_loss = -(per_idx_loss_ground_truth *
                                         torch.log(per_idx_global_pred) +
                                         (1. - per_idx_loss_ground_truth) *
                                         torch.log(1. - per_idx_global_pred))
                    per_idx_bce_loss = per_idx_bce_loss.mean()

                    per_sample_bce_loss += per_idx_bce_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_bce_loss = per_sample_bce_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_bce_loss += per_sample_bce_loss

        if valid_batch_size > 0:
            total_bce_loss = total_bce_loss / valid_batch_size

        return total_bce_loss

    def gloabel_trimap_iou_loss(self, global_pred, alpha, trimap):
        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 1024, 1024])
        # global_pred shape:[b,4,3,h,w] -> [b,4,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.permute(0, 1, 3, 4, 2).contiguous()
        num_classes = global_pred.shape[4]
        global_pred = global_pred.float()
        global_pred = torch.clamp(global_pred, min=1e-4, max=1. - 1e-4)

        trimap = torch.unsqueeze(trimap, dim=1)
        convert_trimap = trimap.clone()
        # 0为背景区域，2为global区域，1为local区域
        convert_trimap[convert_trimap == 0] = 0
        convert_trimap[convert_trimap == 255] = 2
        convert_trimap[convert_trimap > 2] = 1

        alpha = torch.repeat_interleave(alpha, global_pred.shape[1], dim=1)

        batch_size = global_pred.shape[0]
        idx_nums = global_pred.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_iou_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_iou_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_global_pred = global_pred[per_sample_idx]
            per_sample_convert_trimap = convert_trimap[per_sample_idx]
            per_sample_convert_trimap = torch.squeeze(
                per_sample_convert_trimap, dim=0)

            per_sample_alpha = alpha[per_sample_idx]

            per_sample_alpha_h, per_sample_alpha_w = per_sample_alpha.shape[
                -2], per_sample_alpha.shape[-1]

            per_sample_alpha_to_compute_area = per_sample_alpha[0]
            per_sample_alpha_area_ratio = torch.sum(
                torch.where(per_sample_alpha_to_compute_area > 0, 1,
                            0)) / float(
                                per_sample_alpha_h * per_sample_alpha_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_alpha_area_ratio < per_area_range2:
                    per_idx_global_pred = per_sample_global_pred[per_idx, :, :]

                    per_idx_label = F.one_hot(per_sample_convert_trimap.long(),
                                              num_classes=num_classes).float()

                    per_idx_global_pred = per_idx_global_pred.view(
                        -1, num_classes)
                    per_idx_label = per_idx_label.view(-1, num_classes)

                    per_idx_intersection = per_idx_global_pred * per_idx_label
                    per_idx_iou_loss = 1. - (per_idx_intersection + 1e-4) / (
                        per_idx_global_pred + per_idx_label -
                        per_idx_intersection + 1e-4)
                    per_idx_iou_loss = 1. - (
                        torch.sum(per_idx_intersection, dim=1) +
                        1e-4) / (torch.sum(per_idx_global_pred, dim=1) +
                                 torch.sum(per_idx_label, dim=1) -
                                 torch.sum(per_idx_intersection, dim=1) + 1e-4)
                    per_idx_iou_loss = per_idx_iou_loss.mean()

                    per_sample_iou_loss += per_idx_iou_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_iou_loss = per_sample_iou_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_iou_loss += per_sample_iou_loss

        if valid_batch_size > 0:
            total_iou_loss = total_iou_loss / valid_batch_size

        return total_iou_loss

    def local_alpha_loss(self, local_pred, alpha, trimap):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024])
        # local_pred shape:[b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha shape:[b,1,h,w]
        # trimap shape:[b,h,w]->[b,1,h,w]
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        alpha = torch.repeat_interleave(alpha, local_pred.shape[1], dim=1)

        batch_size = local_pred.shape[0]
        idx_nums = local_pred.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_alpha_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_alpha_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_local_pred = local_pred[per_sample_idx]
            per_sample_weighted = weighted[per_sample_idx]

            per_sample_alpha = alpha[per_sample_idx]

            per_sample_alpha_h, per_sample_alpha_w = per_sample_alpha.shape[
                -2], per_sample_alpha.shape[-1]

            per_sample_alpha_to_compute_area = per_sample_alpha[0]
            per_sample_alpha_area_ratio = torch.sum(
                torch.where(per_sample_alpha_to_compute_area > 0, 1,
                            0)) / float(
                                per_sample_alpha_h * per_sample_alpha_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_alpha_area_ratio < per_area_range2:
                    per_idx_local_pred = per_sample_local_pred[per_idx, :, :]
                    per_idx_alpha = per_sample_alpha[per_idx, :, :]
                    per_idx_weighted = per_sample_weighted[per_idx, :, :]

                    diff = per_idx_local_pred - per_idx_alpha
                    diff = diff * per_idx_weighted
                    per_idx_alpha_loss = torch.sqrt(diff**2 + 1e-12)
                    per_idx_alpha_loss = per_idx_alpha_loss.sum() / (
                        per_idx_weighted.sum() + 1.)

                    per_sample_alpha_loss += per_idx_alpha_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_alpha_loss = per_sample_alpha_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_alpha_loss += per_sample_alpha_loss

        if valid_batch_size > 0:
            total_alpha_loss = total_alpha_loss / valid_batch_size

        return total_alpha_loss

    def local_laplacian_loss(self, local_pred, alpha, trimap):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 1024, 1024])
        # local_pred shape:[b,4,1,h,w] -> [b,4,h,w,1]
        # alpha shape:[b,1,h,w]
        # trimap shape:[b,h,w]
        device = local_pred.device
        local_pred = local_pred.permute(0, 1, 3, 4, 2).contiguous()
        local_pred = local_pred.float()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        trimap = torch.unsqueeze(trimap, dim=1)
        trimap = torch.repeat_interleave(trimap, local_pred.shape[1], dim=1)
        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        alpha = torch.repeat_interleave(alpha, local_pred.shape[1], dim=1)

        local_pred = local_pred * weighted
        alpha = alpha * weighted

        batch_size = local_pred.shape[0]
        idx_nums = local_pred.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_laplacian_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_laplacian_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_local_pred = local_pred[per_sample_idx]
            per_sample_alpha = alpha[per_sample_idx]
            per_sample_weighted = weighted[per_sample_idx]

            per_sample_local_pred = torch.unsqueeze(per_sample_local_pred,
                                                    dim=0)
            per_sample_alpha = torch.unsqueeze(per_sample_alpha, dim=0)
            per_sample_weighted = torch.unsqueeze(per_sample_weighted, dim=0)

            per_sample_alpha_h, per_sample_alpha_w = per_sample_alpha.shape[
                -2], per_sample_alpha.shape[-1]

            per_sample_alpha_to_compute_area = per_sample_alpha[0][0]
            per_sample_alpha_area_ratio = torch.sum(
                torch.where(per_sample_alpha_to_compute_area > 0, 1,
                            0)) / float(
                                per_sample_alpha_h * per_sample_alpha_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_alpha_area_ratio < per_area_range2:
                    per_idx_local_pred = per_sample_local_pred[:, per_idx:
                                                               per_idx +
                                                               1, :, :]
                    per_idx_alpha = per_sample_alpha[:,
                                                     per_idx:per_idx + 1, :, :]
                    per_idx_weighted = per_sample_weighted[:, per_idx:per_idx +
                                                           1, :, :]

                    per_idx_local_pred = per_idx_local_pred * per_idx_weighted
                    per_idx_alpha = per_idx_alpha * per_idx_weighted

                    gauss_kernel = self.build_gauss_kernel(
                        size=5, sigma=1.0, n_channels=1).to(device)
                    pyr_per_idx_alpha = self.laplacian_pyramid(
                        per_idx_alpha, gauss_kernel, 5)
                    pyr_per_idx_predict = self.laplacian_pyramid(
                        per_idx_local_pred, gauss_kernel, 5)
                    per_idx_laplacian_loss = sum(
                        F.l1_loss(a, b) for a, b in zip(
                            pyr_per_idx_alpha, pyr_per_idx_predict))

                    per_sample_laplacian_loss += per_idx_laplacian_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_laplacian_loss = per_sample_laplacian_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_laplacian_loss += per_sample_laplacian_loss

        if valid_batch_size > 0:
            total_laplacian_loss = total_laplacian_loss / valid_batch_size

        return total_laplacian_loss

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        gaussian = lambda x: np.exp((x - size // 2)**2 / (-2 * sigma**2))**2
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

    def fusion_alpha_loss(self, fusion_pred, alpha):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024])
        # fusion_pred shape:[b,4,1,h,w] -> [b,4,h,w,1] -> [b,4,h,w]
        # alpha shape:[b,1,h,w]
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)
        weighted = torch.ones_like(alpha)

        batch_size = fusion_pred.shape[0]
        idx_nums = fusion_pred.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_alpha_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_alpha_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_fusion_pred = fusion_pred[per_sample_idx]
            per_sample_weighted = weighted[per_sample_idx]

            per_sample_alpha = alpha[per_sample_idx]

            per_sample_alpha_h, per_sample_alpha_w = per_sample_alpha.shape[
                -2], per_sample_alpha.shape[-1]

            per_sample_alpha_to_compute_area = per_sample_alpha[0]
            per_sample_alpha_area_ratio = torch.sum(
                torch.where(per_sample_alpha_to_compute_area > 0, 1,
                            0)) / float(
                                per_sample_alpha_h * per_sample_alpha_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_alpha_area_ratio < per_area_range2:
                    per_idx_fusion_pred = per_sample_fusion_pred[per_idx, :, :]
                    per_idx_alpha = per_sample_alpha[per_idx, :, :]
                    per_idx_weighted = per_sample_weighted[per_idx, :, :]

                    diff = per_idx_fusion_pred - per_idx_alpha
                    per_idx_alpha_loss = torch.sqrt(diff**2 + 1e-12)
                    per_idx_alpha_loss = per_idx_alpha_loss.sum() / (
                        per_idx_weighted.sum())
                    per_sample_alpha_loss += per_idx_alpha_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_alpha_loss = per_sample_alpha_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_alpha_loss += per_sample_alpha_loss

        if valid_batch_size > 0:
            total_alpha_loss = total_alpha_loss / valid_batch_size

        return total_alpha_loss

    def fusion_laplacian_loss(self, fusion_pred, alpha):
        # torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 1, 1024, 1024])
        # fusion_pred shape:[b,4,1,h,w]
        # alpha shape:[b,1,h,w]
        device = fusion_pred.device
        fusion_pred = fusion_pred.permute(0, 1, 3, 4, 2).contiguous()
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)

        batch_size = fusion_pred.shape[0]
        idx_nums = fusion_pred.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_laplacian_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_laplacian_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_fusion_pred = fusion_pred[per_sample_idx]
            per_sample_alpha = alpha[per_sample_idx]

            per_sample_fusion_pred = torch.unsqueeze(per_sample_fusion_pred,
                                                     dim=0)
            per_sample_alpha = torch.unsqueeze(per_sample_alpha, dim=0)

            per_sample_alpha_h, per_sample_alpha_w = per_sample_alpha.shape[
                -2], per_sample_alpha.shape[-1]

            per_sample_alpha_to_compute_area = per_sample_alpha[0][0]
            per_sample_alpha_area_ratio = torch.sum(
                torch.where(per_sample_alpha_to_compute_area > 0, 1,
                            0)) / float(
                                per_sample_alpha_h * per_sample_alpha_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_alpha_area_ratio < per_area_range2:
                    per_idx_fusion_pred = per_sample_fusion_pred[:, per_idx:
                                                                 per_idx +
                                                                 1, :, :]
                    per_idx_alpha = per_sample_alpha[:,
                                                     per_idx:per_idx + 1, :, :]

                    gauss_kernel = self.build_gauss_kernel(
                        size=5, sigma=1.0, n_channels=1).to(device)
                    pyr_idx_alpha = self.laplacian_pyramid(
                        per_idx_alpha, gauss_kernel, 5)
                    pyr_idx_predict = self.laplacian_pyramid(
                        per_idx_fusion_pred, gauss_kernel, 5)
                    per_idx_laplacian_loss = sum(
                        F.l1_loss(a, b)
                        for a, b in zip(pyr_idx_alpha, pyr_idx_predict))

                    per_sample_laplacian_loss += per_idx_laplacian_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_laplacian_loss = per_sample_laplacian_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_laplacian_loss += per_sample_laplacian_loss

        if valid_batch_size > 0:
            total_laplacian_loss = total_laplacian_loss / valid_batch_size

        return total_laplacian_loss

    def composition_loss(self, image, alpha, fg_map, bg_map, fusion_pred):
        # torch.Size([2, 3, 1024, 1024]) torch.Size([2, 1, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 3, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024])
        # image shape:[b,3,h,w]
        # alpha shape:[b,1,h,w]
        # fg_map shape:[b,3,h,w]
        # bg_map shape:[b,3,h,w]
        # fusion_pred shape:[b,4,1,h,w]
        fusion_pred = fusion_pred.float()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.cat([fusion_pred, fusion_pred, fusion_pred], dim=2)

        alpha = torch.cat([alpha, alpha, alpha], dim=1)
        alpha = torch.unsqueeze(alpha, dim=1)
        alpha = torch.repeat_interleave(alpha, fusion_pred.shape[1], dim=1)
        weighted = torch.ones_like(alpha)

        fg_map = torch.unsqueeze(fg_map, dim=1)
        fg_map = torch.repeat_interleave(fg_map, fusion_pred.shape[1], dim=1)
        bg_map = torch.unsqueeze(bg_map, dim=1)
        bg_map = torch.repeat_interleave(bg_map, fusion_pred.shape[1], dim=1)
        image = torch.unsqueeze(image, dim=1)
        image = torch.repeat_interleave(image, fusion_pred.shape[1], dim=1)

        batch_size = fusion_pred.shape[0]
        idx_nums = fusion_pred.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_composition_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_composition_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_fusion_pred = fusion_pred[per_sample_idx]
            per_sample_alpha = alpha[per_sample_idx]
            per_sample_weighted = weighted[per_sample_idx]

            per_sample_fg_map = fg_map[per_sample_idx]
            per_sample_bg_map = bg_map[per_sample_idx]
            per_sample_image = image[per_sample_idx]

            per_sample_alpha_h, per_sample_alpha_w = per_sample_alpha.shape[
                -2], per_sample_alpha.shape[-1]

            per_sample_alpha_to_compute_area = per_sample_alpha[0][0]
            per_sample_alpha_area_ratio = torch.sum(
                torch.where(per_sample_alpha_to_compute_area > 0, 1,
                            0)) / float(
                                per_sample_alpha_h * per_sample_alpha_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_alpha_area_ratio < per_area_range2:
                    per_idx_fusion_pred = per_sample_fusion_pred[
                        per_idx, :, :, :]
                    per_idx_fg_map = per_sample_fg_map[per_idx, :, :, :]
                    per_idx_bg_map = per_sample_bg_map[per_idx, :, :, :]
                    per_idx_image = per_sample_image[per_idx, :, :, :]
                    per_idx_weighted = per_sample_weighted[per_idx, :, :, :]

                    per_idx_composition = per_idx_fusion_pred * per_idx_fg_map + (
                        1. - per_idx_fusion_pred) * per_idx_bg_map
                    per_idx_composition_loss = torch.sqrt(
                        (per_idx_composition - per_idx_image)**2 + 1e-12)

                    per_idx_composition_loss = per_idx_composition_loss.sum(
                    ) / (per_idx_weighted.sum())

                    per_sample_composition_loss += per_idx_composition_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_composition_loss = per_sample_composition_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_composition_loss += per_sample_composition_loss

        if valid_batch_size > 0:
            total_composition_loss = total_composition_loss / valid_batch_size

        return total_composition_loss

    def fusion_iou_predict_loss(self, inputs, targets, iou_predictions):
        # torch.Size([3, 4, 1, 1024, 1024]) torch.Size([3, 1, 1024, 1024])
        batch_size = inputs.shape[0]
        inputs = torch.squeeze(inputs, dim=2)
        inputs = inputs.float()
        inputs = torch.clamp(inputs, min=1e-4, max=1. - 1e-4)

        targets = torch.repeat_interleave(targets, inputs.shape[1], dim=1)

        inputs = (inputs >= self.mask_threshold).float()
        targets = (targets >= self.mask_threshold).float()

        batch_size = inputs.shape[0]
        idx_nums = inputs.shape[1]
        assert idx_nums == self.idx_nums == len(self.area_ranges)

        total_iou_predict_loss = 0.
        valid_batch_size = 0.
        for per_sample_idx in range(batch_size):
            per_sample_iou_predict_loss = 0.
            per_sample_valid_idx_nums = 0

            per_sample_inputs = inputs[per_sample_idx]
            per_sample_target = targets[per_sample_idx]
            per_sample_iou_predictions = iou_predictions[per_sample_idx]

            per_sample_target_h, per_sample_target_w = per_sample_target.shape[
                -2], per_sample_target.shape[-1]

            per_sample_target_to_compute_area = per_sample_target[0]
            per_sample_target_area_ratio = torch.sum(
                per_sample_target_to_compute_area) / float(
                    per_sample_target_h * per_sample_target_w)

            for per_idx, (per_area_range1,
                          per_area_range2) in enumerate(self.area_ranges):
                if per_area_range1 < per_sample_target_area_ratio < per_area_range2:
                    per_idx_inputs = per_sample_inputs[per_idx:per_idx +
                                                       1, :, :]
                    per_idx_target = per_sample_target[per_idx:per_idx +
                                                       1, :, :]
                    per_idx_iou_predictions = per_sample_iou_predictions[
                        per_idx]

                    per_idx_inputs = per_idx_inputs.reshape(-1)
                    per_idx_target = per_idx_target.reshape(-1)

                    per_idx_intersection = per_idx_inputs * per_idx_target
                    per_idx_iou_gt = (
                        torch.sum(per_idx_intersection, dim=0) + 1e-4) / (
                            (torch.sum(per_idx_inputs, dim=0) +
                             torch.sum(per_idx_target, dim=0) -
                             torch.sum(per_idx_intersection, dim=0)) + 1e-4)

                    per_idx_iou_predict_loss = F.mse_loss(
                        per_idx_iou_predictions,
                        per_idx_iou_gt,
                        reduction='sum')
                    per_sample_iou_predict_loss += per_idx_iou_predict_loss
                    per_sample_valid_idx_nums += 1

            if per_sample_valid_idx_nums > 0:
                per_sample_iou_predict_loss = per_sample_iou_predict_loss / per_sample_valid_idx_nums
                valid_batch_size += 1

            total_iou_predict_loss += per_sample_iou_predict_loss

        if valid_batch_size > 0:
            total_iou_predict_loss = total_iou_predict_loss / valid_batch_size

        return total_iou_predict_loss


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

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from tools.path import interactive_segmentation_dataset_path

    from simpleAICV.interactive_segmentation.datasets.sam_matting_dataset import SAMMattingDataset
    from simpleAICV.interactive_segmentation.common_matting import SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingBatchCollater, load_state_dict

    samdataset = SAMMattingDataset(
        interactive_segmentation_dataset_path,
        set_name=[
            'DIS5K',
            'sa_000020',
        ],
        set_type='train',
        max_side=2048,
        kernel_size_range=[10, 15],
        per_set_image_choose_max_num={
            'DIS5K': 10000000,
            'sa_000020': 1000000,
        },
        per_image_mask_chosse_max_num=1,
        positive_points_num=9,
        negative_points_num=9,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        resample_num=1,
        transform=transforms.Compose([
            SAMMattingResize(resize=1024),
            SAMMattingRandomHorizontalFlip(prob=0.5),
            SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAMMattingBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from simpleAICV.interactive_segmentation.models.segment_anything_matting.sam_matting1 import sam_b_matting1

    net = sam_b_matting1(image_size=1024,
                         frozen_image_encoder=False,
                         frozen_prompt_encoder=False,
                         frozen_mask_decoder=False,
                         use_gradient_checkpoint=True)
    load_state_dict('', net)

    loss = SAMMattingOneLevelLoss(global_pred_trimap_ce_loss_weight=1,
                                  gloabel_pred_trimap_iou_loss_weight=1,
                                  local_pred_alpha_loss_weight=1,
                                  local_pred_laplacian_loss_weight=1,
                                  fusion_pred_alpha_loss_weight=1,
                                  fusion_pred_laplacian_loss_weight=1,
                                  composition_loss_weight=1,
                                  fused_pred_iou_predict_loss_weight=1,
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

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = preds
        print('3333', batch_masks_global_preds.shape,
              batch_masks_local_preds.shape, batch_masks_fused_preds.shape,
              batch_iou_preds.shape)

        loss_dict = loss(input_images, [
            batch_masks_global_preds,
            batch_masks_local_preds,
            batch_masks_fused_preds,
            batch_iou_preds,
        ], [input_masks, trimaps, fg_maps, bg_maps])
        print('4444', loss_dict)

        break

    from simpleAICV.interactive_segmentation.models.segment_anything_matting.sam_matting2 import sam_b_matting2

    net = sam_b_matting2(image_size=1024,
                         frozen_image_encoder=False,
                         frozen_prompt_encoder=False,
                         frozen_mask_decoder=False,
                         use_gradient_checkpoint=True)
    load_state_dict('', net)

    loss = SAMMattingMultiLevelLoss(global_pred_trimap_ce_loss_weight=1,
                                    gloabel_pred_trimap_iou_loss_weight=1,
                                    local_pred_alpha_loss_weight=1,
                                    local_pred_laplacian_loss_weight=1,
                                    fusion_pred_alpha_loss_weight=1,
                                    fusion_pred_laplacian_loss_weight=1,
                                    composition_loss_weight=1,
                                    fused_pred_iou_predict_loss_weight=1,
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

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = preds
        print('3333', batch_masks_global_preds.shape,
              batch_masks_local_preds.shape, batch_masks_fused_preds.shape,
              batch_iou_preds.shape)

        loss_dict = loss(input_images, [
            batch_masks_global_preds,
            batch_masks_local_preds,
            batch_masks_fused_preds,
            batch_iou_preds,
        ], [input_masks, trimaps, fg_maps, bg_maps])
        print('4444', loss_dict)

        break

    from simpleAICV.interactive_segmentation.models.segment_anything_matting.sam_matting2 import sam_b_matting2

    net = sam_b_matting2(image_size=1024,
                         frozen_image_encoder=False,
                         frozen_prompt_encoder=False,
                         frozen_mask_decoder=False,
                         use_gradient_checkpoint=True)
    load_state_dict('', net)

    loss = SAMMattingMultiLevelIoUMaxLoss(
        global_pred_trimap_ce_loss_weight=1,
        gloabel_pred_trimap_iou_loss_weight=1,
        local_pred_alpha_loss_weight=1,
        local_pred_laplacian_loss_weight=1,
        fusion_pred_alpha_loss_weight=1,
        fusion_pred_laplacian_loss_weight=1,
        composition_loss_weight=1,
        fused_pred_iou_predict_loss_weight=1,
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

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = preds
        print('3333', batch_masks_global_preds.shape,
              batch_masks_local_preds.shape, batch_masks_fused_preds.shape,
              batch_iou_preds.shape)

        loss_dict = loss(input_images, [
            batch_masks_global_preds,
            batch_masks_local_preds,
            batch_masks_fused_preds,
            batch_iou_preds,
        ], [input_masks, trimaps, fg_maps, bg_maps])
        print('4444', loss_dict)

        break

    from simpleAICV.interactive_segmentation.models.segment_anything_matting.sam_matting2 import sam_b_matting2

    net = sam_b_matting2(image_size=1024,
                         frozen_image_encoder=False,
                         frozen_prompt_encoder=False,
                         frozen_mask_decoder=False,
                         use_gradient_checkpoint=True)
    load_state_dict('', net)

    loss = SAMMattingMultiLevelAssignLoss(
        global_pred_trimap_ce_loss_weight=1,
        gloabel_pred_trimap_iou_loss_weight=1,
        local_pred_alpha_loss_weight=1,
        local_pred_laplacian_loss_weight=1,
        fusion_pred_alpha_loss_weight=1,
        fusion_pred_laplacian_loss_weight=1,
        composition_loss_weight=1,
        fused_pred_iou_predict_loss_weight=1,
        mask_threshold=0.5,
        idx_nums=4,
        area_ranges=[[0.04, 0.64], [0.0, 0.04], [0.01, 0.25], [0.16, 1.0]])

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

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = preds
        print('3333', batch_masks_global_preds.shape,
              batch_masks_local_preds.shape, batch_masks_fused_preds.shape,
              batch_iou_preds.shape)

        loss_dict = loss(input_images, [
            batch_masks_global_preds,
            batch_masks_local_preds,
            batch_masks_fused_preds,
            batch_iou_preds,
        ], [input_masks, trimaps, fg_maps, bg_maps])
        print('4444', loss_dict)

        break
