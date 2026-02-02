import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'SAM2Loss',
    'SAM2MultiLevelLoss',
]


class SAM2Loss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 class_loss_weight=1,
                 supervise_all_iou=True,
                 mask_threshold=0.):

        super(SAM2Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.class_loss_weight = class_loss_weight

        self.supervise_all_iou = supervise_all_iou
        self.mask_threshold = mask_threshold

    def forward(self, all_frame_preds, targets):
        all_frame_mask_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = all_frame_preds

        assert len(all_frame_mask_preds) == len(all_frame_iou_preds) == len(
            all_frame_pred_object_score_logits) == len(targets)

        focal_loss = 0.
        dice_loss = 0.
        iou_predict_loss = 0.
        cls_loss = 0.
        frame_num = len(all_frame_mask_preds)

        for per_frame_mask_preds, per_frame_iou_preds, per_frame_pred_object_score_logits, per_frame_target_mask in zip(
                all_frame_mask_preds, all_frame_iou_preds,
                all_frame_pred_object_score_logits, targets):
            assert len(per_frame_mask_preds) == len(
                per_frame_iou_preds) == len(per_frame_pred_object_score_logits)

            per_frame_focal_loss = 0.
            per_frame_dice_loss = 0.
            per_frame_iou_predict_loss = 0.
            per_frame_cls_loss = 0.
            iter_num = len(per_frame_mask_preds)

            for per_frame_iter_mask_preds, per_frame_iter_iou_preds, per_frame_iter_pred_object_score_logits in zip(
                    per_frame_mask_preds, per_frame_iou_preds,
                    per_frame_pred_object_score_logits):
                per_frame_iter_focal_loss, per_frame_iter_dice_loss, per_frame_iter_iou_predict_loss, per_frame_iter_cls_loss = self.compute_per_frame_iter_loss(
                    per_frame_iter_mask_preds, per_frame_iter_iou_preds,
                    per_frame_iter_pred_object_score_logits,
                    per_frame_target_mask)

                per_frame_focal_loss += per_frame_iter_focal_loss
                per_frame_dice_loss += per_frame_iter_dice_loss
                per_frame_iou_predict_loss += per_frame_iter_iou_predict_loss
                per_frame_cls_loss += per_frame_iter_cls_loss

            per_frame_focal_loss = per_frame_focal_loss / float(iter_num)
            per_frame_dice_loss = per_frame_dice_loss / float(iter_num)
            per_frame_iou_predict_loss = per_frame_iou_predict_loss / float(
                iter_num)
            per_frame_cls_loss = per_frame_cls_loss / float(iter_num)

            focal_loss += per_frame_focal_loss
            dice_loss += per_frame_dice_loss
            iou_predict_loss += per_frame_iou_predict_loss
            cls_loss += per_frame_cls_loss

        focal_loss = focal_loss / float(frame_num)
        dice_loss = dice_loss / float(frame_num)
        iou_predict_loss = iou_predict_loss / float(frame_num)
        cls_loss = cls_loss / float(frame_num)

        focal_loss = focal_loss * self.focal_loss_weight
        dice_loss = dice_loss * self.dice_loss_weight
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight
        cls_loss = cls_loss * self.class_loss_weight

        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'iou_predict_loss': iou_predict_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict

    def compute_per_frame_iter_loss(self, per_frame_iter_mask_preds,
                                    per_frame_iter_iou_preds,
                                    per_frame_iter_pred_object_score_logits,
                                    per_frame_target_mask):
        # 使 per_frame_target_mask 维度与 per_frame_iter_mask_preds 对齐: [N, M, H, W]
        # M为pred预测每一个object输出的mask数量
        # per_frame_iter_mask_preds: torch.Size([4, 3, 1024, 1024])
        # per_frame_iter_iou_preds: torch.Size([4, 3])
        # per_frame_iter_pred_object_score_logits: torch.Size([4, 1])
        # per_frame_target_mask: torch.Size([4, 3, 1024, 1024])
        per_frame_target_mask = torch.unsqueeze(per_frame_target_mask, dim=1)
        per_frame_target_mask = per_frame_target_mask.expand_as(
            per_frame_iter_mask_preds)

        # focal_loss: torch.Size([4, 3])
        # dice_loss: torch.Size([4, 3])
        # iou_predict_loss: torch.Size([4, 3])
        # cls_loss: torch.Size([])
        # target_object: torch.Size([4, 1])
        focal_loss = self.focal_loss(per_frame_iter_mask_preds,
                                     per_frame_target_mask)
        dice_loss = self.dice_loss(per_frame_iter_mask_preds,
                                   per_frame_target_mask)
        iou_predict_loss = self.iou_predict_loss(per_frame_iter_mask_preds,
                                                 per_frame_target_mask,
                                                 per_frame_iter_iou_preds)

        # target_obj: 是否出现物体
        # per_frame_target_mask[:, 0],0这个维度是为了对齐pred加的,实际上target_mask每个object只有一个mask
        target_object = torch.unsqueeze(torch.any((per_frame_target_mask[:, 0]
                                                   > 0).flatten(1),
                                                  dim=-1),
                                        dim=-1).float()
        cls_loss = self.cls_loss(per_frame_iter_pred_object_score_logits,
                                 target_object)

        # 如果预测了多个mask,只对最优mask回传focal loss和dice loss；
        # 但对于iou_predict_loss,多mask全部回传
        if focal_loss.shape[1] > 1:
            # [N, M], 组合 focal + dice 选最优
            combine_loss = focal_loss * self.focal_loss_weight + dice_loss * self.dice_loss_weight
            best_index = torch.argmin(combine_loss, dim=-1)
            batch_index = torch.arange(combine_loss.shape[0],
                                       device=combine_loss.device)

            # focal loss和dice loss取最优mask的loss
            focal_loss = focal_loss[batch_index, best_index].unsqueeze(1)
            dice_loss = dice_loss[batch_index, best_index].unsqueeze(1)

            # supervise_all_iou为True, iou_predict_loss多mask全监督, 否则只监督最优那个
            if self.supervise_all_iou:
                iou_predict_loss = iou_predict_loss.mean(dim=-1, keepdim=True)
            else:
                iou_predict_loss = iou_predict_loss[batch_index,
                                                    best_index].unsqueeze(1)

        # focal_loss: torch.Size([4, 1])
        # dice_loss: torch.Size([4, 1])
        # iou_predict_loss: torch.Size([4, 1])
        # target_object: torch.Size([4, 1])
        # 只有物体存在时(target_obj=1)时才回传
        focal_loss = (focal_loss * target_object).sum()
        dice_loss = (dice_loss * target_object).sum()
        iou_predict_loss = (iou_predict_loss * target_object).sum()
        cls_loss = cls_loss

        return focal_loss, dice_loss, iou_predict_loss, cls_loss

    def focal_loss(self, inputs, targets):
        object_nums = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction='none')
        pred_prob = torch.sigmoid(inputs)
        pt = pred_prob * targets + (1 - pred_prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)
        focal_loss = focal_weight * bce_loss

        # focal_loss: torch.Size([4, 3, 1024, 1024])
        focal_loss = focal_loss.flatten(2).mean(dim=-1)
        # focal_loss: torch.Size([4, 3])
        focal_loss = focal_loss / object_nums

        return focal_loss

    def dice_loss(self, inputs, targets):
        object_nums = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()
        inputs = torch.sigmoid(inputs)

        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        intersection = (inputs * targets).sum(dim=-1)

        dice_loss = 1. - ((2. * intersection + 1) /
                          (inputs.sum(dim=-1) + targets.sum(dim=-1) + 1))
        # dice_loss: torch.Size([4, 3])
        dice_loss = dice_loss / object_nums

        return dice_loss

    def iou_predict_loss(self, inputs, targets, pred_ious):
        object_nums = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()

        inputs = (inputs > self.mask_threshold)
        targets = (targets > self.mask_threshold)

        inputs = inputs.flatten(2)
        targets = targets.flatten(2)

        intersection = torch.sum(inputs & targets, dim=-1).float()
        union = torch.sum(inputs | targets, dim=-1).float()
        gt_ious = intersection / torch.clamp(union, min=1e-6)
        gt_ious = torch.clamp(gt_ious, min=0.0, max=1.0)

        iou_predict_loss = F.l1_loss(pred_ious, gt_ious, reduction="none")
        # iou_predict_loss: torch.Size([4, 3])
        iou_predict_loss = iou_predict_loss / object_nums

        return iou_predict_loss

    def cls_loss(self, inputs, targets):
        object_nums = inputs.shape[0]
        inputs = inputs.float()
        cls_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction="none")
        cls_loss = cls_loss.mean(dim=1).sum() / object_nums

        return cls_loss


class SAM2MultiLevelLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 class_loss_weight=1,
                 mask_threshold=0.):

        super(SAM2MultiLevelLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight
        self.class_loss_weight = class_loss_weight

        self.mask_threshold = mask_threshold

    def forward(self, all_frame_preds, targets):
        all_frame_mask_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = all_frame_preds

        assert len(all_frame_mask_preds) == len(all_frame_iou_preds) == len(
            all_frame_pred_object_score_logits) == len(targets)

        focal_loss = 0.
        dice_loss = 0.
        iou_predict_loss = 0.
        cls_loss = 0.
        frame_num = len(all_frame_mask_preds)

        for per_frame_mask_preds, per_frame_iou_preds, per_frame_pred_object_score_logits, per_frame_target_mask in zip(
                all_frame_mask_preds, all_frame_iou_preds,
                all_frame_pred_object_score_logits, targets):
            assert len(per_frame_mask_preds) == len(
                per_frame_iou_preds) == len(per_frame_pred_object_score_logits)

            per_frame_focal_loss = 0.
            per_frame_dice_loss = 0.
            per_frame_iou_predict_loss = 0.
            per_frame_cls_loss = 0.
            iter_num = len(per_frame_mask_preds)

            for per_frame_iter_mask_preds, per_frame_iter_iou_preds, per_frame_iter_pred_object_score_logits in zip(
                    per_frame_mask_preds, per_frame_iou_preds,
                    per_frame_pred_object_score_logits):
                per_frame_iter_focal_loss, per_frame_iter_dice_loss, per_frame_iter_iou_predict_loss, per_frame_iter_cls_loss = self.compute_per_frame_iter_loss(
                    per_frame_iter_mask_preds, per_frame_iter_iou_preds,
                    per_frame_iter_pred_object_score_logits,
                    per_frame_target_mask)

                per_frame_focal_loss += per_frame_iter_focal_loss
                per_frame_dice_loss += per_frame_iter_dice_loss
                per_frame_iou_predict_loss += per_frame_iter_iou_predict_loss
                per_frame_cls_loss += per_frame_iter_cls_loss

            per_frame_focal_loss = per_frame_focal_loss / float(iter_num)
            per_frame_dice_loss = per_frame_dice_loss / float(iter_num)
            per_frame_iou_predict_loss = per_frame_iou_predict_loss / float(
                iter_num)
            per_frame_cls_loss = per_frame_cls_loss / float(iter_num)

            focal_loss += per_frame_focal_loss
            dice_loss += per_frame_dice_loss
            iou_predict_loss += per_frame_iou_predict_loss
            cls_loss += per_frame_cls_loss

        focal_loss = focal_loss / float(frame_num)
        dice_loss = dice_loss / float(frame_num)
        iou_predict_loss = iou_predict_loss / float(frame_num)
        cls_loss = cls_loss / float(frame_num)

        focal_loss = focal_loss * self.focal_loss_weight
        dice_loss = dice_loss * self.dice_loss_weight
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight
        cls_loss = cls_loss * self.class_loss_weight

        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'iou_predict_loss': iou_predict_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict

    def compute_per_frame_iter_loss(self, per_frame_iter_mask_preds,
                                    per_frame_iter_iou_preds,
                                    per_frame_iter_pred_object_score_logits,
                                    per_frame_target_mask):
        # 使 per_frame_target_mask 维度与 per_frame_iter_mask_preds 对齐: [N, M, H, W]
        # M为pred预测每一个object输出的mask数量
        # per_frame_iter_mask_preds: torch.Size([2, 4, 1024, 1024])
        # per_frame_iter_iou_preds: torch.Size([2, 4])
        # per_frame_iter_pred_object_score_logits: torch.Size([2, 1])
        # per_frame_target_mask: torch.Size([2, 4, 1024, 1024]))
        per_frame_target_mask = torch.unsqueeze(per_frame_target_mask, dim=1)
        per_frame_target_mask = per_frame_target_mask.expand_as(
            per_frame_iter_mask_preds)

        # focal_loss: torch.Size([2, 4])
        # dice_loss: torch.Size([2, 4])
        # iou_predict_loss: torch.Size([2, 4])
        # cls_loss: torch.Size([])
        # target_object: torch.Size([2, 1])
        focal_loss = self.focal_loss(per_frame_iter_mask_preds,
                                     per_frame_target_mask)
        dice_loss = self.dice_loss(per_frame_iter_mask_preds,
                                   per_frame_target_mask)
        iou_predict_loss = self.iou_predict_loss(per_frame_iter_mask_preds,
                                                 per_frame_target_mask,
                                                 per_frame_iter_iou_preds)

        # target_obj: 是否出现物体
        # per_frame_target_mask[:, 0],0这个维度是为了对齐pred加的,实际上target_mask每个object只有一个mask
        target_object = torch.unsqueeze(torch.any((per_frame_target_mask[:, 0]
                                                   > 0).flatten(1),
                                                  dim=-1),
                                        dim=-1).float()

        cls_loss = self.cls_loss(per_frame_iter_pred_object_score_logits,
                                 target_object)

        # focal_loss: torch.Size([2, 1])
        # dice_loss: torch.Size([2, 1])
        # iou_predict_loss: torch.Size([2, 1])
        # target_object: torch.Size([2, 1])
        focal_loss = focal_loss.mean(dim=-1, keepdim=True)
        dice_loss = dice_loss.mean(dim=-1, keepdim=True)
        iou_predict_loss = iou_predict_loss.mean(dim=-1, keepdim=True)

        # 只有物体存在时(target_obj=1)时才回传
        focal_loss = (focal_loss * target_object).sum()
        dice_loss = (dice_loss * target_object).sum()
        iou_predict_loss = (iou_predict_loss * target_object).sum()
        cls_loss = cls_loss

        return focal_loss, dice_loss, iou_predict_loss, cls_loss

    def focal_loss(self, inputs, targets):
        object_nums = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction='none')
        pred_prob = torch.sigmoid(inputs)
        pt = pred_prob * targets + (1 - pred_prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)
        focal_loss = focal_weight * bce_loss

        # focal_loss: torch.Size([2, 4, 1024, 1024])
        focal_loss = focal_loss.flatten(2).mean(dim=-1)
        # focal_loss: torch.Size([2, 4])
        focal_loss = focal_loss / object_nums

        return focal_loss

    def dice_loss(self, inputs, targets):
        object_nums = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()
        inputs = torch.sigmoid(inputs)

        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        intersection = (inputs * targets).sum(dim=-1)

        dice_loss = 1. - ((2. * intersection + 1) /
                          (inputs.sum(dim=-1) + targets.sum(dim=-1) + 1))
        # dice_loss: torch.Size([2, 4])
        dice_loss = dice_loss / object_nums

        return dice_loss

    def iou_predict_loss(self, inputs, targets, pred_ious):
        object_nums = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()

        inputs = (inputs > self.mask_threshold)
        targets = (targets > self.mask_threshold)

        inputs = inputs.flatten(2)
        targets = targets.flatten(2)

        intersection = torch.sum(inputs & targets, dim=-1).float()
        union = torch.sum(inputs | targets, dim=-1).float()
        gt_ious = intersection / torch.clamp(union, min=1e-6)
        gt_ious = torch.clamp(gt_ious, min=0.0, max=1.0)

        iou_predict_loss = F.l1_loss(pred_ious, gt_ious, reduction="none")
        # iou_predict_loss: torch.Size([2, 4])
        iou_predict_loss = iou_predict_loss / object_nums

        return iou_predict_loss

    def cls_loss(self, inputs, targets):
        object_nums = inputs.shape[0]
        inputs = inputs.float()
        cls_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction="none")
        cls_loss = cls_loss.mean(dim=1).sum() / object_nums

        return cls_loss


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

    from SimpleAICV.video_interactive_segmentation.datasets.sam2_video_segmentation_dataset import SAM2VideoSegmentationDataset
    from SimpleAICV.video_interactive_segmentation.common import Sam2Resize, Sam2RandomHorizontalFlip, Sam2RandomMosaicAug, Sam2RandomRsverseFrameOrder, Sam2Normalize, SAM2VideoBatchCollater, load_state_dict

    sam2_video_dataset = SAM2VideoSegmentationDataset(
        image_root_dir=interactive_segmentation_dataset_path,
        image_set_name=[
            ###########################################
            'sa_000000',
        ],
        image_set_type='train',
        image_per_set_image_choose_max_num={
            ###########################################
            'sa_000000': 1000000,
        },
        per_image_mask_chosse_max_num=16,
        video_root_dir=video_interactive_segmentation_dataset_path,
        video_set_name=[
            ###########################################
            'sav_000',
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
        per_video_choose_frame_nums=4,
        per_video_choose_object_nums=2,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            Sam2Resize(resize=1024),
            Sam2Normalize(mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAM2VideoBatchCollater(resize=1024, use_image_prob=0.0)
    train_loader = DataLoader(sam2_video_dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from SimpleAICV.video_interactive_segmentation.models.segment_anything2.sam2video_train import hiera_b_plus_sam2video
    net = hiera_b_plus_sam2video(use_gradient_checkpoint=True)

    load_state_dict(
        '/root/autodl-tmp/pretrained_models/sam2.1_convert_from_pytorch_official_weights/sam2.1_hiera_base_plus_convert_from_pytorch_official_weight.pth',
        net)

    loss = SAM2Loss(alpha=0.25,
                    gamma=2,
                    focal_loss_weight=20,
                    dice_loss_weight=1,
                    iou_predict_loss_weight=1,
                    class_loss_weight=1,
                    supervise_all_iou=True,
                    mask_threshold=0.)

    for data in tqdm(train_loader):
        input_images, input_masks = data['image'], data['mask']

        input_prompt_points, input_prompt_boxes, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        object_to_frame_idxs = data['object_to_frame_idx']

        print('1111', input_images.shape, input_masks.shape,
              input_prompt_points.shape, input_prompt_boxes.shape,
              input_prompt_masks.shape, object_to_frame_idxs.shape)

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda()

        net = net.cuda()

        images, masks = data['image'], data['mask']

        outs = net(data)
        loss_dict = loss(outs, masks)
        print('2222', loss_dict)

        break

    # 释放显存
    del loss
    del outs
    del loss_dict
    del data
    torch.cuda.empty_cache()

    loss = SAM2MultiLevelLoss(alpha=0.25,
                              gamma=2,
                              focal_loss_weight=20,
                              dice_loss_weight=1,
                              iou_predict_loss_weight=1,
                              class_loss_weight=1,
                              mask_threshold=0.)

    for data in tqdm(train_loader):
        input_images, input_masks = data['image'], data['mask']

        input_prompt_points, input_prompt_boxes, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        object_to_frame_idxs = data['object_to_frame_idx']

        print('1111', input_images.shape, input_masks.shape,
              input_prompt_points.shape, input_prompt_boxes.shape,
              input_prompt_masks.shape, object_to_frame_idxs.shape)

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda()

        net = net.cuda()

        images, masks = data['image'], data['mask']

        outs = net(data)
        loss_dict = loss(outs, masks)
        print('2222', loss_dict)

        break
