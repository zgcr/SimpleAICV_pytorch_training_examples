import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'SAMLoss',
    'SAMMultiLevelLoss',
]


class SAMLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 supervise_all_iou=True,
                 mask_threshold=0.):

        super(SAMLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight

        self.supervise_all_iou = supervise_all_iou
        self.mask_threshold = mask_threshold

    def forward(self, all_iter_preds, targets):
        # all_iter_mask_preds: iter_num, torch.Size([4, 4, 1024, 1024])
        # all_iter_iou_preds: iter_num, torch.Size([4, 4])
        # targets: torch.Size([4, 1, 1024, 1024])
        all_iter_mask_preds, all_iter_iou_preds = all_iter_preds

        assert len(all_iter_mask_preds) == len(all_iter_iou_preds)

        focal_loss = 0.
        dice_loss = 0.
        iou_predict_loss = 0.
        iter_num = len(all_iter_mask_preds)
        for per_iter_mask_preds, per_iter_iou_preds in zip(
                all_iter_mask_preds, all_iter_iou_preds):
            # per_iter_mask_preds: torch.Size([4, 4, 1024, 1024])
            # per_iter_iou_preds: torch.Size([4, 4])

            per_iter_focal_loss, per_iter_dice_loss, per_iter_iou_predict_loss = self.compute_per_iter_loss(
                per_iter_mask_preds,
                per_iter_iou_preds,
                targets,
            )

            focal_loss += per_iter_focal_loss
            dice_loss += per_iter_dice_loss
            iou_predict_loss += per_iter_iou_predict_loss

        focal_loss = focal_loss / float(iter_num)
        dice_loss = dice_loss / float(iter_num)
        iou_predict_loss = iou_predict_loss / float(iter_num)

        focal_loss = focal_loss * self.focal_loss_weight
        dice_loss = dice_loss * self.dice_loss_weight
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight

        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def compute_per_iter_loss(
        self,
        per_iter_mask_preds,
        per_iter_iou_preds,
        targets,
    ):
        # 使 targets 维度与 per_iter_mask_preds 对齐: [B, M, H, W]
        # M为pred预测每一个object输出的mask数量
        # per_iter_mask_preds: torch.Size([4, 4, 1024, 1024])
        # per_iter_iou_preds: torch.Size([4, 4])
        # targets: torch.Size([4, 4, 1024, 1024])
        targets = targets.expand_as(per_iter_mask_preds)

        # focal_loss: torch.Size([4, 4])
        # dice_loss: torch.Size([4, 4])
        # iou_predict_loss: torch.Size([4, 4])
        focal_loss = self.focal_loss(per_iter_mask_preds, targets)
        dice_loss = self.dice_loss(per_iter_mask_preds, targets)
        iou_predict_loss = self.iou_predict_loss(per_iter_mask_preds, targets,
                                                 per_iter_iou_preds)

        # 如果预测了多个mask,只对最优mask回传focal loss和dice loss；
        # 但对于iou_predict_loss,多mask全部回传
        if focal_loss.shape[1] > 1:
            # [B, M], 组合 focal + dice 选最优
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
        focal_loss = focal_loss.sum()
        dice_loss = dice_loss.sum()
        iou_predict_loss = iou_predict_loss.sum()

        return focal_loss, dice_loss, iou_predict_loss

    def focal_loss(self, inputs, targets):
        batch_size = inputs.shape[0]

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

        # h,w维度合成一维然后mean
        focal_loss = focal_loss.flatten(2).mean(dim=-1)
        # batch_size维度上平均,但保留维度
        focal_loss = focal_loss / batch_size

        return focal_loss

    def dice_loss(self, inputs, targets):
        batch_size = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()
        inputs = torch.sigmoid(inputs)

        # h,w维度合成一维
        # inputs: torch.Size([4, 4, h*w])
        # targets: torch.Size([4, 4, h*w])
        # intersection: torch.Size([4, 4])
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        intersection = (inputs * targets).sum(dim=-1)

        # dice_loss: torch.Size([4, 4])
        dice_loss = 1. - ((2. * intersection + 1) /
                          (inputs.sum(dim=-1) + targets.sum(dim=-1) + 1))
        # batch_size维度上平均,但保留维度
        dice_loss = dice_loss / batch_size

        return dice_loss

    def iou_predict_loss(self, inputs, targets, pred_ious):
        batch_size = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()

        inputs = (inputs > self.mask_threshold)
        targets = (targets > self.mask_threshold)

        # h,w维度合成一维
        # inputs: torch.Size([4, 4, h*w])
        # targets: torch.Size([4, 4, h*w])
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)

        # intersection: torch.Size([4, 4])
        # union: torch.Size([4, 4])
        # gt_ious: torch.Size([4, 4])
        intersection = torch.sum(inputs & targets, dim=-1).float()
        union = torch.sum(inputs | targets, dim=-1).float()
        gt_ious = intersection / torch.clamp(union, min=1e-6)
        gt_ious = torch.clamp(gt_ious, min=0.0, max=1.0)

        # iou_predict_loss: torch.Size([4, 4])
        iou_predict_loss = F.mse_loss(pred_ious, gt_ious, reduction="none")
        # batch_size维度上平均,但保留维度
        iou_predict_loss = iou_predict_loss / batch_size

        return iou_predict_loss


class SAMMultiLevelLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 focal_loss_weight=20,
                 dice_loss_weight=1,
                 iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAMMultiLevelLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_predict_loss_weight = iou_predict_loss_weight

        self.mask_threshold = mask_threshold

    def forward(self, all_iter_preds, targets):
        # all_iter_mask_preds: iter_num, torch.Size([4, 4, 1024, 1024])
        # all_iter_iou_preds: iter_num, torch.Size([4, 4])
        # targets: torch.Size([4, 1, 1024, 1024])
        all_iter_mask_preds, all_iter_iou_preds = all_iter_preds

        assert len(all_iter_mask_preds) == len(all_iter_iou_preds)

        focal_loss = 0.
        dice_loss = 0.
        iou_predict_loss = 0.
        iter_num = len(all_iter_mask_preds)
        for per_iter_mask_preds, per_iter_iou_preds in zip(
                all_iter_mask_preds, all_iter_iou_preds):
            # per_iter_mask_preds: torch.Size([4, 4, 1024, 1024])
            # per_iter_iou_preds: torch.Size([4, 4])

            per_iter_focal_loss, per_iter_dice_loss, per_iter_iou_predict_loss = self.compute_per_iter_loss(
                per_iter_mask_preds,
                per_iter_iou_preds,
                targets,
            )

            focal_loss += per_iter_focal_loss
            dice_loss += per_iter_dice_loss
            iou_predict_loss += per_iter_iou_predict_loss

        focal_loss = focal_loss / float(iter_num)
        dice_loss = dice_loss / float(iter_num)
        iou_predict_loss = iou_predict_loss / float(iter_num)

        focal_loss = focal_loss * self.focal_loss_weight
        dice_loss = dice_loss * self.dice_loss_weight
        iou_predict_loss = iou_predict_loss * self.iou_predict_loss_weight

        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'iou_predict_loss': iou_predict_loss,
        }

        return loss_dict

    def compute_per_iter_loss(
        self,
        per_iter_mask_preds,
        per_iter_iou_preds,
        targets,
    ):
        # 使 targets 维度与 per_iter_mask_preds 对齐: [B, M, H, W]
        # M为pred预测每一个object输出的mask数量
        # per_iter_mask_preds: torch.Size([4, 4, 1024, 1024])
        # per_iter_iou_preds: torch.Size([4, 4])
        # targets: torch.Size([4, 4, 1024, 1024])
        targets = targets.expand_as(per_iter_mask_preds)

        # focal_loss: torch.Size([4, 4])
        # dice_loss: torch.Size([4, 4])
        # iou_predict_loss: torch.Size([4, 4])
        focal_loss = self.focal_loss(per_iter_mask_preds, targets)
        dice_loss = self.dice_loss(per_iter_mask_preds, targets)
        iou_predict_loss = self.iou_predict_loss(per_iter_mask_preds, targets,
                                                 per_iter_iou_preds)

        # focal_loss: torch.Size([4, 1])
        # dice_loss: torch.Size([4, 1])
        # iou_predict_loss: torch.Size([4, 1])
        focal_loss = focal_loss.mean(dim=-1, keepdim=True)
        dice_loss = dice_loss.mean(dim=-1, keepdim=True)
        iou_predict_loss = iou_predict_loss.mean(dim=-1, keepdim=True)

        # focal_loss: torch.Size([4, 1])
        # dice_loss: torch.Size([4, 1])
        # iou_predict_loss: torch.Size([4, 1])
        focal_loss = focal_loss.sum()
        dice_loss = dice_loss.sum()
        iou_predict_loss = iou_predict_loss.sum()

        return focal_loss, dice_loss, iou_predict_loss

    def focal_loss(self, inputs, targets):
        batch_size = inputs.shape[0]

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
        focal_loss = focal_loss / batch_size

        return focal_loss

    def dice_loss(self, inputs, targets):
        batch_size = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()
        inputs = torch.sigmoid(inputs)

        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        intersection = (inputs * targets).sum(dim=-1)

        dice_loss = 1. - ((2. * intersection + 1) /
                          (inputs.sum(dim=-1) + targets.sum(dim=-1) + 1))
        # dice_loss: torch.Size([2, 4])
        dice_loss = dice_loss / batch_size

        return dice_loss

    def iou_predict_loss(self, inputs, targets, pred_ious):
        batch_size = inputs.shape[0]

        inputs = inputs.float()
        targets = targets.float()

        inputs = (inputs > self.mask_threshold)
        targets = (targets > self.mask_threshold)

        # h,w维度合成一维
        # inputs: torch.Size([4, 4, h*w])
        # targets: torch.Size([4, 4, h*w])
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)

        # intersection: torch.Size([4, 4])
        # union: torch.Size([4, 4])
        # gt_ious: torch.Size([4, 4])
        intersection = torch.sum(inputs & targets, dim=-1).float()
        union = torch.sum(inputs | targets, dim=-1).float()
        gt_ious = intersection / torch.clamp(union, min=1e-6)
        gt_ious = torch.clamp(gt_ious, min=0.0, max=1.0)

        # iou_predict_loss: torch.Size([4, 4])
        iou_predict_loss = F.mse_loss(pred_ious, gt_ious, reduction="none")
        iou_predict_loss = iou_predict_loss / batch_size

        return iou_predict_loss


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

    from SimpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
    from SimpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

    samdataset = SAMSegmentationDataset(interactive_segmentation_dataset_path,
                                        set_name=[
                                            'DIS5K',
                                            'sa_000000',
                                        ],
                                        set_type='train',
                                        per_set_image_choose_max_num={
                                            'DIS5K': 1000000,
                                            'sa_000000': 1000000,
                                        },
                                        per_image_mask_chosse_max_num=16,
                                        points_num=1,
                                        area_filter_ratio=0.0001,
                                        box_noise_wh_ratio=0.1,
                                        mask_noise_area_ratio=0.04,
                                        transform=transforms.Compose([
                                            SamResize(resize=1024),
                                            SamRandomHorizontalFlip(prob=0.5),
                                            SamNormalize(
                                                mean=[123.675, 116.28, 103.53],
                                                std=[58.395, 57.12, 57.375]),
                                        ]))

    from torch.utils.data import DataLoader

    collater = SAMBatchCollater(resize=1024)
    train_loader = DataLoader(samdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.interactive_segmentation.models.segment_anything.sam import sam_b
    net = sam_b(use_gradient_checkpoint=True)
    load_state_dict(
        '/root/autodl-tmp/pretrained_models/sam_pytorch_official_weights/sam_vit_b_01ec64.pth',
        net)

    loss = SAMLoss(alpha=0.25,
                   gamma=2,
                   focal_loss_weight=20,
                   dice_loss_weight=1,
                   iou_predict_loss_weight=1,
                   supervise_all_iou=True,
                   mask_threshold=0.)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

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

        batch_image_embeddings = net.forward_image_encoder(input_images)

        print('3333', batch_image_embeddings.shape)

        mask_preds, iou_preds = net.forward_prompt_encoder_mask_decoder(
            batch_image_embeddings, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('4444', mask_preds.shape, iou_preds.shape, mask_preds.dtype,
              iou_preds.dtype)

        all_iter_mask_preds, all_iter_iou_preds = [mask_preds], [iou_preds]

        print('5555', len(all_iter_mask_preds), len(all_iter_iou_preds))

        for per_iter_mask_preds, per_iter_iou_preds in zip(
                all_iter_mask_preds, all_iter_iou_preds):
            print('6666', per_iter_mask_preds.shape, per_iter_iou_preds.shape)

        loss_dict = loss([all_iter_mask_preds, all_iter_iou_preds],
                         input_masks)
        print('7777', loss_dict)

        break

    loss = SAMMultiLevelLoss(alpha=0.25,
                             gamma=2,
                             focal_loss_weight=20,
                             dice_loss_weight=1,
                             iou_predict_loss_weight=1,
                             mask_threshold=0.0)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

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

        batch_image_embeddings = net.forward_image_encoder(input_images)

        print('3333', batch_image_embeddings.shape)

        mask_preds, iou_preds = net.forward_prompt_encoder_mask_decoder(
            batch_image_embeddings, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('4444', mask_preds.shape, iou_preds.shape, mask_preds.dtype,
              iou_preds.dtype)

        all_iter_mask_preds, all_iter_iou_preds = [mask_preds], [iou_preds]

        print('5555', len(all_iter_mask_preds), len(all_iter_iou_preds))

        for per_iter_mask_preds, per_iter_iou_preds in zip(
                all_iter_mask_preds, all_iter_iou_preds):
            print('6666', per_iter_mask_preds.shape, per_iter_iou_preds.shape)

        loss_dict = loss([all_iter_mask_preds, all_iter_iou_preds],
                         input_masks)
        print('7777', loss_dict)

        break
