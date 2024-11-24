import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MSELoss',
    'SAMDistillLoss',
]


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, tea_preds, stu_preds):
        loss = self.loss(stu_preds, tea_preds)

        loss_dict = {
            'distill_mse_loss': loss,
        }

        return loss_dict


class SAMDistillLoss(nn.Module):

    def __init__(self,
                 alpha=0.8,
                 gamma=2,
                 smooth=1e-4,
                 distill_focal_loss_weight=20,
                 distill_dice_loss_weight=1,
                 distill_iou_predict_loss_weight=1,
                 mask_threshold=0.0):
        super(SAMDistillLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.distill_focal_loss_weight = distill_focal_loss_weight
        self.distill_dice_loss_weight = distill_dice_loss_weight
        self.distill_iou_predict_loss_weight = distill_iou_predict_loss_weight
        self.mask_threshold = mask_threshold

        self.sigmoid = nn.Sigmoid()

    def forward(self, tea_inputs, stu_inputs, targets):
        tea_pred_masks, tea_pred_ious = tea_inputs
        stu_pred_masks, stu_pred_ious = stu_inputs

        assert tea_pred_masks.shape[1] == tea_pred_ious.shape[1]
        assert stu_pred_masks.shape[1] == stu_pred_ious.shape[1]

        distill_focal_loss = self.distill_focal_loss(stu_pred_masks,
                                                     tea_pred_masks)
        distill_dice_loss = self.distill_dice_loss(stu_pred_masks,
                                                   tea_pred_masks)
        distill_iou_predict_loss = self.distill_iou_predict_loss(
            stu_pred_ious, tea_pred_ious)

        distill_focal_loss = self.distill_focal_loss_weight * distill_focal_loss
        distill_dice_loss = self.distill_dice_loss_weight * distill_dice_loss
        distill_iou_predict_loss = self.distill_iou_predict_loss_weight * distill_iou_predict_loss

        loss_dict = {
            'distill_focal_loss': distill_focal_loss,
            'distill_dice_loss': distill_dice_loss,
            'distill_iou_predict_loss': distill_iou_predict_loss,
        }

        return loss_dict

    def distill_focal_loss(self, stu_preds, tea_preds):
        # stu_preds:torch.Size([1, 4, 1024, 1024])
        # tea_preds:torch.Size([1, 4, 1024, 1024])
        assert stu_preds.shape[1] == tea_preds.shape[1]

        batch_size = stu_preds.shape[0]
        idx_nums = stu_preds.shape[1]

        tea_preds = (tea_preds > self.mask_threshold).float()

        total_distill_focal_loss = 0.
        for per_idx in range(idx_nums):
            per_idx_stu_preds = stu_preds[:, per_idx:per_idx + 1, :, :]
            per_idx_stu_preds = per_idx_stu_preds.reshape(-1)

            per_idx_tea_preds = tea_preds[:, per_idx:per_idx + 1, :, :]
            per_idx_tea_preds = per_idx_tea_preds.reshape(-1)

            assert per_idx_stu_preds.shape[0] == per_idx_tea_preds.shape[0]

            bce_loss = F.binary_cross_entropy_with_logits(per_idx_stu_preds,
                                                          per_idx_tea_preds,
                                                          reduction='none')
            focal_loss = self.alpha * (
                1 - torch.exp(-bce_loss))**self.gamma * bce_loss
            focal_loss = focal_loss.mean()
            total_distill_focal_loss += focal_loss

        total_distill_focal_loss = total_distill_focal_loss / batch_size

        return total_distill_focal_loss

    def distill_dice_loss(self, stu_preds, tea_preds):
        # stu_preds:torch.Size([1, 4, 1024, 1024])
        # tea_preds:torch.Size([1, 4, 1024, 1024])
        assert stu_preds.shape[1] == tea_preds.shape[1]

        batch_size = stu_preds.shape[0]
        idx_nums = stu_preds.shape[1]

        stu_preds = stu_preds.float()
        stu_preds = self.sigmoid(stu_preds)
        stu_preds = torch.clamp(stu_preds, min=1e-4, max=1. - 1e-4)

        tea_preds = (tea_preds > self.mask_threshold).float()

        total_distill_dice_loss = 0.
        for per_idx in range(idx_nums):
            per_idx_stu_preds = stu_preds[:, per_idx:per_idx + 1, :, :]
            per_idx_stu_preds = per_idx_stu_preds.reshape(-1)

            per_idx_tea_preds = tea_preds[:, per_idx:per_idx + 1, :, :]
            per_idx_tea_preds = per_idx_tea_preds.reshape(-1)

            assert per_idx_stu_preds.shape[0] == per_idx_tea_preds.shape[0]

            intersection = (per_idx_stu_preds * per_idx_tea_preds).sum()

            dice_loss = 1. - ((2. * intersection + self.smooth) /
                              (per_idx_stu_preds.sum() +
                               per_idx_tea_preds.sum() + self.smooth))
            total_distill_dice_loss += dice_loss

        total_distill_dice_loss = total_distill_dice_loss / batch_size

        return total_distill_dice_loss

    def distill_iou_predict_loss(self, stu_iou_predictions,
                                 tea_iou_predictions):
        # stu_iou_predictions:torch.Size([1, 4])
        # tea_iou_predictions:torch.Size([1, 4])
        assert stu_iou_predictions.shape[1] == tea_iou_predictions.shape[1]

        batch_size = stu_iou_predictions.shape[0]
        idx_nums = stu_iou_predictions.shape[1]

        total_distill_iou_predict_loss = 0.
        for per_idx in range(idx_nums):
            per_idx_stu_iou_predictions = stu_iou_predictions[:, per_idx]
            per_idx_stu_iou_predictions = per_idx_stu_iou_predictions.reshape(
                -1)

            per_idx_tea_iou_predictions = tea_iou_predictions[:, per_idx]
            per_idx_tea_iou_predictions = per_idx_tea_iou_predictions.reshape(
                -1)

            iou_predict_loss = F.mse_loss(per_idx_stu_iou_predictions,
                                          per_idx_tea_iou_predictions,
                                          reduction='sum')
            total_distill_iou_predict_loss += iou_predict_loss

        total_distill_iou_predict_loss = total_distill_iou_predict_loss / batch_size

        return total_distill_iou_predict_loss


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

    from simpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

    # stu_pred = torch.autograd.Variable(torch.randn(1, 256, 64, 64))
    # tea_pred = torch.autograd.Variable(torch.randn(1, 256, 64, 64))
    # print('1111', stu_pred.shape, tea_pred.shape)

    # loss1 = MSELoss()
    # out = loss1(stu_pred, tea_pred)
    # print('2222', out)

    samdataset = SAMSegmentationDataset(interactive_segmentation_dataset_path,
                                        set_name=[
                                            'sa_000020',
                                        ],
                                        set_type='train',
                                        per_set_image_choose_max_num={
                                            'sa_000020': 1000000,
                                        },
                                        per_image_mask_chosse_max_num=16,
                                        positive_points_num=9,
                                        negative_points_num=9,
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

    collater = SAMBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collater)

    from simpleAICV.interactive_segmentation.distill_model import SAMDistillModel
    net = SAMDistillModel(
        teacher_type='sam_h',
        student_type='sam_b',
        teacher_params={
            'image_size': 1024,
            'use_gradient_checkpoint': False,
            'frozen_image_encoder': True,
            'frozen_prompt_encoder': True,
            'frozen_mask_decoder': True,
            'sigmoid_out': False,
            'binary_mask_out': False,
            'mask_threshold': 0.0,
        },
        student_params={
            'image_size': 1024,
            'use_gradient_checkpoint': False,
            'frozen_image_encoder': False,
            'frozen_prompt_encoder': False,
            'frozen_mask_decoder': False,
            'sigmoid_out': False,
            'binary_mask_out': False,
            'mask_threshold': 0.0,
        },
        teacher_pretrained_path=
        '/root/autodl-tmp/pretrained_models/sam_official_pytorch_weights/sam_vit_h_4b8939.pth',
        student_pretrained_path=
        '/root/autodl-tmp/pretrained_models/sam_official_pytorch_weights/sam_vit_b_01ec64.pth',
        freeze_teacher=True)

    loss2 = SAMDistillLoss(alpha=0.8,
                           gamma=2,
                           smooth=1e-4,
                           distill_focal_loss_weight=20,
                           distill_dice_loss_weight=1,
                           distill_iou_predict_loss_weight=1,
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

        tea_preds, stu_preds = net(input_images,
                                   input_prompts,
                                   mask_out_idxs=[0, 1, 2, 3])

        print('3333', tea_preds[0].shape, tea_preds[1].shape,
              tea_preds[0].dtype, tea_preds[1].dtype)

        print('4444', stu_preds[0].shape, stu_preds[1].shape,
              stu_preds[0].dtype, stu_preds[1].dtype)

        loss_dict = loss2(tea_preds, stu_preds, input_masks)
        print('5555', loss_dict)

        break
