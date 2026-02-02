"""
https://github.com/huggingface/transformers/blob/main/src/transformers/models/mask2former/modeling_mask2former.py
https://github.com/tue-mps/eomt/blob/985630d27fc5adc05afd1a674c0fc0fb307ef928/training/mask_classification_loss.py
"""
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Mask2FormerLoss',
]


class Mask2FormerHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the labels and the predictions of the network.
    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 point_nums=16384,
                 mask_cost=1.0,
                 dice_cost=1.0,
                 class_cost=1.0):
        super(Mask2FormerHungarianMatcher, self).__init__()
        self.point_nums = point_nums
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost
        self.class_cost = class_cost

        self.sigmoid_ce_loss = nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def forward(self, mask_preds, class_preds, mask_gts, class_gts):
        # # num_classes has background class
        # mask_preds:[batch_size, query_nums, height, width]
        # class_preds:[batch_size, query_nums, num_classes]
        # mask_gts[0]:[mask_nums, height, width]
        # class_gts[0]:[mask_nums]

        device = mask_preds.device
        batch_size = mask_preds.shape[0]
        indices = []
        for i in range(batch_size):
            pred_probs = class_preds[i].softmax(dim=-1)
            pred_mask = mask_preds[i]

            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be omitted.
            class_cost = -pred_probs[:, class_gts[i]]
            target_mask = mask_gts[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            # Sample ground truth and predicted masks
            point_coordinates = torch.rand(1, self.point_nums, 2).to(device)

            target_coordinates = point_coordinates.repeat(
                target_mask.shape[0], 1, 1)
            target_mask = self.sample_point(target_mask,
                                            target_coordinates).squeeze(1)

            pred_coordinates = point_coordinates.repeat(
                pred_mask.shape[0], 1, 1)
            pred_mask = self.sample_point(pred_mask,
                                          pred_coordinates).squeeze(1)

            # compute the cross entropy loss between each mask pairs -> shape (query_nums, num_labels)
            mask_cost = self.compute_pair_wise_sigmoid_cross_entropy_loss(
                pred_mask, target_mask)
            # Compute the dice loss between each mask pairs -> shape (query_nums, num_labels)
            dice_cost = self.compute_pair_wise_dice_loss(
                pred_mask, target_mask)

            # final cost matrix
            cost_matrix = self.mask_cost * mask_cost + self.dice_cost * dice_cost + self.class_cost * class_cost
            # eliminate infinite values in cost_matrix to avoid the error ``ValueError: cost matrix is infeasible``
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            # do the assignment using the hungarian algorithm in scipy
            assigned_indices = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [(torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64))
                           for i, j in indices]

        # matched_indices (`list[tuple[Tensor]]`):
        # A list of size batch_size, containing tuples of (index_i, index_j)
        # index_i is the indices of the selected predictions (in order)
        # index_j is the indices of the corresponding selected labels (in order)
        # For each batch element, it holds:
        #     len(index_i) = len(index_j) = min(query_nums, target_box_nums).

        return matched_indices

    def compute_pair_wise_sigmoid_cross_entropy_loss(self, pred_mask,
                                                     target_mask):
        height_and_width = pred_mask.shape[1]

        cross_entropy_loss_pos = self.sigmoid_ce_loss(
            pred_mask, torch.ones_like(pred_mask))
        cross_entropy_loss_neg = self.sigmoid_ce_loss(
            pred_mask, torch.zeros_like(pred_mask))

        loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width,
                                target_mask.T)
        loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width,
                                (1 - target_mask).T)
        loss = loss_pos + loss_neg

        return loss

    def compute_pair_wise_dice_loss(self, pred_mask, target_mask):
        pred_mask = torch.sigmoid(pred_mask)
        numerator = 2 * torch.matmul(pred_mask, target_mask.T)
        denominator = pred_mask.sum(dim=-1)[:, None] + target_mask.sum(
            dim=-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)

        return loss

    def sample_point(self, input_features, point_coordinates):
        point_coordinates = point_coordinates.unsqueeze(2)

        # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
        point_features = F.grid_sample(input_features,
                                       2.0 * point_coordinates - 1.0)
        point_features = point_features.squeeze(3)

        return point_features


class Mask2FormerLoss(nn.Module):

    def __init__(self,
                 point_nums=16384,
                 oversample_ratio=3.0,
                 importance_sample_ratio=0.75,
                 mask_cost=5.0,
                 dice_cost=5.0,
                 class_cost=2.0,
                 num_classes=151,
                 mask_loss_weight=5.0,
                 dice_loss_weight=5.0,
                 class_loss_weight=2.0,
                 no_object_class_weight=0.1):
        super(Mask2FormerLoss, self).__init__()
        self.point_nums = point_nums
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        # num_classes has background class
        self.num_classes = num_classes

        self.mask_loss_weight = mask_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.class_loss_weight = class_loss_weight

        self.hungarian_matcher = Mask2FormerHungarianMatcher(
            point_nums=point_nums,
            mask_cost=mask_cost,
            dice_cost=dice_cost,
            class_cost=class_cost)

        self.sigmoid_ce_loss = nn.BCEWithLogitsLoss(reduction="none")

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

    def sample_point(self, input_features, point_coordinates):
        point_coordinates = point_coordinates.unsqueeze(2)

        # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
        point_features = F.grid_sample(input_features,
                                       2.0 * point_coordinates - 1.0)
        point_features = point_features.squeeze(3)

        return point_features

    def sample_points_using_uncertainty(self, pred_masks, point_nums,
                                        oversample_ratio,
                                        importance_sample_ratio):
        device = pred_masks.device

        box_nums = pred_masks.shape[0]
        sampled_point_nums = int(point_nums * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(box_nums, sampled_point_nums,
                                       2).to(device)
        # Get sampled prediction value for the point coordinates
        point_preds = self.sample_point(pred_masks, point_coordinates)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = -(torch.abs(point_preds))

        uncertain_point_nums = int(importance_sample_ratio * point_nums)
        random_point_nums = point_nums - uncertain_point_nums

        idx = torch.topk(point_uncertainties[:, 0, :],
                         k=uncertain_point_nums,
                         dim=1)[1]
        shift = sampled_point_nums * torch.arange(box_nums,
                                                  dtype=torch.long).to(device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(
            -1, 2)[idx.view(-1), :].view(box_nums, uncertain_point_nums, 2)

        if random_point_nums > 0:
            point_coordinates = torch.cat([
                point_coordinates,
                torch.rand(box_nums, random_point_nums, 2).to(device)
            ],
                                          dim=1)

        return point_coordinates

    def get_point_preds_and_point_targets(self, mask_preds, mask_gts, indices):
        device = mask_preds.device

        pred_idx = self.get_pred_permutation_indices(indices)
        target_idx = self.get_target_permutation_indices(indices)

        pred_masks = mask_preds[pred_idx]

        batch_size, batch_max_object_num, max_height, max_width = len(
            mask_gts), 0, 0, 0
        for per_image_mask_gts in mask_gts:
            object_num, height, width = per_image_mask_gts.shape
            batch_max_object_num = max(batch_max_object_num, object_num)
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        target_masks = torch.zeros(
            [batch_size, batch_max_object_num, max_height, max_width],
            dtype=torch.float32).to(device)
        for idx, per_image_mask_gts in enumerate(mask_gts):
            target_masks[
                idx, :per_image_mask_gts.shape[0], :per_image_mask_gts.
                shape[1], :per_image_mask_gts.shape[2]] = per_image_mask_gts
        target_masks = target_masks[target_idx]

        pred_masks = pred_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1)

        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks, self.point_nums, self.oversample_ratio,
                self.importance_sample_ratio)
            # [mask_nums, point_nums]
            point_targets = self.sample_point(target_masks,
                                              point_coordinates).squeeze(1)
        # [mask_nums, point_nums]
        point_preds = self.sample_point(pred_masks,
                                        point_coordinates).squeeze(1)

        return point_preds, point_targets

    def compute_batch_mask_loss(self, point_preds, point_targets):
        # point_preds:[mask_nums, point_nums]
        # point_targets:[mask_nums, point_nums]

        # [mask_nums, point_nums]
        mask_loss = self.sigmoid_ce_loss(point_preds, point_targets)
        mask_loss = mask_loss.mean()

        return mask_loss

    def compute_batch_dice_loss(self, point_preds, point_targets):
        # point_preds:[mask_nums, point_nums]
        # point_targets:[mask_nums, point_nums]

        # [mask_nums, point_nums]
        point_preds = torch.sigmoid(point_preds)
        # [mask_nums]
        numerator = 2 * (point_preds * point_targets).sum(dim=-1)
        # [mask_nums]
        denominator = point_preds.sum(dim=-1) + point_targets.sum(dim=-1)
        # [mask_nums]
        dice_loss = 1 - (numerator + 1) / (denominator + 1)
        dice_loss = dice_loss.mean()

        return dice_loss

    def compute_batch_class_loss(self, class_preds, class_gts, indices):
        device = class_preds.device
        batch_size, query_nums = class_preds.shape[0], class_preds.shape[1]

        idx = self.get_pred_permutation_indices(indices)
        # [batch_size, query_nums]
        # self.num_classes - 1 is background class index
        class_targets = torch.full((batch_size, query_nums),
                                   fill_value=self.num_classes - 1,
                                   dtype=torch.int64,
                                   device=device)
        class_target_objects = torch.cat(
            [target[j] for target, (_, j) in zip(class_gts, indices)])
        class_targets[idx] = class_target_objects

        # [batch_size, query_nums, num_classes] -> [batch_size, num_classes, query_nums]
        class_preds = class_preds.transpose(1, 2)
        class_loss = self.ce_loss(class_preds, class_targets)

        return class_loss

    def forward(self, mask_preds, class_preds, mask_gts, class_gts):
        mask_preds = mask_preds.float()
        class_preds = class_preds.float()

        device = mask_preds.device

        mask_gts = [
            per_image_mask_gts.float().to(device)
            for per_image_mask_gts in mask_gts
        ]
        class_gts = [
            per_image_class_gts.long().to(device)
            for per_image_class_gts in class_gts
        ]

        indices = self.hungarian_matcher(mask_preds=mask_preds,
                                         mask_gts=mask_gts,
                                         class_preds=class_preds,
                                         class_gts=class_gts)

        point_preds, point_targets = self.get_point_preds_and_point_targets(
            mask_preds, mask_gts, indices)

        mask_loss = self.compute_batch_mask_loss(point_preds, point_targets)
        dice_loss = self.compute_batch_dice_loss(point_preds, point_targets)
        class_loss = self.compute_batch_class_loss(class_preds, class_gts,
                                                   indices)

        mask_loss = self.mask_loss_weight * mask_loss
        dice_loss = self.dice_loss_weight * dice_loss
        class_loss = self.class_loss_weight * class_loss

        loss_dict = {
            'mask_loss': mask_loss,
            'dice_loss': dice_loss,
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

    from tools.path import ADE20Kdataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.datasets.ade20kdataset import ADE20KSemanticSegmentation
    from SimpleAICV.universal_segmentation.semantic_segmentation_common import YoloStyleResize, RandomHorizontalFlip, Normalize, SemanticSegmentationTrainCollater

    ade20kdataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        transform=transforms.Compose([
            YoloStyleResize(resize=512),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = SemanticSegmentationTrainCollater(resize=512)
    train_loader = DataLoader(ade20kdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.universal_segmentation.models.dinov3_universal_segmentation import dinov3_vit_base_patch16_universal_segmentation
    net = dinov3_vit_base_patch16_universal_segmentation(
        backbone_pretrained_path='', image_size=512, num_classes=151)

    loss1 = Mask2FormerLoss(point_nums=16384,
                            oversample_ratio=3.0,
                            importance_sample_ratio=0.75,
                            mask_cost=5.0,
                            dice_cost=5.0,
                            class_cost=2.0,
                            num_classes=151,
                            mask_loss_weight=5.0,
                            dice_loss_weight=5.0,
                            class_loss_weight=2.0,
                            no_object_class_weight=0.1)
    for data in tqdm(train_loader):
        images, masks, labels, sizes = data['image'], data['mask'], data[
            'label'], data['size']

        print('1111', images.shape, len(masks), len(labels), sizes.shape)

        for per_image_masks, per_image_labels in zip(masks, labels):
            print('2222', per_image_masks.shape)
            print('3333', len(per_image_labels), per_image_labels)

        mask_preds, class_preds = net(images)
        out = loss1(mask_preds, class_preds, masks, labels)
        print('3333', out)

        break
