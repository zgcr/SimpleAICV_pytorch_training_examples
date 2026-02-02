import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'UniversalSegmentationDecoder',
]


class UniversalSegmentationDecoder(nn.Module):

    def __init__(self,
                 topk=100,
                 min_score_threshold=0.1,
                 mask_threshold=0.5,
                 binary_mask=True):
        super(UniversalSegmentationDecoder, self).__init__()
        self.topk = topk
        self.min_score_threshold = min_score_threshold
        self.mask_threshold = mask_threshold
        self.binary_mask = binary_mask

    def forward(self, preds, scaled_sizes, origin_sizes):
        with torch.no_grad():
            mask_preds, class_preds = preds

            mask_preds_h, mask_preds_w = mask_preds.shape[2], mask_preds.shape[
                3]

            # mask_preds用sigmoid预测,class_preds用softmax预测
            mask_preds = torch.sigmoid(mask_preds)
            class_preds = torch.softmax(class_preds, dim=-1)

            # 模型训练时背景类index是num_classes-1,预测结果去掉背景类预测值(所有正类预测完后,剩下的像素点默认属于背景类)
            # [b, query_num, num_classes] -> [b, query_num, num_classes-1]
            class_preds = class_preds[:, :, :-1]

            pred_scores, pred_classes = torch.max(class_preds, dim=-1)

            batch_masks, batch_scores, batch_classes = [], [], []
            for per_image_mask_preds, per_image_pred_scores, per_image_pred_classes, per_image_scaled_sizes, per_image_origin_sizes in zip(
                    mask_preds, pred_scores, pred_classes, scaled_sizes,
                    origin_sizes):

                per_image_keep_flag = per_image_pred_scores > self.min_score_threshold
                per_image_mask_preds = per_image_mask_preds[
                    per_image_keep_flag]
                per_image_pred_scores = per_image_pred_scores[
                    per_image_keep_flag]
                per_image_pred_classes = per_image_pred_classes[
                    per_image_keep_flag]

                empty_per_image_mask_preds = np.zeros(
                    (0, mask_preds_h, mask_preds_w), dtype=np.float32)
                empty_per_image_pred_scores = np.zeros((0), dtype=np.float32)
                empty_per_image_pred_classes = np.zeros((0), dtype=np.float32)

                if per_image_pred_scores.shape[0] == 0:
                    batch_masks.append(empty_per_image_mask_preds)
                    batch_scores.append(empty_per_image_pred_scores)
                    batch_classes.append(empty_per_image_pred_classes)
                    continue

                # sort and keep top_k
                sort_indexs = torch.argsort(per_image_pred_scores,
                                            descending=True)
                if sort_indexs.shape[0] > self.topk:
                    sort_indexs = sort_indexs[:self.topk]

                per_image_mask_preds = per_image_mask_preds[sort_indexs]
                per_image_pred_scores = per_image_pred_scores[sort_indexs]
                per_image_pred_classes = per_image_pred_classes[sort_indexs]

                if per_image_pred_scores.shape[0] == 0:
                    batch_masks.append(empty_per_image_mask_preds)
                    batch_scores.append(empty_per_image_pred_scores)
                    batch_classes.append(empty_per_image_pred_classes)
                    continue

                per_image_mask_preds = per_image_mask_preds[:, :int(
                    per_image_scaled_sizes[0]), :int(per_image_scaled_sizes[1]
                                                     )]

                per_image_origin_h, per_image_origin_w = int(
                    per_image_origin_sizes[0]), int(per_image_origin_sizes[1])
                per_image_mask_preds = F.interpolate(
                    per_image_mask_preds.unsqueeze(0),
                    size=[per_image_origin_h, per_image_origin_w],
                    mode='bilinear')
                per_image_mask_preds = per_image_mask_preds.squeeze(0)

                if self.binary_mask:
                    per_image_mask_preds = (per_image_mask_preds
                                            > self.mask_threshold).to(
                                                torch.uint8)

                per_image_mask_preds = per_image_mask_preds.cpu().numpy()
                per_image_pred_scores = per_image_pred_scores.cpu().numpy()
                per_image_pred_classes = per_image_pred_classes.cpu().numpy()

                batch_masks.append(per_image_mask_preds)
                batch_scores.append(per_image_pred_scores)
                batch_classes.append(per_image_pred_classes)

            return batch_masks, batch_scores, batch_classes


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

    from tools.path import COCO2017_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.datasets.cocodataset import CocoInstanceSegmentation
    from SimpleAICV.universal_segmentation.instance_segmentation_common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, InstanceSegmentationTestCollater

    cocodataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='train2017',
        filter_no_object_image=True,
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=1024,
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = InstanceSegmentationTestCollater(resize=1024,
                                                resize_type='yolo_style')
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.universal_segmentation.models.dinov3_universal_segmentation import dinov3_vit_base_patch16_universal_segmentation
    net = dinov3_vit_base_patch16_universal_segmentation(image_size=1024,
                                                         query_num=200,
                                                         num_classes=81)
    decode = UniversalSegmentationDecoder(topk=100,
                                          min_score_threshold=0.1,
                                          mask_threshold=0.5,
                                          binary_mask=True)

    for data in tqdm(train_loader):
        images, masks, sizes, origin_sizes = data['image'], data['mask'], data[
            'size'], data['origin_size']
        print('1111', images.shape, len(masks), sizes.shape,
              origin_sizes.shape)

        preds = net(images)
        batch_masks, batch_scores, batch_classes = decode(
            preds, sizes, origin_sizes)

        for per_image_mask_preds, per_image_pred_scores, per_image_pred_classes in zip(
                batch_masks, batch_scores, batch_classes):
            print('2222', per_image_mask_preds.shape,
                  per_image_pred_scores.shape, per_image_pred_classes.shape)
        break
