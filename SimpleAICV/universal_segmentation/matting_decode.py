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
    'UniversalMattingDecoder',
]


class UniversalMattingDecoder(nn.Module):

    def __init__(self, topk=100, min_score_threshold=0.1):
        super(UniversalMattingDecoder, self).__init__()
        self.topk = topk
        self.min_score_threshold = min_score_threshold

    def forward(self, preds, scaled_sizes, origin_sizes):
        with torch.no_grad():
            global_preds, local_preds, fused_preds, class_preds = preds

            mask_preds_h, mask_preds_w = fused_preds.shape[
                2], fused_preds.shape[3]

            # class_preds用softmax预测
            class_preds = torch.softmax(class_preds, dim=-1)

            # 模型训练时背景类index是num_classes-1,预测结果去掉背景类预测值(所有正类预测完后,剩下的像素点默认属于背景类)
            # [b, query_num, num_classes] -> [b, query_num, num_classes-1]
            class_preds = class_preds[:, :, :-1]

            pred_scores, pred_classes = torch.max(class_preds, dim=-1)

            batch_masks, batch_scores, batch_classes = [], [], []
            for per_image_fused_preds, per_image_pred_scores, per_image_pred_classes, per_image_scaled_sizes, per_image_origin_sizes in zip(
                    fused_preds, pred_scores, pred_classes, scaled_sizes,
                    origin_sizes):

                per_image_fused_preds = per_image_fused_preds.squeeze(1)

                per_image_keep_flag = per_image_pred_scores > self.min_score_threshold
                per_image_fused_preds = per_image_fused_preds[
                    per_image_keep_flag]
                per_image_pred_scores = per_image_pred_scores[
                    per_image_keep_flag]
                per_image_pred_classes = per_image_pred_classes[
                    per_image_keep_flag]

                empty_per_image_fused_preds = np.zeros(
                    (0, mask_preds_h, mask_preds_w), dtype=np.float32)
                empty_per_image_pred_scores = np.zeros((0), dtype=np.float32)
                empty_per_image_pred_classes = np.zeros((0), dtype=np.float32)

                if per_image_pred_scores.shape[0] == 0:
                    batch_masks.append(empty_per_image_fused_preds)
                    batch_scores.append(empty_per_image_pred_scores)
                    batch_classes.append(empty_per_image_pred_classes)
                    continue

                # sort and keep top_k
                sort_indexs = torch.argsort(per_image_pred_scores,
                                            descending=True)
                if sort_indexs.shape[0] > self.topk:
                    sort_indexs = sort_indexs[:self.topk]

                per_image_fused_preds = per_image_fused_preds[sort_indexs]
                per_image_pred_scores = per_image_pred_scores[sort_indexs]
                per_image_pred_classes = per_image_pred_classes[sort_indexs]

                if per_image_pred_scores.shape[0] == 0:
                    batch_masks.append(empty_per_image_fused_preds)
                    batch_scores.append(empty_per_image_pred_scores)
                    batch_classes.append(empty_per_image_pred_classes)
                    continue

                per_image_fused_preds = per_image_fused_preds[:, :int(
                    per_image_scaled_sizes[0]), :int(per_image_scaled_sizes[1]
                                                     )]

                per_image_origin_h, per_image_origin_w = int(
                    per_image_origin_sizes[0]), int(per_image_origin_sizes[1])
                per_image_fused_preds = F.interpolate(
                    per_image_fused_preds.unsqueeze(0),
                    size=[per_image_origin_h, per_image_origin_w],
                    mode='bilinear')
                per_image_fused_preds = per_image_fused_preds.squeeze(0)

                per_image_fused_preds = per_image_fused_preds.cpu().numpy()
                per_image_pred_scores = per_image_pred_scores.cpu().numpy()
                per_image_pred_classes = per_image_pred_classes.cpu().numpy()

                batch_masks.append(per_image_fused_preds)
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

    from tools.path import human_matting_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.datasets.human_matting_dataset import HumanMattingDataset
    from SimpleAICV.universal_segmentation.human_matting_common import RandomHorizontalFlip, Resize, Normalize, HumanMattingTestCollater

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
    collater = HumanMattingTestCollater(resize=512)
    train_loader = DataLoader(human_matting_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from SimpleAICV.universal_segmentation.models.dinov3_universal_matting import dinov3_vit_base_patch16_universal_matting
    net = dinov3_vit_base_patch16_universal_matting(image_size=512,
                                                    query_num=100,
                                                    num_classes=2)
    decode = UniversalMattingDecoder(topk=100, min_score_threshold=0.1)

    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes, origin_sizes = data[
            'image'], data['mask'], data['trimap'], data['fg_map'], data[
                'bg_map'], data['size'], data['origin_size']
        print('1111', images.shape, len(masks), len(trimaps), len(fg_maps),
              len(bg_maps), sizes.shape, origin_sizes.shape)

        preds = net(images)
        batch_masks, batch_scores, batch_classes = decode(
            preds, sizes, origin_sizes)

        for per_image_fused_preds, per_image_pred_scores, per_image_pred_classes in zip(
                batch_masks, batch_scores, batch_classes):
            print('2222', per_image_fused_preds.shape,
                  per_image_pred_scores.shape, per_image_pred_classes.shape)
        break
