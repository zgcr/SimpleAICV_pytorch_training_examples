import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'DBNetLoss',
]


class DBNetLoss(nn.Module):

    def __init__(self,
                 probability_weight=1,
                 threshold_weight=5,
                 binary_weight=1,
                 negative_ratio=3,
                 k=50):
        super(DBNetLoss, self).__init__()
        self.probability_weight = probability_weight
        self.threshold_weight = threshold_weight
        self.binary_weight = binary_weight
        self.negative_ratio = negative_ratio
        self.k = k

    def forward(self, preds, shapes):
        device = preds.device
        probability_map, threshold_map = preds[:, 0, :, :], preds[:, 1, :, :]

        binary_map = self.compute_binary_map(probability_map, threshold_map)

        probability_map = torch.clamp(probability_map, min=1e-4, max=1. - 1e-4)

        probability_mask = shapes['probability_mask'].to(device)
        probability_ignore_mask = shapes['probability_ignore_mask'].to(device)
        threshold_mask = shapes['threshold_mask'].to(device)
        threshold_ignore_mask = shapes['threshold_ignore_mask'].to(device)

        probability_map_loss = self.compute_batch_probability_map_loss(
            probability_map, probability_mask, probability_ignore_mask)
        threshold_map_loss = self.compute_batch_threshold_map_loss(
            threshold_map, threshold_mask, threshold_ignore_mask)

        binary_map_loss = self.compute_batch_binary_map_loss(
            binary_map, probability_mask, probability_ignore_mask)

        loss_dict = {
            'probability_map_loss':
            self.probability_weight * probability_map_loss,
            'threshold_map_loss': self.threshold_weight * threshold_map_loss,
            'binary_map_loss': self.binary_weight * binary_map_loss,
        }

        return loss_dict

    def compute_binary_map(self, probability_map, threshold_map):
        binary_map = torch.reciprocal(
            1 + torch.exp(-self.k * (probability_map - threshold_map)))

        return binary_map

    def compute_batch_probability_map_loss(self, probability_map,
                                           probability_mask,
                                           probability_ignore_mask):
        device = probability_map.device
        # 供loss计算部分权重置为1,过滤掉ignore框部分
        positive_label = probability_mask * probability_ignore_mask
        negative_label = (1. - probability_mask) * probability_ignore_mask
        # 计算正样本和负样本像素点数量
        positive_sample_nums = positive_label.sum()
        negative_sample_nums = min(negative_label.sum(),
                                   positive_sample_nums * self.negative_ratio)

        if positive_sample_nums + negative_sample_nums == 0:
            return torch.tensor(0.).to(device)

        bce_loss = -(probability_mask * torch.log(probability_map) +
                     (1. - probability_mask) * torch.log(1. - probability_map))
        # 正样本和负样本loss中均过滤掉ignore框部分
        positive_loss = bce_loss * positive_label
        negative_loss = bce_loss * negative_label
        negative_loss, _ = torch.topk(negative_loss.view(-1),
                                      int(negative_sample_nums.item()))

        probability_map_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_sample_nums + negative_sample_nums)

        return probability_map_loss

    def compute_batch_threshold_map_loss(self, threshold_map, threshold_mask,
                                         threshold_ignore_mask):
        device = threshold_map.device
        if threshold_ignore_mask.sum() == 0:
            return torch.tensor(0.).to(device)
        # 只计算threshold_ignore_mask中值为1的区域的loss,其他部分被过滤
        threshold_map_loss = (
            torch.abs(threshold_map - threshold_mask) *
            threshold_ignore_mask).sum() / threshold_ignore_mask.sum()

        return threshold_map_loss

    def compute_batch_binary_map_loss(self, binary_map, probability_mask,
                                      probability_ignore_mask):
        device = binary_map.device
        positive_sample_nums = (probability_mask *
                                probability_ignore_mask).sum()
        if positive_sample_nums == 0:
            return torch.tensor(0.).to(device)

        # intersection和union均过滤掉ignore框区域
        intersection = (binary_map * probability_mask *
                        probability_ignore_mask).sum()
        union = (binary_map * probability_ignore_mask).sum() + (
            probability_mask * probability_ignore_mask).sum()

        if intersection == 0 or union == 0:
            return torch.tensor(0.).to(device)

        binary_map_loss = 1 - 2.0 * intersection / union

        return binary_map_loss


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

    from tools.path import text_detection_dataset_path

    from simpleAICV.text_detection.datasets.text_detection_dataset import TextDetection
    from simpleAICV.text_detection.common import RandomRotate, MainDirectionRandomRotate, Resize, Normalize, TextDetectionCollater, DBNetTextDetectionCollater

    textdetectiondataset = TextDetection(text_detection_dataset_path,
                                         set_name=[
                                             'ICDAR2017RCTW_text_detection',
                                             'ICDAR2019ART_text_detection',
                                             'ICDAR2019LSVT_text_detection',
                                             'ICDAR2019MLT_text_detection',
                                             'ICDAR2019ReCTS_text_detection',
                                         ],
                                         set_type='train',
                                         transform=transforms.Compose([
                                             RandomRotate(angle=[-30, 30],
                                                          prob=0.3),
                                             MainDirectionRandomRotate(
                                                 angle=[0, 90, 180, 270],
                                                 prob=[0.7, 0.1, 0.1, 0.1]),
                                             Resize(resize=1024),
                                             Normalize(),
                                         ]))

    count = 0
    for per_sample in tqdm(textdetectiondataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, len(per_sample['annots']),
              per_sample['annots'][0]['points'].shape, per_sample['scale'],
              per_sample['size'])

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DBNetTextDetectionCollater(resize=1024,
                                          min_box_size=3,
                                          min_max_threshold=[0.3, 0.7],
                                          shrink_ratio=0.6)
    train_loader = DataLoader(textdetectiondataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.text_detection.models.dbnet import resnet50_dbnet
    net = resnet50_dbnet()
    loss = DBNetLoss(probability_weight=1,
                     threshold_weight=5,
                     binary_weight=1,
                     negative_ratio=3,
                     k=50)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        out = net(images)
        loss_dict = loss(out, annots)
        print("2222", loss_dict)
        break
