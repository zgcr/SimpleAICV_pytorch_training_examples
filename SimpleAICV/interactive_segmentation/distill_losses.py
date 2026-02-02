import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MSELoss',
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
                                        per_image_mask_chosse_max_num=1,
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

    from SimpleAICV.interactive_segmentation.distill_model import DINOV3ImageEncoderDistillModel
    net = DINOV3ImageEncoderDistillModel(
        teacher_params={
            'image_size': 1024,
            'patch_size': 16,
            'inplanes': 3,
            'embedding_planes': 1280,
            'block_nums': 32,
            'head_nums': 16,
            'mlp_ratio': 4,
            'out_planes': 256,
            'window_size': 14,
            'global_attn_indexes': [7, 15, 23, 31],
            'use_gradient_checkpoint': False,
        },
        student_params={
            'backbone_type': 'dinov3_vit_base_patch16_backbone',
            'image_size': 1024,
            'out_planes': 256,
            'use_gradient_checkpoint': False,
        },
        teacher_pretrained_path='',
        student_pretrained_path='',
        freeze_teacher=True)
    loss = MSELoss()

    for data in tqdm(train_loader):
        images = data['image']
        print('1111', images.shape)

        net = net.cuda()
        images = images.cuda()
        tea_out, stu_out = net(images)
        print('2222', tea_out.shape, stu_out.shape)

        out = loss(tea_out, stu_out)
        print('3333', out)

        break
