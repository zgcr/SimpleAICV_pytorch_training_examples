import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = [
    'GlobalTrimapCELoss',
    'GloabelTrimapIouLoss',
    'LocalAlphaLoss',
    'LocalLaplacianLoss',
    'FusionAlphaLoss',
    'FusionLaplacianLoss',
    'CompositionLoss',
]


class GlobalTrimapCELoss(nn.Module):

    def __init__(self):
        super(GlobalTrimapCELoss, self).__init__()
        pass

    def forward(self, global_pred, trimap):
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]

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


class GloabelTrimapIouLoss(nn.Module):

    def __init__(self, smooth=1e-4):
        super(GloabelTrimapIouLoss, self).__init__()
        self.smooth = smooth

    def forward(self, global_pred, trimap):
        # global_pred shape:[b,3,h,w] -> [b,h,w,3]
        # trimap shape:[b,h,w]
        global_pred = global_pred.permute(0, 2, 3, 1).contiguous()
        num_classes = global_pred.shape[3]

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

        iou_loss = 1. - (torch.sum(intersection, dim=1) + self.smooth) / (
            torch.sum(global_pred, dim=1) + torch.sum(label, dim=1) -
            torch.sum(intersection, dim=1) + self.smooth)
        iou_loss = iou_loss.mean()

        return iou_loss


class LocalAlphaLoss(nn.Module):

    def __init__(self):
        super(LocalAlphaLoss, self).__init__()
        pass

    def forward(self, local_pred, alpha, trimap):
        # local_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,h,w]
        # trimap shape:[b,h,w]
        local_pred = local_pred.permute(0, 2, 3, 1).contiguous()
        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)
        local_pred = torch.squeeze(local_pred, dim=-1)

        weighted = torch.zeros_like(trimap)
        weighted[trimap == 128] = 1.

        diff = local_pred - alpha
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum() + 1.)

        return alpha_loss


class LocalLaplacianLoss(nn.Module):

    def __init__(self):
        super(LocalLaplacianLoss, self).__init__()
        pass

    def forward(self, local_pred, alpha, trimap):
        # local_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,h,w]
        # trimap shape:[b,h,w]
        device = local_pred.device

        local_pred = torch.clamp(local_pred, min=1e-4, max=1. - 1e-4)

        alpha = torch.unsqueeze(alpha, dim=1)
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


class FusionAlphaLoss(nn.Module):

    def __init__(self):
        super(FusionAlphaLoss, self).__init__()
        pass

    def forward(self, fusion_pred, alpha):
        # fusion_pred shape:[b,1,h,w] -> [b,h,w,1] -> [b,h,w]
        # alpha shape:[b,h,w]
        fusion_pred = fusion_pred.permute(0, 2, 3, 1).contiguous()
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.squeeze(fusion_pred, dim=-1)

        weighted = torch.ones_like(alpha)

        diff = fusion_pred - alpha
        alpha_loss = torch.sqrt(diff**2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())

        return alpha_loss


class FusionLaplacianLoss(nn.Module):

    def __init__(self):
        super(FusionLaplacianLoss, self).__init__()
        pass

    def forward(self, fusion_pred, alpha):
        # fusion_pred shape:[b,1,h,w]
        # alpha shape:[b,h,w]
        device = fusion_pred.device

        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)

        alpha = torch.unsqueeze(alpha, dim=1)

        gauss_kernel = self.build_gauss_kernel(size=5, sigma=1.0,
                                               n_channels=1).to(device)
        pyr_alpha = self.laplacian_pyramid(alpha, gauss_kernel, 5)
        pyr_predict = self.laplacian_pyramid(fusion_pred, gauss_kernel, 5)

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


class CompositionLoss(nn.Module):

    def __init__(self):
        super(CompositionLoss, self).__init__()
        pass

    def forward(self, image, alpha, fg_map, bg_map, fusion_pred):
        # image shape:[b,3,h,w]
        # alpha shape:[b,h,w]
        # fg_map shape:[b,3,h,w]
        # bg_map shape:[b,3,h,w]
        # fusion_pred shape:[b,1,h,w]
        fusion_pred = torch.clamp(fusion_pred, min=1e-4, max=1. - 1e-4)
        fusion_pred = torch.cat([fusion_pred, fusion_pred, fusion_pred], dim=1)

        weighted = torch.ones_like(alpha)

        composition = fusion_pred * fg_map + (1. - fusion_pred) * bg_map
        composition_loss = torch.sqrt((composition - image)**2 + 1e-12)
        composition_loss = composition_loss.sum() / (weighted.sum())

        return composition_loss


if __name__ == '__main__':
    import os
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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from tools.path import human_matting_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.human_matting.datasets.human_matting_dataset import HumanMattingDataset
    from simpleAICV.human_matting.common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, HumanMattingCollater

    human_matting_dataset = HumanMattingDataset(
        human_matting_dataset_path,
        set_name_list=[
            'Deep_Automatic_Portrait_Matting',
            'RealWorldPortrait636',
            'P3M10K',
        ],
        set_type='train',
        max_side=2048,
        kernel_size_range=[10, 15],
        transform=transforms.Compose([
            # YoloStyleResize(resize=832),
            Resize(resize=832),
            RandomHorizontalFlip(prob=1.0),
            Normalize(),
        ]))

    from torch.utils.data import DataLoader
    collater = HumanMattingCollater(resize=832)
    train_loader = DataLoader(human_matting_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.human_matting.models import vanb3_pfan_matting
    net = vanb3_pfan_matting()

    loss = GlobalTrimapCELoss()
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map'], data[
                'size']
        print('1111', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape)
        global_preds, local_preds, fused_preds = net(images)
        print('1212', global_preds.shape, local_preds.shape, fused_preds.shape)
        out = loss(global_preds, trimaps)
        print('1313', out)
        break

    loss = GloabelTrimapIouLoss()
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map'], data[
                'size']
        print('1111', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape)
        global_preds, local_preds, fused_preds = net(images)
        print('1212', global_preds.shape, local_preds.shape, fused_preds.shape)
        out = loss(global_preds, trimaps)
        print('1313', out)
        break

    loss = LocalAlphaLoss()
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map'], data[
                'size']
        print('1111', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape)
        global_preds, local_preds, fused_preds = net(images)
        print('1212', global_preds.shape, local_preds.shape, fused_preds.shape)
        out = loss(local_preds, masks, trimaps)
        print('1313', out)
        break

    loss = LocalLaplacianLoss()
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map'], data[
                'size']
        print('1111', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape)
        global_preds, local_preds, fused_preds = net(images)
        print('1212', global_preds.shape, local_preds.shape, fused_preds.shape)
        out = loss(local_preds, masks, trimaps)
        print('1313', out)
        break

    loss = FusionAlphaLoss()
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map'], data[
                'size']
        print('1111', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape)
        global_preds, local_preds, fused_preds = net(images)
        print('1212', global_preds.shape, local_preds.shape, fused_preds.shape)
        out = loss(fused_preds, masks)
        print('1313', out)
        break

    loss = FusionLaplacianLoss()
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map'], data[
                'size']
        print('1111', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape)
        global_preds, local_preds, fused_preds = net(images)
        print('1212', global_preds.shape, local_preds.shape, fused_preds.shape)
        out = loss(fused_preds, masks)
        print('1313', out)
        break

    loss = CompositionLoss()
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map'], data[
                'size']
        print('1111', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape)
        global_preds, local_preds, fused_preds = net(images)
        print('1212', global_preds.shape, local_preds.shape, fused_preds.shape)
        out = loss(images, masks, fg_maps, bg_maps, fused_preds)
        print('1313', out)
        break
