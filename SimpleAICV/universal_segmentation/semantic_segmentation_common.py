import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import numpy as np

import torch

from SimpleAICV.classification.common import load_state_dict


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask, size, origin_size = sample['image'], sample[
            'mask'], sample['size'], sample['origin_size']

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]

        sample['image'], sample['mask'], sample['size'], sample[
            'origin_size'] = image, mask, size, origin_size

        return sample


class RandomVerticalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask, size, origin_size = sample['image'], sample[
            'mask'], sample['size'], sample['origin_size']

        if np.random.uniform(0, 1) < self.prob:
            image = image[::-1, :, :]
            mask = mask[::-1, :]

        sample['image'], sample['mask'], sample['size'], sample[
            'origin_size'] = image, mask, size, origin_size

        return sample


class YoloStyleResize:

    def __init__(self, resize=512):
        self.resize = resize

    def __call__(self, sample):
        image, mask, size, origin_size = sample['image'], sample[
            'mask'], sample['size'], sample['origin_size']

        h, w, _ = image.shape

        factor = self.resize / max(h, w)

        resize_w, resize_h = int(w * float(factor) +
                                 0.5), int(h * float(factor) + 0.5)
        image = cv2.resize(image, (resize_w, resize_h))
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['size'], sample[
            'origin_size'] = image, mask, size, origin_size

        return sample


class Resize:

    def __init__(self, resize=512):
        self.resize = resize

    def __call__(self, sample):
        image, mask, size, origin_size = sample['image'], sample[
            'mask'], sample['size'], sample['origin_size']

        image = cv2.resize(image, (self.resize, self.resize))
        mask = cv2.resize(mask, (self.resize, self.resize),
                          interpolation=cv2.INTER_NEAREST)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['size'], sample[
            'origin_size'] = image, mask, size, origin_size

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask, size, origin_size = sample['image'], sample[
            'mask'], sample['size'], sample['origin_size']

        image = image / 255.

        sample['image'], sample['mask'], sample['size'], sample[
            'origin_size'] = image, mask, size, origin_size

        return sample


class SemanticSegmentationTestCollater:

    def __init__(self, resize=512):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['mask'] for s in data]
        sizes = [s['size'] for s in data]
        origin_sizes = [s['origin_size'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)
        input_images = input_images.float()

        # 0 is background class
        input_masks = np.zeros((len(masks), self.resize, self.resize),
                               dtype=np.float32)
        for i, per_mask in enumerate(masks):
            input_masks[i, 0:per_mask.shape[0], 0:per_mask.shape[1]] = per_mask
        # [B,h,w]
        input_masks = torch.from_numpy(input_masks)
        input_masks = input_masks.float()

        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'mask': input_masks,
            'size': sizes,
            'origin_size': origin_sizes,
        }


class SemanticSegmentationTrainCollater:

    def __init__(self, resize=512):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['mask'] for s in data]
        sizes = [s['size'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)
        input_images = input_images.float()

        input_masks = []
        input_labels = []
        for i, per_mask in enumerate(masks):
            per_mask = per_mask.copy()
            # per_mask中0是背景,其他值是前景
            per_mask_class_ids = list(np.unique(per_mask))
            # remove background 0
            if 0 in per_mask_class_ids:
                per_mask_class_ids.remove(0)
            per_mask_class_ids = np.array(per_mask_class_ids).astype(
                np.float32)
            per_mask_class_ids = torch.from_numpy(per_mask_class_ids).long()

            per_mask_class_masks = []
            for per_class_id in per_mask_class_ids:
                per_class_mask = (per_mask == per_class_id)
                per_mask_class_masks.append(per_class_mask)
            per_mask_class_masks = np.stack(per_mask_class_masks,
                                            dtype=np.float32)

            per_mask_input_class_masks = np.zeros(
                (len(per_mask_class_masks), self.resize, self.resize),
                dtype=np.float32)
            per_mask_input_class_masks[:, 0:per_mask_class_masks.shape[1],
                                       0:per_mask_class_masks.
                                       shape[2]] = per_mask_class_masks
            per_mask_input_class_masks = torch.from_numpy(
                per_mask_input_class_masks)
            per_mask_input_class_masks = per_mask_input_class_masks.float()

            # 将所有前景类别ID - 1,因为预测是前景类别是从0开始,num_classes值才是背景,背景值loss计算时填充
            per_mask_class_ids = per_mask_class_ids - 1

            assert len(per_mask_class_masks) == len(per_mask_class_ids)

            input_masks.append(per_mask_input_class_masks)
            input_labels.append(per_mask_class_ids)

        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'mask': input_masks,
            'label': input_labels,
            'size': sizes,
        }
