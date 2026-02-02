import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.classification.common import load_state_dict


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, masks, trimaps, fg_maps, bg_maps, size, origin_size = sample[
            'image'], sample['mask'], sample['trimap'], sample[
                'fg_map'], sample['bg_map'], sample['size'], sample[
                    'origin_size']

        if np.random.uniform(0, 1) < self.prob:
            # [h,w,3]
            image = image[:, ::-1, :]
            # [h,w,n]
            masks = masks[:, ::-1, :]
            # [h,w,n]
            trimaps = trimaps[:, ::-1, :]
            # [h,w,3,n]
            fg_maps = fg_maps[:, ::-1, :, :]
            # [h,w,3,n]
            bg_maps = bg_maps[:, ::-1, :, :]

        sample['image'], sample['mask'], sample['trimap'], sample[
            'fg_map'], sample['bg_map'], sample['size'], sample[
                'origin_size'] = image, masks, trimaps, fg_maps, bg_maps, size, origin_size

        return sample


class RandomVerticalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, masks, trimaps, fg_maps, bg_maps, size, origin_size = sample[
            'image'], sample['mask'], sample['trimap'], sample[
                'fg_map'], sample['bg_map'], sample['size'], sample[
                    'origin_size']

        if np.random.uniform(0, 1) < self.prob:
            # [h,w,3]
            image = image[::-1, :, :]
            # [h,w,n]
            masks = masks[::-1, :, :]
            # [h,w,n]
            trimaps = trimaps[::-1, :, :]
            # [h,w,3,n]
            fg_maps = fg_maps[::-1, :, :, :]
            # [h,w,3,n]
            bg_maps = bg_maps[::-1, :, :, :]

        sample['image'], sample['mask'], sample['trimap'], sample[
            'fg_map'], sample['bg_map'], sample['size'], sample[
                'origin_size'] = image, masks, trimaps, fg_maps, bg_maps, size, origin_size

        return sample


class YoloStyleResize:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, sample):
        image, masks, trimaps, fg_maps, bg_maps, size, origin_size = sample[
            'image'], sample['mask'], sample['trimap'], sample[
                'fg_map'], sample['bg_map'], sample['size'], sample[
                    'origin_size']

        h, w, _ = image.shape
        factor = self.resize / max(h, w)
        resize_w, resize_h = int(w * float(factor) +
                                 0.5), int(h * float(factor) + 0.5)

        # [h,w,3]
        image = cv2.resize(image, (resize_w, resize_h))
        # [h,w,n]
        masks = cv2.resize(masks, (resize_w, resize_h))
        if len(masks.shape) != 3:
            masks = np.expand_dims(masks, axis=-1)
        # [h,w,n]
        trimaps = cv2.resize(trimaps, (resize_w, resize_h),
                             interpolation=cv2.INTER_NEAREST)
        if len(trimaps.shape) != 3:
            trimaps = np.expand_dims(trimaps, axis=-1)

        # [h,w,3,n]
        resized_fg_maps = []
        # [h,w,3,n]
        resized_bg_maps = []
        object_nums = masks.shape[-1]
        for object_idx in range(object_nums):
            per_object_fg_map = fg_maps[:, :, :, object_idx]
            per_object_bg_map = bg_maps[:, :, :, object_idx]

            per_object_fg_map = cv2.resize(per_object_fg_map,
                                           (resize_w, resize_h))
            per_object_bg_map = cv2.resize(per_object_bg_map,
                                           (resize_w, resize_h))

            resized_fg_maps.append(per_object_fg_map)
            resized_bg_maps.append(per_object_bg_map)
        resized_fg_maps = np.stack(resized_fg_maps, axis=-1)
        resized_bg_maps = np.stack(resized_bg_maps, axis=-1)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['trimap'], sample[
            'fg_map'], sample['bg_map'], sample['size'], sample[
                'origin_size'] = image, masks, trimaps, resized_fg_maps, resized_bg_maps, size, origin_size

        return sample


class Resize:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, sample):
        image, masks, trimaps, fg_maps, bg_maps, size, origin_size = sample[
            'image'], sample['mask'], sample['trimap'], sample[
                'fg_map'], sample['bg_map'], sample['size'], sample[
                    'origin_size']

        # [h,w,3]
        image = cv2.resize(image, (self.resize, self.resize))
        # [h,w,n]
        masks = cv2.resize(masks, (self.resize, self.resize))
        if len(masks.shape) != 3:
            masks = np.expand_dims(masks, axis=-1)
        # [h,w,n]
        trimaps = cv2.resize(trimaps, (self.resize, self.resize),
                             interpolation=cv2.INTER_NEAREST)
        if len(trimaps.shape) != 3:
            trimaps = np.expand_dims(trimaps, axis=-1)

        # [h,w,3,n]
        resized_fg_maps = []
        # [h,w,3,n]
        resized_bg_maps = []
        object_nums = masks.shape[-1]
        for object_idx in range(object_nums):
            per_object_fg_map = fg_maps[:, :, :, object_idx]
            per_object_bg_map = bg_maps[:, :, :, object_idx]

            per_object_fg_map = cv2.resize(per_object_fg_map,
                                           (self.resize, self.resize))
            per_object_bg_map = cv2.resize(per_object_bg_map,
                                           (self.resize, self.resize))

            resized_fg_maps.append(per_object_fg_map)
            resized_bg_maps.append(per_object_bg_map)
        resized_fg_maps = np.stack(resized_fg_maps, axis=-1)
        resized_bg_maps = np.stack(resized_bg_maps, axis=-1)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['trimap'], sample[
            'fg_map'], sample['bg_map'], sample['size'], sample[
                'origin_size'] = image, masks, trimaps, resized_fg_maps, resized_bg_maps, size, origin_size

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        image, masks, trimaps, fg_maps, bg_maps, size, origin_size = sample[
            'image'], sample['mask'], sample['trimap'], sample[
                'fg_map'], sample['bg_map'], sample['size'], sample[
                    'origin_size']

        image = image / 255.
        fg_maps = fg_maps / 255.
        bg_maps = bg_maps / 255.

        sample['image'], sample['mask'], sample['trimap'], sample[
            'fg_map'], sample['bg_map'], sample['size'], sample[
                'origin_size'] = image, masks, trimaps, fg_maps, bg_maps, size, origin_size

        return sample


class HumanInstanceMattingTestCollater:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['mask'] for s in data]
        trimaps = [s['trimap'] for s in data]
        fg_maps = [s['fg_map'] for s in data]
        bg_maps = [s['bg_map'] for s in data]
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

        input_masks = []
        input_trimaps = []
        input_fg_maps = []
        input_bg_maps = []
        input_labels = []
        for i in range(len(images)):
            ########################################################################
            per_image_masks = masks[i]
            per_image_object_nums = per_image_masks.shape[-1]

            per_image_input_masks = np.zeros(
                (per_image_object_nums, self.resize, self.resize),
                dtype=np.float32)
            for per_object_idx in range(per_image_object_nums):
                per_object_mask = per_image_masks[:, :, per_object_idx]
                per_image_input_masks[
                    per_object_idx, 0:per_object_mask.shape[0],
                    0:per_object_mask.shape[1]] = per_object_mask
            per_image_input_masks = torch.from_numpy(per_image_input_masks)
            per_image_input_masks = per_image_input_masks.float()

            ########################################################################
            per_image_trimaps = trimaps[i]
            per_image_object_nums = per_image_trimaps.shape[-1]

            per_image_input_trimaps = np.zeros(
                (per_image_object_nums, self.resize, self.resize),
                dtype=np.uint8)
            for per_object_idx in range(per_image_object_nums):
                per_object_trimap = per_image_trimaps[:, :, per_object_idx]
                per_image_input_trimaps[
                    per_object_idx, 0:per_object_trimap.shape[0],
                    0:per_object_trimap.shape[1]] = per_object_trimap
            per_image_input_trimaps = torch.from_numpy(per_image_input_trimaps)
            per_image_input_trimaps = per_image_input_trimaps.float()

            ########################################################################
            per_image_fg_maps = fg_maps[i]
            per_image_object_nums = per_image_fg_maps.shape[-1]

            per_image_input_fg_maps = np.zeros(
                (per_image_object_nums, self.resize, self.resize, 3),
                dtype=np.float32)
            for per_object_idx in range(per_image_object_nums):
                per_object_fg_map = per_image_fg_maps[:, :, :, per_object_idx]
                per_image_input_fg_maps[
                    per_object_idx, 0:per_object_fg_map.shape[0],
                    0:per_object_fg_map.shape[1], :] = per_object_fg_map
            per_image_input_fg_maps = torch.from_numpy(per_image_input_fg_maps)
            # B H W 3 ->B 3 H W
            per_image_input_fg_maps = per_image_input_fg_maps.permute(
                0, 3, 1, 2)
            per_image_input_fg_maps = per_image_input_fg_maps.float()

            ########################################################################
            per_image_bg_maps = bg_maps[i]
            per_image_object_nums = per_image_bg_maps.shape[-1]

            per_image_input_bg_maps = np.zeros(
                (per_image_object_nums, self.resize, self.resize, 3),
                dtype=np.float32)
            for per_object_idx in range(per_image_object_nums):
                per_object_bg_map = per_image_bg_maps[:, :, :, per_object_idx]
                per_image_input_bg_maps[
                    per_object_idx, 0:per_object_bg_map.shape[0],
                    0:per_object_bg_map.shape[1], :] = per_object_bg_map
            per_image_input_bg_maps = torch.from_numpy(per_image_input_bg_maps)
            # B H W 3 ->B 3 H W
            per_image_input_bg_maps = per_image_input_bg_maps.permute(
                0, 3, 1, 2)
            per_image_input_bg_maps = per_image_input_bg_maps.float()

            ########################################################################
            # 每张图像只有人像实例mask,类别标签统一设为0
            per_image_labels = []
            per_image_object_nums = per_image_masks.shape[-1]
            for _ in range(per_image_object_nums):
                per_image_labels.append(0)
            per_image_labels = np.array(per_image_labels).astype(np.float32)
            per_image_labels = torch.from_numpy(per_image_labels).long()

            input_masks.append(per_image_input_masks)
            input_trimaps.append(per_image_input_trimaps)
            input_fg_maps.append(per_image_input_fg_maps)
            input_bg_maps.append(per_image_input_bg_maps)
            input_labels.append(per_image_labels)

        sizes = np.array(sizes, dtype=np.float32)
        origin_sizes = np.array(origin_sizes, dtype=np.float32)

        return {
            'image': input_images,
            'mask': input_masks,
            'trimap': input_trimaps,
            'fg_map': input_fg_maps,
            'bg_map': input_bg_maps,
            'label': input_labels,
            'size': sizes,
            'origin_size': origin_sizes,
        }


class HumanInstanceMattingTrainCollater:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['mask'] for s in data]
        trimaps = [s['trimap'] for s in data]
        fg_maps = [s['fg_map'] for s in data]
        bg_maps = [s['bg_map'] for s in data]
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
        input_trimaps = []
        input_fg_maps = []
        input_bg_maps = []
        input_labels = []
        for i in range(len(images)):
            ########################################################################
            per_image_masks = masks[i]
            per_image_object_nums = per_image_masks.shape[-1]

            per_image_input_masks = np.zeros(
                (per_image_object_nums, self.resize, self.resize),
                dtype=np.float32)
            for per_object_idx in range(per_image_object_nums):
                per_object_mask = per_image_masks[:, :, per_object_idx]
                per_image_input_masks[
                    per_object_idx, 0:per_object_mask.shape[0],
                    0:per_object_mask.shape[1]] = per_object_mask
            per_image_input_masks = torch.from_numpy(per_image_input_masks)
            per_image_input_masks = per_image_input_masks.float()

            ########################################################################
            per_image_trimaps = trimaps[i]
            per_image_object_nums = per_image_trimaps.shape[-1]

            per_image_input_trimaps = np.zeros(
                (per_image_object_nums, self.resize, self.resize),
                dtype=np.uint8)
            for per_object_idx in range(per_image_object_nums):
                per_object_trimap = per_image_trimaps[:, :, per_object_idx]
                per_image_input_trimaps[
                    per_object_idx, 0:per_object_trimap.shape[0],
                    0:per_object_trimap.shape[1]] = per_object_trimap
            per_image_input_trimaps = torch.from_numpy(per_image_input_trimaps)
            per_image_input_trimaps = per_image_input_trimaps.float()

            ########################################################################
            per_image_fg_maps = fg_maps[i]
            per_image_object_nums = per_image_fg_maps.shape[-1]

            per_image_input_fg_maps = np.zeros(
                (per_image_object_nums, self.resize, self.resize, 3),
                dtype=np.float32)
            for per_object_idx in range(per_image_object_nums):
                per_object_fg_map = per_image_fg_maps[:, :, :, per_object_idx]
                per_image_input_fg_maps[
                    per_object_idx, 0:per_object_fg_map.shape[0],
                    0:per_object_fg_map.shape[1], :] = per_object_fg_map
            per_image_input_fg_maps = torch.from_numpy(per_image_input_fg_maps)
            # B H W 3 ->B 3 H W
            per_image_input_fg_maps = per_image_input_fg_maps.permute(
                0, 3, 1, 2)
            per_image_input_fg_maps = per_image_input_fg_maps.float()

            ########################################################################
            per_image_bg_maps = bg_maps[i]
            per_image_object_nums = per_image_bg_maps.shape[-1]

            per_image_input_bg_maps = np.zeros(
                (per_image_object_nums, self.resize, self.resize, 3),
                dtype=np.float32)
            for per_object_idx in range(per_image_object_nums):
                per_object_bg_map = per_image_bg_maps[:, :, :, per_object_idx]
                per_image_input_bg_maps[
                    per_object_idx, 0:per_object_bg_map.shape[0],
                    0:per_object_bg_map.shape[1], :] = per_object_bg_map
            per_image_input_bg_maps = torch.from_numpy(per_image_input_bg_maps)
            # B H W 3 ->B 3 H W
            per_image_input_bg_maps = per_image_input_bg_maps.permute(
                0, 3, 1, 2)
            per_image_input_bg_maps = per_image_input_bg_maps.float()

            ########################################################################
            # 每张图像只有人像实例mask,类别标签统一设为0
            per_image_labels = []
            per_image_object_nums = per_image_masks.shape[-1]
            for _ in range(per_image_object_nums):
                per_image_labels.append(0)
            per_image_labels = np.array(per_image_labels).astype(np.float32)
            per_image_labels = torch.from_numpy(per_image_labels).long()

            input_masks.append(per_image_input_masks)
            input_trimaps.append(per_image_input_trimaps)
            input_fg_maps.append(per_image_input_fg_maps)
            input_bg_maps.append(per_image_input_bg_maps)
            input_labels.append(per_image_labels)

        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'mask': input_masks,
            'trimap': input_trimaps,
            'fg_map': input_fg_maps,
            'bg_map': input_bg_maps,
            'label': input_labels,
            'size': sizes,
        }
