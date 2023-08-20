import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import numpy as np

import torch

from simpleAICV.classification.common import load_state_dict


class Resize:

    def __init__(self, resize=512):
        self.resize = resize

    def __call__(self, sample):
        image, mask, scale, size = sample['image'], sample['mask'], sample[
            'scale'], sample['size']

        h, w, _ = image.shape

        scale_factor = min(self.resize / max(h, w), self.resize / min(h, w))
        resize_w, resize_h = int(round(w * scale_factor)), int(
            round(h * scale_factor))

        image = cv2.resize(image, (resize_w, resize_h))
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        scale *= np.float32(scale_factor)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        return {
            'image': image,
            'mask': mask,
            'scale': scale,
            'size': size,
        }


class RandomCropResize:

    def __init__(self,
                 image_scale=(2048, 512),
                 multi_scale=False,
                 multi_scale_range=(0.5, 2.0),
                 crop_size=(512, 512),
                 cat_max_ratio=0.75,
                 ignore_index=None):
        self.image_scale = image_scale
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

        assert self.multi_scale_range[0] <= self.multi_scale_range[1]
        assert self.multi_scale_range[0] > 0 and self.multi_scale_range[1] > 0
        assert self.crop_size[0] > 0 and self.crop_size[1] > 0

    def __call__(self, sample):
        image, mask, scale, size = sample['image'], sample['mask'], sample[
            'scale'], sample['size']

        # multi scale resize
        h, w, _ = image.shape
        min_ratio, max_ratio = self.multi_scale_range

        if self.multi_scale:
            random_ratio = np.random.uniform(
                0, 1) * (max_ratio - min_ratio) + min_ratio
        else:
            random_ratio = 1.

        resize_scale = (int(self.image_scale[0] * random_ratio),
                        int(self.image_scale[1] * random_ratio))

        max_long_edge, max_short_edge = max(resize_scale), min(resize_scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
        resize_w, resize_h = int(round(w * scale_factor)), int(
            round(h * scale_factor))

        image = cv2.resize(image, (resize_w, resize_h))
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        scale *= np.float32(scale_factor)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        # random crop
        crop_bbox = self.get_crop_bbox(image)
        if self.cat_max_ratio < 1.:
            for _ in range(10):
                temp_mask = mask[crop_bbox[0]:crop_bbox[1],
                                 crop_bbox[2]:crop_bbox[3]]
                labels, counts = np.unique(temp_mask, return_counts=True)
                if self.ignore_index:
                    counts = counts[labels != self.ignore_index]
                if len(counts) > 1 and np.max(counts) / np.sum(
                        counts) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(image)

        image = image[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3], :]
        mask = mask[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        return {
            'image': image,
            'mask': mask,
            'scale': scale,
            'size': size,
        }

    def get_crop_bbox(self, image):
        margin_h = max(image.shape[0] - self.crop_size[0], 0)
        margin_w = max(image.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask, scale, size = sample['image'], sample['mask'], sample[
            'scale'], sample['size']

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]

        return {
            'image': image,
            'mask': mask,
            'scale': scale,
            'size': size,
        }


class PhotoMetricDistortion:
    """
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 prob=0.5):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob

    def __call__(self, sample):
        image, mask, scale, size = sample['image'], sample['mask'], sample[
            'scale'], sample['size']

        # random brightness
        image = self.brightness(image)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            image = self.contrast(image)

        # random saturation
        image = self.saturation(image)

        # random hue
        image = self.hue(image)

        # random contrast
        if mode == 0:
            image = self.contrast(image)

        return {
            'image': image,
            'mask': mask,
            'scale': scale,
            'size': size,
        }

    def convert(self, image, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255)

        return image.astype(np.float32)

    def brightness(self, image):
        """Brightness distortion."""
        if np.random.uniform(0, 1) < self.prob:
            image = self.convert(image,
                                 beta=np.random.uniform(
                                     -self.brightness_delta,
                                     self.brightness_delta))

        return image

    def contrast(self, image):
        """Contrast distortion."""
        if np.random.uniform(0, 1) < self.prob:
            image = self.convert(image,
                                 alpha=np.random.uniform(
                                     self.contrast_lower, self.contrast_upper))

        return image

    def saturation(self, image):
        """Saturation distortion."""
        if np.random.uniform(0, 1) < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 1] = self.convert(image[:, :, 1],
                                          alpha=np.random.uniform(
                                              self.saturation_lower,
                                              self.saturation_upper))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

        return image

    def hue(self, image):
        """Hue distortion."""
        if np.random.uniform(0, 1) < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 0] = (image[:, :, 0].astype(int) + np.random.randint(
                -self.hue_delta, self.hue_delta)) % 180
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

        return image


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask, scale, size = sample['image'], sample['mask'], sample[
            'scale'], sample['size']

        image = image / 255.

        return {
            'image': image,
            'mask': mask,
            'scale': scale,
            'size': size,
        }


class SemanticSegmentationCollater:

    def __init__(self, resize=512, ignore_index=None):
        self.resize = resize
        self.ignore_index = ignore_index

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['mask'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)

        if self.ignore_index:
            input_masks = np.ones((len(masks), self.resize, self.resize),
                                  dtype=np.float32) * self.ignore_index
        else:
            input_masks = np.zeros((len(masks), self.resize, self.resize),
                                   dtype=np.float32)
        for i, per_mask in enumerate(masks):
            input_masks[i, 0:per_mask.shape[0], 0:per_mask.shape[1]] = per_mask
        # [B,h,w]
        input_masks = torch.from_numpy(input_masks)

        scales = np.array(scales, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'mask': input_masks,
            'scale': scales,
            'size': sizes,
        }