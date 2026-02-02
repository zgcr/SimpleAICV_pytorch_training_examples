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


class SamResize:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        prompt_point, prompt_box, prompt_mask = sample['prompt_point'], sample[
            'prompt_box'], sample['prompt_mask']

        h, w, _ = image.shape
        factor = self.resize / max(h, w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        box[0:4] *= factor
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        prompt_point[:, 0:2] *= factor
        prompt_box[0:4] *= factor
        prompt_mask = cv2.resize(prompt_mask, (resize_w, resize_h),
                                 interpolation=cv2.INTER_NEAREST)

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_point, prompt_box, prompt_mask

        return sample


class SamRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        prompt_point, prompt_box, prompt_mask = sample['prompt_point'], sample[
            'prompt_box'], sample['prompt_mask']

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]

            prompt_mask = prompt_mask[:, ::-1]

            _, w, _ = image.shape

            x1 = box[0].copy()
            x2 = box[2].copy()

            box[0] = w - x2
            box[2] = w - x1

            x1 = prompt_box[0].copy()
            x2 = prompt_box[2].copy()

            prompt_box[0] = w - x2
            prompt_box[2] = w - x1

            _, w, _ = image.shape

            for i in range(len(prompt_point)):
                prompt_point[i][0] = w - prompt_point[i][0]

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_point, prompt_box, prompt_mask

        return sample


class SamNormalize:

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        self.mean = np.expand_dims(np.expand_dims(np.array(mean), axis=0),
                                   axis=0)
        self.std = np.expand_dims(np.expand_dims(np.array(std), axis=0),
                                  axis=0)

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        prompt_point, prompt_box, prompt_mask = sample['prompt_point'], sample[
            'prompt_box'], sample['prompt_mask']

        image = (image - self.mean) / self.std

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_point, prompt_box, prompt_mask

        return sample


class SAMBatchCollater:

    def __init__(self, resize):
        self.resize = resize
        assert resize % 64 == 0

        self.prompt_mask_size = resize // 4

    def __call__(self, data):
        images = [s['image'] for s in data]
        boxes = [x['box'] for x in data]
        masks = [x['mask'] for x in data]
        sizes = [x['size'] for x in data]

        prompt_points = [x['prompt_point'] for x in data]
        prompt_boxs = [x['prompt_box'] for x in data]
        prompt_masks = [x['prompt_mask'] for x in data]

        input_images = []
        for i, per_image in enumerate(images):
            per_input_image = np.zeros((self.resize, self.resize, 3),
                                       dtype=np.float32)
            per_input_image[0:per_image.shape[0],
                            0:per_image.shape[1], :] = per_image
            per_input_image = torch.from_numpy(per_input_image)
            # [3,H,W]
            per_input_image = per_input_image.permute(2, 0, 1)
            input_images.append(per_input_image)
        input_images = torch.stack(input_images, dim=0)
        input_images = input_images.float()

        input_boxes = []
        for i, per_box in enumerate(boxes):
            per_input_box = np.zeros((4), dtype=np.float32)
            per_input_box[0:per_box.shape[0]] = per_box[0:4]
            # [4]
            per_input_box = torch.from_numpy(per_input_box)
            input_boxes.append(per_input_box)
        input_boxes = torch.stack(input_boxes, dim=0)
        input_boxes = input_boxes.float()

        input_masks = []
        for i, per_mask in enumerate(masks):
            per_input_mask = np.zeros((self.resize, self.resize),
                                      dtype=np.float32)
            per_input_mask[0:per_mask.shape[0], 0:per_mask.shape[1]] = per_mask
            # [H,W]
            per_input_mask = torch.from_numpy(per_input_mask)
            input_masks.append(per_input_mask)
        input_masks = torch.stack(input_masks, dim=0)
        input_masks = input_masks.unsqueeze(1)
        input_masks = input_masks.float()

        assert len(input_images) == len(input_boxes) == len(input_masks)

        sizes = np.array(sizes, dtype=np.float32)

        input_prompt_points = []
        for i, per_prompt_point in enumerate(prompt_points):
            per_input_prompt_point = torch.from_numpy(per_prompt_point)
            input_prompt_points.append(per_input_prompt_point)
        input_prompt_points = torch.stack(input_prompt_points, dim=0)
        input_prompt_points = input_prompt_points.float()

        input_prompt_boxs = []
        for i, per_prompt_box in enumerate(prompt_boxs):
            per_input_prompt_box = torch.from_numpy(per_prompt_box)
            input_prompt_boxs.append(per_input_prompt_box)
        input_prompt_boxs = torch.stack(input_prompt_boxs, dim=0)
        input_prompt_boxs = input_prompt_boxs.float()

        input_prompt_masks = []
        for i, per_prompt_mask in enumerate(prompt_masks):
            h, w = per_prompt_mask.shape
            factor = self.prompt_mask_size / max(h, w)
            resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
            per_prompt_mask = cv2.resize(per_prompt_mask, (resize_w, resize_h),
                                         interpolation=cv2.INTER_NEAREST)

            per_input_prompt_mask = np.zeros(
                (self.prompt_mask_size, self.prompt_mask_size),
                dtype=np.float32)
            per_input_prompt_mask[0:per_prompt_mask.shape[0],
                                  0:per_prompt_mask.shape[1]] = per_prompt_mask
            # [H//4,W//4]
            per_input_prompt_mask = torch.from_numpy(per_input_prompt_mask)
            input_prompt_masks.append(per_input_prompt_mask)
        input_prompt_masks = torch.stack(input_prompt_masks, dim=0)
        input_prompt_masks = input_prompt_masks.unsqueeze(1)
        input_prompt_masks = input_prompt_masks.float()

        assert len(input_images) == len(input_boxes) == len(
            input_masks) == len(input_prompt_points) == len(
                input_prompt_boxs) == len(input_prompt_masks)

        return {
            'image': input_images,
            'box': input_boxes,
            'mask': input_masks,
            'size': sizes,
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }
