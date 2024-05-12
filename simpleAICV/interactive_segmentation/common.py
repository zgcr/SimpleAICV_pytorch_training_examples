import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import numpy as np

import torch

from simpleAICV.classification.common import AverageMeter, load_state_dict


class SamResize:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, sample):
        origin_image, origin_bbox, origin_mask, origin_size = sample[
            'origin_image'], sample['origin_bbox'], sample[
                'origin_mask'], sample['origin_size']

        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        h, w, _ = image.shape
        factor = self.resize / max(h, w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        box[0:4] *= factor
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        positive_prompt_point[:, 0:2] *= factor
        negative_prompt_point[:, 0:2] *= factor
        prompt_box[0:4] *= factor
        prompt_mask = cv2.resize(prompt_mask, (resize_w, resize_h),
                                 interpolation=cv2.INTER_NEAREST)

        return {
            'origin_image': origin_image,
            'origin_bbox': origin_bbox,
            'origin_mask': origin_mask,
            'origin_size': origin_size,
            'image': image,
            'box': box,
            'mask': mask,
            'size': size,
            'positive_prompt_point': positive_prompt_point,
            'negative_prompt_point': negative_prompt_point,
            'prompt_box': prompt_box,
            'prompt_mask': prompt_mask,
        }


class SamRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        origin_image, origin_bbox, origin_mask, origin_size = sample[
            'origin_image'], sample['origin_bbox'], sample[
                'origin_mask'], sample['origin_size']

        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
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

            for i in range(len(positive_prompt_point)):
                positive_prompt_point[i][0] = w - positive_prompt_point[i][0]

            for i in range(len(negative_prompt_point)):
                negative_prompt_point[i][0] = w - negative_prompt_point[i][0]

        return {
            'origin_image': origin_image,
            'origin_bbox': origin_bbox,
            'origin_mask': origin_mask,
            'origin_size': origin_size,
            'image': image,
            'box': box,
            'mask': mask,
            'size': size,
            'positive_prompt_point': positive_prompt_point,
            'negative_prompt_point': negative_prompt_point,
            'prompt_box': prompt_box,
            'prompt_mask': prompt_mask,
        }


class SamNormalize:

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        self.mean = np.expand_dims(np.expand_dims(np.array(mean), axis=0),
                                   axis=0)
        self.std = np.expand_dims(np.expand_dims(np.array(std), axis=0),
                                  axis=0)

    def __call__(self, sample):
        origin_image, origin_bbox, origin_mask, origin_size = sample[
            'origin_image'], sample['origin_bbox'], sample[
                'origin_mask'], sample['origin_size']

        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        image = (image - self.mean) / self.std

        return {
            'origin_image': origin_image,
            'origin_bbox': origin_bbox,
            'origin_mask': origin_mask,
            'origin_size': origin_size,
            'image': image,
            'box': box,
            'mask': mask,
            'size': size,
            'positive_prompt_point': positive_prompt_point,
            'negative_prompt_point': negative_prompt_point,
            'prompt_box': prompt_box,
            'prompt_mask': prompt_mask,
        }


class SAMCollater:

    def __init__(self,
                 resize,
                 positive_point_num_range=[1, 9],
                 negative_point_num_range=[1, 9],
                 batch_align_random_point_num=False,
                 positive_negative_point_num_ratio=None):
        self.resize = resize
        assert resize % 64 == 0

        self.prompt_mask_size = resize // 4

        assert isinstance(positive_point_num_range, (int, list))
        assert isinstance(negative_point_num_range, (int, list))
        if isinstance(positive_point_num_range, list):
            assert isinstance(positive_point_num_range[0], int)
            assert isinstance(positive_point_num_range[1], int)
            assert positive_point_num_range[0] <= positive_point_num_range[1]
            assert positive_point_num_range[0] >= 1
        if isinstance(negative_point_num_range, list):
            assert isinstance(negative_point_num_range[0], int)
            assert isinstance(negative_point_num_range[1], int)
            assert negative_point_num_range[0] <= negative_point_num_range[1]
            assert negative_point_num_range[0] >= 1
        if isinstance(positive_point_num_range, int):
            assert positive_point_num_range >= 0
        if isinstance(negative_point_num_range, int):
            assert negative_point_num_range >= 0

        self.positive_point_num_range = positive_point_num_range
        self.negative_point_num_range = negative_point_num_range
        self.batch_align_random_point_num = batch_align_random_point_num
        self.positive_negative_point_num_ratio = positive_negative_point_num_ratio

    def __call__(self, data):
        origin_images = [s['origin_image'] for s in data]
        origin_bboxes = [s['origin_bbox'] for s in data]
        origin_masks = [s['origin_mask'] for s in data]
        origin_sizes = [s['origin_size'] for s in data]

        images = [s['image'] for s in data]
        boxes = [x['box'] for x in data]
        masks = [x['mask'] for x in data]
        sizes = [x['size'] for x in data]

        positive_prompt_points = [x['positive_prompt_point'] for x in data]
        negative_prompt_points = [s['negative_prompt_point'] for s in data]
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

        batch_images = torch.stack(
            [per_input_image for per_input_image in input_images], dim=0)

        input_boxes = []
        for i, per_box in enumerate(boxes):
            per_input_box = np.zeros((4), dtype=np.float32)
            per_input_box[0:per_box.shape[0]] = per_box[0:4]
            # [4]
            per_input_box = torch.from_numpy(per_input_box)
            input_boxes.append(per_input_box)

        input_masks = []
        for i, per_mask in enumerate(masks):
            per_input_mask = np.zeros((self.resize, self.resize),
                                      dtype=np.float32)
            per_input_mask[0:per_mask.shape[0], 0:per_mask.shape[1]] = per_mask
            # [H,W]
            per_input_mask = torch.from_numpy(per_input_mask)
            input_masks.append(per_input_mask)

        batch_masks = torch.stack(
            [per_input_mask for per_input_mask in input_masks], dim=0)
        batch_masks = batch_masks.unsqueeze(1)

        assert len(input_images) == len(input_boxes) == len(input_masks)

        sizes = np.array(sizes, dtype=np.float32)

        input_positive_prompt_points = []
        if isinstance(self.positive_point_num_range, list):
            if self.batch_align_random_point_num:
                random_positive_point_num = np.random.randint(
                    self.positive_point_num_range[0],
                    self.positive_point_num_range[1] + 1)

            for i in range(len(positive_prompt_points)):
                if not self.batch_align_random_point_num:
                    random_positive_point_num = np.random.randint(
                        self.positive_point_num_range[0],
                        self.positive_point_num_range[1] + 1)

                per_input_positive_prompt_points = positive_prompt_points[i][
                    0:random_positive_point_num, :]
                # [random_positive_point_num,3]
                per_input_positive_prompt_points = torch.from_numpy(
                    per_input_positive_prompt_points)
                input_positive_prompt_points.append(
                    per_input_positive_prompt_points)

        elif isinstance(self.positive_point_num_range, int):
            for i in range(len(positive_prompt_points)):
                if self.positive_point_num_range <= 0:
                    per_input_positive_prompt_points = None
                else:
                    per_input_positive_prompt_points = positive_prompt_points[
                        i][0:self.positive_point_num_range, :]
                    # [positive_point_num,3]
                    per_input_positive_prompt_points = torch.from_numpy(
                        per_input_positive_prompt_points)

                input_positive_prompt_points.append(
                    per_input_positive_prompt_points)

        input_negative_prompt_points = []
        if isinstance(self.negative_point_num_range, list):
            if self.batch_align_random_point_num:
                if self.positive_negative_point_num_ratio:
                    random_negative_point_num = min(
                        int(random_positive_point_num *
                            self.positive_negative_point_num_ratio),
                        self.negative_point_num_range[1])
                else:
                    random_negative_point_num = np.random.randint(
                        self.negative_point_num_range[0],
                        self.negative_point_num_range[1] + 1)

            for i in range(len(negative_prompt_points)):
                if not self.batch_align_random_point_num:
                    if self.positive_negative_point_num_ratio:
                        random_negative_point_num = min(
                            int(
                                len(input_positive_prompt_points[i]) *
                                self.positive_negative_point_num_ratio),
                            self.negative_point_num_range[1])
                    else:
                        random_negative_point_num = np.random.randint(
                            self.negative_point_num_range[0],
                            self.negative_point_num_range[1] + 1)

                per_input_negative_prompt_points = negative_prompt_points[i][
                    0:random_negative_point_num, :]
                # [random_negative_point_num,3]
                per_input_negative_prompt_points = torch.from_numpy(
                    per_input_negative_prompt_points)
                input_negative_prompt_points.append(
                    per_input_negative_prompt_points)

        elif isinstance(self.negative_point_num_range, int):
            for i in range(len(negative_prompt_points)):
                if self.negative_point_num_range <= 0:
                    per_input_negative_prompt_points = None
                else:
                    if self.positive_negative_point_num_ratio:
                        negative_point_num = min(
                            int(random_positive_point_num *
                                self.positive_negative_point_num_ratio),
                            self.negative_point_num_range)
                    else:
                        negative_point_num = self.negative_point_num_range

                    per_input_negative_prompt_points = negative_prompt_points[
                        i][0:negative_point_num, :]
                    # [negative_point_num,3]
                    per_input_negative_prompt_points = torch.from_numpy(
                        per_input_negative_prompt_points)

                input_negative_prompt_points.append(
                    per_input_negative_prompt_points)

        assert len(input_positive_prompt_points) == len(
            input_negative_prompt_points)

        input_prompt_points = []
        for per_input_positive_prompt_points, per_input_negative_prompt_points in zip(
                input_positive_prompt_points, input_negative_prompt_points):
            if per_input_positive_prompt_points is not None and per_input_negative_prompt_points is not None:
                per_input_prompt_points = torch.cat([
                    per_input_positive_prompt_points.clone(),
                    per_input_negative_prompt_points.clone(),
                ],
                                                    dim=0)
            elif per_input_positive_prompt_points is not None:
                per_input_prompt_points = per_input_positive_prompt_points.clone(
                )
            elif per_input_negative_prompt_points is not None:
                per_input_prompt_points = per_input_negative_prompt_points.clone(
                )
            else:
                per_input_prompt_points = None
            input_prompt_points.append(per_input_prompt_points)

        input_prompt_boxs = []
        for i, per_prompt_box in enumerate(prompt_boxs):
            per_input_prompt_box = np.zeros((4), dtype=np.float32)
            per_input_prompt_box[0:per_prompt_box.
                                 shape[0]] = per_prompt_box[0:4]
            # [4]
            per_input_prompt_box = torch.from_numpy(per_input_prompt_box)
            input_prompt_boxs.append(per_input_prompt_box)

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

        assert len(input_images) == len(input_boxes) == len(
            input_masks) == len(input_positive_prompt_points) == len(
                input_negative_prompt_points) == len(
                    input_prompt_points) == len(input_prompt_boxs) == len(
                        input_prompt_masks)

        batch_prompts = []
        for per_input_image, per_input_prompt_points, per_input_prompt_box, per_input_prompt_mask, per_image_mask in zip(
                input_images, input_prompt_points, input_prompt_boxs,
                input_prompt_masks, input_masks):
            per_image_prompts = {
                'prompt_point': per_input_prompt_points.unsqueeze(0),
                'prompt_box': per_input_prompt_box.unsqueeze(0),
                'prompt_mask': per_input_prompt_mask.unsqueeze(0).unsqueeze(0),
            }
            batch_prompts.append(per_image_prompts)

        assert len(input_images) == len(input_boxes) == len(
            input_masks) == len(input_positive_prompt_points) == len(
                input_negative_prompt_points) == len(
                    input_prompt_points) == len(input_prompt_boxs) == len(
                        input_prompt_masks) == len(batch_prompts)

        return {
            'origin_image': origin_images,
            'origin_bbox': origin_bboxes,
            'origin_mask': origin_masks,
            'origin_size': origin_sizes,
            'image': input_images,
            'box': input_boxes,
            'mask': input_masks,
            'size': sizes,
            'positive_prompt_point': input_positive_prompt_points,
            'negative_prompt_point': input_negative_prompt_points,
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
            'batch_image': batch_images,
            'batch_mask': batch_masks,
            'batch_prompt': batch_prompts,
        }
