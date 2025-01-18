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

from simpleAICV.classification.common import AverageMeter, load_state_dict


class SAMMattingRandomScale:

    def __init__(self, scale=[0.5, 1.0], area_ratio=0.01, prob=0.5):
        self.scale = scale
        self.area_ratio = area_ratio
        self.prob = prob

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        trimap, fg_map, bg_map = sample['trimap'], sample['fg_map'], sample[
            'bg_map']

        if np.random.uniform(0, 1) < self.prob:
            h, w = image.shape[0], image.shape[1]

            if (np.sum(mask > 0) / float(h * w)) < self.area_ratio:
                return sample

            center = (int(w / 2), int(h / 2))
            scale = np.random.uniform(self.scale[0], self.scale[1])
            M = cv2.getRotationMatrix2D(center, 0, scale)

            radian = np.deg2rad(0)
            new_w = int((abs(np.sin(radian) * h) + abs(np.cos(radian) * w)))
            new_h = int((abs(np.cos(radian) * h) + abs(np.sin(radian) * w)))

            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            image = cv2.warpAffine(image, M, (new_w, new_h))
            mask = cv2.warpAffine(mask, M, (new_w, new_h))
            prompt_mask = cv2.warpAffine(prompt_mask, M, (new_w, new_h))
            trimap = cv2.warpAffine(trimap,
                                    M, (new_w, new_h),
                                    flags=cv2.INTER_NEAREST)
            fg_map = cv2.warpAffine(fg_map, M, (new_w, new_h))
            bg_map = cv2.warpAffine(bg_map, M, (new_w, new_h))

            size = np.array([image.shape[0],
                             image.shape[1]]).astype(np.float32)

            box = self.transform_box(box, M)
            prompt_box = self.transform_box(prompt_box, M)

            new_positive_prompt_point = []
            for per_positive_prompt_point in positive_prompt_point:
                per_positive_prompt_point_label = per_positive_prompt_point[2]
                per_positive_prompt_point_coord = per_positive_prompt_point[
                    0:2]
                per_positive_prompt_point_coord = self.transform_point(
                    per_positive_prompt_point_coord, M)

                new_per_positive_prompt_point = [
                    per_positive_prompt_point_coord[0],
                    per_positive_prompt_point_coord[1],
                    per_positive_prompt_point_label,
                ]
                new_per_positive_prompt_point = np.array(
                    new_per_positive_prompt_point, dtype=np.float32)
                new_positive_prompt_point.append(new_per_positive_prompt_point)

            new_positive_prompt_point = np.array(new_positive_prompt_point,
                                                 dtype=np.float32)
            positive_prompt_point = new_positive_prompt_point

            new_negative_prompt_point = []
            for per_negative_prompt_point in negative_prompt_point:
                per_negative_prompt_point_label = per_negative_prompt_point[2]
                per_negative_prompt_point_coord = per_negative_prompt_point[
                    0:2]
                per_negative_prompt_point_coord = self.transform_point(
                    per_negative_prompt_point_coord, M)

                new_per_negative_prompt_point = [
                    per_negative_prompt_point_coord[0],
                    per_negative_prompt_point_coord[1],
                    per_negative_prompt_point_label,
                ]
                new_per_negative_prompt_point = np.array(
                    new_per_negative_prompt_point, dtype=np.float32)
                new_negative_prompt_point.append(new_per_negative_prompt_point)

            new_negative_prompt_point = np.array(new_negative_prompt_point,
                                                 dtype=np.float32)
            negative_prompt_point = new_negative_prompt_point

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['positive_prompt_point'], sample[
            'negative_prompt_point'], sample['prompt_box'], sample[
                'prompt_mask'] = positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimap, fg_map, bg_map

        return sample

    def transform_box(self, box, M):
        """
        使用仿射矩阵 M 转换边界框坐标。
        Args:
            box (list or np.ndarray): 边界框，格式为 [x_min, y_min, x_max, y_max]
            M (np.ndarray): 仿射变换矩阵 2x3

        Returns:
            list: 转换后的边界框 [new_x_min, new_y_min, new_x_max, new_y_max]
        """
        # 提取四个顶点
        points = np.array([[box[0], box[1]], [box[2], box[1]],
                           [box[2], box[3]], [box[0], box[3]]])

        # 转换为齐次坐标
        ones = np.ones((points.shape[0], 1))
        points_ones = np.hstack([points, ones])

        # 应用仿射变换
        transformed_points = M.dot(points_ones.T).T

        # 计算新的边界框
        x_coords = transformed_points[:, 0]
        y_coords = transformed_points[:, 1]
        new_box = [
            x_coords.min(),
            y_coords.min(),
            x_coords.max(),
            y_coords.max()
        ]
        new_box = np.array(new_box, dtype=np.float32)

        return new_box

    def transform_point(self, point, M):
        """
        使用仿射矩阵 M 转换单个点的坐标。

        Args:
            point (list or tuple or np.ndarray): 点的坐标，格式为 [x, y]
            M (np.ndarray): 仿射变换矩阵 2x3

        Returns:
            list: 转换后的点坐标 [new_x, new_y]
        """
        assert len(point) == 2
        # 转换为齐次坐标
        point_homogeneous = np.array([point[0], point[1], 1])

        # 应用仿射变换
        transformed_point = M.dot(point_homogeneous)

        transformed_point = np.array(transformed_point, dtype=np.float32)

        return transformed_point


class SAMMattingRandomTranslate:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        trimap, fg_map, bg_map = sample['trimap'], sample['fg_map'], sample[
            'bg_map']

        if np.random.uniform(0, 1) < self.prob:
            h, w = image.shape[0], image.shape[1]
            max_left_trans, max_up_trans = box[0], box[1]
            max_right_trans, max_down_trans = w - box[2], h - box[3]
            tx = np.random.uniform(-(max_left_trans - 1),
                                   (max_right_trans - 1))
            ty = np.random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])

            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
            prompt_mask = cv2.warpAffine(prompt_mask, M, (w, h))
            trimap = cv2.warpAffine(trimap, M, (w, h), flags=cv2.INTER_NEAREST)
            fg_map = cv2.warpAffine(fg_map, M, (w, h))
            bg_map = cv2.warpAffine(bg_map, M, (w, h))

            size = np.array([image.shape[0],
                             image.shape[1]]).astype(np.float32)

            box = self.transform_box(box, M)
            prompt_box = self.transform_box(prompt_box, M)

            new_positive_prompt_point = []
            for per_positive_prompt_point in positive_prompt_point:
                per_positive_prompt_point_label = per_positive_prompt_point[2]
                per_positive_prompt_point_coord = per_positive_prompt_point[
                    0:2]
                per_positive_prompt_point_coord = self.transform_point(
                    per_positive_prompt_point_coord, M)

                new_per_positive_prompt_point = [
                    per_positive_prompt_point_coord[0],
                    per_positive_prompt_point_coord[1],
                    per_positive_prompt_point_label,
                ]
                new_per_positive_prompt_point = np.array(
                    new_per_positive_prompt_point, dtype=np.float32)
                new_positive_prompt_point.append(new_per_positive_prompt_point)

            new_positive_prompt_point = np.array(new_positive_prompt_point,
                                                 dtype=np.float32)
            positive_prompt_point = new_positive_prompt_point

            new_negative_prompt_point = []
            for per_negative_prompt_point in negative_prompt_point:
                per_negative_prompt_point_label = per_negative_prompt_point[2]
                per_negative_prompt_point_coord = per_negative_prompt_point[
                    0:2]
                per_negative_prompt_point_coord = self.transform_point(
                    per_negative_prompt_point_coord, M)

                new_per_negative_prompt_point = [
                    per_negative_prompt_point_coord[0],
                    per_negative_prompt_point_coord[1],
                    per_negative_prompt_point_label,
                ]
                new_per_negative_prompt_point = np.array(
                    new_per_negative_prompt_point, dtype=np.float32)
                new_negative_prompt_point.append(new_per_negative_prompt_point)

            new_negative_prompt_point = np.array(new_negative_prompt_point,
                                                 dtype=np.float32)
            negative_prompt_point = new_negative_prompt_point

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['positive_prompt_point'], sample[
            'negative_prompt_point'], sample['prompt_box'], sample[
                'prompt_mask'] = positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimap, fg_map, bg_map

        return sample

    def transform_box(self, box, M):
        """
        使用仿射矩阵 M 转换边界框坐标。
        Args:
            box (list or np.ndarray): 边界框，格式为 [x_min, y_min, x_max, y_max]
            M (np.ndarray): 仿射变换矩阵 2x3

        Returns:
            list: 转换后的边界框 [new_x_min, new_y_min, new_x_max, new_y_max]
        """
        # 提取四个顶点
        points = np.array([[box[0], box[1]], [box[2], box[1]],
                           [box[2], box[3]], [box[0], box[3]]])

        # 转换为齐次坐标
        ones = np.ones((points.shape[0], 1))
        points_ones = np.hstack([points, ones])

        # 应用仿射变换
        transformed_points = M.dot(points_ones.T).T

        # 计算新的边界框
        x_coords = transformed_points[:, 0]
        y_coords = transformed_points[:, 1]
        new_box = [
            x_coords.min(),
            y_coords.min(),
            x_coords.max(),
            y_coords.max()
        ]
        new_box = np.array(new_box, dtype=np.float32)

        return new_box

    def transform_point(self, point, M):
        """
        使用仿射矩阵 M 转换单个点的坐标。

        Args:
            point (list or tuple or np.ndarray): 点的坐标，格式为 [x, y]
            M (np.ndarray): 仿射变换矩阵 2x3

        Returns:
            list: 转换后的点坐标 [new_x, new_y]
        """
        assert len(point) == 2
        # 转换为齐次坐标
        point_homogeneous = np.array([point[0], point[1], 1])

        # 应用仿射变换
        transformed_point = M.dot(point_homogeneous)

        transformed_point = np.array(transformed_point, dtype=np.float32)

        return transformed_point


class SAMMattingResize:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        trimap, fg_map, bg_map = sample['trimap'], sample['fg_map'], sample[
            'bg_map']

        h, w, _ = image.shape
        factor = self.resize / max(h, w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        box[0:4] *= factor
        mask = cv2.resize(mask, (resize_w, resize_h))

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        positive_prompt_point[:, 0:2] *= factor
        negative_prompt_point[:, 0:2] *= factor
        prompt_box[0:4] *= factor
        prompt_mask = cv2.resize(prompt_mask, (resize_w, resize_h))

        trimap = cv2.resize(trimap, (resize_w, resize_h),
                            interpolation=cv2.INTER_NEAREST)
        fg_map = cv2.resize(fg_map, (resize_w, resize_h))
        bg_map = cv2.resize(bg_map, (resize_w, resize_h))

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['positive_prompt_point'], sample[
            'negative_prompt_point'], sample['prompt_box'], sample[
                'prompt_mask'] = positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimap, fg_map, bg_map

        return sample


class SAMMattingRandomRGBToGRAY:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        trimap, fg_map, bg_map = sample['trimap'], sample['fg_map'], sample[
            'bg_map']

        if np.random.uniform(0, 1) < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            fg_map = cv2.cvtColor(fg_map, cv2.COLOR_RGB2GRAY)
            fg_map = cv2.cvtColor(fg_map, cv2.COLOR_GRAY2RGB)

            bg_map = cv2.cvtColor(bg_map, cv2.COLOR_RGB2GRAY)
            bg_map = cv2.cvtColor(bg_map, cv2.COLOR_GRAY2RGB)

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['positive_prompt_point'], sample[
            'negative_prompt_point'], sample['prompt_box'], sample[
                'prompt_mask'] = positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimap, fg_map, bg_map

        return sample


class SAMMattingRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        trimap, fg_map, bg_map = sample['trimap'], sample['fg_map'], sample[
            'bg_map']

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]

            trimap = trimap[:, ::-1]
            fg_map = fg_map[:, ::-1, :]
            bg_map = bg_map[:, ::-1, :]

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

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['positive_prompt_point'], sample[
            'negative_prompt_point'], sample['prompt_box'], sample[
                'prompt_mask'] = positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimap, fg_map, bg_map

        return sample


class SAMMattingRandomVerticalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, box, mask, size = sample['image'], sample['box'], sample[
            'mask'], sample['size']

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        trimap, fg_map, bg_map = sample['trimap'], sample['fg_map'], sample[
            'bg_map']

        if np.random.uniform(0, 1) < self.prob:
            image = image[::-1, :, :]
            mask = mask[::-1, :]

            trimap = trimap[::-1, :]
            fg_map = fg_map[::-1, :, :]
            bg_map = bg_map[::-1, :, :]

            prompt_mask = prompt_mask[::-1, :]

            h, _, _ = image.shape

            y1 = box[1].copy()
            y2 = box[3].copy()

            box[1] = h - y2
            box[3] = h - y1

            y1 = prompt_box[1].copy()
            y2 = prompt_box[3].copy()

            prompt_box[1] = h - y2
            prompt_box[3] = h - y1

            h, _, _ = image.shape

            for i in range(len(positive_prompt_point)):
                positive_prompt_point[i][1] = h - positive_prompt_point[i][1]

            for i in range(len(negative_prompt_point)):
                negative_prompt_point[i][1] = h - negative_prompt_point[i][1]

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['positive_prompt_point'], sample[
            'negative_prompt_point'], sample['prompt_box'], sample[
                'prompt_mask'] = positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimap, fg_map, bg_map

        return sample


class SAMMattingNormalize:

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

        positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask = sample[
            'positive_prompt_point'], sample['negative_prompt_point'], sample[
                'prompt_box'], sample['prompt_mask']

        trimap, fg_map, bg_map = sample['trimap'], sample['fg_map'], sample[
            'bg_map']

        image = (image - self.mean) / self.std
        fg_map = (fg_map - self.mean) / self.std
        bg_map = (bg_map - self.mean) / self.std

        sample['image'], sample['box'], sample['mask'], sample[
            'size'] = image, box, mask, size

        sample['positive_prompt_point'], sample[
            'negative_prompt_point'], sample['prompt_box'], sample[
                'prompt_mask'] = positive_prompt_point, negative_prompt_point, prompt_box, prompt_mask

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimap, fg_map, bg_map

        return sample


class SAMMattingBatchCollater:

    def __init__(self, resize, positive_point_num_range=[1, 9]):
        self.resize = resize
        assert resize % 64 == 0

        self.prompt_mask_size = resize // 4

        assert isinstance(positive_point_num_range, (int, list))
        if isinstance(positive_point_num_range, list):
            assert isinstance(positive_point_num_range[0], int)
            assert isinstance(positive_point_num_range[1], int)
            assert positive_point_num_range[0] <= positive_point_num_range[1]
            assert positive_point_num_range[0] >= 1
        if isinstance(positive_point_num_range, int):
            assert positive_point_num_range >= 0

        self.positive_point_num_range = positive_point_num_range

    def __call__(self, data):
        images = [s['image'] for s in data]
        boxes = [x['box'] for x in data]
        masks = [x['mask'] for x in data]
        sizes = [x['size'] for x in data]

        positive_prompt_points = [x['positive_prompt_point'] for x in data]
        prompt_boxs = [x['prompt_box'] for x in data]
        prompt_masks = [x['prompt_mask'] for x in data]

        trimaps = [x['trimap'] for x in data]
        fg_maps = [x['fg_map'] for x in data]
        bg_maps = [x['bg_map'] for x in data]

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

        input_boxes = []
        for i, per_box in enumerate(boxes):
            per_input_box = np.zeros((4), dtype=np.float32)
            per_input_box[0:per_box.shape[0]] = per_box[0:4]
            # [4]
            per_input_box = torch.from_numpy(per_input_box)
            input_boxes.append(per_input_box)
        input_boxes = torch.stack(input_boxes, dim=0)

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

        assert len(input_images) == len(input_boxes) == len(input_masks)

        sizes = np.array(sizes, dtype=np.float32)

        input_positive_prompt_points = []
        if isinstance(self.positive_point_num_range, list):
            random_positive_point_num = np.random.randint(
                self.positive_point_num_range[0],
                self.positive_point_num_range[1] + 1)

            for i in range(len(positive_prompt_points)):
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
        if len(input_positive_prompt_points) > 0:
            input_positive_prompt_points = torch.stack(
                input_positive_prompt_points, dim=0)

        input_prompt_points = input_positive_prompt_points

        input_prompt_boxs = []
        for i, per_prompt_box in enumerate(prompt_boxs):
            per_input_prompt_box = np.zeros((4), dtype=np.float32)
            per_input_prompt_box[0:per_prompt_box.
                                 shape[0]] = per_prompt_box[0:4]
            # [4]
            per_input_prompt_box = torch.from_numpy(per_input_prompt_box)
            input_prompt_boxs.append(per_input_prompt_box)
        input_prompt_boxs = torch.stack(input_prompt_boxs, dim=0)

        input_prompt_masks = []
        for i, per_prompt_mask in enumerate(prompt_masks):
            h, w = per_prompt_mask.shape
            factor = self.prompt_mask_size / max(h, w)
            resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
            per_prompt_mask = cv2.resize(per_prompt_mask, (resize_w, resize_h))

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

        assert len(input_images) == len(input_boxes) == len(
            input_masks) == len(input_positive_prompt_points) == len(
                input_prompt_points) == len(input_prompt_boxs) == len(
                    input_prompt_masks)

        input_trimaps = np.zeros((len(trimaps), self.resize, self.resize),
                                 dtype=np.uint8)
        for i, trimap in enumerate(trimaps):
            input_trimaps[i, 0:trimap.shape[0], 0:trimap.shape[1]] = trimap
        input_trimaps = torch.from_numpy(input_trimaps)

        input_fg_maps = np.zeros((len(fg_maps), self.resize, self.resize, 3),
                                 dtype=np.float32)
        for i, fg_map in enumerate(fg_maps):
            input_fg_maps[i, 0:fg_map.shape[0], 0:fg_map.shape[1], :] = fg_map
        input_fg_maps = torch.from_numpy(input_fg_maps)
        # B H W 3 ->B 3 H W
        input_fg_maps = input_fg_maps.permute(0, 3, 1, 2)

        input_bg_maps = np.zeros((len(bg_maps), self.resize, self.resize, 3),
                                 dtype=np.float32)
        for i, bg_map in enumerate(bg_maps):
            input_bg_maps[i, 0:bg_map.shape[0], 0:bg_map.shape[1], :] = bg_map
        input_bg_maps = torch.from_numpy(input_bg_maps)
        # B H W 3 ->B 3 H W
        input_bg_maps = input_bg_maps.permute(0, 3, 1, 2)

        assert len(input_images) == len(input_boxes) == len(
            input_masks) == len(input_prompt_points) == len(
                input_prompt_boxs) == len(input_prompt_masks) == len(
                    input_trimaps) == len(input_fg_maps) == len(input_bg_maps)

        return {
            'image': input_images,
            'box': input_boxes,
            'mask': input_masks,
            'size': sizes,
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
            'trimap': input_trimaps,
            'fg_map': input_fg_maps,
            'bg_map': input_bg_maps,
        }
