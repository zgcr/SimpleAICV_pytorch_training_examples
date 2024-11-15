import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import numpy as np

import torch

from simpleAICV.classification.common import load_state_dict


class RandomCrop:

    def __init__(self, prob=0.5, filter_percent=0.1, random_range=[0, 0.25]):
        self.prob = prob
        self.filter_percent = filter_percent
        self.random_range = random_range

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        if np.random.uniform(0, 1) > self.prob:
            return sample

        crop_origin_mask = mask.copy()
        crop_origin_mask[crop_origin_mask > 0] = 1
        crop_origin_mask = (crop_origin_mask * 255.).astype(np.uint8)

        mask_h, mask_w = crop_origin_mask.shape[0], crop_origin_mask.shape[1]
        ret, binary = cv2.threshold(crop_origin_mask, 128, 255,
                                    cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        result_area = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w / mask_w > self.filter_percent and h / mask_h > self.filter_percent:
                # x_min,y_min,x_max,y_max
                result_area.append([x, y, x + w, y + h])

        if len(result_area) == 0:
            return sample

        total_x_min, total_y_min, total_x_max, total_y_max = result_area[0]
        for per_area in result_area:
            if per_area[0] < total_x_min:
                total_x_min = per_area[0]
            if per_area[1] < total_y_min:
                total_y_min = per_area[1]
            if per_area[2] > total_x_max:
                total_x_max = per_area[2]
            if per_area[3] > total_y_max:
                total_y_max = per_area[3]

        total_x_min = max(total_x_min, 0)
        total_y_min = max(total_y_min, 0)
        total_x_max = min(total_x_max, mask_w)
        total_y_max = min(total_y_max, mask_h)

        w_random_expand_ratio = np.random.uniform(self.random_range[0],
                                                  self.random_range[1])
        h_random_expand_ratio = np.random.uniform(self.random_range[0],
                                                  self.random_range[1])
        total_x_center, total_y_center = int(
            (total_x_min + total_x_max) / 2.), int(
                (total_y_min + total_y_max) / 2.)
        total_w, total_h = total_x_max - total_x_min, total_y_max - total_y_min
        expand_w, expand_h = int(total_w * (1. + w_random_expand_ratio)), int(
            total_h * (1. + h_random_expand_ratio))

        final_total_x_min, final_total_y_min, final_total_x_max, final_total_y_max = total_x_center - int(
            expand_w / 2.), total_y_center - int(
                expand_h / 2.), total_x_center + int(
                    expand_w / 2.), total_y_center + int(expand_h / 2.)

        final_total_x_min = max(final_total_x_min, 0)
        final_total_y_min = max(final_total_y_min, 0)
        final_total_x_max = min(final_total_x_max, mask_w)
        final_total_y_max = min(final_total_y_max, mask_h)

        image = image[final_total_y_min:final_total_y_max,
                      final_total_x_min:final_total_x_max, :]
        mask = mask[final_total_y_min:final_total_y_max,
                    final_total_x_min:final_total_x_max]

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class RandomShrink:

    def __init__(self, prob=0.3, random_resize_range=[320, 832]):
        self.prob = prob
        self.random_resize_range = random_resize_range

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        if np.random.uniform(0, 1) > self.prob:
            return sample

        h, w, _ = image.shape

        scale_range = [
            int(self.random_resize_range[0]),
            int(self.random_resize_range[1])
        ]
        resize_list = [
            i // 64 * 64 for i in range(scale_range[0], scale_range[1] + 64)
        ]
        resize_list = list(set(resize_list))

        random_idx = np.random.randint(0, len(resize_list))
        final_resize = resize_list[random_idx]

        factor = final_resize / max(h, w)

        resize_w, resize_h = int(w * float(factor) +
                                 0.5), int(h * float(factor) + 0.5)
        # print(image.shape, resize_w, resize_h, image_path)
        image = cv2.resize(image, (resize_w, resize_h))
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class RandomRGBToGRAY:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        if np.random.uniform(0, 1) < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class RandomVerticalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        if np.random.uniform(0, 1) < self.prob:
            image = image[::-1, :, :]
            mask = mask[::-1, :]

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class RandomRotate:

    def __init__(self, angle=[-5, 5], prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        if np.random.uniform(0, 1) > self.prob:
            return sample

        matrix = np.eye(3, dtype=np.float32)
        h, w = image.shape[0], image.shape[1]
        center_matrix = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]],
                                 dtype=np.float32)

        matrix = np.matmul(center_matrix, matrix)
        angle = np.random.uniform(self.angle[0], self.angle[1])
        rad = -1.0 * np.deg2rad(angle)
        rad_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                               [-np.sin(rad), np.cos(rad), 0], [0, 0, 1]],
                              dtype=np.float32)
        matrix = np.matmul(rad_matrix, matrix)

        invert_center_matrix = np.array(
            [[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], dtype=np.float32)
        matrix = np.matmul(invert_center_matrix, matrix)

        corners_matrix = np.array([[0, 0], [0, h - 1], [w - 1, h - 1],
                                   [w - 1, 0]])
        x, y = np.transpose(corners_matrix)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.T, matrix.T)
        dst[dst[:, 2] == 0, 2] = np.finfo(float).eps
        dst[:, :2] /= dst[:, 2:3]
        corners_matrix = dst[:, :2]

        minc, minr, maxc, maxr = corners_matrix[:, 0].min(
        ), corners_matrix[:, 1].min(), corners_matrix[:, 0].max(
        ), corners_matrix[:, 1].max()
        new_h, new_w = int(np.round(maxr - minr + 1)), int(
            np.round(maxc - minc + 1))

        translate_matrix = np.array([[1, 0, -minc], [0, 1, -minr], [0, 0, 1]])
        matrix = np.matmul(translate_matrix, matrix)

        image = cv2.warpAffine(image,
                               matrix[0:2, :], (new_w, new_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        mask = cv2.warpAffine(mask,
                              matrix[0:2, :], (new_w, new_h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class YoloStyleResize:

    def __init__(self, resize=832):
        self.resize = resize

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        h, w, _ = image.shape

        factor = self.resize / max(h, w)

        resize_w, resize_h = int(w * float(factor) +
                                 0.5), int(h * float(factor) + 0.5)
        image = cv2.resize(image, (resize_w, resize_h))
        mask = cv2.resize(mask, (resize_w, resize_h),
                          interpolation=cv2.INTER_NEAREST)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class Resize:

    def __init__(self, resize=832):
        self.resize = resize

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        image = cv2.resize(image, (self.resize, self.resize))
        mask = cv2.resize(mask, (self.resize, self.resize),
                          interpolation=cv2.INTER_NEAREST)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask, size = sample['image'], sample['mask'], sample['size']

        image = image / 255.

        sample['image'], sample['mask'], sample['size'] = image, mask, size

        return sample


class HumanParsingCollater:

    def __init__(self, resize=832):
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

        input_masks = np.zeros((len(masks), self.resize, self.resize),
                               dtype=np.float32)
        for i, mask in enumerate(masks):
            input_masks[i, 0:mask.shape[0], 0:mask.shape[1]] = mask
        input_masks = torch.from_numpy(input_masks)

        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'mask': input_masks,
            'size': sizes,
        }
