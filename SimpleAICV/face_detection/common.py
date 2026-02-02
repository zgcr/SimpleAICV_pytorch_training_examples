import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.classification.common import load_state_dict


class YoloStyleResize:

    def __init__(self,
                 resize=640,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=[0.5, 1.0]):
        self.resize = resize
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        h, w = image.shape[0], image.shape[1]

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            final_resize = resize_list[random_idx]
        else:
            final_resize = self.resize

        factor = final_resize / max(h, w)

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        size = np.array([resize_h, resize_w]).astype(np.float32)
        factor = np.float32(factor)
        annots[:, :4] *= factor
        scale *= factor

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, annots, scale, size

        return sample


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1]
            w = image.shape[1]

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = w - x2
            annots[:, 2] = w - x1

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, annots, scale, size

        return sample


class RandomVerticalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            image = image[::-1, :]
            h = image.shape[0]

            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()

            annots[:, 1] = h - y2
            annots[:, 3] = h - y1

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, annots, scale, size

        return sample


class RandomCrop:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            h, w = image.shape[0], image.shape[1]
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            crop_xmin = max(
                0, int(max_bbox[0] - np.random.uniform(0, max_left_trans)))
            crop_ymin = max(
                0, int(max_bbox[1] - np.random.uniform(0, max_up_trans)))
            crop_xmax = min(
                w, int(max_bbox[2] + np.random.uniform(0, max_right_trans)))
            crop_ymax = min(
                h, int(max_bbox[3] + np.random.uniform(0, max_down_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_xmin
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_ymin

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, annots, scale, size

        return sample


class RandomTranslate:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            h, w = image.shape[0], image.shape[1]
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            tx = np.random.uniform(-(max_left_trans - 1),
                                   (max_right_trans - 1))
            ty = np.random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, annots, scale, size

        return sample


class RandomGaussianBlur:

    def __init__(self, sigma=[0.5, 1.5], prob=0.5):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            ksize = int(2 * ((sigma - 0.8) / 0.3 + 1) + 1)
            if ksize % 2 == 0:
                ksize += 1

            image = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, annots, scale, size

        return sample


class MainDirectionRandomRotate:

    def __init__(self, angle=[0, 90, 180, 270], prob=[0.55, 0.15, 0.15, 0.15]):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        matrix = np.eye(3, dtype=np.float32)
        h, w = image.shape[0], image.shape[1]

        center_matrix = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]],
                                 dtype=np.float32)

        matrix = np.matmul(center_matrix, matrix)

        angle = np.random.choice(self.angle, p=self.prob)

        if angle == 0:
            return sample

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

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        new_annots = copy.deepcopy(annots)

        for i in range(len(new_annots)):
            bbox = new_annots[i]
            label = new_annots[i][4]
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                             [bbox[0], bbox[3]], [bbox[2], bbox[3]]])
            box = np.array([np.append(xy, 1) for xy in bbox])
            t_matrix = matrix[0:2, :].transpose(1, 0)
            box = np.dot(box, t_matrix)  #[[x1,y1],[x2,y2]]
            box = np.array([
                np.min(box[:, 0]),
                np.min(box[:, 1]),
                np.max(box[:, 0]),
                np.max(box[:, 1]),
                label,
            ])
            new_annots[i] = box.astype(np.float32)

        new_annots = np.array(new_annots, dtype=np.float32)

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, new_annots, scale, size

        return sample


class RandomRotate:

    def __init__(self, angle=[-15, 15], prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
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

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        new_annots = copy.deepcopy(annots)

        for i in range(len(new_annots)):
            bbox = new_annots[i]
            label = new_annots[i][4]
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                             [bbox[0], bbox[3]], [bbox[2], bbox[3]]])
            box = np.array([np.append(xy, 1) for xy in bbox])
            t_matrix = matrix[0:2, :].transpose(1, 0)
            box = np.dot(box, t_matrix)  #[[x1,y1],[x2,y2]]
            box = np.array([
                np.min(box[:, 0]),
                np.min(box[:, 1]),
                np.max(box[:, 0]),
                np.max(box[:, 1]),
                label,
            ])
            new_annots[i] = box.astype(np.float32)

        new_annots = np.array(new_annots, dtype=np.float32)

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, new_annots, scale, size

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        image = image / 255.

        sample['image'], sample['annots'], sample['scale'], sample[
            'size'] = image, annots, scale, size

        return sample


class FaceDetectionCollater:

    def __init__(self, resize=640):
        self.resize = resize

    def __call__(self, data):
        images = [s['image'] for s in data]
        annots = [s['annots'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        input_images = np.zeros((len(images), self.resize, self.resize, 3),
                                dtype=np.float32)

        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)
        input_images = input_images.float()

        max_annots_num = max(annot.shape[0] for annot in annots)
        if max_annots_num > 0:
            input_annots = np.ones(
                (len(annots), max_annots_num, 5), dtype=np.float32) * (-1)
            for i, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    input_annots[i, :annot.shape[0], :] = annot
        else:
            input_annots = np.ones(
                (len(annots), 1, 5), dtype=np.float32) * (-1)

        input_annots = torch.from_numpy(input_annots)
        input_annots = input_annots.float()

        scales = np.array(scales, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'annots': input_annots,
            'scale': scales,
            'size': sizes,
        }
