import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import copy
import numpy as np

import pyclipper
from shapely.geometry import Polygon

import torch
import torch.nn.functional as F

from simpleAICV.classification.common import load_state_dict, AverageMeter


class MainDirectionRandomRotate:

    def __init__(self, angle=[0, 90, 180, 270], prob=[0.55, 0.15, 0.15, 0.15]):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        h, w, _ = image.shape
        matrix = np.eye(3, dtype=np.float32)
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

        size = [image.shape[0], image.shape[1]]

        new_annots = copy.deepcopy(annots)

        for i in range(len(new_annots)):
            box = np.array([
                np.append(per_coord, 1)
                for per_coord in new_annots[i]['points']
            ])
            box = box.transpose(1, 0)
            box = np.dot(matrix[0:2, :], box)
            box = box.transpose(1, 0)
            new_annots[i]['points'] = box.astype(np.float32)

        sample = {
            'image': image,
            'annots': new_annots,
            'scale': scale,
            'size': size,
        }

        return sample


class RandomRotate:

    def __init__(self, angle=[-15, 15], prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if np.random.uniform(0, 1) > self.prob:
            return sample

        h, w, _ = image.shape
        matrix = np.eye(3, dtype=np.float32)
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

        size = [image.shape[0], image.shape[1]]

        new_annots = copy.deepcopy(annots)

        for i in range(len(new_annots)):
            box = np.array([
                np.append(per_coord, 1)
                for per_coord in new_annots[i]['points']
            ])
            box = box.transpose(1, 0)
            box = np.dot(matrix[0:2, :], box)
            box = box.transpose(1, 0)
            new_annots[i]['points'] = box.astype(np.float32)

        sample = {
            'image': image,
            'annots': new_annots,
            'scale': scale,
            'size': size,
        }

        return sample


class Resize:

    def __init__(self, resize=960):
        self.resize = resize

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        h, w, _ = image.shape
        factor = self.resize / max(h, w)

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        factor = np.float32(factor)
        scale *= factor

        size = [image.shape[0], image.shape[1]]

        new_annots = copy.deepcopy(annots)

        for i in range(len(new_annots)):
            new_annots[i]['points'] = np.array(
                new_annots[i]['points']) * factor

        sample = {
            'image': image,
            'annots': new_annots,
            'scale': scale,
            'size': size,
        }

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        image = image / 255.
        image = image.astype(np.float32)

        new_annots = copy.deepcopy(annots)

        sample = {
            'image': image,
            'annots': new_annots,
            'scale': scale,
            'size': size,
        }

        return sample


class TextDetectionCollater:

    def __init__(self, resize=960):
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

        new_annots = copy.deepcopy(annots)

        for i in range(len(new_annots)):
            for j in range(len(new_annots[i])):
                new_annots[i][j]['points'] = np.array(
                    new_annots[i][j]['points']).astype(np.float32)

        return {
            'image': input_images,
            'annots': new_annots,
            'scale': scales,
            'size': sizes,
        }


class GenerateProbabilityThresholdMask:

    def __init__(self,
                 min_box_size=3,
                 min_max_threshold=[0.3, 0.7],
                 shrink_ratio=0.6):
        self.min_box_size = min_box_size
        self.min_max_threshold = min_max_threshold
        self.shrink_ratio = shrink_ratio

    def __call__(self, images, annots, sizes):
        b, _, h, w = images.shape

        probability_mask = np.zeros((b, h, w), dtype=np.float32)
        probability_ignore_mask = np.ones((b, h, w), dtype=np.float32)

        threshold_mask = np.zeros((b, h, w), dtype=np.float32)
        threshold_ignore_mask = np.zeros((b, h, w), dtype=np.float32)

        new_annots = copy.deepcopy(annots)

        for idx, per_image_annots in enumerate(new_annots):
            new_per_image_annots = copy.deepcopy(per_image_annots)
            for per_box_label in new_per_image_annots:
                per_box, label, ignore = per_box_label[
                    'points'], per_box_label['label'], per_box_label['ignore']
                per_box = np.array(per_box, dtype=np.float32)
                height, width = max(per_box[:, 1]) - min(per_box[:, 1]), max(
                    per_box[:, 0]) - min(per_box[:, 0])
                per_box_area = cv2.contourArea(per_box)

                if ignore or min(
                        height, width
                ) < self.min_box_size or per_box_area < self.min_box_size**2:
                    cv2.fillPoly(probability_ignore_mask[idx],
                                 [per_box.astype(np.int32)], 0.0)
                    continue

                shrinked_box, border_box, distance = self.generate_shrink_border_polygon_by_pyclipper(
                    per_box)
                if len(shrinked_box) != 1 or len(border_box) != 1:
                    cv2.fillPoly(probability_ignore_mask[idx],
                                 [per_box.astype(np.int32)], 0.0)
                    continue

                # cut超出边界的框，检测不合法多边形
                h, w = sizes[idx]
                image_matrix = np.array([[0, 0], [w, 0], [w, h], [0, h]])
                border_box = self.cut_box_from_image_border(
                    border_box, image_matrix)
                if len(border_box) != 1:
                    cv2.fillPoly(probability_ignore_mask[idx],
                                 [per_box.astype(np.int32)], 0.0)
                    continue

                # 检测不合法多边形
                shrinked_box = shrinked_box[0]
                shrinked_box = pyclipper.SimplifyPolygon(shrinked_box)
                if len(shrinked_box) != 1:
                    cv2.fillPoly(probability_ignore_mask[idx],
                                 [per_box.astype(np.int32)], 0.0)
                    continue

                shrinked_box, border_box = np.array(shrinked_box), np.array(
                    border_box)
                if shrinked_box.shape[2] != 2 or border_box.shape[2] != 2:
                    cv2.fillPoly(probability_ignore_mask[idx],
                                 [per_box.astype(np.int32)], 0.0)
                    continue

                shrinked_box, border_box = shrinked_box[0], border_box[0]

                cv2.fillPoly(probability_mask[idx],
                             [shrinked_box.astype(np.int32)], 1.0)
                cv2.fillPoly(threshold_ignore_mask[idx],
                             [border_box.astype(np.int32)], 1.0)

                threshold_mask[idx] = self.draw_threshold_mask(
                    distance, per_box, border_box, threshold_mask[idx])

        threshold_mask = threshold_mask * (
            self.min_max_threshold[1] -
            self.min_max_threshold[0]) + self.min_max_threshold[0]

        return probability_mask, probability_ignore_mask, threshold_mask, threshold_ignore_mask

    def cut_box_from_image_border(self, box, image_matrix):
        pc = pyclipper.Pyclipper()
        pc.AddPath(image_matrix, pyclipper.PT_CLIP, True)
        pc.AddPaths(box, pyclipper.PT_SUBJECT, True)
        box = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD,
                         pyclipper.PFT_EVENODD)

        return box

    def generate_shrink_border_polygon_by_pyclipper(self, polygon_box):
        polygon_shape = Polygon(polygon_box)
        distance = polygon_shape.area * (
            1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon_box]
        list_box = [list(coord) for coord in subject]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(list_box, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)

        shrinked_box = padding.Execute(-distance)
        border_box = padding.Execute(distance)

        return shrinked_box, border_box, distance

    def draw_threshold_mask(self, distance, box, border_box, threshold_mask):
        xmin, ymin = border_box[:, 0].min(), border_box[:, 1].min()
        xmax, ymax = border_box[:, 0].max(), border_box[:, 1].max()
        height, width = ymax - ymin + 1, xmax - xmin + 1
        point_nums = box.shape[0]

        box[:, 0], box[:, 1] = box[:, 0] - xmin, box[:, 1] - ymin

        xs = np.repeat(np.linspace(0, width - 1, num=width).reshape(1, width),
                       height,
                       axis=0)
        ys = np.repeat(np.linspace(0, height - 1,
                                   num=height).reshape(height, 1),
                       width,
                       axis=1)
        distance_map = np.zeros((point_nums, height, width), dtype=np.float32)

        for i in range(point_nums):
            j = (i + 1) % point_nums
            absolute_distance = self.compute_distance(xs, ys, box[i], box[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        valid_xmin = min(max(0, xmin), threshold_mask.shape[1] - 1)
        valid_ymin = min(max(0, ymin), threshold_mask.shape[0] - 1)
        valid_xmax = min(max(0, xmax), threshold_mask.shape[1] - 1)
        valid_ymax = min(max(0, ymax), threshold_mask.shape[0] - 1)

        threshold_mask[valid_ymin:valid_ymax + 1,
                       valid_xmin:valid_xmax + 1] = np.fmax(
                           1 - distance_map[valid_ymin - ymin:valid_ymax -
                                            ymax + height, valid_xmin -
                                            xmin:valid_xmax - xmax + width],
                           threshold_mask[valid_ymin:valid_ymax + 1,
                                          valid_xmin:valid_xmax + 1])

        return threshold_mask

    def compute_distance(self, xs, ys, point1, point2):
        '''
        compute the distance from point to a line
        '''
        square_coords_point1_distance = np.square(xs - point1[0]) + np.square(
            ys - point1[1])
        square_coords_point2_distance = np.square(xs - point2[0]) + np.square(
            ys - point2[1])
        square_line_distance = np.square(point1[0] -
                                         point2[0]) + np.square(point1[1] -
                                                                point2[1])
        # 余弦定理
        cosin = (square_line_distance - square_coords_point1_distance -
                 square_coords_point2_distance) / (
                     2 * np.sqrt(square_coords_point1_distance *
                                 square_coords_point2_distance) + 1e-4)
        cosin = np.clip(cosin, -1, 1)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)

        # 求三角形面积
        result = np.sqrt(square_coords_point1_distance *
                         square_coords_point2_distance * square_sin /
                         square_line_distance)
        result[cosin < 0] = np.sqrt(
            np.fmin(square_coords_point1_distance,
                    square_coords_point2_distance))[cosin < 0]

        return result


class DBNetTextDetectionCollater:

    def __init__(self,
                 resize=960,
                 min_box_size=3,
                 min_max_threshold=[0.3, 0.7],
                 shrink_ratio=0.6):
        self.resize = resize
        self.generate_mask = GenerateProbabilityThresholdMask(
            min_box_size=min_box_size,
            min_max_threshold=min_max_threshold,
            shrink_ratio=shrink_ratio)

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

        new_annots = copy.deepcopy(annots)

        for i in range(len(new_annots)):
            for j in range(len(new_annots[i])):
                new_annots[i][j]['points'] = np.array(
                    new_annots[i][j]['points']).astype(np.float32)

        probability_mask, probability_ignore_mask, threshold_mask, threshold_ignore_mask = self.generate_mask(
            input_images, annots, sizes)
        probability_mask, probability_ignore_mask, threshold_mask, threshold_ignore_mask = torch.from_numpy(
            probability_mask), torch.from_numpy(
                probability_ignore_mask), torch.from_numpy(
                    threshold_mask), torch.from_numpy(threshold_ignore_mask)

        all_shapes = {
            'shape': new_annots,
            'probability_mask': probability_mask,
            'probability_ignore_mask': probability_ignore_mask,
            'threshold_mask': threshold_mask,
            'threshold_ignore_mask': threshold_ignore_mask,
        }

        return {
            'image': input_images,
            'annots': all_shapes,
            'scale': scales,
            'size': sizes,
        }


class PrecisionRecallMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_correct_num = 0
        self.gt_correct_num = 0
        self.pred_num = 0
        self.gt_num = 0
        self.precision = 0
        self.recall = 0

    def update(self, pred_correct_num, gt_correct_num, pred_num, gt_num):
        self.pred_correct_num += pred_correct_num
        self.gt_correct_num += gt_correct_num
        self.pred_num += pred_num
        self.gt_num += gt_num

    def compute(self):
        self.precision = float(
            self.pred_correct_num) / self.pred_num if self.pred_num != 0 else 0
        self.recall = float(
            self.gt_correct_num) / self.gt_num if self.gt_num != 0 else 0
