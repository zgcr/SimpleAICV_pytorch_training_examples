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


class Sam2Resize:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']

        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        frame_nums = images.shape[0]

        resized_images = []
        resized_masks = []
        resized_prompt_points = []
        resized_prompt_boxes = []
        resized_prompt_masks = []
        for frame_idx in range(frame_nums):
            per_image = images[frame_idx]
            origin_h, origin_w = per_image.shape[0], per_image.shape[1]

            per_image = cv2.resize(per_image, (self.resize, self.resize))
            resized_images.append(per_image)

            per_mask = masks[frame_idx]
            per_mask = cv2.resize(per_mask, (self.resize, self.resize),
                                  interpolation=cv2.INTER_NEAREST)
            if len(per_mask.shape) != 3:
                per_mask = np.expand_dims(per_mask, axis=-1)
            resized_masks.append(per_mask)

            per_prompt_mask = prompt_masks[frame_idx]
            per_prompt_mask = cv2.resize(per_prompt_mask,
                                         (self.resize, self.resize),
                                         interpolation=cv2.INTER_NEAREST)
            if len(per_prompt_mask.shape) != 3:
                per_prompt_mask = np.expand_dims(per_prompt_mask, axis=-1)
            resized_prompt_masks.append(per_prompt_mask)

            h_factor, w_factor = self.resize / origin_h, self.resize / origin_w

            per_prompt_points = prompt_points[frame_idx]
            per_prompt_points[:, :, 0] = per_prompt_points[:, :, 0] * w_factor
            per_prompt_points[:, :, 1] = per_prompt_points[:, :, 1] * h_factor
            resized_prompt_points.append(per_prompt_points)

            per_prompt_boxes = prompt_boxes[frame_idx]
            per_prompt_boxes[:, 0] = per_prompt_boxes[:, 0] * w_factor
            per_prompt_boxes[:, 1] = per_prompt_boxes[:, 1] * h_factor
            per_prompt_boxes[:, 2] = per_prompt_boxes[:, 2] * w_factor
            per_prompt_boxes[:, 3] = per_prompt_boxes[:, 3] * h_factor
            resized_prompt_boxes.append(per_prompt_boxes)

        resized_images = np.stack(resized_images, axis=0)
        resized_masks = np.stack(resized_masks, axis=0)

        resized_prompt_masks = np.stack(resized_prompt_masks, axis=0)
        resized_prompt_points = np.stack(resized_prompt_points, axis=0)
        resized_prompt_boxes = np.stack(resized_prompt_boxes, axis=0)

        resized_size = np.array(
            [resized_images.shape[1],
             resized_images.shape[2]]).astype(np.float32)

        sample['image'], sample['mask'], sample[
            'size'] = resized_images, resized_masks, resized_size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = resized_prompt_points, resized_prompt_boxes, resized_prompt_masks

        return sample


class Sam2RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']

        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        if np.random.uniform(0, 1) < self.prob:
            # [T,H,W,3]
            images = images[:, :, ::-1, :].copy()
            masks = masks[:, :, ::-1, :].copy()
            prompt_masks = prompt_masks[:, :, ::-1, :].copy()

            _, _, w, _ = images.shape

            prompt_boxes_x1 = prompt_boxes[:, :, 0].copy()
            prompt_boxes_x2 = prompt_boxes[:, :, 2].copy()

            prompt_boxes[:, :, 0] = w - prompt_boxes_x2
            prompt_boxes[:, :, 2] = w - prompt_boxes_x1

            _, _, w, _ = images.shape
            prompt_points[:, :, :, 0] = w - prompt_points[:, :, :, 0]

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_points, prompt_boxes, prompt_masks

        return sample


class Sam2RandomMosaicAug:

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']

        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        if np.random.uniform(0, 1) < self.prob:
            T, h, w, _ = images.shape
            h_half, w_half = h // 2, w // 2

            # 缩小图像到一半
            images_small = np.zeros((T, h_half, w_half, 3), dtype=images.dtype)
            for t in range(T):
                images_small[t] = cv2.resize(images[t], (w_half, h_half))

            # 创建四宫格图像
            new_images = np.zeros_like(images)
            for t in range(T):
                # 左上
                new_images[t, :h_half, :w_half] = images_small[t]
                # 右上
                new_images[t, :h_half, w_half:] = images_small[t]
                # 左下
                new_images[t, h_half:, :w_half] = images_small[t]
                # 右下
                new_images[t, h_half:, w_half:] = images_small[t]

            new_size = np.array([new_images.shape[1],
                                 new_images.shape[2]]).astype(np.float32)

            # 随机选择区域及其偏移量
            regions = [
                # 左上
                (0, 0),
                # 右上
                (w_half, 0),
                # 左下
                (0, h_half),
                # 右下
                (w_half, h_half),
            ]
            region_idx = np.random.randint(0, 4)
            x_offset, y_offset = regions[region_idx]

            # 调整提示点坐标
            new_prompt_points = prompt_points.copy()
            # 调整x坐标
            new_prompt_points[...,
                              0] = (prompt_points[..., 0] * 0.5) + x_offset
            # 调整y坐标
            new_prompt_points[...,
                              1] = (prompt_points[..., 1] * 0.5) + y_offset

            # 调整边界框坐标（假设为xyxy格式）
            new_prompt_boxes = prompt_boxes.copy()
            # 调整x1, x2
            new_prompt_boxes[..., [0, 2]] = (prompt_boxes[..., [0, 2]] *
                                             0.5) + x_offset
            # 调整y1, y2
            new_prompt_boxes[..., [1, 3]] = (prompt_boxes[..., [1, 3]] *
                                             0.5) + y_offset

            # 调整mask和提示mask
            new_masks = np.zeros_like(masks)
            new_prompt_masks = np.zeros_like(prompt_masks)
            for t in range(T):
                for obj_idx in range(masks.shape[3]):
                    # 处理原始mask
                    mask = masks[t, ..., obj_idx]
                    resized_mask = cv2.resize(mask.astype(np.float32),
                                              (w_half, h_half),
                                              interpolation=cv2.INTER_NEAREST)
                    new_masks[t, :, :, obj_idx] = self.place_patch(
                        resized_mask, h, w, x_offset, y_offset)

                    # 处理提示mask
                    pmask = prompt_masks[t, ..., obj_idx]
                    resized_prompt_mask = cv2.resize(
                        pmask.astype(np.float32), (w_half, h_half),
                        interpolation=cv2.INTER_NEAREST)
                    new_prompt_masks[t, :, :, obj_idx] = self.place_patch(
                        resized_prompt_mask, h, w, x_offset, y_offset)

            images, masks, size = new_images, new_masks, new_size
            prompt_points, prompt_boxes, prompt_masks = new_prompt_points, new_prompt_boxes, new_prompt_masks

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_points, prompt_boxes, prompt_masks

        return sample

    def place_patch(self, patch, h, w, x_offset, y_offset):
        """将调整后的patch放置在指定区域"""
        canvas = np.zeros((h, w), dtype=patch.dtype)
        h_patch, w_patch = patch.shape[0], patch.shape[1]
        canvas[y_offset:y_offset + h_patch,
               x_offset:x_offset + w_patch] = patch

        return canvas


class Sam2RandomRsverseFrameOrder:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']

        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        if np.random.uniform(0, 1) < self.prob:
            images = images[::-1].copy()
            masks = masks[::-1].copy()
            prompt_points = prompt_points[::-1].copy()
            prompt_boxes = prompt_boxes[::-1].copy()
            prompt_masks = prompt_masks[::-1].copy()

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_points, prompt_boxes, prompt_masks

        return sample


class Sam2Normalize:

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        self.mean = np.expand_dims(np.expand_dims(np.expand_dims(
            np.array(mean), axis=0),
                                                  axis=0),
                                   axis=0)
        self.std = np.expand_dims(np.expand_dims(np.expand_dims(np.array(std),
                                                                axis=0),
                                                 axis=0),
                                  axis=0)

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']

        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        images = (images - self.mean) / self.std

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_points, prompt_boxes, prompt_masks

        return sample


class SAM2BatchCollater:

    def __init__(self, resize):
        self.resize = resize

    def __call__(self, data):
        # List of [T, H, W, 3],长度B
        images = [s['image'] for s in data]
        # List of [T, H, W, object_num],长度B,注意每个视频的object_num可能不一样
        masks = [s['mask'] for s in data]
        # List of [T, object_num, point_num, 3],长度B
        prompt_points = [s['prompt_point'] for s in data]
        # List of [T, object_num, 4],长度B
        prompt_boxes = [s['prompt_box'] for s in data]
        # List of [T, H, W, object_num],长度B
        prompt_masks = [s['prompt_mask'] for s in data]

        input_images = []
        for video_idx, per_video_images in enumerate(images):
            per_video_input_images = []
            frame_nums = per_video_images.shape[0]
            for frame_idx in range(frame_nums):
                per_video_frame_image = per_video_images[frame_idx]
                per_video_frame_input_image = np.zeros(
                    (self.resize, self.resize, 3), dtype=np.float32)
                per_video_frame_input_image[
                    0:per_video_frame_image.shape[0], 0:per_video_frame_image.
                    shape[1], :] = per_video_frame_image
                per_video_frame_input_image = torch.from_numpy(
                    per_video_frame_input_image)
                # [H,W,3]->[3,H,W]
                per_video_frame_input_image = per_video_frame_input_image.permute(
                    2, 0, 1)
                per_video_input_images.append(per_video_frame_input_image)
            per_video_input_images = torch.stack(per_video_input_images, dim=0)
            input_images.append(per_video_input_images)
        input_images = torch.stack(input_images, dim=0)
        # [B, T, 3, H, W]->[T, B, 3, H, W],T为帧数,B为视频数
        input_images = input_images.permute(1, 0, 2, 3, 4).contiguous()

        # 时间步数 T
        frame_nums = input_images.shape[0]
        # 批次大小 B
        video_nums = input_images.shape[1]

        # 初始化用于存储掩码和索引的列表
        frame_step_masks = [[] for _ in range(frame_nums)]
        frame_step_prompt_points = [[] for _ in range(frame_nums)]
        frame_step_prompt_boxes = [[] for _ in range(frame_nums)]
        frame_step_prompt_masks = [[] for _ in range(frame_nums)]
        frame_step_masks_idx = [[] for _ in range(frame_nums)]

        for video_idx in range(video_nums):
            # [T, H, W, object_num]
            per_video_mask = masks[video_idx]
            per_video_object_nums = per_video_mask.shape[3]

            # [T, object_num, point_num, 3]
            per_video_prompt_points = prompt_points[video_idx]
            # [T, object_num, 4]
            per_video_prompt_boxes = prompt_boxes[video_idx]
            # [T, H, W, object_num]
            per_video_prompt_masks = prompt_masks[video_idx]
            for frame_idx in range(frame_nums):
                # [H, W, object_num]
                per_frame_video_mask = per_video_mask[frame_idx]
                # [object_num, point_num, 3]
                per_frame_video_prompt_point = per_video_prompt_points[
                    frame_idx]
                # [object_num, 4]
                per_frame_video_prompt_box = per_video_prompt_boxes[frame_idx]
                # [H, W, object_num]
                per_frame_video_prompt_mask = per_video_prompt_masks[frame_idx]

                for object_idx in range(per_video_object_nums):
                    # [H, W]
                    per_object_frame_video_mask = per_frame_video_mask[:, :,
                                                                       object_idx]
                    per_object_frame_video_mask = torch.from_numpy(
                        per_object_frame_video_mask)
                    frame_step_masks[frame_idx].append(
                        per_object_frame_video_mask)

                    # [point_num, 3]
                    per_object_frame_video_prompt_point = per_frame_video_prompt_point[
                        object_idx]
                    per_object_frame_video_prompt_point = torch.from_numpy(
                        per_object_frame_video_prompt_point)
                    frame_step_prompt_points[frame_idx].append(
                        per_object_frame_video_prompt_point)

                    # [4]
                    per_object_frame_video_prompt_box = per_frame_video_prompt_box[
                        object_idx]
                    per_object_frame_video_prompt_box = torch.from_numpy(
                        per_object_frame_video_prompt_box)
                    frame_step_prompt_boxes[frame_idx].append(
                        per_object_frame_video_prompt_box)

                    # [H, W]
                    per_object_frame_video_prompt_mask = per_frame_video_prompt_mask[:, :,
                                                                                     object_idx]
                    per_object_frame_video_prompt_mask = torch.from_numpy(
                        per_object_frame_video_prompt_mask)
                    frame_step_prompt_masks[frame_idx].append(
                        per_object_frame_video_prompt_mask)

                    # 每帧的idx是一个list,每个list中包含若干个object的idx
                    frame_step_masks_idx[frame_idx].append(
                        torch.tensor([frame_idx, video_idx],
                                     dtype=torch.int64))

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_masks = torch.stack(
            [torch.stack(masks_t, dim=0) for masks_t in frame_step_masks],
            dim=0)

        # [T, N_o, point_num, 3],T为帧数,N_o为B个视频中object_num的总数
        input_prompt_points = torch.stack([
            torch.stack(per_frame_prompt_points, dim=0)
            for per_frame_prompt_points in frame_step_prompt_points
        ],
                                          dim=0)

        # [T, N_o, 4],T为帧数,N_o为B个视频中object_num的总数
        input_prompt_boxes = torch.stack([
            torch.stack(per_frame_prompt_boxes, dim=0)
            for per_frame_prompt_boxes in frame_step_prompt_boxes
        ],
                                         dim=0)

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_prompt_masks = torch.stack([
            torch.stack(per_frame_prompt_masks, dim=0)
            for per_frame_prompt_masks in frame_step_prompt_masks
        ],
                                         dim=0)

        # [T, N_o, 2],T为帧数,N_o为B个视频中object_num的总数
        object_to_frame_idx = torch.stack([
            torch.stack(per_frame_mask_idx, dim=0)
            for per_frame_mask_idx in frame_step_masks_idx
        ],
                                          dim=0)

        assert input_images.shape[0] == input_masks.shape[
            0] == object_to_frame_idx.shape[0]

        return {
            # [T, B, 3, H, W]
            'image': input_images,
            # [T, N_o, H, W]
            'mask': input_masks,
            # [T, N_o, point_num, 3]
            'prompt_point': input_prompt_points,
            # [T, N_o, 4]
            'prompt_box': input_prompt_boxes,
            # [T, N_o, H, W]
            'prompt_mask': input_prompt_masks,
            # [T, N_o, 2]
            'object_to_frame_idx': object_to_frame_idx,
            # int
            'frame_num': frame_nums,
        }


class SAM2VideoBatchCollater:

    def __init__(self, resize, use_image_prob=0.1):
        self.resize = resize
        self.use_image_prob = use_image_prob

    def __call__(self, data):
        use_image_flag = False
        if np.random.uniform(0, 1) < self.use_image_prob:
            use_image_flag = True

        if use_image_flag:
            data = [s['image_sample'] for s in data]
        else:
            data = [s['video_sample'] for s in data]

        # List of [T, H, W, 3],长度B
        images = [s['image'] for s in data]
        # List of [T, H, W, object_num],长度B,注意每个视频的object_num可能不一样
        masks = [s['mask'] for s in data]
        # List of [T, object_num, point_num, 3],长度B
        prompt_points = [s['prompt_point'] for s in data]
        # List of [T, object_num, 4],长度B
        prompt_boxes = [s['prompt_box'] for s in data]
        # List of [T, H, W, object_num],长度B
        prompt_masks = [s['prompt_mask'] for s in data]

        input_images = []
        for video_idx, per_video_images in enumerate(images):
            per_video_input_images = []
            frame_nums = per_video_images.shape[0]
            for frame_idx in range(frame_nums):
                per_video_frame_image = per_video_images[frame_idx]
                per_video_frame_input_image = np.zeros(
                    (self.resize, self.resize, 3), dtype=np.float32)
                per_video_frame_input_image[
                    0:per_video_frame_image.shape[0], 0:per_video_frame_image.
                    shape[1], :] = per_video_frame_image
                per_video_frame_input_image = torch.from_numpy(
                    per_video_frame_input_image)
                # [H,W,3]->[3,H,W]
                per_video_frame_input_image = per_video_frame_input_image.permute(
                    2, 0, 1)
                per_video_input_images.append(per_video_frame_input_image)
            per_video_input_images = torch.stack(per_video_input_images, dim=0)
            input_images.append(per_video_input_images)
        input_images = torch.stack(input_images, dim=0)
        # [B, T, 3, H, W]->[T, B, 3, H, W],T为帧数,B为视频数
        input_images = input_images.permute(1, 0, 2, 3, 4).contiguous()

        # 时间步数 T
        frame_nums = input_images.shape[0]
        # 批次大小 B
        video_nums = input_images.shape[1]

        # 初始化用于存储掩码和索引的列表
        frame_step_masks = [[] for _ in range(frame_nums)]
        frame_step_prompt_points = [[] for _ in range(frame_nums)]
        frame_step_prompt_boxes = [[] for _ in range(frame_nums)]
        frame_step_prompt_masks = [[] for _ in range(frame_nums)]
        frame_step_masks_idx = [[] for _ in range(frame_nums)]

        for video_idx in range(video_nums):
            # [T, H, W, object_num]
            per_video_mask = masks[video_idx]
            per_video_object_nums = per_video_mask.shape[3]

            # [T, object_num, point_num, 3]
            per_video_prompt_points = prompt_points[video_idx]
            # [T, object_num, 4]
            per_video_prompt_boxes = prompt_boxes[video_idx]
            # [T, H, W, object_num]
            per_video_prompt_masks = prompt_masks[video_idx]
            for frame_idx in range(frame_nums):
                # [H, W, object_num]
                per_frame_video_mask = per_video_mask[frame_idx]
                # [object_num, point_num, 3]
                per_frame_video_prompt_point = per_video_prompt_points[
                    frame_idx]
                # [object_num, 4]
                per_frame_video_prompt_box = per_video_prompt_boxes[frame_idx]
                # [H, W, object_num]
                per_frame_video_prompt_mask = per_video_prompt_masks[frame_idx]

                for object_idx in range(per_video_object_nums):
                    # [H, W]
                    per_object_frame_video_mask = per_frame_video_mask[:, :,
                                                                       object_idx]
                    per_object_frame_video_mask = torch.from_numpy(
                        per_object_frame_video_mask)
                    frame_step_masks[frame_idx].append(
                        per_object_frame_video_mask)

                    # [point_num, 3]
                    per_object_frame_video_prompt_point = per_frame_video_prompt_point[
                        object_idx]
                    per_object_frame_video_prompt_point = torch.from_numpy(
                        per_object_frame_video_prompt_point)
                    frame_step_prompt_points[frame_idx].append(
                        per_object_frame_video_prompt_point)

                    # [4]
                    per_object_frame_video_prompt_box = per_frame_video_prompt_box[
                        object_idx]
                    per_object_frame_video_prompt_box = torch.from_numpy(
                        per_object_frame_video_prompt_box)
                    frame_step_prompt_boxes[frame_idx].append(
                        per_object_frame_video_prompt_box)

                    # [H, W]
                    per_object_frame_video_prompt_mask = per_frame_video_prompt_mask[:, :,
                                                                                     object_idx]
                    per_object_frame_video_prompt_mask = torch.from_numpy(
                        per_object_frame_video_prompt_mask)
                    frame_step_prompt_masks[frame_idx].append(
                        per_object_frame_video_prompt_mask)

                    # 每帧的idx是一个list,每个list中包含若干个object的idx
                    frame_step_masks_idx[frame_idx].append(
                        torch.tensor([frame_idx, video_idx],
                                     dtype=torch.int64))

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_masks = torch.stack(
            [torch.stack(masks_t, dim=0) for masks_t in frame_step_masks],
            dim=0)

        # [T, N_o, point_num, 3],T为帧数,N_o为B个视频中object_num的总数
        input_prompt_points = torch.stack([
            torch.stack(per_frame_prompt_points, dim=0)
            for per_frame_prompt_points in frame_step_prompt_points
        ],
                                          dim=0)

        # [T, N_o, 4],T为帧数,N_o为B个视频中object_num的总数
        input_prompt_boxes = torch.stack([
            torch.stack(per_frame_prompt_boxes, dim=0)
            for per_frame_prompt_boxes in frame_step_prompt_boxes
        ],
                                         dim=0)

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_prompt_masks = torch.stack([
            torch.stack(per_frame_prompt_masks, dim=0)
            for per_frame_prompt_masks in frame_step_prompt_masks
        ],
                                         dim=0)

        # [T, N_o, 2],T为帧数,N_o为B个视频中object_num的总数
        object_to_frame_idx = torch.stack([
            torch.stack(per_frame_mask_idx, dim=0)
            for per_frame_mask_idx in frame_step_masks_idx
        ],
                                          dim=0)

        assert input_images.shape[0] == input_masks.shape[
            0] == object_to_frame_idx.shape[0]

        return {
            # [T, B, 3, H, W]
            'image': input_images,
            # [T, N_o, H, W]
            'mask': input_masks,
            # [T, N_o, point_num, 3]
            'prompt_point': input_prompt_points,
            # [T, N_o, 4]
            'prompt_box': input_prompt_boxes,
            # [T, N_o, H, W]
            'prompt_mask': input_prompt_masks,
            # [T, N_o, 2]
            'object_to_frame_idx': object_to_frame_idx,
            # int
            'frame_num': frame_nums,
        }
