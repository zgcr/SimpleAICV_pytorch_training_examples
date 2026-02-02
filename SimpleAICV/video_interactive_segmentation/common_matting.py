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


class Sam2MattingResize:

    def __init__(self, resize=1024):
        self.resize = resize

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']
        trimaps, fg_maps, bg_maps = sample['trimap'], sample['fg_map'], sample[
            'bg_map']
        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        frame_nums, origin_h, origin_w, _ = images.shape
        h_factor = self.resize / origin_h
        w_factor = self.resize / origin_w

        resized_images = np.array([
            cv2.resize(image, (self.resize, self.resize)) for image in images
        ])

        resized_masks = [
            cv2.resize(mask, (self.resize, self.resize)) for mask in masks
        ]
        resized_masks = np.stack(resized_masks, axis=0)
        if len(resized_masks.shape) == 3:
            resized_masks = np.expand_dims(resized_masks, axis=-1)

        resized_trimaps = [
            cv2.resize(trimap, (self.resize, self.resize),
                       interpolation=cv2.INTER_NEAREST) for trimap in trimaps
        ]
        resized_trimaps = np.stack(resized_trimaps, axis=0)
        if len(resized_trimaps.shape) == 3:
            resized_trimaps = np.expand_dims(resized_trimaps, axis=-1)

        resized_fg_maps = []
        resized_bg_maps = []
        object_nums = fg_maps.shape[-1]
        for frame_idx in range(frame_nums):
            per_frame_fg_maps = fg_maps[frame_idx]
            per_frame_bg_maps = bg_maps[frame_idx]

            per_frame_resized_fg_maps = [
                cv2.resize(per_frame_fg_maps[:, :, :, object_idx],
                           (self.resize, self.resize))
                for object_idx in range(object_nums)
            ]
            per_frame_resized_fg_maps = np.stack(per_frame_resized_fg_maps,
                                                 axis=-1)

            per_frame_resized_bg_maps = [
                cv2.resize(per_frame_bg_maps[:, :, :, object_idx],
                           (self.resize, self.resize))
                for object_idx in range(object_nums)
            ]
            per_frame_resized_bg_maps = np.stack(per_frame_resized_bg_maps,
                                                 axis=-1)
            resized_fg_maps.append(per_frame_resized_fg_maps)
            resized_bg_maps.append(per_frame_resized_bg_maps)
        resized_fg_maps = np.stack(resized_fg_maps, axis=0)
        resized_bg_maps = np.stack(resized_bg_maps, axis=0)

        resized_prompt_masks = [
            cv2.resize(prompt_mask, (self.resize, self.resize))
            for prompt_mask in prompt_masks
        ]
        resized_prompt_masks = np.stack(resized_prompt_masks, axis=0)
        if len(resized_prompt_masks.shape) == 3:
            resized_prompt_masks = np.expand_dims(resized_prompt_masks,
                                                  axis=-1)

        resized_prompt_points = prompt_points.copy()
        resized_prompt_points[:, :, :, 0] *= w_factor
        resized_prompt_points[:, :, :, 1] *= h_factor

        resized_prompt_boxes = prompt_boxes.copy()
        resized_prompt_boxes[:, :, [0, 2]] *= w_factor
        resized_prompt_boxes[:, :, [1, 3]] *= h_factor

        resized_size = np.array(
            [resized_images.shape[1],
             resized_images.shape[2]]).astype(np.float32)

        sample['image'], sample['mask'], sample[
            'size'] = resized_images, resized_masks, resized_size

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = resized_trimaps, resized_fg_maps, resized_bg_maps

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = resized_prompt_points, resized_prompt_boxes, resized_prompt_masks

        return sample


class Sam2MattingRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']
        trimaps, fg_maps, bg_maps = sample['trimap'], sample['fg_map'], sample[
            'bg_map']
        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        if np.random.uniform(0, 1) < self.prob:
            # [T,H,W,3]
            images = images[:, :, ::-1, :].copy()
            masks = masks[:, :, ::-1, :].copy()

            trimaps = trimaps[:, :, ::-1, :].copy()
            fg_maps = fg_maps[:, :, ::-1, :].copy()
            bg_maps = bg_maps[:, :, ::-1, :].copy()

            prompt_masks = prompt_masks[:, :, ::-1, :].copy()

            _, _, w, _ = images.shape

            prompt_boxes_x1 = prompt_boxes[:, :, 0].copy()
            prompt_boxes_x2 = prompt_boxes[:, :, 2].copy()

            prompt_boxes[:, :, 0] = w - prompt_boxes_x2
            prompt_boxes[:, :, 2] = w - prompt_boxes_x1

            _, _, w, _ = images.shape
            prompt_points[:, :, :, 0] = w - prompt_points[:, :, :, 0]

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimaps, fg_maps, bg_maps

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_points, prompt_boxes, prompt_masks

        return sample


class Sam2MattingRandomMosaicAug:

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']
        trimaps, fg_maps, bg_maps = sample['trimap'], sample['fg_map'], sample[
            'bg_map']
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

            # 调整masks/prompt_masks/trimaps/fg_maps/bg_maps
            new_masks = np.zeros_like(masks)
            new_prompt_masks = np.zeros_like(prompt_masks)
            new_trimaps = np.zeros_like(trimaps)
            new_fg_maps = np.zeros_like(fg_maps)
            new_bg_maps = np.zeros_like(bg_maps)
            for t in range(T):
                # 以mask的object_nums为基准
                for obj_idx in range(masks.shape[3]):
                    # 处理原始mask
                    mask = masks[t, ..., obj_idx]
                    resized_mask = cv2.resize(mask.astype(np.float32),
                                              (w_half, h_half),
                                              interpolation=cv2.INTER_NEAREST)
                    new_masks[t, :, :, obj_idx] = self.place_patch(
                        resized_mask, h, w, x_offset, y_offset)

                    # 处理prompt_mask
                    pmask = prompt_masks[t, ..., obj_idx]
                    resized_prompt_mask = cv2.resize(
                        pmask.astype(np.float32), (w_half, h_half),
                        interpolation=cv2.INTER_NEAREST)
                    new_prompt_masks[t, :, :, obj_idx] = self.place_patch(
                        resized_prompt_mask, h, w, x_offset, y_offset)

                    # trimap处理
                    trimap = trimaps[t, ..., obj_idx]
                    resized_trimap = cv2.resize(
                        trimap.astype(np.float32), (w_half, h_half),
                        interpolation=cv2.INTER_NEAREST)
                    new_trimaps[t, ..., obj_idx] = self.place_patch(
                        resized_trimap, h, w, x_offset, y_offset)

                    # fg_map处理
                    fg_map = fg_maps[t, ..., obj_idx]
                    resized_fg_map = cv2.resize(fg_map.astype(np.float32),
                                                (w_half, h_half))
                    # 分通道处理
                    for c in range(3):
                        new_fg_maps[t, :, :, c, obj_idx] = self.place_patch(
                            resized_fg_map[..., c], h, w, x_offset, y_offset)

                    # bg_map处理
                    bg_map = bg_maps[t, ..., obj_idx]
                    resized_bg_map = cv2.resize(bg_map.astype(np.float32),
                                                (w_half, h_half))
                    # 分通道处理
                    for c in range(3):
                        new_bg_maps[t, :, :, c, obj_idx] = self.place_patch(
                            resized_bg_map[..., c], h, w, x_offset, y_offset)

            images, masks, size = new_images, new_masks, new_size
            trimaps, fg_maps, bg_maps = new_trimaps, new_fg_maps, new_bg_maps
            prompt_points, prompt_boxes, prompt_masks = new_prompt_points, new_prompt_boxes, new_prompt_masks

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimaps, fg_maps, bg_maps

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


class Sam2MattingRandomRsverseFrameOrder:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']
        trimaps, fg_maps, bg_maps = sample['trimap'], sample['fg_map'], sample[
            'bg_map']
        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        if np.random.uniform(0, 1) < self.prob:
            images = images[::-1].copy()
            masks = masks[::-1].copy()

            trimaps = trimaps[::-1].copy()
            fg_maps = fg_maps[::-1].copy()
            bg_maps = bg_maps[::-1].copy()

            prompt_points = prompt_points[::-1].copy()
            prompt_boxes = prompt_boxes[::-1].copy()
            prompt_masks = prompt_masks[::-1].copy()

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimaps, fg_maps, bg_maps

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_points, prompt_boxes, prompt_masks

        return sample


class Sam2MattingNormalize:

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        # [1,1,1,3]
        self.mean = np.expand_dims(np.expand_dims(np.expand_dims(
            np.array(mean), axis=0),
                                                  axis=0),
                                   axis=0)
        # [1,1,1,3]
        self.std = np.expand_dims(np.expand_dims(np.expand_dims(np.array(std),
                                                                axis=0),
                                                 axis=0),
                                  axis=0)

    def __call__(self, sample):
        images, masks, size = sample['image'], sample['mask'], sample['size']
        trimaps, fg_maps, bg_maps = sample['trimap'], sample['fg_map'], sample[
            'bg_map']
        prompt_points, prompt_boxes, prompt_masks = sample[
            'prompt_point'], sample['prompt_box'], sample['prompt_mask']

        images = (images - self.mean) / self.std
        images = images.astype(np.float32)

        fg_maps = (fg_maps - np.expand_dims(
            self.mean, axis=-1)) / np.expand_dims(self.std, axis=-1)
        fg_maps = fg_maps.astype(np.float32)

        bg_maps = (bg_maps - np.expand_dims(
            self.mean, axis=-1)) / np.expand_dims(self.std, axis=-1)
        bg_maps = bg_maps.astype(np.float32)

        sample['image'], sample['mask'], sample['size'] = images, masks, size

        sample['trimap'], sample['fg_map'], sample[
            'bg_map'] = trimaps, fg_maps, bg_maps

        sample['prompt_point'], sample['prompt_box'], sample[
            'prompt_mask'] = prompt_points, prompt_boxes, prompt_masks

        return sample


class SAM2MattingBatchCollater:

    def __init__(self, resize):
        self.resize = resize

    def __call__(self, data):
        # List of [T, H, W, 3],长度B
        images = [s['image'] for s in data]
        # List of [T, H, W, object_num],长度B,注意每个视频的object_num可能不一样
        masks = [s['mask'] for s in data]
        # List of [T, H, W, object_num],长度B,注意每个视频的object_num可能不一样
        trimaps = [x['trimap'] for x in data]
        # List of [T, H, W, 3, object_nums],长度B,注意每个视频的object_num可能不一样
        fg_maps = [x['fg_map'] for x in data]
        # List of [T, H, W, 3, object_nums],长度B,注意每个视频的object_num可能不一样
        bg_maps = [x['bg_map'] for x in data]
        # List of [T, object_num, point_num, 3],长度B
        prompt_points = [s['prompt_point'] for s in data]
        # List of [T, object_num, 4],长度B
        prompt_boxes = [s['prompt_box'] for s in data]
        # List of [T, H, W, object_num],长度B
        prompt_masks = [s['prompt_mask'] for s in data]

        batch_images = []
        for per_video_images in images:
            T, H, W, C = per_video_images.shape
            resized = np.zeros((T, self.resize, self.resize, C),
                               dtype=np.float32)
            resized[:, :H, :W, :] = per_video_images
            # [T, C, H, W]
            resized = torch.from_numpy(resized).permute(0, 3, 1, 2)
            batch_images.append(resized)
        # [B, T, C, H, W]
        batch_images = torch.stack(batch_images, dim=0)
        # [B, T, C, H, W]->[T, B, C, H, W],T为帧数,B为视频数
        batch_images = batch_images.permute(1, 0, 2, 3, 4).contiguous()

        # 时间步数 T
        frame_nums = batch_images.shape[0]
        # 批次大小 B
        video_nums = batch_images.shape[1]

        # 初始化用于存储掩码和索引的列表
        frame_step_images = [[] for _ in range(frame_nums)]
        frame_step_masks = [[] for _ in range(frame_nums)]
        frame_step_trimaps = [[] for _ in range(frame_nums)]
        frame_step_fg_maps = [[] for _ in range(frame_nums)]
        frame_step_bg_maps = [[] for _ in range(frame_nums)]
        frame_step_prompt_points = [[] for _ in range(frame_nums)]
        frame_step_prompt_boxes = [[] for _ in range(frame_nums)]
        frame_step_prompt_masks = [[] for _ in range(frame_nums)]
        frame_step_masks_idx = [[] for _ in range(frame_nums)]

        for video_idx in range(video_nums):
            # [T, H, W, 3]
            per_video_image = images[video_idx]
            # [T, H, W, object_num]
            per_video_mask = masks[video_idx]
            per_video_object_nums = per_video_mask.shape[3]

            # [T, H, W, object_num]
            per_video_trimap = trimaps[video_idx]
            # [T, H, W, 3, object_nums]
            per_video_fg_map = fg_maps[video_idx]
            # [T, H, W, 3, object_nums]
            per_video_bg_map = bg_maps[video_idx]

            # [T, object_num, point_num, 3]
            per_video_prompt_points = prompt_points[video_idx]
            # [T, object_num, 4]
            per_video_prompt_boxes = prompt_boxes[video_idx]
            # [T, H, W, object_num]
            per_video_prompt_masks = prompt_masks[video_idx]
            for frame_idx in range(frame_nums):
                # [H, W, 3]
                per_frame_video_image = per_video_image[frame_idx]
                # [H, W, object_num]
                per_frame_video_mask = per_video_mask[frame_idx]
                # [H, W, object_num]
                per_frame_video_trimap = per_video_trimap[frame_idx]
                # [H, W, 3, object_nums]
                per_frame_video_fg_map = per_video_fg_map[frame_idx]
                # [H, W, 3, object_nums]
                per_frame_video_bg_map = per_video_bg_map[frame_idx]

                # [object_num, point_num, 3]
                per_frame_video_prompt_point = per_video_prompt_points[
                    frame_idx]
                # [object_num, 4]
                per_frame_video_prompt_box = per_video_prompt_boxes[frame_idx]
                # [H, W, object_num]
                per_frame_video_prompt_mask = per_video_prompt_masks[frame_idx]

                for object_idx in range(per_video_object_nums):
                    # [H, W, 3]
                    per_object_frame_video_image = per_frame_video_image.copy()
                    per_object_frame_video_image = torch.from_numpy(
                        per_object_frame_video_image)
                    frame_step_images[frame_idx].append(
                        per_object_frame_video_image)

                    # [H, W]
                    per_object_frame_video_mask = per_frame_video_mask[:, :,
                                                                       object_idx]
                    per_object_frame_video_mask = torch.from_numpy(
                        per_object_frame_video_mask)
                    frame_step_masks[frame_idx].append(
                        per_object_frame_video_mask)

                    # [H, W]
                    per_object_frame_video_trimap = per_frame_video_trimap[:, :,
                                                                           object_idx]
                    per_object_frame_video_trimap = torch.from_numpy(
                        per_object_frame_video_trimap)
                    frame_step_trimaps[frame_idx].append(
                        per_object_frame_video_trimap)

                    # [H, W, 3]
                    per_object_frame_video_fg_map = per_frame_video_fg_map[:, :, :,
                                                                           object_idx]
                    per_object_frame_video_fg_map = torch.from_numpy(
                        per_object_frame_video_fg_map)
                    frame_step_fg_maps[frame_idx].append(
                        per_object_frame_video_fg_map)

                    # [H, W, 3]
                    per_object_frame_video_bg_map = per_frame_video_bg_map[:, :, :,
                                                                           object_idx]
                    per_object_frame_video_bg_map = torch.from_numpy(
                        per_object_frame_video_bg_map)
                    frame_step_bg_maps[frame_idx].append(
                        per_object_frame_video_bg_map)

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

        # [T, N_o, H, W, 3],T为帧数,N_o为B个视频中object_num的总数
        input_images = torch.stack(
            [torch.stack(images_t, dim=0) for images_t in frame_step_images],
            dim=0)

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_masks = torch.stack(
            [torch.stack(masks_t, dim=0) for masks_t in frame_step_masks],
            dim=0)

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_trimaps = torch.stack([
            torch.stack(trimaps_t, dim=0) for trimaps_t in frame_step_trimaps
        ],
                                    dim=0)

        # [T, N_o, H, W, 3],T为帧数,N_o为B个视频中object_num的总数
        input_fg_maps = torch.stack([
            torch.stack(fg_maps_t, dim=0) for fg_maps_t in frame_step_fg_maps
        ],
                                    dim=0)

        # [T, N_o, H, W, 3],T为帧数,N_o为B个视频中object_num的总数
        input_bg_maps = torch.stack([
            torch.stack(bg_maps_t, dim=0) for bg_maps_t in frame_step_bg_maps
        ],
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

        assert batch_images.shape[0] == input_masks.shape[
            0] == object_to_frame_idx.shape[0]

        return {
            # [T, B, 3, H, W]
            'batch_image': batch_images,
            # [T, N_o, H, W]
            'mask': input_masks,
            # [T, N_o, H, W]
            'trimap': input_trimaps,
            # [T, N_o, H, W, 3]
            'input_image': input_images,
            # [T, N_o, H, W, 3]
            'fg_map': input_fg_maps,
            # [T, N_o, H, W, 3]
            'bg_map': input_bg_maps,
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


class SAM2MattingVideoBatchCollater:

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
        # List of [T, H, W, object_num],长度B,注意每个视频的object_num可能不一样
        trimaps = [x['trimap'] for x in data]
        # List of [T, H, W, 3, object_nums],长度B,注意每个视频的object_num可能不一样
        fg_maps = [x['fg_map'] for x in data]
        # List of [T, H, W, 3, object_nums],长度B,注意每个视频的object_num可能不一样
        bg_maps = [x['bg_map'] for x in data]
        # List of [T, object_num, point_num, 3],长度B
        prompt_points = [s['prompt_point'] for s in data]
        # List of [T, object_num, 4],长度B
        prompt_boxes = [s['prompt_box'] for s in data]
        # List of [T, H, W, object_num],长度B
        prompt_masks = [s['prompt_mask'] for s in data]

        batch_images = []
        for per_video_images in images:
            T, H, W, C = per_video_images.shape
            resized = np.zeros((T, self.resize, self.resize, C),
                               dtype=np.float32)
            resized[:, :H, :W, :] = per_video_images
            # [T, C, H, W]
            resized = torch.from_numpy(resized).permute(0, 3, 1, 2)
            batch_images.append(resized)
        # [B, T, C, H, W]
        batch_images = torch.stack(batch_images, dim=0)
        # [B, T, C, H, W]->[T, B, C, H, W],T为帧数,B为视频数
        batch_images = batch_images.permute(1, 0, 2, 3, 4).contiguous()

        # 时间步数 T
        frame_nums = batch_images.shape[0]
        # 批次大小 B
        video_nums = batch_images.shape[1]

        # 初始化用于存储掩码和索引的列表
        frame_step_images = [[] for _ in range(frame_nums)]
        frame_step_masks = [[] for _ in range(frame_nums)]
        frame_step_trimaps = [[] for _ in range(frame_nums)]
        frame_step_fg_maps = [[] for _ in range(frame_nums)]
        frame_step_bg_maps = [[] for _ in range(frame_nums)]
        frame_step_prompt_points = [[] for _ in range(frame_nums)]
        frame_step_prompt_boxes = [[] for _ in range(frame_nums)]
        frame_step_prompt_masks = [[] for _ in range(frame_nums)]
        frame_step_masks_idx = [[] for _ in range(frame_nums)]

        for video_idx in range(video_nums):
            # [T, H, W, 3]
            per_video_image = images[video_idx]
            # [T, H, W, object_num]
            per_video_mask = masks[video_idx]
            per_video_object_nums = per_video_mask.shape[3]

            # [T, H, W, object_num]
            per_video_trimap = trimaps[video_idx]
            # [T, H, W, 3, object_nums]
            per_video_fg_map = fg_maps[video_idx]
            # [T, H, W, 3, object_nums]
            per_video_bg_map = bg_maps[video_idx]

            # [T, object_num, point_num, 3]
            per_video_prompt_points = prompt_points[video_idx]
            # [T, object_num, 4]
            per_video_prompt_boxes = prompt_boxes[video_idx]
            # [T, H, W, object_num]
            per_video_prompt_masks = prompt_masks[video_idx]
            for frame_idx in range(frame_nums):
                # [H, W, 3]
                per_frame_video_image = per_video_image[frame_idx]
                # [H, W, object_num]
                per_frame_video_mask = per_video_mask[frame_idx]
                # [H, W, object_num]
                per_frame_video_trimap = per_video_trimap[frame_idx]
                # [H, W, 3, object_nums]
                per_frame_video_fg_map = per_video_fg_map[frame_idx]
                # [H, W, 3, object_nums]
                per_frame_video_bg_map = per_video_bg_map[frame_idx]

                # [object_num, point_num, 3]
                per_frame_video_prompt_point = per_video_prompt_points[
                    frame_idx]
                # [object_num, 4]
                per_frame_video_prompt_box = per_video_prompt_boxes[frame_idx]
                # [H, W, object_num]
                per_frame_video_prompt_mask = per_video_prompt_masks[frame_idx]

                for object_idx in range(per_video_object_nums):
                    # [H, W, 3]
                    per_object_frame_video_image = per_frame_video_image.copy()
                    per_object_frame_video_image = torch.from_numpy(
                        per_object_frame_video_image)
                    frame_step_images[frame_idx].append(
                        per_object_frame_video_image)

                    # [H, W]
                    per_object_frame_video_mask = per_frame_video_mask[:, :,
                                                                       object_idx]
                    per_object_frame_video_mask = torch.from_numpy(
                        per_object_frame_video_mask)
                    frame_step_masks[frame_idx].append(
                        per_object_frame_video_mask)

                    # [H, W]
                    per_object_frame_video_trimap = per_frame_video_trimap[:, :,
                                                                           object_idx]
                    per_object_frame_video_trimap = torch.from_numpy(
                        per_object_frame_video_trimap)
                    frame_step_trimaps[frame_idx].append(
                        per_object_frame_video_trimap)

                    # [H, W, 3]
                    per_object_frame_video_fg_map = per_frame_video_fg_map[:, :, :,
                                                                           object_idx]
                    per_object_frame_video_fg_map = torch.from_numpy(
                        per_object_frame_video_fg_map)
                    frame_step_fg_maps[frame_idx].append(
                        per_object_frame_video_fg_map)

                    # [H, W, 3]
                    per_object_frame_video_bg_map = per_frame_video_bg_map[:, :, :,
                                                                           object_idx]
                    per_object_frame_video_bg_map = torch.from_numpy(
                        per_object_frame_video_bg_map)
                    frame_step_bg_maps[frame_idx].append(
                        per_object_frame_video_bg_map)

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

        # [T, N_o, H, W, 3],T为帧数,N_o为B个视频中object_num的总数
        input_images = torch.stack(
            [torch.stack(images_t, dim=0) for images_t in frame_step_images],
            dim=0)

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_masks = torch.stack(
            [torch.stack(masks_t, dim=0) for masks_t in frame_step_masks],
            dim=0)

        # [T, N_o, H, W],T为帧数,N_o为B个视频中object_num的总数
        input_trimaps = torch.stack([
            torch.stack(trimaps_t, dim=0) for trimaps_t in frame_step_trimaps
        ],
                                    dim=0)

        # [T, N_o, H, W, 3],T为帧数,N_o为B个视频中object_num的总数
        input_fg_maps = torch.stack([
            torch.stack(fg_maps_t, dim=0) for fg_maps_t in frame_step_fg_maps
        ],
                                    dim=0)

        # [T, N_o, H, W, 3],T为帧数,N_o为B个视频中object_num的总数
        input_bg_maps = torch.stack([
            torch.stack(bg_maps_t, dim=0) for bg_maps_t in frame_step_bg_maps
        ],
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

        assert batch_images.shape[0] == input_masks.shape[
            0] == object_to_frame_idx.shape[0]

        return {
            # [T, B, 3, H, W]
            'batch_image': batch_images,
            # [T, N_o, H, W]
            'mask': input_masks,
            # [T, N_o, H, W]
            'trimap': input_trimaps,
            # [T, N_o, H, W, 3]
            'input_image': input_images,
            # [T, N_o, H, W, 3]
            'fg_map': input_fg_maps,
            # [T, N_o, H, W, 3]
            'bg_map': input_bg_maps,
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
