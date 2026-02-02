import os
import collections
import cv2
import imagesize
import json
import math
import numpy as np
import random

from PIL import Image

from tqdm import tqdm
from pycocotools import mask as mask_utils
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset


class SAM2ImageSegmentationDataset(Dataset):

    def __init__(self,
                 image_root_dir,
                 image_set_name=[
                    'DIS5K',
                    'sa_000000',
                 ],
                 image_set_type='train',
                 image_per_set_image_choose_max_num={
                    'DIS5K': 1000000,
                    'sa_000000': 1000000,
                 },
                 per_image_mask_chosse_max_num=16,
                 points_num=1,
                 area_filter_ratio=0.0001,
                 box_noise_wh_ratio=0.1,
                 mask_noise_area_ratio=0.04,
                 transform=None):

        self.all_set_image_path_list = collections.OrderedDict()
        self.all_set_image_nums = collections.OrderedDict()

        # for per_set_name in tqdm(image_set_name):
        #     per_set_dir = os.path.join(image_root_dir, per_set_name, image_set_type)
        #     self.all_set_image_nums[per_set_name] = 0
        #     self.all_set_image_path_list[per_set_name] = []
        #     for root, folders, files in os.walk(per_set_dir):
        #         for file_name in files:
        #             if '.jpg' in file_name:
        #                 per_image_path = os.path.join(root, file_name)

        #                 png_name = file_name.split('.')[0] + '.png'
        #                 per_mask_label_path = os.path.join(root, png_name)

        #                 if not os.path.exists(per_mask_label_path):
        #                     json_name = file_name.split('.')[0] + '.json'
        #                     per_mask_label_path = os.path.join(root, json_name)

        #                 if os.path.exists(per_image_path) and os.path.exists(
        #                         per_mask_label_path):
        #                     self.all_set_image_nums[per_set_name] += 1
        #                     self.all_set_image_path_list[per_set_name].append([
        #                         file_name,
        #                         per_image_path,
        #                         per_mask_label_path,
        #                     ])

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_set, image_root_dir, per_set_name, image_set_type):
                per_set_name
                for per_set_name in image_set_name
            }

            for future in tqdm(as_completed(futures), total=len(image_set_name),
                               desc="Processing sam image sets"):
                per_set_name = futures[future]
                per_set_image_paths, per_set_count = future.result()
                self.all_set_image_nums[per_set_name] = per_set_count
                self.all_set_image_path_list[
                    per_set_name] = per_set_image_paths

        for key, value in self.all_set_image_path_list.items():
            print(f'set_name:{key},origin_image_num:{len(value)}')

        self.image_path_list = []
        for per_set_name, per_set_image_path_list in self.all_set_image_path_list.items(
        ):
            per_set_image_path_list = sorted(per_set_image_path_list)
            per_set_image_max_num = image_per_set_image_choose_max_num[per_set_name]
            if len(per_set_image_path_list) > per_set_image_max_num:
                per_set_image_path_list = per_set_image_path_list[
                    0:per_set_image_max_num]

            print(
                f'set_name:{per_set_name},choose_image_num:{len(per_set_image_path_list)}'
            )

            for per_image_info in per_set_image_path_list:
                self.image_path_list.append(per_image_info)
        self.image_path_list = sorted(self.image_path_list)

        # get all mask for all images
        self.all_image_mask_path_list = []

        # for per_image_name, per_image_path, per_mask_label_path in tqdm(
        #         self.image_path_list):
        #     if per_mask_label_path.endswith('.png'):
        #         mask_list_idx = 0
        #         per_image_w, per_image_h = imagesize.get(per_image_path)
        #         self.all_image_mask_path_list.append([
        #             per_image_name,
        #             mask_list_idx,
        #             per_image_path,
        #             per_mask_label_path,
        #             per_image_h,
        #             per_image_w,
        #         ])
        #     elif per_mask_label_path.endswith('.json'):
        #         with open(per_mask_label_path, encoding='utf-8') as f:
        #             per_image_json_data = json.load(f)
        #             per_image_annotation = per_image_json_data['annotations']

        #             per_image_h, per_image_w = per_image_json_data['image'][
        #                 'height'], per_image_json_data['image']['width']

        #             per_image_annotation_num = len(per_image_annotation)
        #             if per_image_annotation_num > per_image_mask_chosse_max_num:
        #                 per_image_annotation = per_image_annotation[
        #                     0:per_image_mask_chosse_max_num]

        #             for mask_list_idx, per_annot in enumerate(
        #                     per_image_annotation):

        #                 # bbox format:[x_min, y_min, w, h]
        #                 per_box = per_annot['bbox']

        #                 x_min = math.ceil(max(per_box[0], 0))
        #                 y_min = math.ceil(max(per_box[1], 0))
        #                 x_max = math.ceil(
        #                     min(per_box[0] + per_box[2], per_image_w))
        #                 y_max = math.ceil(
        #                     min(per_box[1] + per_box[3], per_image_h))
        #                 box_w = math.ceil(x_max - x_min)
        #                 box_h = math.ceil(y_max - y_min)

        #                 if box_w / per_image_w < math.sqrt(
        #                         area_filter_ratio
        #                 ) and box_h / per_image_h < math.sqrt(
        #                         area_filter_ratio):
        #                     continue

        #                 if (box_w * box_h) / float(
        #                         per_image_h * per_image_w) < area_filter_ratio:
        #                     continue

        #                 if per_annot['area'] / float(
        #                         per_image_h * per_image_w
        #                 ) < area_filter_ratio or per_annot['area'] / float(
        #                         per_image_h * per_image_w) > 0.9:
        #                     continue

        #                 self.all_image_mask_path_list.append([
        #                     per_image_name,
        #                     mask_list_idx,
        #                     per_image_path,
        #                     per_mask_label_path,
        #                     per_image_h,
        #                     per_image_w,
        #                 ])

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_image_mask_pairs, per_image_name, per_image_path, per_mask_label_path, per_image_mask_chosse_max_num, area_filter_ratio):
                idx
                for idx, (
                    per_image_name, per_image_path,
                    per_mask_label_path) in enumerate(self.image_path_list)
            }

            results = [None] * len(self.image_path_list)
            for future in tqdm(as_completed(futures),
                               total=len(self.image_path_list)):
                idx = futures[future]
                result = future.result()
                results[idx] = result

        for result in results:
            if result is not None:
                self.all_image_mask_path_list.extend(result)

        self.points_num = points_num
        self.area_filter_ratio = area_filter_ratio
        self.box_noise_wh_ratio = box_noise_wh_ratio
        self.mask_noise_area_ratio = mask_noise_area_ratio
        self.transform = transform

        print(f'Image Size:{len(self.image_path_list)}')
        print(f'Dataset Size:{len(self.all_image_mask_path_list)}')

    def process_set(self, root_dir, per_set_name, set_type):
        per_set_dir = os.path.join(root_dir, per_set_name, set_type)
        per_set_image_paths = []
        per_set_count = 0

        for root, folders, files in os.walk(per_set_dir):
            for file_name in files:
                if '.jpg' in file_name:
                    per_image_path = os.path.join(root, file_name)

                    png_name = file_name.split('.')[0] + '.png'
                    per_mask_label_path = os.path.join(root, png_name)

                    if not os.path.exists(per_mask_label_path):
                        json_name = file_name.split('.')[0] + '.json'
                        per_mask_label_path = os.path.join(root, json_name)

                    if os.path.exists(per_image_path) and os.path.exists(
                            per_mask_label_path):
                        per_set_count += 1
                        per_set_image_paths.append([
                            file_name,
                            per_image_path,
                            per_mask_label_path,
                        ])

        return per_set_image_paths, per_set_count

    def process_image_mask_pairs(self, per_image_name, per_image_path,
                                 per_mask_label_path,
                                 per_image_mask_chosse_max_num,
                                 area_filter_ratio):
        result_list = []

        if per_mask_label_path.endswith('.png'):
            mask_list_idx = 0
            per_image_w, per_image_h = imagesize.get(per_image_path)
            result_list.append([
                per_image_name,
                mask_list_idx,
                per_image_path,
                per_mask_label_path,
                per_image_h,
                per_image_w,
            ])
        elif per_mask_label_path.endswith('.json'):
            with open(per_mask_label_path, encoding='utf-8') as f:
                per_image_json_data = json.load(f)
                per_image_annotation = per_image_json_data['annotations']

                per_image_h, per_image_w = per_image_json_data['image'][
                    'height'], per_image_json_data['image']['width']

                per_image_annotation_num = len(per_image_annotation)
                if per_image_annotation_num > per_image_mask_chosse_max_num:
                    per_image_annotation = per_image_annotation[
                        0:per_image_mask_chosse_max_num]

                for mask_list_idx, per_annot in enumerate(
                        per_image_annotation):

                    # bbox format:[x_min, y_min, w, h]
                    per_box = per_annot['bbox']

                    x_min = math.ceil(max(per_box[0], 0))
                    y_min = math.ceil(max(per_box[1], 0))
                    x_max = math.ceil(min(per_box[0] + per_box[2],
                                          per_image_w))
                    y_max = math.ceil(min(per_box[1] + per_box[3],
                                          per_image_h))
                    box_w = math.ceil(x_max - x_min)
                    box_h = math.ceil(y_max - y_min)

                    if box_w / per_image_w < math.sqrt(
                            area_filter_ratio
                    ) and box_h / per_image_h < math.sqrt(area_filter_ratio):
                        continue

                    if (box_w * box_h) / float(
                            per_image_h * per_image_w) < area_filter_ratio:
                        continue

                    if per_annot['area'] / float(
                            per_image_h * per_image_w
                    ) < area_filter_ratio or per_annot['area'] / float(
                            per_image_h * per_image_w) > 0.9:
                        continue

                    result_list.append([
                        per_image_name,
                        mask_list_idx,
                        per_image_path,
                        per_mask_label_path,
                        per_image_h,
                        per_image_w,
                    ])

        return result_list

    def __len__(self):
        return len(self.all_image_mask_path_list)

    def __getitem__(self, idx):
        per_image_name, mask_list_idx, _, _, _, _ = self.all_image_mask_path_list[idx]

        video_name = f'{per_image_name.split('.')[0]}_{mask_list_idx}'

        image = self.load_image(idx)
        mask = self.load_mask(idx)

        # [1,h,w,3]
        frames_images = np.expand_dims(image, axis=0).astype(np.float32)
        # [1,h,w,1]
        frames_masks = np.expand_dims(np.expand_dims(mask, axis=-1),
                                      axis=0).astype(np.float32)

        size = np.array([frames_images.shape[1],
                         frames_images.shape[2]]).astype(np.float32)

        assert frames_images.shape[0] == frames_masks.shape[
            0] and frames_images.shape[1] == frames_masks.shape[
                1] and frames_images.shape[2] == frames_masks.shape[2]

        # # [1,object_nums,point_nums,3]
        # frame_prompt_points = []
        # for per_frame_idx in range(frames_masks.shape[0]):
        #     # [h,w,object_nums]
        #     per_frame_mask = frames_masks[per_frame_idx]
        #     per_frame_prompt_point = self.load_frame_points((per_frame_mask > 0.5).astype(np.float32))
        #     frame_prompt_points.append(per_frame_prompt_point)
        # frame_prompt_points = np.stack(frame_prompt_points,
        #                                axis=0).astype(np.float32)

        # # [1,object_nums,4]
        # frame_prompt_boxes = []
        # for per_frame_idx in range(frames_masks.shape[0]):
        #     # [h,w,object_nums]
        #     per_frame_mask = frames_masks[per_frame_idx]
        #     per_frame_mask_size = [
        #         per_frame_mask.shape[0], per_frame_mask.shape[1]
        #     ]

        #     per_frame_prompt_box = self.load_frame_box((per_frame_mask > 0.5).astype(np.float32))
        #     if self.box_noise_wh_ratio > 0:
        #         per_frame_prompt_box = self.noise_frame_box(
        #             per_frame_prompt_box, per_frame_mask_size)
        #     frame_prompt_boxes.append(per_frame_prompt_box)
        # frame_prompt_boxes = np.stack(frame_prompt_boxes,
        #                               axis=0).astype(np.float32)

        # # [1,h,w,object_nums]
        # frame_prompt_masks = []
        # for per_frame_idx in range(frames_masks.shape[0]):
        #     # [h,w,object_nums]
        #     per_frame_prompt_mask = frames_masks[per_frame_idx]
        #     per_frame_prompt_mask = self.noise_frame_mask(
        #         (per_frame_prompt_mask > 0.2).astype(np.float32))
        #     frame_prompt_masks.append(per_frame_prompt_mask)
        # frame_prompt_masks = np.stack(frame_prompt_masks,
        #                               axis=0).astype(np.float32)

        # [1,object_nums,point_nums,3]
        with ThreadPoolExecutor(max_workers=4) as executor:
            frame_prompt_points_list = list(
                executor.map(self.load_frame_points, (frames_masks > 0.5).astype(np.float32)))
        frame_prompt_points = np.stack(frame_prompt_points_list,
                                       axis=0).astype(np.float32)

        # [1,object_nums,4]
        with ThreadPoolExecutor(max_workers=4) as executor:
            frame_prompt_boxes_list = list(
                executor.map(
                    lambda m: (lambda box: self.noise_frame_box(
                        box, [m.shape[0], m.shape[1]])
                               if self.box_noise_wh_ratio > 0 else box)
                    (self.load_frame_box((m > 0.5).astype(np.float32))), frames_masks))
        frame_prompt_boxes = np.stack(frame_prompt_boxes_list,
                                      axis=0).astype(np.float32)

        # [1,h,w,object_nums]
        with ThreadPoolExecutor(max_workers=4) as executor:
            frame_prompt_masks_list = list(
                executor.map(self.noise_frame_mask, (frames_masks > 0.2).astype(np.float32)))
        frame_prompt_masks = np.stack(frame_prompt_masks_list,
                                      axis=0).astype(np.float32)

        sample = {
            'video_name': video_name,
            'image': frames_images,
            'mask': frames_masks,
            'size': size,
            'prompt_point': frame_prompt_points,
            'prompt_box': frame_prompt_boxes,
            'prompt_mask': frame_prompt_masks,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        _, _, per_image_path, _, _, _ = self.all_image_mask_path_list[idx]
        image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        _, mask_list_idx, _, per_mask_label_path, _, _ = self.all_image_mask_path_list[
            idx]
        if per_mask_label_path.endswith('.png'):
            target_mask = np.array(
                Image.open(per_mask_label_path).convert('L'), dtype=np.uint8)
            # 0.9*255
            target_mask[target_mask >= 230] = 255
            # 0.1*255
            target_mask[target_mask <= 25] = 0
            target_mask = target_mask / 255.
            # 二值化
            target_mask = (target_mask > 0.5).astype(np.float32)

        elif per_mask_label_path.endswith('.json'):
            with open(per_mask_label_path, encoding='utf-8') as f:
                per_image_json_data = json.load(f)
                per_image_annotation = per_image_json_data['annotations']
                per_annot = per_image_annotation[mask_list_idx]

                target_mask = mask_utils.decode(per_annot['segmentation'])
                target_mask[target_mask > 0] = 1

        return target_mask.astype(np.float32)

    def load_frame_points(self, per_frame_mask):
        h, w, object_nums = per_frame_mask.shape

        per_frame_point = []
        for object_idx in range(object_nums):
            # 每一个object的mask
            per_object_mask = per_frame_mask[:, :, object_idx]

            if np.count_nonzero(per_object_mask) == 0:
                # 若没有前景区域,取所有背景区域点
                per_object_all_point_coords = np.argwhere(per_object_mask == 0)
                point_label = 0
            else:
                # 若前景区域,取所有前景区域点
                per_object_all_point_coords = np.argwhere(per_object_mask != 0)
                point_label = 1

            per_object_point = []
            all_points_num = len(per_object_all_point_coords)
            points_index = np.random.choice(all_points_num,
                                            self.points_num,
                                            replace=False)
            for per_point_idx in points_index:
                per_object_point.append([
                    per_object_all_point_coords[per_point_idx][1],
                    per_object_all_point_coords[per_point_idx][0],
                    point_label,
                ])
            per_object_point = np.array(per_object_point, dtype=np.float32)
            per_frame_point.append(per_object_point)

        per_frame_point = np.stack(per_frame_point, axis=0)
        per_frame_point = per_frame_point.astype(np.float32)

        return per_frame_point

    def load_frame_box(self, per_frame_mask):
        h, w, object_nums = per_frame_mask.shape

        # 创建网格坐标
        xs = np.arange(w, dtype=np.int32)
        ys = np.arange(h, dtype=np.int32)
        # grid_xs:[h,w],grid_ys[h,w]
        grid_xs, grid_ys = np.meshgrid(xs, ys, indexing='xy')
        # grid_xs:[1,h,w],grid_ys[1,h,w]
        grid_xs = np.expand_dims(grid_xs, axis=0)
        grid_ys = np.expand_dims(grid_ys, axis=0)

        # 将mask转换为布尔类型
        per_frame_mask = per_frame_mask.astype(bool)

        # 初始化结果
        xs_min = np.full(object_nums, w, dtype=np.int32)
        ys_min = np.full(object_nums, h, dtype=np.int32)
        xs_max = np.full(object_nums, -1, dtype=np.int32)
        ys_max = np.full(object_nums, -1, dtype=np.int32)

        for object_idx in range(object_nums):
            # 每一个object的mask
            per_object_mask = per_frame_mask[:, :, object_idx]

            # 只有当per_object_mask有前景时才计算边界框坐标
            if per_object_mask.any():
                x_min = np.min(np.where(per_object_mask, grid_xs[0], w),
                               axis=(0, 1))
                y_min = np.min(np.where(per_object_mask, grid_ys[0], h),
                               axis=(0, 1))
                x_max = np.max(np.where(per_object_mask, grid_xs[0], -1),
                               axis=(0, 1))
                y_max = np.max(np.where(per_object_mask, grid_ys[0], -1),
                               axis=(0, 1))

                xs_min[object_idx] = x_min
                ys_min[object_idx] = y_min
                xs_max[object_idx] = x_max
                ys_max[object_idx] = y_max

        per_frame_box = np.stack((xs_min, ys_min, xs_max, ys_max), axis=-1)

        return per_frame_box

    def noise_frame_box(self, properties_boxes, mask_np_shape):
        post_properties_boxes = []
        box_num = properties_boxes.shape[0]
        for box_idx in range(box_num):
            per_properties_box = properties_boxes[box_idx]

            if -1 in per_properties_box:
                post_properties_boxes.append(per_properties_box)
                continue

            w, h = per_properties_box[2] - per_properties_box[
                0], per_properties_box[3] - per_properties_box[1]

            # 根据现有条件判断是否进行抖动
            if h / mask_np_shape[0] <= math.sqrt(
                    self.area_filter_ratio
            ) or w / mask_np_shape[1] <= math.sqrt(self.area_filter_ratio):
                post_properties_boxes.append(per_properties_box)
                continue

            noise_x, noise_y = w * self.box_noise_wh_ratio, h * self.box_noise_wh_ratio
            noise_x, noise_y = min(int(mask_np_shape[1] * 0.02),
                                   noise_x), min(int(mask_np_shape[0] * 0.02),
                                                 noise_y)

            if noise_x <= 1 or noise_y <= 1:
                post_properties_boxes.append(per_properties_box)
                continue

            # 随机生成抖动的边界框
            x0 = per_properties_box[0] + max(
                min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
            y0 = per_properties_box[1] + max(
                min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)
            x1 = per_properties_box[2] + max(
                min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
            y1 = per_properties_box[3] + max(
                min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)

            # 限制坐标范围，避免越界
            x0 = x0 if x0 >= 0 else 0
            y0 = y0 if y0 >= 0 else 0
            x1 = x1 if x1 <= mask_np_shape[1] else mask_np_shape[1]
            y1 = y1 if y1 <= mask_np_shape[0] else mask_np_shape[0]

            post_per_properties_box = np.array([x0, y0, x1, y1])
            post_per_properties_box = np.where(post_per_properties_box > 0,
                                               post_per_properties_box, 0)

            # 如果新的框无效，则返回原始框
            if x0 >= x1 or y0 >= y1:
                post_properties_boxes.append(per_properties_box)
            else:
                post_properties_boxes.append(post_per_properties_box)

        post_properties_boxes = np.stack(post_properties_boxes, axis=0)
        post_properties_boxes = post_properties_boxes.astype(np.float32)

        return post_properties_boxes

    def noise_frame_mask(self, properties_masks):
        mask_h, mask_w, object_nums = properties_masks.shape

        post_properties_masks = []
        for mask_idx in range(object_nums):
            per_properties_mask = properties_masks[:, :, mask_idx]

            origin_mask_area = np.count_nonzero(per_properties_mask)
            total_mask_area = float(mask_h * mask_w)

            mask_area_ratio = origin_mask_area / total_mask_area

            if mask_area_ratio < self.area_filter_ratio:
                post_properties_masks.append(per_properties_mask)
                continue

            reduce_mask_area = origin_mask_area * self.mask_noise_area_ratio
            reduce_area_ratio = reduce_mask_area / total_mask_area
            if reduce_area_ratio < self.area_filter_ratio:
                post_properties_masks.append(per_properties_mask)
                continue

            max_kernel = np.sqrt(reduce_mask_area) / 2.
            if int(max_kernel) > 1:
                kernel = np.random.randint(1, max_kernel)
                kernel = np.ones((kernel, kernel), np.uint8)
                if np.random.uniform(0, 1) < 0.5:
                    post_per_properties_mask = cv2.erode(per_properties_mask,
                                                         kernel,
                                                         iterations=1)
                else:
                    post_per_properties_mask = cv2.dilate(per_properties_mask,
                                                          kernel,
                                                          iterations=1)
            else:
                post_per_properties_mask = per_properties_mask

            if np.count_nonzero(post_per_properties_mask
                                ) / total_mask_area > self.area_filter_ratio:
                post_properties_masks.append(post_per_properties_mask)
            else:
                post_properties_masks.append(per_properties_mask)

        post_properties_masks = np.stack(post_properties_masks, axis=-1)
        post_properties_masks = post_properties_masks.astype(np.float32)

        return post_properties_masks

if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import interactive_segmentation_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.video_interactive_segmentation.common import Sam2Resize, Sam2RandomHorizontalFlip, Sam2RandomMosaicAug, Sam2RandomRsverseFrameOrder, Sam2Normalize, SAM2BatchCollater

    sam2_video_dataset = SAM2ImageSegmentationDataset(
        interactive_segmentation_dataset_path,
        image_set_name=[
            # 'COCO2017',
            'SAMA-COCO',
            'lvisv1.0',
            # 'lvisv1.0_filter_part_object',
            ###########################################
            'HIM2K',
            'I-HIM50K',
            'RefMatte',
            'MAGICK',
            'AM2K',
            'DIS5K',
            'HRS10K',
            'HRSOD',
            'UHRSD',
            ###########################################
            'matting_human_half',
            'Deep_Automatic_Portrait_Matting',
            'RealWorldPortrait636',
            'P3M10K',
            ###########################################
            'sa_000000',
            # 'sa_000000_filter_part_object',
        ],
        image_set_type='train',
        image_per_set_image_choose_max_num={
            'COCO2017': 1000000,
            'SAMA-COCO': 1000000,
            'lvisv1.0': 1000000,
            'lvisv1.0_filter_part_object': 1000000,
            ###########################################
            'HIM2K': 1000000,
            'I-HIM50K': 1000000,
            'RefMatte': 1000000,
            'MAGICK': 1000000,
            'AM2K': 1000000,
            'DIS5K': 1000000,
            'HRS10K': 1000000,
            'HRSOD': 1000000,
            'UHRSD': 1000000,
            ###########################################
            'matting_human_half': 1000000,
            'Deep_Automatic_Portrait_Matting': 1000000,
            'RealWorldPortrait636': 1000000,
            'P3M10K': 1000000,
            ###########################################
            'sa_000000': 1000000,
            'sa_000000_filter_part_object': 1000000,
        },
        per_image_mask_chosse_max_num=16,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            Sam2Resize(resize=1024),
            Sam2RandomHorizontalFlip(prob=0.5),
            Sam2RandomMosaicAug(prob=0.5),
            Sam2RandomRsverseFrameOrder(prob=0.5),
            # Sam2Normalize(mean=[123.675, 116.28, 103.53],
            #               std=[58.395, 57.12, 57.375]),
        ]))
    
    video_count = 0
    for per_sample in tqdm(sam2_video_dataset):
        print('1111', per_sample['video_name'])
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['size'], per_sample['prompt_point'].shape,
              per_sample['prompt_box'].shape, per_sample['prompt_mask'].shape)
        print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['size'].dtype, per_sample['prompt_point'].dtype,
              per_sample['prompt_box'].dtype, per_sample['prompt_mask'].dtype)
        print('1111', np.max(per_sample['mask']), np.min(per_sample['mask']),
              np.unique(per_sample['mask']))
        print('1111', per_sample['mask'].shape[0])
        print('1111', per_sample['mask'].shape[-1])

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # per_video_dir = os.path.join(temp_dir, f'video_{video_count}')
        # if not os.path.exists(per_video_dir):
        #     os.makedirs(per_video_dir)

        # video_name = per_sample['video_name']
        # per_sample_images = per_sample['image']
        # per_sample_masks = per_sample['mask']
        # per_sample_prompt_points = per_sample['prompt_point']
        # per_sample_prompt_boxes = per_sample['prompt_box']
        # per_sample_prompt_masks = per_sample['prompt_mask']

        # object_nums = per_sample_masks.shape[-1]
        # frame_nums = per_sample_masks.shape[0]

        # for per_object_idx in tqdm(range(object_nums)):
        #     per_object_masks = per_sample_masks[:, :, :, per_object_idx]
        #     per_object_prompt_points = per_sample_prompt_points[:,
        #                                                         per_object_idx, :, :]
        #     per_object_prompt_boxes = per_sample_prompt_boxes[:,
        #                                                       per_object_idx, :]
        #     per_object_prompt_masks = per_sample_prompt_masks[:, :, :,
        #                                                       per_object_idx]

        #     per_video_object_dir = os.path.join(per_video_dir,
        #                                         f'object_{per_object_idx}')
        #     if not os.path.exists(per_video_object_dir):
        #         os.makedirs(per_video_object_dir)

        #     positive_prompt_point_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     negative_prompt_point_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     prompt_box_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     prompt_mask_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     for per_frame_idx in range(frame_nums):
        #         per_object_frame_mask = per_object_masks[per_frame_idx]
        #         per_object_frame_mask = (per_object_frame_mask * 255.).astype(
        #             np.uint8)

        #         per_object_frame_image = per_sample_images[per_frame_idx]
        #         per_object_frame_image = np.ascontiguousarray(
        #             per_object_frame_image, dtype=np.uint8)
        #         per_object_frame_image = cv2.cvtColor(per_object_frame_image,
        #                                               cv2.COLOR_RGB2BGR)

        #         b_channel, g_channel, r_channel = cv2.split(
        #             per_object_frame_image)
        #         per_object_frame_image_with_mask = cv2.merge(
        #             [b_channel, g_channel, r_channel, per_object_frame_mask])

        #         cv2.imencode('.jpg', per_object_frame_image)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_image.jpg'
        #             ))
        #         cv2.imencode('.png', per_object_frame_mask)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_mask.jpg'
        #             ))
        #         cv2.imencode(
        #             '.png', per_object_frame_image_with_mask
        #         )[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_image_with_mask.jpg'
        #             ))

        #         per_object_frame_prompt_points = per_object_prompt_points[
        #             per_frame_idx]
        #         per_object_frame_prompt_box = per_object_prompt_boxes[
        #             per_frame_idx]

        #         per_object_frame_image_for_prompt_point_box = per_object_frame_image.copy(
        #         ).astype(np.uint8)

        #         for per_point in per_object_frame_prompt_points:
        #             if per_point[2] == 1:
        #                 cv2.circle(per_object_frame_image_for_prompt_point_box,
        #                            (int(per_point[0]), int(per_point[1])), 10,
        #                            positive_prompt_point_color, -1)
        #             elif per_point[2] == 0:
        #                 cv2.circle(per_object_frame_image_for_prompt_point_box,
        #                            (int(per_point[0]), int(per_point[1])), 10,
        #                            negative_prompt_point_color, -1)

        #         left_top, right_bottom = (
        #             int(per_object_frame_prompt_box[0]),
        #             int(per_object_frame_prompt_box[1])), (
        #                 int(per_object_frame_prompt_box[2]),
        #                 int(per_object_frame_prompt_box[3]))
        #         cv2.rectangle(per_object_frame_image_for_prompt_point_box,
        #                       left_top,
        #                       right_bottom,
        #                       color=prompt_box_color,
        #                       thickness=2,
        #                       lineType=cv2.LINE_AA)
        #         text = 'object_box'
        #         text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #         fill_right_bottom = (max(left_top[0] + text_size[0],
        #                                  right_bottom[0]),
        #                              left_top[1] - text_size[1] - 3)
        #         cv2.rectangle(per_object_frame_image_for_prompt_point_box,
        #                       left_top,
        #                       fill_right_bottom,
        #                       color=prompt_box_color,
        #                       thickness=-1,
        #                       lineType=cv2.LINE_AA)
        #         cv2.putText(per_object_frame_image_for_prompt_point_box,
        #                     text, (left_top[0], left_top[1] - 2),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     color=(0, 0, 0),
        #                     thickness=1,
        #                     lineType=cv2.LINE_AA)

        #         cv2.imencode(
        #             '.jpg', per_object_frame_image_for_prompt_point_box
        #         )[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_image_with_prompt_point_box.jpg'
        #             ))

        #         per_object_frame_image_for_prompt_mask = per_object_frame_image.copy(
        #         ).astype(np.uint8)
        #         per_object_frame_prompt_mask = per_object_prompt_masks[
        #             per_frame_idx].astype(np.uint8)

        #         per_image_prompt_mask = np.zeros(
        #             (per_object_frame_image_for_prompt_mask.shape[0],
        #              per_object_frame_image_for_prompt_mask.shape[1], 3))
        #         per_image_prompt_contours = []
        #         per_object_frame_prompt_mask = np.nonzero(
        #             per_object_frame_prompt_mask == 1.)
        #         if len(per_object_frame_prompt_mask[0]) > 0:
        #             per_image_prompt_mask[
        #                 per_object_frame_prompt_mask[0],
        #                 per_object_frame_prompt_mask[1]] = prompt_mask_color
        #         new_per_image_prompt_mask = np.zeros(
        #             (per_object_frame_image_for_prompt_mask.shape[0],
        #              per_object_frame_image_for_prompt_mask.shape[1]))
        #         new_per_image_prompt_mask[
        #             per_object_frame_prompt_mask[0],
        #             per_object_frame_prompt_mask[1]] = 255
        #         contours, _ = cv2.findContours(
        #             new_per_image_prompt_mask.astype(np.uint8), cv2.RETR_TREE,
        #             cv2.CHAIN_APPROX_SIMPLE)
        #         per_image_prompt_contours.append(contours)
        #         per_image_prompt_mask = per_image_prompt_mask.astype(np.uint8)
        #         per_image_prompt_mask = cv2.cvtColor(per_image_prompt_mask,
        #                                              cv2.COLOR_RGBA2BGR)
        #         all_classes_mask = np.nonzero(per_image_prompt_mask != 0)
        #         if len(all_classes_mask[0]) > 0:
        #             per_image_prompt_mask[
        #                 all_classes_mask[0],
        #                 all_classes_mask[1]] = cv2.addWeighted(
        #                     per_object_frame_image_for_prompt_mask[
        #                         all_classes_mask[0], all_classes_mask[1]], 0.5,
        #                     per_image_prompt_mask[all_classes_mask[0],
        #                                           all_classes_mask[1]], 1, 0)
        #         no_class_mask = np.nonzero(per_image_prompt_mask == 0)
        #         if len(no_class_mask[0]) > 0:
        #             per_image_prompt_mask[no_class_mask[0], no_class_mask[
        #                 1]] = per_object_frame_image_for_prompt_mask[
        #                     no_class_mask[0], no_class_mask[1]]
        #         for contours in per_image_prompt_contours:
        #             cv2.drawContours(per_image_prompt_mask, contours, -1,
        #                              (255, 255, 255), 2)

        #         cv2.imencode('.jpg', per_image_prompt_mask)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_image_with_prompt_mask.jpg'
        #             ))

        if video_count < 2:
            video_count += 1
        else:
            break


    from torch.utils.data import DataLoader

    collater = SAM2BatchCollater(resize=1024)
    train_loader = DataLoader(sam2_video_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        input_images, input_masks = data['image'], data['mask']

        input_prompt_points, input_prompt_boxes, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        object_to_frame_idxs = data['object_to_frame_idx']

        print('2222', input_images.shape, input_masks.shape,
              input_prompt_points.shape, input_prompt_boxes.shape,
              input_prompt_masks.shape, object_to_frame_idxs.shape)
        print('2222', input_images.dtype, input_masks.dtype,
              input_prompt_points.dtype, input_prompt_boxes.dtype,
              input_prompt_masks.dtype, object_to_frame_idxs.dtype)

        if count < 2:
            count += 1
        else:
            break