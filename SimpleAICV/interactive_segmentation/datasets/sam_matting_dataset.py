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


class SAMMattingDataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name=[
                     'DIS5K',
                     'sa_000000',
                 ],
                 set_type='train',
                 per_set_image_choose_max_num={
                     'DIS5K': 1000000,
                     'sa_000000': 1000000,
                 },
                 max_side=2048,
                 kernel_size_range=[15, 15],
                 per_image_mask_chosse_max_num=16,
                 points_num=1,
                 area_filter_ratio=0.0001,
                 box_noise_wh_ratio=0.1,
                 mask_noise_area_ratio=0.04,
                 transform=None):

        self.all_set_image_path_list = collections.OrderedDict()
        self.all_set_image_nums = collections.OrderedDict()

        # for per_set_name in tqdm(set_name):
        #     per_set_dir = os.path.join(root_dir, per_set_name, set_type)
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
                executor.submit(self.process_set, root_dir, per_set_name, set_type):
                per_set_name
                for per_set_name in set_name
            }

            for future in tqdm(as_completed(futures), total=len(set_name)):
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
            per_set_image_max_num = per_set_image_choose_max_num[per_set_name]
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

        self.max_side = max_side
        self.kernel_size_range = kernel_size_range
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
        _, _, image_path, _, _, _ = self.all_image_mask_path_list[idx]

        image = self.load_image(idx)
        mask = self.load_mask(idx)

        box = self.load_box((mask > 0.5).astype(np.float32))

        image_h, image_w = image.shape[0], image.shape[1]
        if max(image_h, image_w) > self.max_side:
            factor = self.max_side / max(image_h, image_w)
            resize_w, resize_h = int(image_w * float(factor) +
                                     0.5), int(image_h * float(factor) + 0.5)
            image = cv2.resize(image, (resize_w, resize_h))
            mask = cv2.resize(mask, (resize_w, resize_h))
            box = box * factor

        trimap = self.generate_trimap_from_mask(mask)
        fg_map, bg_map = self.generate_fg_bg_map_from_mask(image, mask)

        prompt_point = self.load_points((mask > 0.5).astype(np.float32))

        mask_size = [mask.shape[0], mask.shape[1]]
        prompt_box = self.noise_box(box, mask_size)

        prompt_mask = self.noise_mask((mask > 0.2).astype(np.float32))

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'image_path': image_path,
            'image': image,
            'box': box,
            'mask': mask,
            'size': size,
            'prompt_point': prompt_point,
            'prompt_box': prompt_box,
            'prompt_mask': prompt_mask,
            'trimap': trimap,
            'fg_map': fg_map,
            'bg_map': bg_map,
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

        elif per_mask_label_path.endswith('.json'):
            with open(per_mask_label_path, encoding='utf-8') as f:
                per_image_json_data = json.load(f)
                per_image_annotation = per_image_json_data['annotations']
                per_annot = per_image_annotation[mask_list_idx]

                target_mask = mask_utils.decode(per_annot['segmentation'])
                target_mask[target_mask > 0] = 1

        return target_mask.astype(np.float32)

    def generate_trimap_from_mask(self, alpha):
        alpha_h, alpha_w = alpha.shape[0], alpha.shape[1]
        long_size_length = max(alpha_h, alpha_w)
        side_scale = long_size_length / self.max_side

        if self.kernel_size_range[0] == self.kernel_size_range[1]:
            kernel_size = int(self.kernel_size_range[0] * side_scale)
        else:
            kernel_size = int(
                np.random.randint(self.kernel_size_range[0],
                                  self.kernel_size_range[1]) * side_scale)
        kernel_size = max(3, kernel_size)

        alpha_clone = alpha.copy() * 255.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        fg_and_unknown = np.array(
            np.not_equal(alpha_clone, 0).astype(np.float32))
        fg = np.array(np.equal(alpha_clone, 255).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
        erode = cv2.erode(fg, kernel, iterations=1)
        trimap = erode * 255 + (dilate - erode) * 128

        return trimap.astype(np.uint8)

    def generate_fg_bg_map_from_mask(self, image, alpha):
        expand_dim_mask = np.expand_dims(alpha.copy(),
                                         axis=2).astype(np.float32)
        fg_map = image * expand_dim_mask
        bg_map = image * (1. - expand_dim_mask)

        return fg_map.astype(np.float32), bg_map.astype(np.float32)

    def load_points(self, mask):
        if np.count_nonzero(mask) == 0:
            # 若没有前景区域,取所有背景区域点
            all_point_coords = np.argwhere(mask == 0)
            point_label = 0
        else:
            # 若前景区域,取所有前景区域点
            all_point_coords = np.argwhere(mask != 0)
            point_label = 1

        points = []
        all_points_num = len(all_point_coords)
        points_index = np.random.choice(all_points_num,
                                        self.points_num,
                                        replace=False)
        for per_point_idx in points_index:
            points.append([
                all_point_coords[per_point_idx][1],
                all_point_coords[per_point_idx][0],
                point_label,
            ])
        points = np.array(points, dtype=np.float32)

        return points

    def load_box(self, mask):
        h, w = mask.shape[0], mask.shape[1]

        # 创建网格坐标
        xs = np.arange(w, dtype=np.int32)
        ys = np.arange(h, dtype=np.int32)
        # grid_xs:[h,w],grid_ys[h,w]
        grid_xs, grid_ys = np.meshgrid(xs, ys, indexing='xy')

        # 将mask转换为布尔类型
        mask = mask.astype(bool)

        # 使用向量化操作计算边界框
        if mask.any():
            # 计算最小和最大坐标
            x_min = np.min(np.where(mask, grid_xs, w))
            y_min = np.min(np.where(mask, grid_ys, h))
            x_max = np.max(np.where(mask, grid_xs, -1))
            y_max = np.max(np.where(mask, grid_ys, -1))
        else:
            # 如果没有前景像素，返回无效的边界框
            x_min, y_min, x_max, y_max = w, h, -1, -1

        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        return box

    def noise_box(self, properties_box, mask_np_shape):
        if -1 in properties_box:
            return properties_box.astype(np.float32)

        w, h = properties_box[2] - properties_box[0], properties_box[
            3] - properties_box[1]

        if h / mask_np_shape[0] <= math.sqrt(
                self.area_filter_ratio) or w / mask_np_shape[1] <= math.sqrt(
                    self.area_filter_ratio):
            return properties_box.astype(np.float32)

        noise_x, noise_y = w * self.box_noise_wh_ratio, h * self.box_noise_wh_ratio
        noise_x, noise_y = min(int(mask_np_shape[1] * 0.02),
                               noise_x), min(int(mask_np_shape[0] * 0.02),
                                             noise_y)

        if noise_x <= 1 or noise_y <= 1:
            return properties_box.astype(np.float32)

        # 随机生成抖动的边界框
        x0 = properties_box[0] + max(
            min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
        y0 = properties_box[1] + max(
            min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)
        x1 = properties_box[2] + max(
            min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
        y1 = properties_box[3] + max(
            min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)

        # 限制坐标范围，避免越界
        x0 = x0 if x0 >= 0 else 0
        y0 = y0 if y0 >= 0 else 0
        x1 = x1 if x1 <= mask_np_shape[1] else mask_np_shape[1]
        y1 = y1 if y1 <= mask_np_shape[0] else mask_np_shape[0]

        post_properties_box = np.array([x0, y0, x1, y1])
        post_properties_box = np.where(post_properties_box > 0,
                                       post_properties_box, 0)

        if x0 >= x1 or y0 >= y1:
            return properties_box.astype(np.float32)
        else:
            return post_properties_box.astype(np.float32)

    def noise_mask(self, properties_mask):
        mask_h, mask_w = properties_mask.shape[0], properties_mask.shape[1]

        origin_mask_area = np.count_nonzero(properties_mask)
        total_mask_area = float(mask_h * mask_w)

        mask_area_ratio = origin_mask_area / total_mask_area

        if mask_area_ratio < self.area_filter_ratio:
            return properties_mask.astype(np.float32)

        reduce_mask_area = origin_mask_area * self.mask_noise_area_ratio
        reduce_area_ratio = reduce_mask_area / total_mask_area

        if reduce_area_ratio < self.area_filter_ratio:
            return properties_mask.astype(np.float32)

        max_kernel = np.sqrt(reduce_mask_area) / 2.
        if int(max_kernel) > 1:
            kernel = np.random.randint(1, max_kernel)
            kernel = np.ones((kernel, kernel), np.uint8)
            if np.random.uniform(0, 1) < 0.5:
                post_properties_mask = cv2.erode(properties_mask,
                                                 kernel,
                                                 iterations=1)
            else:
                post_properties_mask = cv2.dilate(properties_mask,
                                                  kernel,
                                                  iterations=1)
        else:
            post_properties_mask = properties_mask

        if np.count_nonzero(post_properties_mask
                            ) / total_mask_area > self.area_filter_ratio:
            return post_properties_mask.astype(np.float32)
        else:
            return properties_mask.astype(np.float32)


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

    import copy

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.interactive_segmentation.common_matting import SAMMattingRandomScale, SAMMattingRandomTranslate, SAMMattingRandomRGBToGRAY, SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingRandomVerticalFlip, SAMMattingBatchCollater

    samdataset = SAMMattingDataset(
        interactive_segmentation_dataset_path,
        set_name=[
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
        set_type='train',
        per_set_image_choose_max_num={
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
        max_side=2048,
        kernel_size_range=[15, 15],
        per_image_mask_chosse_max_num=16,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            SAMMattingResize(resize=1024),
            SAMMattingRandomScale(scale=[0.5, 1.0], area_ratio=0.01, prob=0.5),
            SAMMattingRandomTranslate(prob=0.5),
            SAMMattingRandomHorizontalFlip(prob=0.5),
            SAMMattingRandomVerticalFlip(prob=0.5),
            SAMMattingRandomRGBToGRAY(prob=0.5),
            # SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
            #                     std=[58.395, 57.12, 57.375]),
        ]))

    count = 0
    for per_sample in tqdm(samdataset):
        print('1111', per_sample['image_path'])
        print('1111', per_sample['image'].shape, per_sample['box'].shape,
              per_sample['mask'].shape, per_sample['size'],
              per_sample['prompt_point'].shape, per_sample['prompt_box'].shape,
              per_sample['prompt_mask'].shape, per_sample['trimap'].shape,
              per_sample['fg_map'].shape, per_sample['bg_map'].shape)
        print('2222', per_sample['image'].dtype, per_sample['box'].dtype,
              per_sample['mask'].dtype, per_sample['size'].dtype,
              per_sample['prompt_point'].dtype, per_sample['prompt_box'].dtype,
              per_sample['prompt_mask'].dtype, per_sample['trimap'].dtype,
              per_sample['fg_map'].dtype, per_sample['bg_map'].dtype)
        print('3333', per_sample['box'], per_sample['size'],
              per_sample['prompt_point'][0], per_sample['prompt_box'])
        print('1212', np.max(per_sample['mask']), np.min(per_sample['mask']))
        print('1313', np.unique(per_sample['trimap']))
        print('1414', np.max(per_sample['fg_map']),
              np.min(per_sample['fg_map']))
        print('1515', np.max(per_sample['bg_map']),
              np.min(per_sample['bg_map']))

        # temp_dir = f'./temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # box = per_sample['box']
        # mask = per_sample['mask']

        # image_for_box = copy.deepcopy(image)
        # per_image_box = (box[0:4]).astype(np.int32)

        # if -1 not in per_image_box:
        #     box_color = [int(np.random.choice(range(256))) for _ in range(3)]
        #     left_top, right_bottom = (per_image_box[0],
        #                               per_image_box[1]), (per_image_box[2],
        #                                                   per_image_box[3])
        #     cv2.rectangle(image_for_box,
        #                   left_top,
        #                   right_bottom,
        #                   color=box_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)
        #     text = 'object_box'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(image_for_box,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=box_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(image_for_box,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        # image_for_mask = copy.deepcopy(image).astype(np.uint8)
        # mask = mask.astype(np.uint8)
        # per_image_mask = np.zeros(
        #     (image_for_mask.shape[0], image_for_mask.shape[1], 3))
        # per_image_contours = []
        # mask = np.nonzero(mask == 1.)
        # mask_color = [int(np.random.choice(range(256))) for _ in range(3)]
        # per_image_mask[mask[0], mask[1]] = mask_color
        # new_per_image_mask = np.zeros(
        #     (image_for_mask.shape[0], image_for_mask.shape[1]))
        # new_per_image_mask[mask[0], mask[1]] = 255
        # contours, _ = cv2.findContours(new_per_image_mask.astype(np.uint8),
        #                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # per_image_contours.append(contours)
        # per_image_mask = per_image_mask.astype(np.uint8)
        # per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_image_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_image_mask[all_classes_mask[0],
        #                    all_classes_mask[1]] = cv2.addWeighted(
        #                        image_for_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 0.5,
        #                        per_image_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_image_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_image_mask[no_class_mask[0],
        #                    no_class_mask[1]] = image_for_mask[no_class_mask[0],
        #                                                       no_class_mask[1]]
        # for contours in per_image_contours:
        #     cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 2)

        # cv2.imencode('.jpg', image_for_box)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_box.jpg'))
        # cv2.imencode('.jpg', per_image_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_mask.jpg'))

        # temp_dir = f'./temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # prompt_point = per_sample['prompt_point']
        # prompt_box = per_sample['prompt_box']
        # prompt_mask = per_sample['prompt_mask']
        # positive_prompt_point_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # negative_prompt_point_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # prompt_box_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # prompt_mask_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]

        # image_for_prompt_box = copy.deepcopy(image)

        # for per_point in prompt_point:
        #     point_label = per_point[2]
        #     if point_label == 1:
        #         cv2.circle(image_for_prompt_box,
        #                    (int(per_point[0]), int(per_point[1])), 10,
        #                    positive_prompt_point_color, -1)
        #     elif point_label == 0:
        #         cv2.circle(image_for_prompt_box,
        #                    (int(per_point[0]), int(per_point[1])), 10,
        #                    negative_prompt_point_color, -1)

        # per_image_prompt_box = (prompt_box[0:4]).astype(np.int32)

        # if -1 not in per_image_prompt_box:
        #     left_top, right_bottom = (per_image_prompt_box[0],
        #                               per_image_prompt_box[1]), (
        #                                   per_image_prompt_box[2],
        #                                   per_image_prompt_box[3])
        #     cv2.rectangle(image_for_prompt_box,
        #                   left_top,
        #                   right_bottom,
        #                   color=prompt_box_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)
        #     text = f'prompt_box'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(image_for_prompt_box,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=prompt_box_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(image_for_prompt_box,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        # image_for_prompt_mask = copy.deepcopy(image).astype(np.uint8)

        # for per_point in prompt_point:
        #     point_label = per_point[2]
        #     if point_label == 1:
        #         cv2.circle(image_for_prompt_box,
        #                    (int(per_point[0]), int(per_point[1])), 10,
        #                    positive_prompt_point_color, -1)
        #     elif point_label == 0:
        #         cv2.circle(image_for_prompt_box,
        #                    (int(per_point[0]), int(per_point[1])), 10,
        #                    negative_prompt_point_color, -1)

        # prompt_mask = prompt_mask.astype(np.uint8)
        # per_image_prompt_mask = np.zeros((image_for_prompt_mask.shape[0],
        #                                   image_for_prompt_mask.shape[1], 3))
        # per_image_prompt_contours = []
        # prompt_mask = np.nonzero(prompt_mask == 1.)
        # if len(prompt_mask[0]) > 0:
        #     per_image_prompt_mask[prompt_mask[0],
        #                           prompt_mask[1]] = prompt_mask_color
        # new_per_image_prompt_mask = np.zeros(
        #     (image_for_prompt_mask.shape[0], image_for_prompt_mask.shape[1]))
        # new_per_image_prompt_mask[prompt_mask[0], prompt_mask[1]] = 255
        # contours, _ = cv2.findContours(
        #     new_per_image_prompt_mask.astype(np.uint8), cv2.RETR_TREE,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # per_image_prompt_contours.append(contours)
        # per_image_prompt_mask = per_image_prompt_mask.astype(np.uint8)
        # per_image_prompt_mask = cv2.cvtColor(per_image_prompt_mask,
        #                                      cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_image_prompt_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_image_prompt_mask[
        #         all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
        #             image_for_prompt_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 0.5,
        #             per_image_prompt_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_image_prompt_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_image_prompt_mask[no_class_mask[0],
        #                           no_class_mask[1]] = image_for_prompt_mask[
        #                               no_class_mask[0], no_class_mask[1]]
        # for contours in per_image_prompt_contours:
        #     cv2.drawContours(per_image_prompt_mask, contours, -1,
        #                      (255, 255, 255), 2)

        # cv2.imencode('.jpg', image_for_prompt_box)[1].tofile(
        #     os.path.join(temp_dir,
        #                  f'idx_{count}_image_with_prompt_point_box.jpg'))
        # cv2.imencode('.jpg', per_image_prompt_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_prompt_mask.jpg'))

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # mask = per_sample['mask'] * 255.

        # trimap = per_sample['trimap']
        # fg_map = per_sample['fg_map']
        # fg_map = cv2.cvtColor(fg_map, cv2.COLOR_RGB2BGR)
        # bg_map = per_sample['bg_map']
        # bg_map = cv2.cvtColor(bg_map, cv2.COLOR_RGB2BGR)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image.jpg'))
        # cv2.imencode('.jpg', mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask.jpg'))
        # cv2.imencode('.jpg', trimap)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_trimap.jpg'))
        # cv2.imencode('.jpg', fg_map)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_fg_map.jpg'))
        # cv2.imencode('.jpg', bg_map)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_bg_map.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader

    collater = SAMMattingBatchCollater(resize=1024)
    train_loader = DataLoader(samdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
            'bg_map']

        print('4444', input_images.shape, input_boxs.shape, input_masks.shape,
              sizes)
        print('5555', input_images.dtype, input_boxs.dtype, input_masks.dtype,
              sizes.dtype)
        print('6666', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)
        print('7777', input_prompt_points.dtype, input_prompt_boxs.dtype,
              input_prompt_masks.dtype)
        print('8888', trimaps.shape, fg_maps.shape, bg_maps.shape)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # for i, (per_image, per_image_box, per_image_mask) in enumerate(
        #         zip(input_images, input_boxs, input_masks)):
        #     per_image = per_image.permute(1, 2, 0).cpu().numpy()
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_box = per_image_box.cpu().numpy()
        #     per_image_mask = per_image_mask.cpu().numpy()
        #     per_image_mask = np.squeeze(per_image_mask, axis=0)

        #     box_color = [int(np.random.choice(range(256))) for _ in range(3)]
        #     mask_color = [int(np.random.choice(range(256))) for _ in range(3)]

        #     image_for_box = copy.deepcopy(per_image).astype(np.uint8)
        #     per_image_box = (per_image_box[0:4]).astype(np.int32)

        #     if -1 not in per_image_box:
        #         left_top, right_bottom = (per_image_box[0],
        #                                   per_image_box[1]), (per_image_box[2],
        #                                                       per_image_box[3])
        #         cv2.rectangle(image_for_box,
        #                       left_top,
        #                       right_bottom,
        #                       color=box_color,
        #                       thickness=2,
        #                       lineType=cv2.LINE_AA)
        #         text = f'object_box'
        #         text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #         fill_right_bottom = (max(left_top[0] + text_size[0],
        #                                  right_bottom[0]),
        #                              left_top[1] - text_size[1] - 3)
        #         cv2.rectangle(image_for_box,
        #                       left_top,
        #                       fill_right_bottom,
        #                       color=box_color,
        #                       thickness=-1,
        #                       lineType=cv2.LINE_AA)
        #         cv2.putText(image_for_box,
        #                     text, (left_top[0], left_top[1] - 2),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     color=(0, 0, 0),
        #                     thickness=1,
        #                     lineType=cv2.LINE_AA)

        #     image_for_mask = copy.deepcopy(per_image).astype(np.uint8)
        #     per_image_mask = per_image_mask.astype(np.uint8)

        #     per_image_draw_mask = np.zeros(
        #         (image_for_mask.shape[0], image_for_mask.shape[1], 3))
        #     per_image_contours = []
        #     per_image_mask = np.nonzero(per_image_mask == 1.)
        #     if len(per_image_mask[0]) > 0:
        #         per_image_draw_mask[per_image_mask[0],
        #                             per_image_mask[1]] = mask_color
        #     new_per_image_draw_mask = np.zeros(
        #         (image_for_mask.shape[0], image_for_mask.shape[1]))
        #     if len(per_image_mask[0]) > 0:
        #         new_per_image_draw_mask[per_image_mask[0],
        #                                 per_image_mask[1]] = 255
        #     contours, _ = cv2.findContours(
        #         new_per_image_draw_mask.astype(np.uint8), cv2.RETR_TREE,
        #         cv2.CHAIN_APPROX_SIMPLE)
        #     per_image_contours.append(contours)
        #     per_image_draw_mask = per_image_draw_mask.astype(np.uint8)
        #     per_image_draw_mask = cv2.cvtColor(per_image_draw_mask,
        #                                        cv2.COLOR_RGBA2BGR)
        #     all_classes_mask = np.nonzero(per_image_draw_mask != 0)
        #     if len(all_classes_mask[0]) > 0:
        #         per_image_draw_mask[all_classes_mask[0],
        #                             all_classes_mask[1]] = cv2.addWeighted(
        #                                 image_for_mask[all_classes_mask[0],
        #                                                all_classes_mask[1]],
        #                                 0.5, per_image_draw_mask[
        #                                     all_classes_mask[0],
        #                                     all_classes_mask[1]], 1, 0)
        #     no_class_mask = np.nonzero(per_image_draw_mask == 0)
        #     if len(no_class_mask[0]) > 0:
        #         per_image_draw_mask[no_class_mask[0],
        #                             no_class_mask[1]] = image_for_mask[
        #                                 no_class_mask[0], no_class_mask[1]]
        #     for contours in per_image_contours:
        #         cv2.drawContours(per_image_draw_mask, contours, -1,
        #                          (255, 255, 255), 2)

        #     cv2.imencode('.jpg', image_for_box)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_image_with_box.jpg'))
        #     cv2.imencode('.jpg', per_image_draw_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_image_with_mask.jpg'))

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # for i, (per_image, per_image_prompt_points, per_image_prompt_box,
        #         per_image_prompt_mask) in enumerate(
        #             zip(input_images, input_prompt_points, input_prompt_boxs,
        #                 input_prompt_masks)):
        #     per_image = per_image.permute(1, 2, 0).cpu().numpy()
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_prompt_points = per_image_prompt_points.cpu().numpy()
        #     per_image_prompt_box = per_image_prompt_box.cpu().numpy()
        #     per_image_prompt_mask = per_image_prompt_mask.cpu().numpy()
        #     per_image_prompt_mask = np.squeeze(per_image_prompt_mask, axis=0)

        #     per_image_prompt_mask = cv2.resize(
        #         per_image_prompt_mask, (per_image_prompt_mask.shape[1] * 4,
        #                                 per_image_prompt_mask.shape[0] * 4))

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

        #     image_for_prompt_box = copy.deepcopy(per_image)

        #     for per_point in per_image_prompt_points:
        #         point_label = per_point[2]
        #         if point_label == 1:
        #             cv2.circle(image_for_prompt_box,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        positive_prompt_point_color, -1)
        #         elif point_label == 0:
        #             cv2.circle(image_for_prompt_box,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        negative_prompt_point_color, -1)

        #     per_image_prompt_box = (per_image_prompt_box[0:4]).astype(np.int32)

        #     if -1 not in per_image_prompt_box:
        #         left_top, right_bottom = (per_image_prompt_box[0],
        #                                   per_image_prompt_box[1]), (
        #                                       per_image_prompt_box[2],
        #                                       per_image_prompt_box[3])
        #         cv2.rectangle(image_for_prompt_box,
        #                       left_top,
        #                       right_bottom,
        #                       color=prompt_box_color,
        #                       thickness=2,
        #                       lineType=cv2.LINE_AA)
        #         text = f'prompt_box'
        #         text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #         fill_right_bottom = (max(left_top[0] + text_size[0],
        #                                  right_bottom[0]),
        #                              left_top[1] - text_size[1] - 3)
        #         cv2.rectangle(image_for_prompt_box,
        #                       left_top,
        #                       fill_right_bottom,
        #                       color=prompt_box_color,
        #                       thickness=-1,
        #                       lineType=cv2.LINE_AA)
        #         cv2.putText(image_for_prompt_box,
        #                     text, (left_top[0], left_top[1] - 2),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     color=(0, 0, 0),
        #                     thickness=1,
        #                     lineType=cv2.LINE_AA)

        #     image_for_prompt_mask = copy.deepcopy(per_image).astype(np.uint8)

        #     for per_point in per_image_prompt_points:
        #         point_label = per_point[2]
        #         if point_label == 1:
        #             cv2.circle(image_for_prompt_mask,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        positive_prompt_point_color, -1)
        #         elif point_label == 0:
        #             cv2.circle(image_for_prompt_mask,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        negative_prompt_point_color, -1)

        #     per_image_prompt_mask = per_image_prompt_mask.astype(np.uint8)
        #     per_image_prompt_draw_mask = np.zeros(
        #         (image_for_prompt_mask.shape[0],
        #          image_for_prompt_mask.shape[1], 3))
        #     per_image_prompt_contours = []
        #     per_image_prompt_mask = np.nonzero(per_image_prompt_mask == 1.)
        #     if len(per_image_prompt_mask[0]) > 0:
        #         per_image_prompt_draw_mask[
        #             per_image_prompt_mask[0],
        #             per_image_prompt_mask[1]] = prompt_mask_color
        #     new_per_image_prompt_draw_mask = np.zeros(
        #         (image_for_prompt_mask.shape[0],
        #          image_for_prompt_mask.shape[1]))
        #     if len(per_image_prompt_mask[0]) > 0:
        #         new_per_image_prompt_draw_mask[per_image_prompt_mask[0],
        #                                        per_image_prompt_mask[1]] = 255
        #     contours, _ = cv2.findContours(
        #         new_per_image_prompt_draw_mask.astype(np.uint8), cv2.RETR_TREE,
        #         cv2.CHAIN_APPROX_SIMPLE)
        #     per_image_prompt_contours.append(contours)
        #     per_image_prompt_draw_mask = per_image_prompt_draw_mask.astype(
        #         np.uint8)
        #     per_image_prompt_draw_mask = cv2.cvtColor(
        #         per_image_prompt_draw_mask, cv2.COLOR_RGBA2BGR)
        #     all_classes_mask = np.nonzero(per_image_prompt_draw_mask != 0)
        #     if len(all_classes_mask[0]) > 0:
        #         per_image_prompt_draw_mask[
        #             all_classes_mask[0],
        #             all_classes_mask[1]] = cv2.addWeighted(
        #                 image_for_prompt_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 0.5,
        #                 per_image_prompt_draw_mask[all_classes_mask[0],
        #                                            all_classes_mask[1]], 1, 0)
        #     no_class_mask = np.nonzero(per_image_prompt_draw_mask == 0)
        #     if len(no_class_mask[0]) > 0:
        #         per_image_prompt_draw_mask[
        #             no_class_mask[0],
        #             no_class_mask[1]] = image_for_prompt_mask[no_class_mask[0],
        #                                                       no_class_mask[1]]
        #     for contours in per_image_prompt_contours:
        #         cv2.drawContours(per_image_prompt_draw_mask, contours, -1,
        #                          (255, 255, 255), 2)

        #     cv2.imencode('.jpg', image_for_prompt_box)[1].tofile(
        #         os.path.join(
        #             temp_dir,
        #             f'idx_{count}_{i}_image_with_prompt_point_box.jpg'))
        #     cv2.imencode('.jpg', per_image_prompt_draw_mask)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count}_{i}_image_with_prompt_mask.jpg'))

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # for i, (per_image, per_mask, per_trimap, per_fg_map,
        #         per_bg_map) in enumerate(
        #             zip(input_images, input_masks, trimaps, fg_maps, bg_maps)):
        #     per_image = per_image.permute(1, 2, 0).cpu().numpy()
        #     per_fg_map = per_fg_map.permute(1, 2, 0).cpu().numpy()
        #     per_bg_map = per_bg_map.permute(1, 2, 0).cpu().numpy()
        #     per_mask = per_mask.cpu().numpy()
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)
        #     per_mask = per_mask * 255.
        #     per_mask = np.squeeze(per_mask, axis=0)

        #     per_trimap = np.ascontiguousarray(per_trimap, dtype=np.uint8)
        #     per_fg_map = np.ascontiguousarray(per_fg_map, dtype=np.uint8)
        #     per_fg_map = cv2.cvtColor(per_fg_map, cv2.COLOR_RGB2BGR)
        #     per_bg_map = np.ascontiguousarray(per_bg_map, dtype=np.uint8)
        #     per_bg_map = cv2.cvtColor(per_bg_map, cv2.COLOR_RGB2BGR)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_image.jpg'))
        #     cv2.imencode('.jpg', per_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))
        #     cv2.imencode('.jpg', per_trimap)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_trimap.jpg'))
        #     cv2.imencode('.jpg', per_fg_map)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_fg_map.jpg'))
        #     cv2.imencode('.jpg', per_bg_map)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_bg_map.jpg'))

        if count < 2:
            count += 1
        else:
            break
