import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

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

from tools.path import interactive_segmentation_dataset_path, video_interactive_segmentation_dataset_path, background_video_dataset_path


class SAM2VideoMattingDataset(Dataset):

    def __init__(
            self,
            image_root_dir=interactive_segmentation_dataset_path,
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
            video_root_dir=video_interactive_segmentation_dataset_path,
            video_set_name=[
                'sav_000',
            ],
            video_set_type='train',
            video_matting_root_dir=video_interactive_segmentation_dataset_path,
            video_matting_set_name_list=[
                'VideoMatte240K',
            ],
            video_matting_use_background_video_prob={
                'VideoMatte240K': 1.0,
            },
            video_matting_set_type='train',
            video_matting_background_dir=background_video_dataset_path,
            video_matting_background_set_type='train',
            per_video_choose_frame_nums=8,
            per_video_choose_object_nums=2,
            max_side=2048,
            kernel_size_range=[15, 15],
            points_num=1,
            area_filter_ratio=0.0001,
            box_noise_wh_ratio=0.1,
            mask_noise_area_ratio=0.04,
            transform=None):

        # 加载所有sam image数据集
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################
        self.all_set_image_path_list = collections.OrderedDict()
        self.all_set_image_nums = collections.OrderedDict()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.process_sam_image_set, image_root_dir, per_set_name, image_set_type):
                per_set_name
                for per_set_name in image_set_name
            }

            for future in tqdm(as_completed(futures),
                               total=len(image_set_name),
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
            per_set_image_max_num = image_per_set_image_choose_max_num[
                per_set_name]
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

        with ThreadPoolExecutor(max_workers=8) as executor:
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

        print(f'Add all image dataset!')
        print(f'Image mask pair num:{len(self.all_image_mask_path_list)}')
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

        self.all_video_name_list = set()
        self.all_video_set_dict = collections.OrderedDict()
        self.all_video_image_path_dict = collections.OrderedDict()
        self.all_video_mask_dict = collections.OrderedDict()
        self.all_video_mask_type_dict = collections.OrderedDict()

        # 加载所有sav格式数据集
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################
        min_video_frame_num, avg_video_frame_num, max_video_frame_num, video_num = 1000000, 0, 0, 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务，并用索引保持顺序
            future_to_set = {
                executor.submit(self.process_sav_video_set, video_root_dir, per_set_name, video_set_type, per_video_choose_frame_nums): (idx, per_set_name)
                for idx, per_set_name in enumerate(video_set_name)
            }
            
            # 收集结果并保持原始顺序
            set_results = [None] * len(video_set_name)
            for future in tqdm(as_completed(future_to_set), total=len(video_set_name), desc="Processing sav video sets"):
                idx, per_set_name = future_to_set[future]
                result = future.result()
                set_results[idx] = result

        # 按原始顺序合并结果
        for results in set_results:
            for video_info in results:
                per_video_name = video_info['video_name']
                per_video_frame_nums = video_info['frame_num']
                
                self.all_video_name_list.add(per_video_name)
                self.all_video_set_dict[per_video_name] = video_info['set_name']
                self.all_video_image_path_dict[per_video_name] = video_info['image_path_list']
                self.all_video_mask_dict[per_video_name] = video_info['mask_list']
                self.all_video_mask_type_dict[per_video_name] = video_info['mask_type']

                if per_video_frame_nums > max_video_frame_num:
                    max_video_frame_num = per_video_frame_nums
                if per_video_frame_nums < min_video_frame_num:
                    min_video_frame_num = per_video_frame_nums

                avg_video_frame_num += per_video_frame_nums
                video_num += 1

        if video_num > 0:
            avg_video_frame_num = avg_video_frame_num / float(video_num)
        else:
            avg_video_frame_num = 0

        print(f'Add all sav video dataset!')
        print(
            f'min video frame num:{min_video_frame_num}, avg video frame num:{avg_video_frame_num}, max video frame num:{max_video_frame_num}, video num:{video_num}'
        )
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

        assert len(self.all_video_name_list) == len(self.all_video_set_dict) == len(
            self.all_video_image_path_dict) == len(
                self.all_video_mask_dict) == len(self.all_video_mask_type_dict)

        # 加载所有video_matting格式数据集
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################
        min_video_frame_num, avg_video_frame_num, max_video_frame_num, video_num = 1000000, 0, 0, 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务，并用索引保持顺序
            future_to_set = {
                executor.submit(self.process_video_matting_set, video_matting_root_dir, per_set_name, video_matting_set_type, per_video_choose_frame_nums): (idx, per_set_name)
                for idx, per_set_name in enumerate(video_matting_set_name_list)
            }
            
            # 收集结果并保持原始顺序
            set_results = [None] * len(video_matting_set_name_list)
            for future in tqdm(as_completed(future_to_set), total=len(video_matting_set_name_list), desc="Processing video matting sets"):
                idx, per_set_name = future_to_set[future]
                result = future.result()
                set_results[idx] = result

        # 按原始顺序合并结果
        for results in set_results:
            for video_info in results:
                per_video_name = video_info['video_name']
                per_video_frame_nums = video_info['frame_num']
                
                self.all_video_name_list.add(per_video_name)
                self.all_video_set_dict[per_video_name] = video_info['set_name']
                self.all_video_image_path_dict[per_video_name] = video_info['image_path_list']
                self.all_video_mask_dict[per_video_name] = video_info['mask_path_list']
                self.all_video_mask_type_dict[per_video_name] = video_info['mask_type']

                if per_video_frame_nums > max_video_frame_num:
                    max_video_frame_num = per_video_frame_nums
                if per_video_frame_nums < min_video_frame_num:
                    min_video_frame_num = per_video_frame_nums

                avg_video_frame_num += per_video_frame_nums
                video_num += 1

        if video_num > 0:
            avg_video_frame_num = avg_video_frame_num / float(video_num)
        else:
            avg_video_frame_num = 0

        print(f'Add all video matting video dataset!')
        print(
            f'min video frame num:{min_video_frame_num}, avg video frame num:{avg_video_frame_num}, max video frame num:{max_video_frame_num}, video num:{video_num}'
        )
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

        assert len(self.all_video_name_list) == len(self.all_video_set_dict) == len(
            self.all_video_image_path_dict) == len(
                self.all_video_mask_dict) == len(self.all_video_mask_type_dict)

        ##################################################################################################
        ##################################################################################################
        ##################################################################################################
        min_background_video_frame_num, avg_background_video_frame_num, max_background_video_frame_num, background_video_num = 1000000, 0, 0, 0
        self.all_background_video_name_list = set()
        self.all_background_video_image_path_dict = collections.OrderedDict()

        background_video_dir = os.path.join(video_matting_background_dir, video_matting_background_set_type)

        # 先排序背景视频名称列表，确保顺序一致
        background_video_names = sorted(os.listdir(background_video_dir))

        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务，并用索引保持顺序
            future_to_video = {
                executor.submit(self.process_background_video, background_video_dir, per_background_video_name, per_video_choose_frame_nums): (idx, per_background_video_name)
                for idx, per_background_video_name in enumerate(background_video_names)
            }
            
            # 收集结果并保持原始顺序
            video_results = [None] * len(background_video_names)
            for future in tqdm(as_completed(future_to_video), total=len(background_video_names), desc="Processing background video sets"):
                idx, per_background_video_name = future_to_video[future]
                result = future.result()
                video_results[idx] = result

        # 按原始顺序处理结果
        for result in video_results:
            if result is not None:
                per_video_frame_nums = result['frame_num']  # 获取帧数
                
                self.all_background_video_name_list.add(result['video_name'])
                self.all_background_video_image_path_dict[result['video_name']] = result['image_path_list']
                
                # 更新统计信息
                if per_video_frame_nums > max_background_video_frame_num:
                    max_background_video_frame_num = per_video_frame_nums
                if per_video_frame_nums < min_background_video_frame_num:
                    min_background_video_frame_num = per_video_frame_nums
                
                avg_background_video_frame_num += per_video_frame_nums
                background_video_num += 1

        if background_video_num > 0:
            avg_background_video_frame_num = avg_background_video_frame_num / float(background_video_num)
        else:
            avg_background_video_frame_num = 0

        self.all_background_video_name_list = sorted(
            list(self.all_background_video_name_list))

        print(f'Add all background video dataset!')
        print(
            f'min background video frame num:{min_background_video_frame_num}, avg background video frame num:{avg_background_video_frame_num}, max background video frame num:{max_background_video_frame_num}, background video num:{background_video_num}'
        )
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

        self.all_video_name_list = sorted(list(self.all_video_name_list))

        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

        self.video_matting_use_background_video_prob = video_matting_use_background_video_prob
        self.per_video_choose_frame_nums = per_video_choose_frame_nums
        self.per_video_choose_object_nums = per_video_choose_object_nums
        self.max_side = max_side
        self.kernel_size_range = kernel_size_range
        self.points_num = points_num
        self.area_filter_ratio = area_filter_ratio
        self.box_noise_wh_ratio = box_noise_wh_ratio
        self.mask_noise_area_ratio = mask_noise_area_ratio
        self.transform = transform

        print(f'Image Dataset Size:{len(self.all_image_mask_path_list)}')
        print(f'Video Dataset Size:{len(self.all_video_name_list)}')
        print(
            f'Background Video Dataset Size:{len(self.all_background_video_name_list)}'
        )

    def process_sam_image_set(self, root_dir, per_set_name, set_type):
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

    def process_sav_video_set(self, video_root_dir, per_set_name, video_set_type, per_video_choose_frame_nums):
        per_set_dir = os.path.join(video_root_dir, per_set_name, video_set_type)
        
        results = []
        video_names_in_set = sorted(os.listdir(per_set_dir))
        
        for per_video_name in video_names_in_set:
            per_video_path = os.path.join(per_set_dir, per_video_name)
            if os.path.isdir(per_video_path):
                per_video_frame_nums = 0
                per_video_frame_image_path_list = []
                for per_frame_image_name in os.listdir(per_video_path):
                    if '.jpg' in per_frame_image_name:
                        per_frame_image_path = os.path.join(
                            per_video_path, per_frame_image_name)
                        if os.path.exists(per_frame_image_path):
                            per_video_frame_nums += 1
                            per_video_frame_image_path_list.append(
                                per_frame_image_path)
                per_video_frame_image_path_list = sorted(
                    per_video_frame_image_path_list)

                per_video_json_path = os.path.join(
                    per_video_path, f'{per_video_name}_manual.json')
                
                with open(per_video_json_path, encoding='utf-8') as f:
                    per_image_annotation = json.load(f)
                    per_image_annotation = per_image_annotation['masklet']

                if per_video_frame_nums >= per_video_choose_frame_nums and per_video_frame_nums == len(
                        per_image_annotation):
                    results.append({
                        'video_name': per_video_name,
                        'set_name': per_set_name,
                        'image_path_list': per_video_frame_image_path_list,
                        'mask_list': per_image_annotation,
                        'mask_type': 'sam2',
                        'frame_num': per_video_frame_nums
                    })

        return results

    def process_video_matting_set(self, video_matting_root_dir, per_set_name, video_matting_set_type, per_video_choose_frame_nums):
        per_set_dir = os.path.join(video_matting_root_dir, per_set_name, video_matting_set_type)
        
        results = []
        video_names_in_set = sorted(os.listdir(per_set_dir))
        
        for per_video_name in video_names_in_set:
            per_video_path = os.path.join(per_set_dir, per_video_name)
            per_video_image_path = os.path.join(per_video_path, 'image')
            per_video_mask_path = os.path.join(per_video_path, 'mask')

            if os.path.isdir(per_video_image_path) and os.path.isdir(
                    per_video_mask_path):
                per_video_frame_nums = 0
                per_video_frame_image_path_list = []
                per_video_frame_mask_path_list = []
                for per_frame_image_name in sorted(
                        os.listdir(per_video_image_path)):
                    if '.jpg' in per_frame_image_name:
                        per_frame_mask_name = per_frame_image_name.split(
                            ".")[0] + '.png'

                        per_frame_image_path = os.path.join(
                            per_video_image_path, per_frame_image_name)
                        per_frame_mask_path = os.path.join(
                            per_video_mask_path, per_frame_mask_name)
                        if os.path.exists(
                                per_frame_image_path) and os.path.exists(
                                    per_frame_mask_path):
                            per_video_frame_nums += 1
                            per_video_frame_image_path_list.append(
                                per_frame_image_path)
                            per_video_frame_mask_path_list.append(
                                per_frame_mask_path)
                per_video_frame_image_path_list = sorted(
                    per_video_frame_image_path_list)
                per_video_frame_mask_path_list = sorted(
                    per_video_frame_mask_path_list)

                if per_video_frame_nums >= per_video_choose_frame_nums:
                    results.append({
                        'video_name': per_video_name,
                        'set_name': per_set_name,
                        'image_path_list': per_video_frame_image_path_list,
                        'mask_path_list': per_video_frame_mask_path_list,
                        'mask_type': 'video_matting',
                        'frame_num': per_video_frame_nums
                    })
        
        return results

    def process_background_video(self, background_video_dir, per_background_video_name, per_video_choose_frame_nums):
        per_background_video_path = os.path.join(background_video_dir, per_background_video_name)
        
        if os.path.isdir(per_background_video_path):
            per_background_video_frame_nums = 0
            per_background_video_frame_image_path_list = []
            
            for per_frame_image_name in sorted(os.listdir(per_background_video_path)):
                if '.jpg' in per_frame_image_name:
                    per_frame_image_path = os.path.join(per_background_video_path, per_frame_image_name)
                    if os.path.exists(per_frame_image_path):
                        per_background_video_frame_nums += 1
                        per_background_video_frame_image_path_list.append(per_frame_image_path)
            
            per_background_video_frame_image_path_list = sorted(per_background_video_frame_image_path_list)
            
            if per_background_video_frame_nums >= per_video_choose_frame_nums:
                return {
                    'video_name': per_background_video_name,
                    'image_path_list': per_background_video_frame_image_path_list,
                    'frame_num': per_background_video_frame_nums
                }
        
        return None

    def __len__(self):
        return len(self.all_video_name_list)

    def __getitem__(self, idx):
        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################
        # get image data
        image_idx = np.random.choice(len(self.all_image_mask_path_list))
        per_image_name, mask_list_idx, _, _, _, _ = self.all_image_mask_path_list[
            image_idx]

        video_name = f'{per_image_name.split('.')[0]}_{mask_list_idx}'

        image = self.load_image(image_idx)
        mask = self.load_mask(image_idx)

        trimap = self.generate_trimap_from_single_object_mask(mask)
        fg_map, bg_map = self.generate_fg_bg_map_from_single_object_mask(
            image, mask)

        # [1,h,w,3]
        frames_images = np.expand_dims(image, axis=0).astype(np.float32)
        # [1,h,w,1]
        frames_masks = np.expand_dims(np.expand_dims(mask, axis=-1),
                                    axis=0).astype(np.float32)

        # [1,h,w,1]
        frames_trimaps = np.expand_dims(np.expand_dims(trimap, axis=-1),
                                        axis=0).astype(np.float32)
        # [1,h,w,3,1]
        frames_fg_maps = np.expand_dims(np.expand_dims(fg_map, axis=-1),
                                        axis=0).astype(np.float32)
        # [1,h,w,3,1]
        frames_bg_maps = np.expand_dims(np.expand_dims(bg_map, axis=-1),
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

        image_sample = {
            'video_name': video_name,
            'image': frames_images,
            'mask': frames_masks,
            'size': size,
            'prompt_point': frame_prompt_points,
            'prompt_box': frame_prompt_boxes,
            'prompt_mask': frame_prompt_masks,
            'trimap': frames_trimaps,
            'fg_map': frames_fg_maps,
            'bg_map': frames_bg_maps,
        }

        if self.transform:
            image_sample = self.transform(image_sample)

        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################
        # get video_data
        # 获取首帧范围
        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################
        video_idx = idx
        for _ in range(10):
            video_name = self.all_video_name_list[video_idx]
            video_image_path = self.all_video_image_path_dict[video_name]
            video_mask_list = self.all_video_mask_dict[video_name]
            video_type = self.all_video_mask_type_dict[video_name]
            assert video_type in ['sam2', 'video_matting']

            assert len(video_image_path) == len(video_mask_list)

            if video_type == 'sam2':
                # 首先根据取帧长度判断可作为首帧的idx范围
                # 其次判断可作为首帧的frame是否包含至少一个object，只有至少包含一个object才可作为首帧
                end_idx = len(video_image_path
                            ) - self.per_video_choose_frame_nums + 1
                if end_idx > 0:
                    candidate_start_frame_idx_list = list(
                        range(
                            0,
                            len(video_image_path) -
                            self.per_video_choose_frame_nums + 1))
                else:
                    candidate_start_frame_idx_list = [0]

                # start_frame_idx_list = []
                # for per_frame_idx in candidate_start_frame_idx_list:
                #     per_frame_mask = self.load_frame_sam2_mask(
                #         video_mask_list[per_frame_idx])
                #     per_frame_object_id_nums = per_frame_mask.shape[-1]
                #     final_object_id_nums = 0
                #     for per_object_idx in range(per_frame_object_id_nums):
                #         per_obejct_mask_in_start_frame = per_frame_mask[:, :,
                #                                                         per_object_idx]
                #         per_obejct_mask_area = per_obejct_mask_in_start_frame.shape[
                #             0] * per_obejct_mask_in_start_frame.shape[1]
                #         if np.sum(per_obejct_mask_in_start_frame) / float(
                #                 per_obejct_mask_area
                #         ) < self.area_filter_ratio or np.sum(
                #                 per_obejct_mask_in_start_frame) / float(
                #                     per_obejct_mask_area) > 0.9:
                #             continue
                #         final_object_id_nums += 1

                #     if final_object_id_nums > 0:
                #         start_frame_idx_list.append(per_frame_idx)

                # assert len(start_frame_idx_list) >= 1

                # 并行检查每个候选首帧是否包含至少一个满足面积比例要求的对象
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(
                        executor.map(
                            lambda per_frame_idx: per_frame_idx
                            if ((lambda m: sum(
                                1 for i in range(m.shape[-1])
                                if (np.sum(m[:, :, i]) /
                                    (m.shape[0] * m.shape[1]) >= self.
                                    area_filter_ratio and np.sum(m[:, :, i
                                                                ]) /
                                    (m.shape[0] * m.shape[1]) <= 0.9)))
                                (self.load_frame_sam2_mask(video_mask_list[
                                    per_frame_idx])) > 0) else None,
                            candidate_start_frame_idx_list))
                start_frame_idx_list = [
                    r for r in results if r is not None
                ]

                if len(start_frame_idx_list) >= 1:
                    break
                else:
                    video_idx = np.random.choice(
                        range(len(self.all_video_name_list)))

            elif video_type == 'video_matting':
                # 首先根据取帧长度判断可作为首帧的idx范围
                # 其次判断可作为首帧的frame是否包含至少一个object，只有至少包含一个object才可作为首帧
                end_idx = len(video_image_path
                            ) - self.per_video_choose_frame_nums + 1
                if end_idx > 0:
                    candidate_start_frame_idx_list = list(
                        range(
                            0,
                            len(video_image_path) -
                            self.per_video_choose_frame_nums + 1))
                else:
                    candidate_start_frame_idx_list = [0]

                # 随机打乱候选列表顺序
                np.random.shuffle(candidate_start_frame_idx_list)

                # 随机取5帧作为候选起始帧
                start_frame_idx_list = []
                candidate_frame_num = 5
                for per_frame_idx in candidate_start_frame_idx_list:
                    per_frame_mask = self.load_frame_video_matting_mask(
                        video_mask_list[per_frame_idx])
                    per_frame_mask = (per_frame_mask
                                    > 0.5).astype(np.float32)

                    per_frame_object_id_nums = per_frame_mask.shape[-1]
                    final_object_id_nums = 0
                    for per_object_idx in range(per_frame_object_id_nums):
                        per_obejct_mask_in_start_frame = per_frame_mask[:, :,
                                                                        per_object_idx]
                        per_obejct_mask_area = per_obejct_mask_in_start_frame.shape[
                            0] * per_obejct_mask_in_start_frame.shape[1]
                        if np.sum(per_obejct_mask_in_start_frame) / float(
                                per_obejct_mask_area
                        ) < self.area_filter_ratio or np.sum(
                                per_obejct_mask_in_start_frame) / float(
                                    per_obejct_mask_area) > 0.9:
                            continue
                        final_object_id_nums += 1

                    if final_object_id_nums > 0:
                        start_frame_idx_list.append(per_frame_idx)

                    if len(start_frame_idx_list) >= candidate_frame_num:
                        break

                if len(start_frame_idx_list) >= candidate_frame_num:
                    start_frame_idx_list = start_frame_idx_list[
                        0:candidate_frame_num]

                if len(start_frame_idx_list) >= 1:
                    break
                else:
                    video_idx = np.random.choice(
                        range(len(self.all_video_name_list)))

        assert len(start_frame_idx_list) >= 1

        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################

        if video_type == 'sam2':
            # 随机抽取一帧为首帧
            start_frame_idx = np.random.choice(start_frame_idx_list)
            start_frame_mask = self.load_frame_sam2_mask(
                video_mask_list[start_frame_idx])

            # 获取首帧包含的object_ids
            start_frame_object_id_nums = start_frame_mask.shape[-1]
            start_frame_object_idx = []
            for per_object_idx in range(start_frame_object_id_nums):
                per_obejct_mask_in_start_frame = start_frame_mask[:, :,
                                                                per_object_idx]

                per_obejct_mask_area = per_obejct_mask_in_start_frame.shape[
                    0] * per_obejct_mask_in_start_frame.shape[1]
                if np.sum(per_obejct_mask_in_start_frame) / float(
                        per_obejct_mask_area
                ) < self.area_filter_ratio or np.sum(
                        per_obejct_mask_in_start_frame) / float(
                            per_obejct_mask_area) > 0.9:
                    continue

                start_frame_object_idx.append(per_object_idx)

            assert len(start_frame_object_idx) >= 1

            # 选择首帧读取的object_id
            if len(start_frame_object_idx
                ) > self.per_video_choose_object_nums:
                start_frame_object_idx = np.random.choice(
                    start_frame_object_idx,
                    size=self.per_video_choose_object_nums,
                    replace=False)
                start_frame_object_idx = sorted(start_frame_object_idx)

            choose_frames_image_path = video_image_path[
                start_frame_idx:start_frame_idx +
                self.per_video_choose_frame_nums]
            choose_frames_mask = video_mask_list[
                start_frame_idx:start_frame_idx +
                self.per_video_choose_frame_nums]

            assert len(choose_frames_image_path) == len(choose_frames_mask)

            # # [T,h,w,3]
            # frames_images = []
            # for per_frame_image_path in choose_frames_image_path:
            #     per_frame_image = self.load_frame_image(per_frame_image_path)
            #     frames_images.append(per_frame_image)
            # frames_images = np.stack(frames_images, axis=0).astype(np.float32)

            # # [T,h,w,object_nums]
            # frames_masks = []
            # for per_frame_mask in choose_frames_mask:
            #     per_frame_mask = self.load_frame_sam2_mask(per_frame_mask)
            #     per_frame_mask = per_frame_mask[:, :, start_frame_object_idx]
            #     frames_masks.append(per_frame_mask)
            # frames_masks = np.stack(frames_masks, axis=0).astype(np.float32)

            # # [T,h,w,object_nums]
            # frames_trimaps = []
            # for frame_idx in range(frames_masks.shape[0]):
            #     per_frame_mask = frames_masks[frame_idx]
            #     per_frame_trimap = []
            #     for per_object_idx in range(per_frame_mask.shape[-1]):
            #         per_object_mask = per_frame_mask[:, :, per_object_idx]
            #         per_object_trimap = self.generate_trimap_from_single_object_mask(
            #             per_object_mask)
            #         per_frame_trimap.append(per_object_trimap)
            #     per_frame_trimap = np.stack(per_frame_trimap, axis=-1)
            #     frames_trimaps.append(per_frame_trimap)
            # frames_trimaps = np.stack(frames_trimaps, axis=0).astype(np.float32)

            # # [T,h,w,3,object_nums] [T,h,w,3,object_nums]
            # frames_fg_maps, frames_bg_maps = [], []
            # for frame_idx in range(frames_images.shape[0]):
            #     per_frame_image = frames_images[frame_idx]
            #     per_frame_mask = frames_masks[frame_idx]
            #     per_frame_fg_map = []
            #     per_frame_bg_map = []
            #     for per_object_idx in range(per_frame_mask.shape[-1]):
            #         per_object_mask = per_frame_mask[:, :, per_object_idx]
            #         per_object_fg_map, per_object_bg_map = self.generate_fg_bg_map_from_single_object_mask(
            #             per_frame_image, per_object_mask)
            #         per_frame_fg_map.append(per_object_fg_map)
            #         per_frame_bg_map.append(per_object_bg_map)
            #     per_frame_fg_map = np.stack(per_frame_fg_map, axis=-1)
            #     per_frame_bg_map = np.stack(per_frame_bg_map, axis=-1)
            #     frames_fg_maps.append(per_frame_fg_map)
            #     frames_bg_maps.append(per_frame_bg_map)
            # frames_fg_maps = np.stack(frames_fg_maps, axis=0).astype(np.float32)
            # frames_bg_maps = np.stack(frames_bg_maps, axis=0).astype(np.float32)

            # 并行读取帧图像 [T,h,w,3]
            with ThreadPoolExecutor(max_workers=4) as executor:
                frames_images = list(
                    executor.map(
                        lambda p: self.load_frame_image(p).astype(
                            np.float32), choose_frames_image_path))
            frames_images = np.stack(frames_images, axis=0)

            # 并行读取帧掩码 [T, h, w, object_nums]，并仅保留首帧中确定的对象通道
            with ThreadPoolExecutor(max_workers=4) as executor:
                frames_masks = list(
                    executor.map(
                        lambda per_frame_mask_annotation: self.
                        load_frame_sam2_mask(per_frame_mask_annotation)
                        [:, :, start_frame_object_idx],
                        choose_frames_mask))
            frames_masks = np.stack(frames_masks,
                                    axis=0).astype(np.float32)

            # [T,h,w,object_nums]
            frames_trimaps = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for frame_idx in range(frames_masks.shape[0]):
                    per_frame_mask = frames_masks[frame_idx]
                    per_object_masks = [
                        per_frame_mask[:, :, i]
                        for i in range(per_frame_mask.shape[-1])
                    ]
                    per_frame_trimap = list(
                        executor.map(self.generate_trimap_from_single_object_mask,
                                    per_object_masks))
                    per_frame_trimap = np.stack(per_frame_trimap, axis=-1)
                    frames_trimaps.append(per_frame_trimap)
            frames_trimaps = np.stack(frames_trimaps, axis=0).astype(np.float32)

            # [T,h,w,3,object_nums] [T,h,w,3,object_nums]
            frames_fg_maps, frames_bg_maps = [], []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for frame_idx in range(frames_images.shape[0]):
                    per_frame_image = frames_images[frame_idx]
                    per_frame_mask = frames_masks[frame_idx]
                    object_count = per_frame_mask.shape[-1]

                    args = [(per_frame_image, per_frame_mask[:, :, obj_idx])
                            for obj_idx in range(object_count)]

                    results = list(
                        executor.map(
                            lambda x: self.
                            generate_fg_bg_map_from_single_object_mask(x[0], x[1]),
                            args))

                    per_frame_fg_map = np.stack([res[0] for res in results],
                                                axis=-1)
                    per_frame_bg_map = np.stack([res[1] for res in results],
                                                axis=-1)
                    frames_fg_maps.append(per_frame_fg_map)
                    frames_bg_maps.append(per_frame_bg_map)
            frames_fg_maps = np.stack(frames_fg_maps, axis=0).astype(np.float32)
            frames_bg_maps = np.stack(frames_bg_maps, axis=0).astype(np.float32)

            size = np.array(
                [frames_images.shape[1],
                frames_images.shape[2]]).astype(np.float32)

            assert frames_images.shape[0] == frames_masks.shape[
                0] and frames_images.shape[1] == frames_masks.shape[
                    1] and frames_images.shape[2] == frames_masks.shape[2]

            # # [T,object_nums,point_nums,3]
            # frame_prompt_points = []
            # for per_frame_idx in range(frames_masks.shape[0]):
            #     # [h,w,object_nums]
            #     per_frame_mask = frames_masks[per_frame_idx]
            #     per_frame_prompt_point = self.load_frame_points((per_frame_mask > 0.5).astype(np.float32))
            #     frame_prompt_points.append(per_frame_prompt_point)
            # frame_prompt_points = np.stack(frame_prompt_points,
            #                                axis=0).astype(np.float32)

            # # [T,object_nums,4]
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

            # # [T,h,w,object_nums]
            # frame_prompt_masks = []
            # for per_frame_idx in range(frames_masks.shape[0]):
            #     # [h,w,object_nums]
            #     per_frame_prompt_mask = frames_masks[per_frame_idx]
            #     per_frame_prompt_mask = self.noise_frame_mask(
            #         (per_frame_prompt_mask > 0.2).astype(np.float32))
            #     frame_prompt_masks.append(per_frame_prompt_mask)
            # frame_prompt_masks = np.stack(frame_prompt_masks,
            #                               axis=0).astype(np.float32)

            # [T,object_nums,point_nums,3]
            with ThreadPoolExecutor(max_workers=4) as executor:
                frame_prompt_points_list = list(
                    executor.map(self.load_frame_points, (frames_masks > 0.5).astype(np.float32)))
            frame_prompt_points = np.stack(frame_prompt_points_list,
                                        axis=0).astype(np.float32)

            # [T,object_nums,4]
            with ThreadPoolExecutor(max_workers=4) as executor:
                frame_prompt_boxes_list = list(
                    executor.map(
                        lambda m: (lambda box: self.noise_frame_box(
                            box, [m.shape[0], m.shape[1]])
                                if self.box_noise_wh_ratio > 0 else box)
                        (self.load_frame_box((m > 0.5).astype(np.float32))), frames_masks))
            frame_prompt_boxes = np.stack(frame_prompt_boxes_list,
                                        axis=0).astype(np.float32)

            # [T,h,w,object_nums]
            with ThreadPoolExecutor(max_workers=4) as executor:
                frame_prompt_masks_list = list(
                    executor.map(self.noise_frame_mask, (frames_masks > 0.2).astype(np.float32)))
            frame_prompt_masks = np.stack(frame_prompt_masks_list,
                                        axis=0).astype(np.float32)

            video_sample = {
                'video_name': video_name,
                'image': frames_images,
                'mask': frames_masks,
                'size': size,
                'prompt_point': frame_prompt_points,
                'prompt_box': frame_prompt_boxes,
                'prompt_mask': frame_prompt_masks,
                'trimap': frames_trimaps,
                'fg_map': frames_fg_maps,
                'bg_map': frames_bg_maps,
            }

        elif video_type == 'video_matting':
            # 随机抽取一帧为首帧
            start_frame_idx = np.random.choice(start_frame_idx_list)
            start_frame_mask = self.load_frame_video_matting_mask(
                video_mask_list[start_frame_idx])
            start_frame_mask = (start_frame_mask > 0.5).astype(np.float32)

            # 获取首帧包含的object_ids
            start_frame_object_id_nums = start_frame_mask.shape[-1]
            start_frame_object_idx = []
            for per_object_idx in range(start_frame_object_id_nums):
                per_obejct_mask_in_start_frame = start_frame_mask[:, :,
                                                                per_object_idx]

                per_obejct_mask_area = per_obejct_mask_in_start_frame.shape[
                    0] * per_obejct_mask_in_start_frame.shape[1]
                if np.sum(per_obejct_mask_in_start_frame) / float(
                        per_obejct_mask_area
                ) < self.area_filter_ratio or np.sum(
                        per_obejct_mask_in_start_frame) / float(
                            per_obejct_mask_area) > 0.9:
                    continue

                start_frame_object_idx.append(per_object_idx)

            assert len(start_frame_object_idx) >= 1

            choose_frames_image_path = video_image_path[
                start_frame_idx:start_frame_idx +
                self.per_video_choose_frame_nums]
            choose_frames_mask_path = video_mask_list[
                start_frame_idx:start_frame_idx +
                self.per_video_choose_frame_nums]

            assert len(choose_frames_image_path) == len(
                choose_frames_mask_path)

            video_set_name = self.all_video_set_dict[video_name]
            video_matting_use_background_video_prob = self.video_matting_use_background_video_prob[
                video_set_name]
            if np.random.uniform(
                    0, 1) < video_matting_use_background_video_prob:
                background_video_name = np.random.choice(
                    self.all_background_video_name_list)
                background_video_image_path = self.all_background_video_image_path_dict[
                    background_video_name]

                # 根据取帧长度判断可作为首帧的backbround frame idx范围
                background_end_idx = len(
                    background_video_image_path
                ) - self.per_video_choose_frame_nums + 1
                if background_end_idx > 0:
                    background_start_frame_idx_list = list(
                        range(
                            0,
                            len(background_video_image_path) -
                            self.per_video_choose_frame_nums + 1))
                else:
                    background_start_frame_idx_list = [0]
                assert len(background_start_frame_idx_list) >= 1

                # 随机抽取一帧为首帧
                start_background_frame_idx = np.random.choice(
                    background_start_frame_idx_list)

                choose_background_frames_image_path = background_video_image_path[
                    start_background_frame_idx:start_background_frame_idx +
                    self.per_video_choose_frame_nums]

                # # [T,h,w,3]
                # frames_background_images = []
                # for per_background_frame_image_path in choose_background_frames_image_path:
                #     per_background_frame_image = self.load_frame_image(
                #         per_background_frame_image_path)
                #     frames_background_images.append(per_background_frame_image)
                # frames_background_images = np.stack(frames_background_images,
                #                                     axis=0).astype(np.float32)

                # # [T,h,w,3]
                # frames_images = []
                # for frame_idx in range(len(choose_frames_image_path)):
                #     per_frame_image = self.load_frame_image(
                #         choose_frames_image_path[frame_idx])
                #     per_frame_mask = self.load_frame_mask(
                #         choose_frames_mask_path[frame_idx])
                #     per_background_frame_image = frames_background_images[
                #         frame_idx]
                #     frame_image_h, frame_image_w = per_frame_image.shape[
                #         0], per_frame_image.shape[1]
                #     frame_mask_h, frame_mask_w = per_frame_mask.shape[
                #         0], per_frame_mask.shape[1]
                #     assert frame_image_h == frame_mask_h and frame_image_w == frame_mask_w

                #     per_background_frame_image = cv2.resize(
                #         per_background_frame_image, (frame_image_w, frame_image_h))
                #     background_frame_image_h, background_frame_image_w = per_background_frame_image.shape[
                #         0], per_background_frame_image.shape[1]

                #     assert frame_image_h == background_frame_image_h and frame_image_w == background_frame_image_w

                #     per_frame_image = per_frame_image * per_frame_mask + per_background_frame_image * (
                #         1 - per_frame_mask)

                #     frames_images.append(per_frame_image)
                # frames_images = np.stack(frames_images, axis=0).astype(np.float32)

                # [T,h,w,3]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    frames_background_images = list(
                        executor.map(
                            lambda p: self.load_frame_image(p).astype(
                                np.float32),
                            choose_background_frames_image_path))
                frames_background_images = np.stack(
                    frames_background_images, axis=0)

                # [T,h,w,3]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    frame_args = zip(choose_frames_image_path,
                                    choose_frames_mask_path,
                                    frames_background_images)
                    frames_images = list(
                        executor.map(
                            lambda args: self.process_single_frame_image(
                                *args), frame_args))
                frames_images = np.stack(frames_images, axis=0)
            else:
                # # [T,h,w,3]
                # frames_images = []
                # for per_frame_image_path in choose_frames_image_path:
                #     per_frame_image = self.load_frame_image(per_frame_image_path)
                #     frames_images.append(per_frame_image)
                # frames_images = np.stack(frames_images, axis=0).astype(np.float32)

                # 并行读取帧图像 [T,h,w,3]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    frames_images = list(
                        executor.map(
                            lambda p: self.load_frame_image(p).astype(
                                np.float32), choose_frames_image_path))
                frames_images = np.stack(frames_images, axis=0)

            # # [T,h,w,1],object_nums=1
            # frames_masks = []
            # for per_frame_mask_path in choose_frames_mask_path:
            #     per_frame_mask = self.load_frame_video_matting_mask(per_frame_mask_path)
            #     per_frame_mask = per_frame_mask[:, :, start_frame_object_idx]
            #     frames_masks.append(per_frame_mask)
            # frames_masks = np.stack(frames_masks, axis=0).astype(np.float32)

            # # [T,h,w,1],object_nums=1
            # frames_trimaps = []
            # for frame_idx in range(frames_masks.shape[0]):
            #     per_frame_mask = frames_masks[frame_idx]
            #     per_frame_trimap = []
            #     for per_object_idx in range(per_frame_mask.shape[-1]):
            #         per_object_mask = per_frame_mask[:, :, per_object_idx]
            #         per_object_trimap = self.generate_trimap_from_single_object_mask(
            #             per_object_mask)
            #         per_frame_trimap.append(per_object_trimap)
            #     per_frame_trimap = np.stack(per_frame_trimap, axis=-1)
            #     frames_trimaps.append(per_frame_trimap)
            # frames_trimaps = np.stack(frames_trimaps, axis=0).astype(np.float32)

            # # [T,h,w,3,object_nums] [T,h,w,3,object_nums]
            # frames_fg_maps, frames_bg_maps = [], []
            # for frame_idx in range(frames_images.shape[0]):
            #     per_frame_image = frames_images[frame_idx]
            #     per_frame_mask = frames_masks[frame_idx]
            #     per_frame_fg_map = []
            #     per_frame_bg_map = []
            #     for per_object_idx in range(per_frame_mask.shape[-1]):
            #         per_object_mask = per_frame_mask[:, :, per_object_idx]
            #         per_object_fg_map, per_object_bg_map = self.generate_fg_bg_map_from_single_object_mask(
            #             per_frame_image, per_object_mask)
            #         per_frame_fg_map.append(per_object_fg_map)
            #         per_frame_bg_map.append(per_object_bg_map)
            #     per_frame_fg_map = np.stack(per_frame_fg_map, axis=-1)
            #     per_frame_bg_map = np.stack(per_frame_bg_map, axis=-1)
            #     frames_fg_maps.append(per_frame_fg_map)
            #     frames_bg_maps.append(per_frame_bg_map)
            # frames_fg_maps = np.stack(frames_fg_maps, axis=0).astype(np.float32)
            # frames_bg_maps = np.stack(frames_bg_maps, axis=0).astype(np.float32)

            # 并行读取帧掩码 [T, h, w, object_nums]，并仅保留首帧中确定的对象通道
            with ThreadPoolExecutor(max_workers=4) as executor:
                frames_masks = list(
                    executor.map(
                        lambda per_frame_mask_annotation: self.
                        load_frame_video_matting_mask(
                            per_frame_mask_annotation
                        )[:, :, start_frame_object_idx],
                        choose_frames_mask_path))
            frames_masks = np.stack(frames_masks,
                                    axis=0).astype(np.float32)

            # [T,h,w,object_nums]
            frames_trimaps = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for frame_idx in range(frames_masks.shape[0]):
                    per_frame_mask = frames_masks[frame_idx]
                    per_object_masks = [
                        per_frame_mask[:, :, i]
                        for i in range(per_frame_mask.shape[-1])
                    ]
                    per_frame_trimap = list(
                        executor.map(self.generate_trimap_from_single_object_mask,
                                    per_object_masks))
                    per_frame_trimap = np.stack(per_frame_trimap, axis=-1)
                    frames_trimaps.append(per_frame_trimap)
            frames_trimaps = np.stack(frames_trimaps, axis=0).astype(np.float32)

            # [T,h,w,3,object_nums] [T,h,w,3,object_nums]
            frames_fg_maps, frames_bg_maps = [], []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for frame_idx in range(frames_images.shape[0]):
                    per_frame_image = frames_images[frame_idx]
                    per_frame_mask = frames_masks[frame_idx]
                    object_count = per_frame_mask.shape[-1]

                    args = [(per_frame_image, per_frame_mask[:, :, obj_idx])
                            for obj_idx in range(object_count)]

                    results = list(
                        executor.map(
                            lambda x: self.
                            generate_fg_bg_map_from_single_object_mask(x[0], x[1]),
                            args))

                    per_frame_fg_map = np.stack([res[0] for res in results],
                                                axis=-1)
                    per_frame_bg_map = np.stack([res[1] for res in results],
                                                axis=-1)
                    frames_fg_maps.append(per_frame_fg_map)
                    frames_bg_maps.append(per_frame_bg_map)
            frames_fg_maps = np.stack(frames_fg_maps, axis=0).astype(np.float32)
            frames_bg_maps = np.stack(frames_bg_maps, axis=0).astype(np.float32)

            size = np.array(
                [frames_images.shape[1],
                frames_images.shape[2]]).astype(np.float32)

            assert frames_images.shape[0] == frames_masks.shape[
                0] and frames_images.shape[1] == frames_masks.shape[
                    1] and frames_images.shape[2] == frames_masks.shape[2]

            # # [T,object_nums,point_nums,3]
            # frame_prompt_points = []
            # for per_frame_idx in range(frames_masks.shape[0]):
            #     # [h,w,object_nums]
            #     per_frame_mask = frames_masks[per_frame_idx]
            #     per_frame_prompt_point = self.load_frame_points((per_frame_mask > 0.5).astype(np.float32))
            #     frame_prompt_points.append(per_frame_prompt_point)
            # frame_prompt_points = np.stack(frame_prompt_points,
            #                                axis=0).astype(np.float32)

            # # [T,object_nums,4]
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

            # # [T,h,w,object_nums]
            # frame_prompt_masks = []
            # for per_frame_idx in range(frames_masks.shape[0]):
            #     # [h,w,object_nums]
            #     per_frame_prompt_mask = frames_masks[per_frame_idx]
            #     per_frame_prompt_mask = self.noise_frame_mask(
            #         (per_frame_prompt_mask > 0.2).astype(np.float32))
            #     frame_prompt_masks.append(per_frame_prompt_mask)
            # frame_prompt_masks = np.stack(frame_prompt_masks,
            #                               axis=0).astype(np.float32)

            # [T,object_nums,point_nums,3]
            with ThreadPoolExecutor(max_workers=4) as executor:
                frame_prompt_points_list = list(
                    executor.map(self.load_frame_points, (frames_masks > 0.5).astype(np.float32)))
            frame_prompt_points = np.stack(frame_prompt_points_list,
                                        axis=0).astype(np.float32)

            # [T,object_nums,4]
            with ThreadPoolExecutor(max_workers=4) as executor:
                frame_prompt_boxes_list = list(
                    executor.map(
                        lambda m: (lambda box: self.noise_frame_box(
                            box, [m.shape[0], m.shape[1]])
                                if self.box_noise_wh_ratio > 0 else box)
                        (self.load_frame_box((m > 0.5).astype(np.float32))), frames_masks))
            frame_prompt_boxes = np.stack(frame_prompt_boxes_list,
                                        axis=0).astype(np.float32)

            # [T,h,w,object_nums]
            with ThreadPoolExecutor(max_workers=4) as executor:
                frame_prompt_masks_list = list(
                    executor.map(self.noise_frame_mask, (frames_masks > 0.2).astype(np.float32)))
            frame_prompt_masks = np.stack(frame_prompt_masks_list,
                                        axis=0).astype(np.float32)

            video_sample = {
                'video_name': video_name,
                'image': frames_images,
                'mask': frames_masks,
                'size': size,
                'prompt_point': frame_prompt_points,
                'prompt_box': frame_prompt_boxes,
                'prompt_mask': frame_prompt_masks,
                'trimap': frames_trimaps,
                'fg_map': frames_fg_maps,
                'bg_map': frames_bg_maps,
            }

        if self.transform:
            video_sample = self.transform(video_sample)

        sample = {
                'image_sample':image_sample,
                'video_sample':video_sample,
            }

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

    def load_frame_image(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def process_single_frame_image(self, image_path, mask_path,
                                background_image):
        # 加载前景图像和掩码
        per_frame_image = self.load_frame_image(image_path)
        per_frame_mask = self.load_frame_video_matting_mask(mask_path)

        # 检查尺寸一致性
        frame_h, frame_w = per_frame_image.shape[0], per_frame_image.shape[1]
        mask_h, mask_w = per_frame_mask.shape[0], per_frame_mask.shape[1]
        assert frame_h == mask_h and frame_w == mask_w, "Image and mask dimension mismatch"

        # 调整背景尺寸并与前景对齐
        per_background_frame = cv2.resize(background_image, (frame_w, frame_h))
        bg_h, bg_w = per_background_frame.shape[0], per_background_frame.shape[
            1]
        assert frame_h == bg_h and frame_w == bg_w, "Background resize dimension mismatch"

        # 融合前景和背景
        blended_frame = per_frame_image * per_frame_mask + per_background_frame * (
            1 - per_frame_mask)

        return blended_frame.astype(np.float32)

    def load_frame_sam2_mask(self, per_frame_mask_list):
        object_nums = len(per_frame_mask_list)
        mask = []
        for per_object_idx in range(object_nums):
            per_object_frame_mask = per_frame_mask_list[per_object_idx]
            per_object_frame_mask = mask_utils.decode(per_object_frame_mask)
            per_object_frame_mask[per_object_frame_mask > 0] = 1
            mask.append(per_object_frame_mask)
        mask = np.stack(mask, axis=-1)

        return mask.astype(np.float32)

    def load_frame_video_matting_mask(self, mask_path):
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)

        # 0.9*255
        mask[mask >= 230] = 255
        # 0.1*255
        mask[mask <= 25] = 0
        mask = mask / 255.
        mask = np.expand_dims(mask, axis=-1)

        return mask.astype(np.float32)

    def generate_trimap_from_single_object_mask(self, alpha):
        alpha_h, alpha_w = alpha.shape[0], alpha.shape[1]

        if np.count_nonzero(alpha) == 0:
            single_object_trimap = np.zeros((alpha_h, alpha_w),
                                            dtype=np.uint8)

            return single_object_trimap.astype(np.uint8)

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
        single_object_trimap = erode * 255 + (dilate - erode) * 128

        return single_object_trimap.astype(np.uint8)

    def generate_fg_bg_map_from_single_object_mask(self, image, alpha):
        expand_dim_mask = np.expand_dims(alpha.copy(),
                                        axis=2).astype(np.float32)
        single_object_fg_map = image * expand_dim_mask
        single_object_bg_map = image * (1. - expand_dim_mask)

        return single_object_fg_map.astype(
            np.float32), single_object_bg_map.astype(np.float32)

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

    from tools.path import interactive_segmentation_dataset_path, video_interactive_segmentation_dataset_path, background_video_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.video_interactive_segmentation.common_matting import Sam2MattingResize, Sam2MattingRandomHorizontalFlip, Sam2MattingRandomMosaicAug, Sam2MattingRandomRsverseFrameOrder, Sam2MattingNormalize, SAM2MattingVideoBatchCollater

    sam2_video_dataset = SAM2VideoMattingDataset(
        image_root_dir=interactive_segmentation_dataset_path,
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
        video_root_dir=video_interactive_segmentation_dataset_path,
        video_set_name=[
            # 'MOSEv1',
            'MOSEv2',
            'DAVIS2017',
            'YouTubeVOS2019',
            ###########################################
            'sav_000',
            # 'sav_000_filter_part_object',
        ],
        video_set_type='train',
        video_matting_root_dir=video_interactive_segmentation_dataset_path,
        video_matting_set_name_list=[
            'V-HIM2K5',
            'V-HIM60_comp_easy',
            'V-HIM60_comp_medium',
            'V-HIM60_comp_hard',
            'VideoMatte240K',
        ],
        video_matting_use_background_video_prob={
            'V-HIM2K5': 0.0,
            'V-HIM60_comp_easy': 0.0,
            'V-HIM60_comp_medium': 0.0,
            'V-HIM60_comp_hard': 0.0,
            'VideoMatte240K': 1.0,
        },
        video_matting_set_type='train',
        video_matting_background_dir=background_video_dataset_path,
        video_matting_background_set_type='train',
        per_video_choose_frame_nums=8,
        per_video_choose_object_nums=2,
        max_side=2048,
        kernel_size_range=[15, 15],
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            Sam2MattingResize(resize=1024),
            Sam2MattingRandomHorizontalFlip(prob=1.0),
            Sam2MattingRandomMosaicAug(prob=0.1),
            Sam2MattingRandomRsverseFrameOrder(prob=1.0),
            # Sam2MattingNormalize(mean=[123.675, 116.28, 103.53],
            #                 std=[58.395, 57.12, 57.375]),
        ]))

    video_count = 0
    for per_sample in tqdm(sam2_video_dataset):
        image_sample = per_sample['image_sample']
        video_sample = per_sample['video_sample']

        # 可视化image_sample
        ###############################################################################
        ###############################################################################
        ###############################################################################
        print('1111', image_sample['video_name'])
        print('1111', image_sample['image'].shape, image_sample['mask'].shape,
              image_sample['size'], image_sample['prompt_point'].shape,
              image_sample['prompt_box'].shape, image_sample['prompt_mask'].shape,
              image_sample['trimap'].shape, image_sample['fg_map'].shape,
              image_sample['bg_map'].shape)
        print('1111', image_sample['image'].dtype, image_sample['mask'].dtype,
              image_sample['size'].dtype, image_sample['prompt_point'].dtype,
              image_sample['prompt_box'].dtype, image_sample['prompt_mask'].dtype,
              image_sample['trimap'].dtype, image_sample['fg_map'].dtype,
              image_sample['bg_map'].dtype)
        print('1111', np.max(image_sample['mask']), np.min(image_sample['mask']))
        print('1111', np.max(image_sample['trimap']),
              np.min(image_sample['trimap']), np.unique(image_sample['trimap']))
        print('1111', np.max(image_sample['fg_map']),
              np.min(image_sample['fg_map']))
        print('1111', np.max(image_sample['bg_map']),
              np.min(image_sample['bg_map']))
        print('1111', image_sample['mask'].shape[0])
        print('1111', image_sample['mask'].shape[-1])

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # per_video_dir = os.path.join(temp_dir, f'video_{video_count}')
        # if not os.path.exists(per_video_dir):
        #     os.makedirs(per_video_dir)

        # video_name = image_sample['video_name']
        # image_sample_images = image_sample['image']
        # image_sample_masks = image_sample['mask']
        # image_sample_trimaps = image_sample['trimap']
        # image_sample_fg_maps = image_sample['fg_map']
        # image_sample_bg_maps = image_sample['bg_map']
        # image_sample_prompt_points = image_sample['prompt_point']
        # image_sample_prompt_boxes = image_sample['prompt_box']
        # image_sample_prompt_masks = image_sample['prompt_mask']

        # object_nums = image_sample_masks.shape[-1]
        # frame_nums = image_sample_masks.shape[0]

        # for per_object_idx in tqdm(range(object_nums)):
        #     per_object_masks = image_sample_masks[:, :, :, per_object_idx]
        #     per_object_trimaps = image_sample_trimaps[:, :, :, per_object_idx]
        #     per_object_fg_maps = image_sample_fg_maps[:, :, :, :, per_object_idx]
        #     per_object_bg_maps = image_sample_bg_maps[:, :, :, :, per_object_idx]
        #     per_object_prompt_points = image_sample_prompt_points[:,
        #                                                         per_object_idx, :, :]
        #     per_object_prompt_boxes = image_sample_prompt_boxes[:,
        #                                                       per_object_idx, :]
        #     per_object_prompt_masks = image_sample_prompt_masks[:, :, :,
        #                                                       per_object_idx]

        #     per_video_object_dir = os.path.join(per_video_dir, 'image_sample',
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

        #         per_object_frame_image = image_sample_images[per_frame_idx]
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

        #         per_object_frame_trimap = per_object_trimaps[per_frame_idx]
        #         cv2.imencode('.png', per_object_frame_trimap)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_trimap.jpg'
        #             ))

        #         per_object_frame_fg_map = per_object_fg_maps[per_frame_idx]
        #         per_object_frame_fg_map = np.ascontiguousarray(
        #             per_object_frame_fg_map, dtype=np.uint8)
        #         per_object_frame_fg_map = cv2.cvtColor(per_object_frame_fg_map,
        #                                                cv2.COLOR_RGB2BGR)
        #         cv2.imencode('.jpg', per_object_frame_fg_map)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_fg_map.jpg'
        #             ))

        #         per_object_frame_bg_map = per_object_bg_maps[per_frame_idx]
        #         per_object_frame_bg_map = np.ascontiguousarray(
        #             per_object_frame_bg_map, dtype=np.uint8)
        #         per_object_frame_bg_map = cv2.cvtColor(per_object_frame_bg_map,
        #                                                cv2.COLOR_RGB2BGR)
        #         cv2.imencode('.jpg', per_object_frame_bg_map)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_bg_map.jpg'
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

        # 可视化video_sample
        ###############################################################################
        ###############################################################################
        ###############################################################################
        print('2222', video_sample['video_name'])
        print('2222', video_sample['image'].shape, video_sample['mask'].shape,
              video_sample['size'], video_sample['prompt_point'].shape,
              video_sample['prompt_box'].shape, video_sample['prompt_mask'].shape,
              video_sample['trimap'].shape, video_sample['fg_map'].shape,
              video_sample['bg_map'].shape)
        print('2222', video_sample['image'].dtype, video_sample['mask'].dtype,
              video_sample['size'].dtype, video_sample['prompt_point'].dtype,
              video_sample['prompt_box'].dtype, video_sample['prompt_mask'].dtype,
              video_sample['trimap'].dtype, video_sample['fg_map'].dtype,
              video_sample['bg_map'].dtype)
        print('2222', np.max(video_sample['mask']), np.min(video_sample['mask']))
        print('2222', np.max(video_sample['trimap']),
              np.min(video_sample['trimap']), np.unique(video_sample['trimap']))
        print('2222', np.max(video_sample['fg_map']),
              np.min(video_sample['fg_map']))
        print('2222', np.max(video_sample['bg_map']),
              np.min(video_sample['bg_map']))
        print('2222', video_sample['mask'].shape[0])
        print('2222', video_sample['mask'].shape[-1])

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # per_video_dir = os.path.join(temp_dir, f'video_{video_count}')
        # if not os.path.exists(per_video_dir):
        #     os.makedirs(per_video_dir)

        # video_name = video_sample['video_name']
        # video_sample_images = video_sample['image']
        # video_sample_masks = video_sample['mask']
        # video_sample_trimaps = video_sample['trimap']
        # video_sample_fg_maps = video_sample['fg_map']
        # video_sample_bg_maps = video_sample['bg_map']
        # video_sample_prompt_points = video_sample['prompt_point']
        # video_sample_prompt_boxes = video_sample['prompt_box']
        # video_sample_prompt_masks = video_sample['prompt_mask']

        # object_nums = video_sample_masks.shape[-1]
        # frame_nums = video_sample_masks.shape[0]

        # for per_object_idx in tqdm(range(object_nums)):
        #     per_object_masks = video_sample_masks[:, :, :, per_object_idx]
        #     per_object_trimaps = video_sample_trimaps[:, :, :, per_object_idx]
        #     per_object_fg_maps = video_sample_fg_maps[:, :, :, :, per_object_idx]
        #     per_object_bg_maps = video_sample_bg_maps[:, :, :, :, per_object_idx]
        #     per_object_prompt_points = video_sample_prompt_points[:,
        #                                                         per_object_idx, :, :]
        #     per_object_prompt_boxes = video_sample_prompt_boxes[:,
        #                                                       per_object_idx, :]
        #     per_object_prompt_masks = video_sample_prompt_masks[:, :, :,
        #                                                       per_object_idx]

        #     per_video_object_dir = os.path.join(per_video_dir, 'video_sample',
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

        #         per_object_frame_image = video_sample_images[per_frame_idx]
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

        #         per_object_frame_trimap = per_object_trimaps[per_frame_idx]
        #         cv2.imencode('.png', per_object_frame_trimap)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_trimap.jpg'
        #             ))

        #         per_object_frame_fg_map = per_object_fg_maps[per_frame_idx]
        #         per_object_frame_fg_map = np.ascontiguousarray(
        #             per_object_frame_fg_map, dtype=np.uint8)
        #         per_object_frame_fg_map = cv2.cvtColor(per_object_frame_fg_map,
        #                                                cv2.COLOR_RGB2BGR)
        #         cv2.imencode('.jpg', per_object_frame_fg_map)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_fg_map.jpg'
        #             ))

        #         per_object_frame_bg_map = per_object_bg_maps[per_frame_idx]
        #         per_object_frame_bg_map = np.ascontiguousarray(
        #             per_object_frame_bg_map, dtype=np.uint8)
        #         per_object_frame_bg_map = cv2.cvtColor(per_object_frame_bg_map,
        #                                                cv2.COLOR_RGB2BGR)
        #         cv2.imencode('.jpg', per_object_frame_bg_map)[1].tofile(
        #             os.path.join(
        #                 per_video_object_dir,
        #                 f'{video_name}_{per_object_idx}_{per_frame_idx}_bg_map.jpg'
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

    collater = SAM2MattingVideoBatchCollater(resize=1024,use_image_prob=0.1)
    train_loader = DataLoader(sam2_video_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        batch_images, input_masks, input_images = data['batch_image'], data[
            'mask'], data['input_image']

        input_trimaps, input_fg_maps, input_bg_maps = data['trimap'], data[
            'fg_map'], data['bg_map']

        input_prompt_points, input_prompt_boxes, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        object_to_frame_idxs = data['object_to_frame_idx']

        print('3333', batch_images.shape, input_masks.shape,
              input_images.shape, input_trimaps.shape, input_fg_maps.shape,
              input_bg_maps.shape, input_prompt_points.shape,
              input_prompt_boxes.shape, input_prompt_masks.shape,
              object_to_frame_idxs.shape)
        print('3333', batch_images.dtype, input_masks.dtype,
              input_images.dtype, input_trimaps.dtype, input_fg_maps.dtype,
              input_bg_maps.dtype, input_prompt_points.dtype,
              input_prompt_boxes.dtype, input_prompt_masks.dtype,
              object_to_frame_idxs.dtype)

        if count < 10:
            count += 1
        else:
            break