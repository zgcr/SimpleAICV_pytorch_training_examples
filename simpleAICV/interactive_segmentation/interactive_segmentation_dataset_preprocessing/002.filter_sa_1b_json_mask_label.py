import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import cv2
import copy
import json
import math
import numpy as np
import random
from tqdm import tqdm
import torch
import pycocotools.mask as mask_utils
from concurrent.futures import ThreadPoolExecutor, as_completed


def calculate_recall_iou(mask1, mask2):
    insection = np.logical_and(mask1, mask2).astype(np.uint8)
    insection_area = np.count_nonzero(insection)
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)

    if mask1_area == 0 or mask2_area == 0:
        return 0, 0, 0

    insection_mask1_recall = insection_area / mask1_area
    insection_mask2_recall = insection_area / mask2_area

    union = np.logical_or(mask1, mask2).astype(np.uint8)
    union_area = np.count_nonzero(union)
    mask1_mask2_iou = 0.0 if union_area == 0 else insection_area / union_area

    return insection_mask1_recall, insection_mask2_recall, mask1_mask2_iou


def process_single_image(per_image_name, per_image_path, per_label_name,
                         per_label_path, save_image_path):
    """
    将原先 preprocess_image 中处理单张图像的逻辑抽取到这里。
    方便多线程/多进程并行调用。
    """
    # 读取图像
    per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
    per_image = per_image.astype(np.uint8)

    # 保存图像到目标目录
    save_per_image_path = os.path.join(save_image_path, per_image_name)
    cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    # 读取 json
    with open(per_label_path, encoding='utf-8') as f:
        per_image_json_data = json.load(f)
    per_image_annotation = per_image_json_data['annotations']
    per_image_h = per_image_json_data['image']['height']
    per_image_w = per_image_json_data['image']['width']

    # 1. 根据一些规则先做初筛
    keep_image_annotation = []
    for per_annot in per_image_annotation:
        per_box = per_annot['bbox']
        x_min = math.ceil(max(per_box[0], 0))
        y_min = math.ceil(max(per_box[1], 0))
        x_max = math.ceil(min(per_box[0] + per_box[2], per_image_w))
        y_max = math.ceil(min(per_box[1] + per_box[3], per_image_h))
        box_w = math.ceil(x_max - x_min)
        box_h = math.ceil(y_max - y_min)

        if box_w / per_image_w < 0.01 and box_h / per_image_h < 0.01:
            continue

        if (box_w * box_h) / float(per_image_h * per_image_w) < 0.00005:
            continue

        if per_annot['area'] / float(per_image_h * per_image_w) < 0.0001 or \
           per_annot['area'] / float(per_image_h * per_image_w) > 0.9:
            continue

        keep_image_annotation.append(per_annot)

    keep_per_image_json_data = copy.deepcopy(per_image_json_data)
    keep_per_image_json_data['annotations'] = keep_image_annotation

    # 2. 将对应的 mask 解码出来，做重复/包含过滤
    keep_image_masks = []
    for per_annot in keep_per_image_json_data['annotations']:
        per_mask = mask_utils.decode(per_annot['segmentation'])
        keep_image_masks.append([per_annot, per_mask])

    if len(keep_image_masks) <= 1:
        total_masks = keep_image_masks
    else:
        keep_flag = [True] * len(keep_image_masks)
        for i in range(len(keep_image_masks)):
            if not keep_flag[i]:
                continue
            for j in range(i + 1, len(keep_image_masks)):
                if not keep_flag[j]:
                    continue
                r1, r2, iou = calculate_recall_iou(keep_image_masks[i][1],
                                                   keep_image_masks[j][1])
                if iou > 0.9:
                    # 重复，保留 i, 去除 j
                    keep_flag[j] = False
                else:
                    # 判断包含
                    if r1 > 0.9 and r2 < 0.5:
                        # mask_i 被 mask_j 包含 => 去除 i
                        keep_flag[i] = False
                        break
                    elif r2 > 0.9 and r1 < 0.5:
                        # mask_j 被 mask_i 包含 => 去除 j
                        keep_flag[j] = False
        total_masks = [
            keep_image_masks[i] for i in range(len(keep_image_masks))
            if keep_flag[i]
        ]

    total_json_masks = [m[0] for m in total_masks]
    total_per_image_json_data = copy.deepcopy(per_image_json_data)
    total_per_image_json_data['annotations'] = total_json_masks

    # 3. 把过滤后的 JSON 写回
    save_per_json_path = os.path.join(save_image_path, per_label_name)
    with open(save_per_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(total_per_image_json_data, json_file, ensure_ascii=False)

    # 返回一些信息(可选)
    return per_image_name, len(keep_image_annotation), len(total_json_masks)


@torch.no_grad
def preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=8):
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    root_image_path = os.path.join(root_dataset_path, dataset_type)
    save_image_path = os.path.join(save_dataset_path, dataset_type)
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    # 收集所有需要处理的图像和 JSON
    all_image_name_path_list = []
    for per_image_name in os.listdir(root_image_path):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(root_image_path, per_image_name)
            per_label_name = per_image_name.split('.')[0] + '.json'
            per_label_path = os.path.join(root_image_path, per_label_name)
            if not os.path.exists(per_image_path) or not os.path.exists(
                    per_label_path):
                continue
            all_image_name_path_list.append([
                per_image_name,
                per_image_path,
                per_label_name,
                per_label_path,
            ])

    print(f"待处理图像数量: {len(all_image_name_path_list)}")

    # 使用 ThreadPoolExecutor 并行处理
    # 如果不关心返回值，可以直接用 executor.map；若需要更灵活控制可用 as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for per_image_info in all_image_name_path_list:
            per_image_name, per_image_path, per_label_name, per_label_path = per_image_info
            futures.append(
                executor.submit(process_single_image, per_image_name,
                                per_image_path, per_label_name, per_label_path,
                                save_image_path))

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Processing"):
            # 拿到子任务的返回值
            result = future.result()
            print(f"[主进程] 已完成处理：{result}")

    print("全部处理完成！")


if __name__ == '__main__':
    # root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000020'
    # save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000020_filter_duplicated'
    # dataset_type = 'train'

    # preprocess_image(root_dataset_path,
    #                  save_dataset_path,
    #                  dataset_type,
    #                  max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000021'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000021_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000022'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000022_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000023'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000023_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000024'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000024_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000025'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000025_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000026'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000026_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000027'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000027_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000028'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000028_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000029'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/sa_000029_filter_duplicated'
    dataset_type = 'train'

    preprocess_image(root_dataset_path,
                     save_dataset_path,
                     dataset_type,
                     max_workers=18)
