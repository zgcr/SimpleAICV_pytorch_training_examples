import os
import cv2
import copy
import json
import math
import numpy as np

from tqdm import tqdm
from pycocotools import mask as mask_utils

from multiprocessing import Pool


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


def process_single_image(args):
    file_name, jpg_path, json_name, json_path, original_root, root_dataset_path, save_dataset_path = args
    try:
        per_image = cv2.imdecode(np.fromfile(jpg_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        # 读取json
        with open(json_path, encoding='utf-8') as f:
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

            if (box_w * box_h) / float(per_image_h * per_image_w) < 0.0001:
                continue

            if per_annot['area'] / float(
                    per_image_h *
                    per_image_w) < 0.0001 or per_annot['area'] / float(
                        per_image_h * per_image_w) > 0.9:
                continue

            keep_image_annotation.append(per_annot)

        if len(keep_image_annotation) == 0:
            return False

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
                        if r1 > 0.9:
                            # mask_i 被 mask_j 包含 => 去除 i
                            keep_flag[i] = False
                            break
                        elif r2 > 0.9:
                            # mask_j 被 mask_i 包含 => 去除 j
                            keep_flag[j] = False
            total_masks = [
                keep_image_masks[i] for i in range(len(keep_image_masks))
                if keep_flag[i]
            ]

        total_json_masks = [m[0] for m in total_masks]

        if len(total_json_masks) == 0:
            return False

        total_per_image_json_data = copy.deepcopy(per_image_json_data)
        total_per_image_json_data['annotations'] = total_json_masks

        # 创建保存目录（保持原始目录结构）
        relative_path = os.path.relpath(original_root, root_dataset_path)
        save_dir = os.path.join(save_dataset_path, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        # 保存图像到目标目录
        save_jpg_path = os.path.join(save_dir, os.path.basename(jpg_path))
        cv2.imencode('.jpg', per_image)[1].tofile(save_jpg_path)

        # 保存过滤后的JSON
        save_json_path = os.path.join(save_dir, os.path.basename(json_path))
        with open(save_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(total_per_image_json_data, json_file, ensure_ascii=False)
        return True
    except Exception as e:
        print(f'Error processing {file_name}: {e}')
        return False


def preprocess_image(root_dataset_path, save_dataset_path):
    os.makedirs(save_dataset_path, exist_ok=True)

    all_image_path_list = []
    for root, dirs, files in os.walk(root_dataset_path):
        for file_name in files:
            if file_name.lower().endswith('.jpg'):
                file_path = os.path.join(root, file_name)
                per_png_label_name = file_name.split('.')[0] + '.json'
                per_png_label_path = os.path.join(root, per_png_label_name)

                if os.path.exists(file_path) and os.path.exists(
                        per_png_label_path):
                    all_image_path_list.append([
                        file_name,
                        file_path,
                        per_png_label_name,
                        per_png_label_path,
                        root,
                    ])

    print(f"1111", len(all_image_path_list), all_image_path_list[0])

    # for file_name, jpg_path, json_name, json_path, original_root in tqdm(
    #         all_image_path_list):
    #     try:
    #         per_image = cv2.imdecode(np.fromfile(jpg_path, dtype=np.uint8),
    #                                  cv2.IMREAD_COLOR)

    #         # 读取json
    #         with open(json_path, encoding='utf-8') as f:
    #             per_image_json_data = json.load(f)
    #         per_image_annotation = per_image_json_data['annotations']
    #         per_image_h = per_image_json_data['image']['height']
    #         per_image_w = per_image_json_data['image']['width']

    #         # 1. 根据一些规则先做初筛
    #         keep_image_annotation = []
    #         for per_annot in per_image_annotation:
    #             per_box = per_annot['bbox']
    #             x_min = math.ceil(max(per_box[0], 0))
    #             y_min = math.ceil(max(per_box[1], 0))
    #             x_max = math.ceil(min(per_box[0] + per_box[2], per_image_w))
    #             y_max = math.ceil(min(per_box[1] + per_box[3], per_image_h))
    #             box_w = math.ceil(x_max - x_min)
    #             box_h = math.ceil(y_max - y_min)

    #             if box_w / per_image_w < 0.01 and box_h / per_image_h < 0.01:
    #                 continue

    #             if (box_w * box_h) / float(per_image_h * per_image_w) < 0.0001:
    #                 continue

    #             if per_annot['area'] / float(
    #                     per_image_h *
    #                     per_image_w) < 0.0001 or per_annot['area'] / float(
    #                         per_image_h * per_image_w) > 0.9:
    #                 continue

    #             keep_image_annotation.append(per_annot)

    #         if len(keep_image_annotation) == 0:
    #             continue

    #         keep_per_image_json_data = copy.deepcopy(per_image_json_data)
    #         keep_per_image_json_data['annotations'] = keep_image_annotation

    #         # 2. 将对应的 mask 解码出来，做重复/包含过滤
    #         keep_image_masks = []
    #         for per_annot in keep_per_image_json_data['annotations']:
    #             per_mask = mask_utils.decode(per_annot['segmentation'])
    #             keep_image_masks.append([per_annot, per_mask])

    #         if len(keep_image_masks) <= 1:
    #             total_masks = keep_image_masks
    #         else:
    #             keep_flag = [True] * len(keep_image_masks)
    #             for i in range(len(keep_image_masks)):
    #                 if not keep_flag[i]:
    #                     continue
    #                 for j in range(i + 1, len(keep_image_masks)):
    #                     if not keep_flag[j]:
    #                         continue
    #                     r1, r2, iou = calculate_recall_iou(
    #                         keep_image_masks[i][1], keep_image_masks[j][1])
    #                     if iou > 0.9:
    #                         # 重复，保留 i, 去除 j
    #                         keep_flag[j] = False
    #                     else:
    #                         # 判断包含
    #                         if r1 > 0.9:
    #                             # mask_i 被 mask_j 包含 => 去除 i
    #                             keep_flag[i] = False
    #                             break
    #                         elif r2 > 0.9:
    #                             # mask_j 被 mask_i 包含 => 去除 j
    #                             keep_flag[j] = False
    #             total_masks = [
    #                 keep_image_masks[i] for i in range(len(keep_image_masks))
    #                 if keep_flag[i]
    #             ]

    #         total_json_masks = [m[0] for m in total_masks]

    #         if len(total_json_masks) == 0:
    #             continue

    #         total_per_image_json_data = copy.deepcopy(per_image_json_data)
    #         total_per_image_json_data['annotations'] = total_json_masks

    #         # 创建保存目录（保持原始目录结构）
    #         relative_path = os.path.relpath(original_root, root_dataset_path)
    #         save_dir = os.path.join(save_dataset_path, relative_path)
    #         os.makedirs(save_dir, exist_ok=True)

    #         # 保存图像到目标目录
    #         save_jpg_path = os.path.join(save_dir, os.path.basename(jpg_path))
    #         cv2.imencode('.jpg', per_image)[1].tofile(save_jpg_path)

    #         # 保存过滤后的JSON
    #         save_json_path = os.path.join(save_dir,
    #                                       os.path.basename(json_path))
    #         with open(save_json_path, 'w', encoding='utf-8') as json_file:
    #             json.dump(total_per_image_json_data,
    #                       json_file,
    #                       ensure_ascii=False)
    #     except:
    #         print('1313', file_name)
    #         continue

    # 准备多进程参数
    process_args = [(file_name, jpg_path, json_name, json_path, original_root,
                     root_dataset_path, save_dataset_path)
                    for file_name, jpg_path, json_name, json_path,
                    original_root in all_image_path_list]

    # 使用多进程处理
    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(process_single_image, process_args),
                 total=len(process_args)))

    # 统计成功和失败的数量
    successful = sum(results)
    failed = len(results) - successful
    print(f"Processing completed: {successful} successful, {failed} failed")


if __name__ == '__main__':
    for i in range(0, 50):
        file_index = f"{i:06d}"

        root_dataset_path = f'/root/autodl-tmp/interactive_segmentation_dataset/sa_{file_index}'
        save_dataset_path = f'/root/autodl-tmp/interactive_segmentation_dataset/sa_{file_index}_filter_part_object'

        print(f"\n{'=' * 50}")
        print(f"🚀 开始处理文件：{root_dataset_path}")
        print(f"{'=' * 50}")

        preprocess_image(root_dataset_path, save_dataset_path)
