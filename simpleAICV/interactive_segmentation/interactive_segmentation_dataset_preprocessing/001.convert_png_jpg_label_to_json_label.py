import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import collections
import json
import math
import numpy as np

from PIL import Image
from tqdm import tqdm

import pycocotools.mask as mask_utils


def convert_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_int64(v) for k, v in obj.items()}
    else:
        return obj


def preprocess_image(root_dataset_path, save_dataset_path, dataset_type):
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    root_image_path = os.path.join(root_dataset_path, dataset_type)

    save_image_path = os.path.join(save_dataset_path, dataset_type)
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    all_image_name_path_list = []
    for per_image_name in tqdm(os.listdir(root_image_path)):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(root_image_path, per_image_name)
            per_png_label_name = per_image_name.split('.')[0] + '.png'
            per_png_label_path = os.path.join(root_image_path,
                                              per_png_label_name)
            if not os.path.exists(per_image_path) or not os.path.exists(
                    per_png_label_path):
                continue
            all_image_name_path_list.append([
                per_image_name,
                per_image_path,
                per_png_label_name,
                per_png_label_path,
            ])

    print('1111', len(all_image_name_path_list), all_image_name_path_list[0])

    for per_image_name, per_image_path, per_png_label_name, per_png_label_path in tqdm(
            all_image_name_path_list):
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        per_image = per_image.astype(np.uint8)

        per_mask = cv2.imdecode(
            np.fromfile(per_png_label_path, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE)
        per_mask = per_mask / 255.
        per_mask[per_mask >= 0.5] = 1
        per_mask[per_mask < 0.5] = 0
        per_mask = np.asfortranarray(per_mask)
        per_mask = per_mask.astype(np.uint8)

        size = np.array([per_image.shape[0],
                         per_image.shape[1]]).astype(np.float32)

        per_image_json = {}
        per_image_json_image_info = {}

        per_image_json_image_info["image_id"] = per_image_name[:-4]
        per_image_json_image_info["width"] = int(size[1])
        per_image_json_image_info["height"] = int(size[0])
        per_image_json_image_info["file_name"] = per_image_name[:-4] + '.jpg'

        per_image_json["image"] = per_image_json_image_info

        x_min, y_min, w, h = cv2.boundingRect(per_mask)

        per_mask2image = {}
        per_mask2image["id"] = 0
        rle = mask_utils.encode(per_mask)
        rle["counts"] = rle["counts"].decode('utf-8')
        per_mask2image["segmentation"] = rle

        per_mask2image["bbox"] = [x_min, y_min, w, h]
        per_mask2image["area"] = int(np.sum(per_mask))
        per_mask2image["predicted_iou"] = 1
        per_mask2image["stability_score"] = 1
        per_mask2image["point_coords"] = None
        per_image_json["annotations"] = [per_mask2image]

        per_mask2image["crop_box"] = [x_min, y_min, w, h]

        per_mask_all_points_coords = np.argwhere(per_mask.astype(np.float32))
        per_mask_all_points_num = len(per_mask_all_points_coords)
        choose_points_index = np.random.choice(per_mask_all_points_num, 1)
        all_point_coords = []
        for positive_point_idx in choose_points_index:
            all_point_coords.append([
                per_mask_all_points_coords[positive_point_idx][1],
                per_mask_all_points_coords[positive_point_idx][0],
            ])
        per_mask2image["point_coords"] = all_point_coords

        if len(per_image_json) == 0:
            continue

        save_per_image_path = os.path.join(save_image_path, per_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

        per_image_json = convert_int64(per_image_json)
        per_json_name = per_image_name[:-4] + '.json'
        save_json_path = os.path.join(save_image_path, per_json_name)
        with open(save_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(per_image_json, json_file, ensure_ascii=False)


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AIM500'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AIM500_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AM2K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AM2K_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AM2K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AM2K_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/Deep_Automatic_Portrait_Matting'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/Deep_Automatic_Portrait_Matting_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/Deep_Automatic_Portrait_Matting'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/Deep_Automatic_Portrait_Matting_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K_seg'
    dataset_type = 'test1'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K_seg'
    dataset_type = 'test2'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K_seg'
    dataset_type = 'test3'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K_seg'
    dataset_type = 'test4'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRS10K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRS10K_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRS10K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRS10K_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRSOD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRSOD_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRSOD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRSOD_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-NP'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-NP_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-P'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-P_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M10K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M10K_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/RealWorldPortrait636'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/RealWorldPortrait636_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/UHRSD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/UHRSD_seg'
    dataset_type = 'train'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/UHRSD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/UHRSD_seg'
    dataset_type = 'val'
    preprocess_image(root_dataset_path, save_dataset_path, dataset_type)
