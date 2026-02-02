import os
import json
import cv2
import math
import numpy as np

from tqdm import tqdm
from pycocotools import mask as mask_utils

from multiprocessing import Pool


def process_single_image(args):
    file_name, jpg_path, json_name, json_path, original_root, root_dataset_path, save_dataset_path = args
    try:
        per_image = cv2.imdecode(np.fromfile(jpg_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

        factor = 1080.0 / max(per_image_h, per_image_w)
        resize_h, resize_w = int(round(per_image_h * factor)), int(
            round(per_image_w * factor))
        per_image = cv2.resize(per_image, (resize_w, resize_h))

        # 读取JSON文件
        with open(json_path, encoding='utf-8') as f:
            per_image_json_data = json.load(f)

        # 更新image信息
        per_image_json_data['image']['width'] = resize_w
        per_image_json_data['image']['height'] = resize_h

        per_image_new_annotations = []
        # 处理annotations中的每个标注
        for annotation in per_image_json_data['annotations']:
            # 处理bbox
            bbox = annotation['bbox']
            annotation['bbox'] = [
                bbox[0] * factor,
                bbox[1] * factor,
                bbox[2] * factor,
                bbox[3] * factor,
            ]

            # 处理point_coords
            point_coords = annotation['point_coords']
            annotation['point_coords'] = [[
                point[0] * factor,
                point[1] * factor,
            ] for point in point_coords]

            # 处理crop_box
            crop_box = annotation['crop_box']
            annotation['crop_box'] = [
                crop_box[0] * factor,
                crop_box[1] * factor,
                crop_box[2] * factor,
                crop_box[3] * factor,
            ]

            # 处理segmentation (RLE格式)
            segmentation = annotation['segmentation']
            if 'counts' in segmentation.keys() and 'size' in segmentation.keys(
            ):
                rle_mask = mask_utils.decode(segmentation)
                resized_mask = cv2.resize(rle_mask, (resize_w, resize_h),
                                          interpolation=cv2.INTER_NEAREST)

                # 重新编码为RLE
                resized_rle = mask_utils.encode(
                    np.asfortranarray(resized_mask))
                segmentation['counts'] = resized_rle['counts'].decode('utf-8')
                segmentation['size'] = [resize_h, resize_w]

                # 更新area
                annotation['area'] = int(np.count_nonzero(resized_mask))

            per_image_new_annotations.append(annotation)

        if len(per_image_new_annotations) == 0:
            return False

        per_image_json_data['annotations'] = per_image_new_annotations

        keep_image_annotation = []
        for per_annot in per_image_json_data['annotations']:
            per_box = per_annot['bbox']
            x_min = math.ceil(max(per_box[0], 0))
            y_min = math.ceil(max(per_box[1], 0))
            x_max = math.ceil(min(per_box[0] + per_box[2], resize_w))
            y_max = math.ceil(min(per_box[1] + per_box[3], resize_h))
            box_w = math.ceil(x_max - x_min)
            box_h = math.ceil(y_max - y_min)

            if box_w / resize_w < 0.01 and box_h / resize_h < 0.01:
                continue

            if (box_w * box_h) / float(resize_h * resize_w) < 0.0001:
                continue

            if per_annot['area'] / float(
                    resize_h * resize_w) < 0.0001 or per_annot['area'] / float(
                        resize_h * resize_w) > 0.9:
                continue

            keep_image_annotation.append(per_annot)

        if len(keep_image_annotation) == 0:
            return False

        per_image_json_data['annotations'] = keep_image_annotation

        # 创建保存目录（保持原始目录结构，并在上面添加train文件夹）
        relative_path = os.path.relpath(original_root, root_dataset_path)
        save_dir = os.path.join(save_dataset_path, 'train', relative_path)
        os.makedirs(save_dir, exist_ok=True)

        # 保存resize后的图片
        save_jpg_path = os.path.join(save_dir, os.path.basename(jpg_path))
        cv2.imencode('.jpg', per_image)[1].tofile(save_jpg_path)

        # 保存更新后的JSON
        save_json_path = os.path.join(save_dir, os.path.basename(json_path))
        with open(save_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(per_image_json_data, json_file, ensure_ascii=False)
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

    #         per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    #         factor = 1080.0 / max(per_image_h, per_image_w)
    #         resize_h, resize_w = int(round(per_image_h * factor)), int(
    #             round(per_image_w * factor))
    #         per_image = cv2.resize(per_image, (resize_w, resize_h))

    #         # 读取JSON文件
    #         with open(json_path, encoding='utf-8') as f:
    #             per_image_json_data = json.load(f)

    #         # 更新image信息
    #         per_image_json_data['image']['width'] = resize_w
    #         per_image_json_data['image']['height'] = resize_h

    #         per_image_new_annotations = []
    #         # 处理annotations中的每个标注
    #         for annotation in per_image_json_data['annotations']:
    #             # 处理bbox
    #             bbox = annotation['bbox']
    #             annotation['bbox'] = [
    #                 bbox[0] * factor,
    #                 bbox[1] * factor,
    #                 bbox[2] * factor,
    #                 bbox[3] * factor,
    #             ]

    #             # 处理point_coords
    #             point_coords = annotation['point_coords']
    #             annotation['point_coords'] = [[
    #                 point[0] * factor,
    #                 point[1] * factor,
    #             ] for point in point_coords]

    #             # 处理crop_box
    #             crop_box = annotation['crop_box']
    #             annotation['crop_box'] = [
    #                 crop_box[0] * factor,
    #                 crop_box[1] * factor,
    #                 crop_box[2] * factor,
    #                 crop_box[3] * factor,
    #             ]

    #             # 处理segmentation (RLE格式)
    #             segmentation = annotation['segmentation']
    #             if 'counts' in segmentation.keys(
    #             ) and 'size' in segmentation.keys():
    #                 rle_mask = mask_utils.decode(segmentation)
    #                 resized_mask = cv2.resize(rle_mask, (resize_w, resize_h),
    #                                           interpolation=cv2.INTER_NEAREST)

    #                 # 重新编码为RLE
    #                 resized_rle = mask_utils.encode(
    #                     np.asfortranarray(resized_mask))
    #                 segmentation['counts'] = resized_rle['counts'].decode(
    #                     'utf-8')
    #                 segmentation['size'] = [resize_h, resize_w]

    #                 # 更新area
    #                 annotation['area'] = int(np.count_nonzero(resized_mask))

    #             per_image_new_annotations.append(annotation)

    #         if len(per_image_new_annotations) == 0:
    #             return False

    #         per_image_json_data['annotations'] = per_image_new_annotations

    #         keep_image_annotation = []
    #         for per_annot in per_image_json_data['annotations']:
    #             per_box = per_annot['bbox']
    #             x_min = math.ceil(max(per_box[0], 0))
    #             y_min = math.ceil(max(per_box[1], 0))
    #             x_max = math.ceil(min(per_box[0] + per_box[2], resize_w))
    #             y_max = math.ceil(min(per_box[1] + per_box[3], resize_h))
    #             box_w = math.ceil(x_max - x_min)
    #             box_h = math.ceil(y_max - y_min)

    #             if box_w / resize_w < 0.01 and box_h / resize_h < 0.01:
    #                 continue

    #             if (box_w * box_h) / float(resize_h * resize_w) < 0.0001:
    #                 continue

    #             if per_annot['area'] / float(
    #                     resize_h *
    #                     resize_w) < 0.0001 or per_annot['area'] / float(
    #                         resize_h * resize_w) > 0.9:
    #                 continue

    #             keep_image_annotation.append(per_annot)

    #         if len(keep_image_annotation) == 0:
    #             return False

    #         per_image_json_data['annotations'] = keep_image_annotation

    #         # 创建保存目录（保持原始目录结构，并在上面添加train文件夹）
    #         relative_path = os.path.relpath(original_root, root_dataset_path)
    #         save_dir = os.path.join(save_dataset_path, 'train', relative_path)
    #         os.makedirs(save_dir, exist_ok=True)

    #         # 保存resize后的图片
    #         save_jpg_path = os.path.join(save_dir, os.path.basename(jpg_path))
    #         cv2.imencode('.jpg', per_image)[1].tofile(save_jpg_path)

    #         # 保存更新后的JSON
    #         save_json_path = os.path.join(save_dir,
    #                                       os.path.basename(json_path))
    #         with open(save_json_path, 'w', encoding='utf-8') as json_file:
    #             json.dump(per_image_json_data, json_file, ensure_ascii=False)
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
                 total=len(process_args),
                 desc="Processing images"))

    # 统计成功和失败的数量
    successful = sum(results)
    failed = len(results) - successful
    print(f"Processing completed: {successful} successful, {failed} failed")


if __name__ == '__main__':
    for i in range(0, 50):
        file_index = f"{i:06d}"

        root_dataset_path = f'/root/autodl-tmp/interactive_segmentation_dataset/sa_{file_index}'
        save_dataset_path = f'/root/autodl-tmp/interactive_segmentation_dataset_new2/sa_{file_index}'

        print(f"\n{'=' * 50}")
        print(f"🚀 开始处理文件：{root_dataset_path}")
        print(f"{'=' * 50}")

        preprocess_image(root_dataset_path, save_dataset_path)
