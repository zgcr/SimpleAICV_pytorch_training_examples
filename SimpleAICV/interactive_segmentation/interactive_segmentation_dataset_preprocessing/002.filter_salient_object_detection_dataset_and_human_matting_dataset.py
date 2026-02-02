import os
import numpy as np
import shutil

from PIL import Image
from tqdm import tqdm

from multiprocessing import Pool


def validate_single_image(item):
    file_name, file_path, per_png_label_name, per_png_label_path = item

    try:
        per_mask = np.array(Image.open(per_png_label_path).convert('L'),
                            dtype=np.uint8)
        per_mask = per_mask / 255.
        per_mask[per_mask > 0.5] = 1
        per_mask[per_mask <= 0.5] = 0
        per_mask = per_mask.astype(np.uint8)

        per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]
        total_mask_area = float(per_mask_h * per_mask_w)

        # 计算前景像素面积
        foreground_mask_area = np.count_nonzero(per_mask)
        foreground_ratio = foreground_mask_area / total_mask_area

        # 检查前景区域面积比例
        if foreground_ratio < 0.0001 or foreground_ratio > 0.9:
            return None

        # 寻找前景区域的包围框
        foreground_coords = np.where(per_mask == 1)
        if len(foreground_coords[0]) == 0:
            return None

        y_min, y_max = np.min(foreground_coords[0]), np.max(
            foreground_coords[0])
        x_min, x_max = np.min(foreground_coords[1]), np.max(
            foreground_coords[1])

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # 检查包围框尺寸
        if bbox_width / per_mask_w < 0.01 or bbox_height / per_mask_h < 0.01:
            return None

        # 检查包围框面积比例
        bbox_area = bbox_width * bbox_height
        if bbox_area / total_mask_area < 0.0001:
            return None

        # 所有条件都满足，保留这个文件对
        return [file_name, file_path, per_png_label_name, per_png_label_path]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def copy_single_image(item):
    jpg_name, jpg_path, png_name, png_path = item

    # 获取相对路径
    jpg_relative_path = os.path.relpath(jpg_path, root_dataset_path)
    png_relative_path = os.path.relpath(png_path, root_dataset_path)

    # 构建目标路径
    jpg_target_path = os.path.join(save_dataset_path, jpg_relative_path)
    png_target_path = os.path.join(save_dataset_path, png_relative_path)

    # 创建目标目录
    os.makedirs(os.path.dirname(jpg_target_path), exist_ok=True)

    # 复制文件
    shutil.copy2(jpg_path, jpg_target_path)
    shutil.copy2(png_path, png_target_path)


def preprocess_image(root_dataset_path, save_dataset_path):
    os.makedirs(save_dataset_path, exist_ok=True)

    all_image_path_list = []

    for root, dirs, files in os.walk(root_dataset_path):
        for file_name in files:
            if file_name.lower().endswith('.jpg'):
                file_path = os.path.join(root, file_name)
                per_png_label_name = file_name.split('.')[0] + '.png'
                per_png_label_path = os.path.join(root, per_png_label_name)

                if os.path.exists(file_path) and os.path.exists(
                        per_png_label_path):
                    all_image_path_list.append([
                        file_name,
                        file_path,
                        per_png_label_name,
                        per_png_label_path,
                    ])

    print(f"1111", len(all_image_path_list), all_image_path_list[0])

    # valid_image_path_list = []
    # for file_name, file_path, per_png_label_name, per_png_label_path in tqdm(
    #         all_image_path_list):
    #     per_mask = np.array(Image.open(per_png_label_path).convert('L'),
    #                         dtype=np.uint8)
    #     per_mask = per_mask / 255.
    #     per_mask[per_mask > 0.5] = 1
    #     per_mask[per_mask <= 0.5] = 0
    #     per_mask = per_mask.astype(np.uint8)

    #     per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]
    #     total_mask_area = float(per_mask_h * per_mask_w)

    #     # 计算前景像素面积
    #     foreground_mask_area = np.count_nonzero(per_mask)
    #     foreground_ratio = foreground_mask_area / total_mask_area

    #     # 检查前景区域面积比例
    #     if foreground_ratio < 0.0001 or foreground_ratio > 0.9:
    #         continue

    #     # 寻找前景区域的包围框
    #     foreground_coords = np.where(per_mask == 1)
    #     if len(foreground_coords[0]) == 0:
    #         continue

    #     y_min, y_max = np.min(foreground_coords[0]), np.max(
    #         foreground_coords[0])
    #     x_min, x_max = np.min(foreground_coords[1]), np.max(
    #         foreground_coords[1])

    #     bbox_width = x_max - x_min
    #     bbox_height = y_max - y_min

    #     # 检查包围框尺寸
    #     if bbox_width / per_mask_w < 0.01 or bbox_height / per_mask_h < 0.01:
    #         continue

    #     # 检查包围框面积比例
    #     bbox_area = bbox_width * bbox_height
    #     if bbox_area / total_mask_area < 0.0001:
    #         continue

    #     # 所有条件都满足，保留这个文件对
    #     valid_image_path_list.append([
    #         file_name,
    #         file_path,
    #         per_png_label_name,
    #         per_png_label_path,
    #     ])

    # print(f"2222", len(valid_image_path_list), valid_image_path_list[0])

    # for item in tqdm(valid_image_path_list):
    #     jpg_name, jpg_path, png_name, png_path = item

    #     # 获取相对路径
    #     jpg_relative_path = os.path.relpath(jpg_path, root_dataset_path)
    #     png_relative_path = os.path.relpath(png_path, root_dataset_path)

    #     # 构建目标路径
    #     jpg_target_path = os.path.join(save_dataset_path, jpg_relative_path)
    #     png_target_path = os.path.join(save_dataset_path, png_relative_path)

    #     # 创建目标目录
    #     os.makedirs(os.path.dirname(jpg_target_path), exist_ok=True)

    #     # 复制文件
    #     shutil.copy2(jpg_path, jpg_target_path)
    #     shutil.copy2(png_path, png_target_path)

    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(validate_single_image, all_image_path_list),
                 total=len(all_image_path_list)))

    valid_image_path_list = [
        result for result in results if result is not None
    ]

    print(f"2222", len(valid_image_path_list), valid_image_path_list[0])

    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(copy_single_image, valid_image_path_list),
                 total=len(valid_image_path_list)))


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AIM500'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/AIM500'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AM2K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/AM2K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/DIS5K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRS10K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/HRS10K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRSOD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/HRSOD'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/MAGICK'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/MAGICK'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/UHRSD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/UHRSD'
    preprocess_image(root_dataset_path, save_dataset_path)

    #######################################################################################
    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/Deep_Automatic_Portrait_Matting'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/Deep_Automatic_Portrait_Matting'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/matting_human_half'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/matting_human_half'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-NP'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/P3M-500-NP'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-P'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/P3M-500-P'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M10K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/P3M10K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/RealWorldPortrait636'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset_new2/RealWorldPortrait636'
    preprocess_image(root_dataset_path, save_dataset_path)
