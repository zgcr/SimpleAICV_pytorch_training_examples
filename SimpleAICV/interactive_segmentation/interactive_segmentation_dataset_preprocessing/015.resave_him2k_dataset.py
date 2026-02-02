import os
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


def process_image_mask_pair(args):
    """处理单个图像-mask对的函数"""
    per_image_name_prefix, per_image_name, per_image_path, per_mask_name, per_mask_path, save_dataset_path, is_composition = args

    try:
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

        per_mask = np.array(Image.open(per_mask_path).convert('L'),
                            dtype=np.uint8)

        per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]

        assert per_image_h == per_mask_h and per_image_w == per_mask_w

        factor = 1080 / max(per_image_h, per_image_w)
        resize_h, resize_w = int(round(per_image_h * factor)), int(
            round(per_image_w * factor))
        per_image = cv2.resize(per_image, (resize_w, resize_h))
        per_mask = cv2.resize(per_mask, (resize_w, resize_h))

        per_mask_compute_area = per_mask.copy() / 255.
        per_mask_compute_area = (per_mask_compute_area > 0.5).astype(np.uint8)

        per_mask_compute_area_h, per_mask_compute_area_w = per_mask_compute_area.shape[
            0], per_mask_compute_area.shape[1]
        total_mask_area = float(per_mask_compute_area_h *
                                per_mask_compute_area_w)

        # 计算前景像素面积
        foreground_mask_area = np.count_nonzero(per_mask_compute_area)
        foreground_ratio = foreground_mask_area / total_mask_area

        # 检查前景区域面积比例
        if foreground_ratio < 0.0001 or foreground_ratio > 0.9:
            print('3333', per_image_name_prefix)
            return

        # 寻找前景区域的包围框
        foreground_coords = np.where(per_mask_compute_area == 1)
        if len(foreground_coords[0]) == 0:
            print('4444', per_image_name_prefix)
            return

        y_min, y_max = np.min(foreground_coords[0]), np.max(
            foreground_coords[0])
        x_min, x_max = np.min(foreground_coords[1]), np.max(
            foreground_coords[1])

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # 检查包围框尺寸
        if bbox_width / per_mask_compute_area_w < 0.01 or bbox_height / per_mask_compute_area_h < 0.01:
            print('5555', per_image_name_prefix)
            return

        # 检查包围框面积比例
        bbox_area = bbox_width * bbox_height
        if bbox_area / total_mask_area < 0.0001:
            print('6666', per_image_name_prefix)
            return

        prefix = 'composition_' if is_composition else ''
        save_image_name = prefix + per_image_name_prefix + '_' + per_mask_name.split(
            '.')[0] + '.jpg'
        save_image_path = os.path.join(save_dataset_path, save_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

        save_mask_name = prefix + per_image_name_prefix + '_' + per_mask_name.split(
            '.')[0] + '.png'
        save_mask_path = os.path.join(save_dataset_path, save_mask_name)
        cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)
    except Exception as e:
        print(f'Error processing {per_image_name_prefix}: {e}')


def preprocess_image(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path, 'train')
    os.makedirs(save_dataset_path, exist_ok=True)

    root_image_path = os.path.join(root_dataset_path, 'images')
    root_mask_path = os.path.join(root_dataset_path, 'alphas')

    natural_root_image_path = os.path.join(root_image_path, 'natural')
    natural_root_mask_path = os.path.join(root_mask_path, 'natural')

    natural_image_mask_pair_path_list = []
    for per_image_name in sorted(os.listdir(natural_root_image_path)):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(natural_root_image_path,
                                          per_image_name)
            per_image_name_prefix = per_image_name.split('.')[0]

            per_image_mask_dir_path = os.path.join(natural_root_mask_path,
                                                   per_image_name_prefix)

            for per_mask_name in sorted(os.listdir(per_image_mask_dir_path)):
                if '.png' in per_mask_name:
                    per_mask_path = os.path.join(per_image_mask_dir_path,
                                                 per_mask_name)
                    natural_image_mask_pair_path_list.append([
                        per_image_name_prefix,
                        per_image_name,
                        per_image_path,
                        per_mask_name,
                        per_mask_path,
                    ])

    print('1111', len(natural_image_mask_pair_path_list),
          natural_image_mask_pair_path_list[0])

    composition_root_image_path = os.path.join(root_image_path, 'comp')
    composition_root_mask_path = os.path.join(root_mask_path, 'comp')

    composition_image_mask_pair_path_list = []
    for per_image_name in sorted(os.listdir(composition_root_image_path)):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(composition_root_image_path,
                                          per_image_name)
            per_image_name_prefix = per_image_name.split('.')[0]

            per_image_mask_dir_path = os.path.join(composition_root_mask_path,
                                                   per_image_name_prefix)

            for per_mask_name in sorted(os.listdir(per_image_mask_dir_path)):
                if '.png' in per_mask_name:
                    per_mask_path = os.path.join(per_image_mask_dir_path,
                                                 per_mask_name)
                    composition_image_mask_pair_path_list.append([
                        per_image_name_prefix,
                        per_image_name,
                        per_image_path,
                        per_mask_name,
                        per_mask_path,
                    ])

    print('2222', len(composition_image_mask_pair_path_list),
          composition_image_mask_pair_path_list[0])

    # for per_image_name_prefix, per_image_name, per_image_path, per_mask_name, per_mask_path in tqdm(
    #         natural_image_mask_pair_path_list):
    #     per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
    #                              cv2.IMREAD_COLOR)

    #     per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    #     per_mask = np.array(Image.open(per_mask_path).convert('L'),
    #                         dtype=np.uint8)

    #     per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]

    #     assert per_image_h == per_mask_h and per_image_w == per_mask_w

    #     factor = 1080 / max(per_image_h, per_image_w)
    #     resize_h, resize_w = int(round(per_image_h * factor)), int(
    #         round(per_image_w * factor))
    #     per_image = cv2.resize(per_image, (resize_w, resize_h))
    #     per_mask = cv2.resize(per_mask, (resize_w, resize_h))

    #     per_mask_compute_area = per_mask.copy() / 255.
    #     per_mask_compute_area = (per_mask_compute_area > 0.5).astype(np.uint8)

    #     per_mask_compute_area_h, per_mask_compute_area_w = per_mask_compute_area.shape[
    #         0], per_mask_compute_area.shape[1]
    #     total_mask_area = float(per_mask_compute_area_h *
    #                             per_mask_compute_area_w)

    #     # 计算前景像素面积
    #     foreground_mask_area = np.count_nonzero(per_mask_compute_area)
    #     foreground_ratio = foreground_mask_area / total_mask_area

    #     # 检查前景区域面积比例
    #     if foreground_ratio < 0.0001 or foreground_ratio > 0.9:
    #         print('3333', per_image_name_prefix)
    #         continue

    #     # 寻找前景区域的包围框
    #     foreground_coords = np.where(per_mask_compute_area == 1)
    #     if len(foreground_coords[0]) == 0:
    #         print('4444', per_image_name_prefix)
    #         continue

    #     y_min, y_max = np.min(foreground_coords[0]), np.max(
    #         foreground_coords[0])
    #     x_min, x_max = np.min(foreground_coords[1]), np.max(
    #         foreground_coords[1])

    #     bbox_width = x_max - x_min
    #     bbox_height = y_max - y_min

    #     # 检查包围框尺寸
    #     if bbox_width / per_mask_compute_area_w < 0.01 or bbox_height / per_mask_compute_area_h < 0.01:
    #         print('5555', per_image_name_prefix)
    #         continue

    #     # 检查包围框面积比例
    #     bbox_area = bbox_width * bbox_height
    #     if bbox_area / total_mask_area < 0.0001:
    #         print('6666', per_image_name_prefix)
    #         continue

    #     save_image_name = per_image_name_prefix + '_' + per_mask_name.split(
    #         '.')[0] + '.jpg'
    #     save_image_path = os.path.join(save_dataset_path, save_image_name)
    #     cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    #     save_mask_name = per_image_name_prefix + '_' + per_mask_name.split(
    #         '.')[0] + '.png'
    #     save_mask_path = os.path.join(save_dataset_path, save_mask_name)
    #     cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)

    # for per_image_name_prefix, per_image_name, per_image_path, per_mask_name, per_mask_path in tqdm(
    #         composition_image_mask_pair_path_list):
    #     per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
    #                              cv2.IMREAD_COLOR)

    #     per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    #     per_mask = np.array(Image.open(per_mask_path).convert('L'),
    #                         dtype=np.uint8)

    #     per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]

    #     assert per_image_h == per_mask_h and per_image_w == per_mask_w

    #     factor = 1080 / max(per_image_h, per_image_w)
    #     resize_h, resize_w = int(round(per_image_h * factor)), int(
    #         round(per_image_w * factor))
    #     per_image = cv2.resize(per_image, (resize_w, resize_h))
    #     per_mask = cv2.resize(per_mask, (resize_w, resize_h))

    #     per_mask_compute_area = per_mask.copy() / 255.
    #     per_mask_compute_area = (per_mask_compute_area > 0.5).astype(np.uint8)

    #     per_mask_compute_area_h, per_mask_compute_area_w = per_mask_compute_area.shape[
    #         0], per_mask_compute_area.shape[1]
    #     total_mask_area = float(per_mask_compute_area_h *
    #                             per_mask_compute_area_w)

    #     # 计算前景像素面积
    #     foreground_mask_area = np.count_nonzero(per_mask_compute_area)
    #     foreground_ratio = foreground_mask_area / total_mask_area

    #     # 检查前景区域面积比例
    #     if foreground_ratio < 0.0001 or foreground_ratio > 0.9:
    #         print('3333', per_image_name_prefix)
    #         continue

    #     # 寻找前景区域的包围框
    #     foreground_coords = np.where(per_mask_compute_area == 1)
    #     if len(foreground_coords[0]) == 0:
    #         print('4444', per_image_name_prefix)
    #         continue

    #     y_min, y_max = np.min(foreground_coords[0]), np.max(
    #         foreground_coords[0])
    #     x_min, x_max = np.min(foreground_coords[1]), np.max(
    #         foreground_coords[1])

    #     bbox_width = x_max - x_min
    #     bbox_height = y_max - y_min

    #     # 检查包围框尺寸
    #     if bbox_width / per_mask_compute_area_w < 0.01 or bbox_height / per_mask_compute_area_h < 0.01:
    #         print('5555', per_image_name_prefix)
    #         continue

    #     # 检查包围框面积比例
    #     bbox_area = bbox_width * bbox_height
    #     if bbox_area / total_mask_area < 0.0001:
    #         print('6666', per_image_name_prefix)
    #         continue

    #     save_image_name = 'composition_' + per_image_name_prefix + '_' + per_mask_name.split(
    #         '.')[0] + '.jpg'
    #     save_image_path = os.path.join(save_dataset_path, save_image_name)
    #     cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    #     save_mask_name = 'composition_' + per_image_name_prefix + '_' + per_mask_name.split(
    #         '.')[0] + '.png'
    #     save_mask_path = os.path.join(save_dataset_path, save_mask_name)
    #     cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)

    natural_args = [(item[0], item[1], item[2], item[3], item[4],
                     save_dataset_path, False)
                    for item in natural_image_mask_pair_path_list]

    composition_args = [(item[0], item[1], item[2], item[3], item[4],
                         save_dataset_path, True)
                        for item in composition_image_mask_pair_path_list]

    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_image_mask_pair, natural_args),
                 total=len(natural_args),
                 desc='Processing natural images'))

    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_image_mask_pair, composition_args),
                 total=len(composition_args),
                 desc='Processing composition images'))


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/MaGGIe-HIM/HIM2K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HIM2K'
    preprocess_image(root_dataset_path, save_dataset_path)
