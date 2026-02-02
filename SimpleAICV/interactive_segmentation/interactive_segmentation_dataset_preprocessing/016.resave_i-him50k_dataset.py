import os
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def process_image_mask_pair(item, save_dataset_path):
    per_image_name_prefix, per_image_name, per_image_path, per_mask_name, per_mask_path = item

    per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)

    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    per_mask = np.array(Image.open(per_mask_path).convert('L'), dtype=np.uint8)

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
    total_mask_area = float(per_mask_compute_area_h * per_mask_compute_area_w)

    # 计算前景像素面积
    foreground_mask_area = np.count_nonzero(per_mask_compute_area)
    foreground_ratio = foreground_mask_area / total_mask_area

    # 检查前景区域面积比例
    if foreground_ratio < 0.0001 or foreground_ratio > 0.9:
        print('4444', per_image_path)
        return

    # 寻找前景区域的包围框
    foreground_coords = np.where(per_mask_compute_area == 1)
    if len(foreground_coords[0]) == 0:
        print('5555', per_image_path)
        return

    y_min, y_max = np.min(foreground_coords[0]), np.max(foreground_coords[0])
    x_min, x_max = np.min(foreground_coords[1]), np.max(foreground_coords[1])

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # 检查包围框尺寸
    if bbox_width / per_mask_compute_area_w < 0.01 or bbox_height / per_mask_compute_area_h < 0.01:
        print('6666', per_image_path)
        return

    # 检查包围框面积比例
    bbox_area = bbox_width * bbox_height
    if bbox_area / total_mask_area < 0.0001:
        print('7777', per_image_path)
        return

    save_image_name = f'i_him50k_{int(per_image_name.split(".")[0]):06d}_{int(per_mask_name.split(".")[0]):02d}.jpg'
    save_image_path = os.path.join(save_dataset_path, save_image_name)
    cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    save_mask_name = f'i_him50k_{int(per_image_name.split(".")[0]):06d}_{int(per_mask_name.split(".")[0]):02d}.png'
    save_mask_path = os.path.join(save_dataset_path, save_mask_name)
    cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)


def preprocess_image(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path, 'train')
    os.makedirs(save_dataset_path, exist_ok=True)

    root_image_path = os.path.join(root_dataset_path, 'images')
    root_mask_path = os.path.join(root_dataset_path, 'alphas')

    image_mask_pair_path_list = []
    for per_image_name in sorted(os.listdir(root_image_path)):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(root_image_path, per_image_name)
            per_image_name_prefix = per_image_name.split('.')[0]

            per_image_mask_dir_path = os.path.join(root_mask_path,
                                                   per_image_name_prefix)

            for per_mask_name in sorted(os.listdir(per_image_mask_dir_path)):
                if '.png' in per_mask_name:
                    per_mask_path = os.path.join(per_image_mask_dir_path,
                                                 per_mask_name)
                    image_mask_pair_path_list.append([
                        per_image_name_prefix,
                        per_image_name,
                        per_image_path,
                        per_mask_name,
                        per_mask_path,
                    ])

    print('1111', len(image_mask_pair_path_list), image_mask_pair_path_list[0])

    # for per_image_name_prefix, per_image_name, per_image_path, per_mask_name, per_mask_path in tqdm(
    #         image_mask_pair_path_list):
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
    #         print('4444', per_image_path)
    #         continue

    #     # 寻找前景区域的包围框
    #     foreground_coords = np.where(per_mask_compute_area == 1)
    #     if len(foreground_coords[0]) == 0:
    #         print('5555', per_image_path)
    #         continue

    #     y_min, y_max = np.min(foreground_coords[0]), np.max(
    #         foreground_coords[0])
    #     x_min, x_max = np.min(foreground_coords[1]), np.max(
    #         foreground_coords[1])

    #     bbox_width = x_max - x_min
    #     bbox_height = y_max - y_min

    #     # 检查包围框尺寸
    #     if bbox_width / per_mask_compute_area_w < 0.01 or bbox_height / per_mask_compute_area_h < 0.01:
    #         print('6666', per_image_path)
    #         continue

    #     # 检查包围框面积比例
    #     bbox_area = bbox_width * bbox_height
    #     if bbox_area / total_mask_area < 0.0001:
    #         print('7777', per_image_path)
    #         continue

    #     save_image_name =f'i_him50k_{int(per_image_name.split('.')[0]):06d}_{int(per_mask_name.split('.')[0]):02d}.jpg'
    #     save_image_path = os.path.join(save_dataset_path, save_image_name)
    #     cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    #     save_mask_name = f'i_him50k_{int(per_image_name.split('.')[0]):06d}_{int(per_mask_name.split('.')[0]):02d}.png'
    #     save_mask_path = os.path.join(save_dataset_path, save_mask_name)
    #     cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)

    process_func = partial(process_image_mask_pair,
                           save_dataset_path=save_dataset_path)

    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_func, image_mask_pair_path_list),
                 total=len(image_mask_pair_path_list)))


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/MaGGIe-HIM/I-HIM50K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/I-HIM50K'
    preprocess_image(root_dataset_path, save_dataset_path)
