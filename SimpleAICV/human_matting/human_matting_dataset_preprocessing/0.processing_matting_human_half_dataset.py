import cv2
import os
import numpy as np
import re

from tqdm import tqdm
from multiprocessing import Pool
import functools


def process_single_image(args, save_dataset_path):
    per_image_name, per_image_path, png_file_name, png_file_path = args

    if not os.path.exists(per_image_path) or not os.path.exists(png_file_path):
        return

    per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    per_png = cv2.imdecode(np.fromfile(png_file_path, dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
    _, _, _, alpha = cv2.split(per_png)

    per_alpha_h, per_alpha_w = alpha.shape[0], alpha.shape[1]

    assert per_image_h == per_alpha_h and per_image_w == per_alpha_w

    per_image_prefix = per_image_name.split('.')[0]

    save_image_name = per_image_prefix + '.jpg'
    save_image_path = os.path.join(save_dataset_path, save_image_name)
    cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    save_mask_name = per_image_prefix + '.png'
    save_mask_path = os.path.join(save_dataset_path, save_mask_name)
    cv2.imencode('.png', alpha)[1].tofile(save_mask_path)


def process_data(root_dataset_path, save_dataset_path):
    os.makedirs(save_dataset_path, exist_ok=True)

    root_image_path = os.path.join(root_dataset_path, 'clip_img')

    all_image_path_list = []
    for root, dirs, files in os.walk(root_image_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)

                png_file_path = file_path.replace('clip_img', 'matting')
                png_file_path = png_file_path.replace('.jpg', '.png')
                png_file_path = re.sub(r'(clip_)(\d{8})', r'matting_\2',
                                       png_file_path)

                file_name_prefix = file_path.split('.jpg')[0]
                png_file_name = file_name_prefix + '.png'

                if os.path.exists(file_path) and os.path.exists(png_file_path):
                    all_image_path_list.append(
                        [file, file_path, png_file_name, png_file_path])

    print(f"1111", len(all_image_path_list), all_image_path_list[0])

    # for per_image_name, per_image_path, png_file_name, png_file_path in tqdm(
    #         all_image_path_list):
    #     if not os.path.exists(per_image_path) or not os.path.exists(
    #             png_file_path):
    #         continue

    #     per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
    #                              cv2.IMREAD_COLOR)
    #     per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    #     per_png = cv2.imdecode(np.fromfile(png_file_path, dtype=np.uint8),
    #                            cv2.IMREAD_UNCHANGED)
    #     _, _, _, alpha = cv2.split(per_png)

    #     per_alpha_h, per_alpha_w = alpha.shape[0], alpha.shape[1]

    #     assert per_image_h == per_alpha_h and per_image_w == per_alpha_w

    #     per_image_prefix = per_image_name.split('.')[0]

    #     save_image_name = per_image_prefix + '.jpg'
    #     save_image_path = os.path.join(save_dataset_path, save_image_name)
    #     cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    #     save_mask_name = per_image_prefix + '.png'
    #     save_mask_path = os.path.join(save_dataset_path, save_mask_name)
    #     cv2.imencode('.png', alpha)[1].tofile(save_mask_path)

    with Pool(processes=16) as pool:
        process_func = functools.partial(process_single_image,
                                         save_dataset_path=save_dataset_path)

        # 使用tqdm显示进度
        list(
            tqdm(pool.imap(process_func, all_image_path_list),
                 total=len(all_image_path_list)))


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/human_matting_dataset/matting_human_half/'
    save_dataset_path = r'/root/autodl-tmp/human_matting_dataset/matting_human_half/train/'
    process_data(root_dataset_path, save_dataset_path)
