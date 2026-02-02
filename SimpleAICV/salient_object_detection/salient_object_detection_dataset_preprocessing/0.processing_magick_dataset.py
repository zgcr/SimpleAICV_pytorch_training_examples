import collections
import cv2
import os
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
import functools


def process_single_image(args, save_dataset_path):
    per_image_name, per_image_path = args
    if not os.path.exists(per_image_path):
        return

    try:
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_UNCHANGED)
        b, g, r, a = cv2.split(per_image)
        bgr_img = cv2.merge([b, g, r])

        per_image_prefix = per_image_name.split('.')[0]

        save_image_name = per_image_prefix + '.jpg'
        save_image_path = os.path.join(save_dataset_path, save_image_name)
        cv2.imencode('.jpg', bgr_img)[1].tofile(save_image_path)

        save_mask_name = per_image_prefix + '.png'
        save_mask_path = os.path.join(save_dataset_path, save_mask_name)
        cv2.imencode('.png', a)[1].tofile(save_mask_path)
    except Exception as e:
        print(f"Error processing {per_image_path}: {str(e)}")


def process_data(root_dataset_path, save_dataset_path):
    os.makedirs(save_dataset_path, exist_ok=True)

    all_image_path_list = []

    for root, dirs, files in os.walk(root_dataset_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                all_image_path_list.append([file, file_path])

    print(f"1111", len(all_image_path_list), all_image_path_list[0])

    # for per_image_name, per_image_path in tqdm(all_image_path_list):
    #     if not os.path.exists(per_image_path):
    #         continue
    #     per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
    #                              cv2.IMREAD_UNCHANGED)
    #     b, g, r, a = cv2.split(per_image)
    #     bgr_img = cv2.merge([b, g, r])

    #     per_image_prefix = per_image_name.split('.')[0]

    #     save_image_name = per_image_prefix + '.jpg'
    #     save_image_path = os.path.join(save_dataset_path, save_image_name)
    #     cv2.imencode('.jpg', bgr_img)[1].tofile(save_image_path)

    #     save_mask_name = per_image_prefix + '.png'
    #     save_mask_path = os.path.join(save_dataset_path, save_mask_name)
    #     cv2.imencode('.png', a)[1].tofile(save_mask_path)

    task_args = [(name, path) for name, path in all_image_path_list]
    process_func = functools.partial(process_single_image,
                                     save_dataset_path=save_dataset_path)

    # 使用多进程池处理
    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(process_func, task_args),
                 total=len(task_args),
                 desc="Processing images"))


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/MAGICK/images/'
    save_dataset_path = r'/root/autodl-tmp/MAGICK_new/train/'
    process_data(root_dataset_path, save_dataset_path)
