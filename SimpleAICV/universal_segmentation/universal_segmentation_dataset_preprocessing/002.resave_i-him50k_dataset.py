import os
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def process_single_image(per_image_mask_pair_path_list, save_dataset_path):
    """处理单个图像及其所有mask的函数"""
    if len(per_image_mask_pair_path_list[3]) == 0:
        return

    per_image_name_prefix, per_image_name, per_image_path, per_image_all_mask_path_list = per_image_mask_pair_path_list
    per_image_save_path = os.path.join(
        save_dataset_path, f'i_him50k_{int(per_image_name.split(".")[0]):06d}')
    os.makedirs(per_image_save_path, exist_ok=True)

    per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)

    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    factor = 1080 / max(per_image_h, per_image_w)
    resize_h, resize_w = int(round(per_image_h * factor)), int(
        round(per_image_w * factor))

    per_image = cv2.resize(per_image, (resize_w, resize_h))

    save_image_name = f'i_him50k_{int(per_image_name.split(".")[0]):06d}.jpg'
    save_image_path = os.path.join(per_image_save_path, save_image_name)
    cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    for per_mask_name, per_mask_path in per_image_all_mask_path_list:
        per_mask = np.array(Image.open(per_mask_path).convert('L'),
                            dtype=np.uint8)

        per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]

        assert per_image_h == per_mask_h and per_image_w == per_mask_w

        per_mask = cv2.resize(per_mask, (resize_w, resize_h))

        save_mask_name = f'i_him50k_{int(per_image_name.split(".")[0]):06d}_{int(per_mask_name.split(".")[0]):03d}.png'
        save_mask_path = os.path.join(per_image_save_path, save_mask_name)
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

            per_image_mask_pair_path_list = [
                per_image_name_prefix,
                per_image_name,
                per_image_path,
                [],
            ]

            for per_mask_name in sorted(os.listdir(per_image_mask_dir_path)):
                if '.png' in per_mask_name:
                    per_mask_path = os.path.join(per_image_mask_dir_path,
                                                 per_mask_name)
                    per_image_mask_pair_path_list[3].append([
                        per_mask_name,
                        per_mask_path,
                    ])

            image_mask_pair_path_list.append(per_image_mask_pair_path_list)

    print('1111', len(image_mask_pair_path_list), image_mask_pair_path_list[0])

    # for per_image_mask_pair_path_list in tqdm(
    #         image_mask_pair_path_list):
    #     if len(per_image_mask_pair_path_list[3]) == 0:
    #         continue

    #     per_image_name_prefix, per_image_name, per_image_path, per_image_all_mask_path_list = per_image_mask_pair_path_list
    #     per_image_save_path = os.path.join(save_dataset_path,
    #                                        f'i_him50k_{int(per_image_name.split('.')[0]):06d}')
    #     os.makedirs(per_image_save_path, exist_ok=True)

    #     per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
    #                              cv2.IMREAD_COLOR)

    #     per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    #     factor = 1080 / max(per_image_h, per_image_w)
    #     resize_h, resize_w = int(round(per_image_h * factor)), int(
    #         round(per_image_w * factor))

    #     per_image = cv2.resize(per_image, (resize_w, resize_h))

    #     save_image_name =f'i_him50k_{int(per_image_name.split('.')[0]):06d}.jpg'
    #     save_image_path = os.path.join(per_image_save_path, save_image_name)
    #     cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

    #     for per_mask_name, per_mask_path in per_image_all_mask_path_list:
    #         per_mask = np.array(Image.open(per_mask_path).convert('L'),
    #                             dtype=np.uint8)

    #         per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]

    #         assert per_image_h == per_mask_h and per_image_w == per_mask_w

    #         per_mask = cv2.resize(per_mask, (resize_w, resize_h))

    #         save_mask_name = f'i_him50k_{int(per_image_name.split('.')[0]):06d}_{int(per_mask_name.split('.')[0]):03d}.png'
    #         save_mask_path = os.path.join(per_image_save_path, save_mask_name)
    #         cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)

    process_func = partial(process_single_image,
                           save_dataset_path=save_dataset_path)

    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_func, image_mask_pair_path_list),
                 total=len(image_mask_pair_path_list)))


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/MaGGIe-HIM/I-HIM50K'
    save_dataset_path = r'/root/autodl-tmp/human_instance_matting_dataset/I-HIM50K'
    preprocess_image(root_dataset_path, save_dataset_path)
