import os
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm

from multiprocessing import Pool


def process_single_image(args):
    """处理单个图像的函数，用于多进程调用"""
    file_name, file_path, png_label_name, png_label_path, root_dataset_path, save_dataset_path = args

    relative_path = os.path.relpath(os.path.dirname(file_path),
                                    root_dataset_path)

    # 在目标目录中创建相同的目录结构
    target_dir = os.path.join(save_dataset_path, relative_path)
    os.makedirs(target_dir, exist_ok=True)

    # 目标文件路径
    target_jpg_path = os.path.join(target_dir, file_name)
    target_png_path = os.path.join(target_dir, png_label_name)

    per_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)

    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    per_mask = np.array(Image.open(png_label_path).convert('L'),
                        dtype=np.uint8)

    per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]

    assert per_image_h == per_mask_h and per_image_w == per_mask_w

    factor = 1080 / max(per_image_h, per_image_w)
    resize_h, resize_w = int(round(per_image_h * factor)), int(
        round(per_image_w * factor))
    per_image = cv2.resize(per_image, (resize_w, resize_h))
    per_mask = cv2.resize(per_mask, (resize_w, resize_h))

    cv2.imencode('.jpg', per_image)[1].tofile(target_jpg_path)
    cv2.imencode('.png', per_mask)[1].tofile(target_png_path)


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

    # for file_name, file_path, png_label_name, png_label_path in tqdm(
    #         all_image_path_list):
    #     relative_path = os.path.relpath(os.path.dirname(file_path),
    #                                     root_dataset_path)

    #     # 在目标目录中创建相同的目录结构
    #     target_dir = os.path.join(save_dataset_path, relative_path)
    #     os.makedirs(target_dir, exist_ok=True)

    #     # 目标文件路径
    #     target_jpg_path = os.path.join(target_dir, file_name)
    #     target_png_path = os.path.join(target_dir, png_label_name)

    #     per_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
    #                              cv2.IMREAD_COLOR)

    #     per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    #     per_mask = np.array(Image.open(png_label_path).convert('L'),
    #                         dtype=np.uint8)

    #     per_mask_h, per_mask_w = per_mask.shape[0], per_mask.shape[1]

    #     assert per_image_h == per_mask_h and per_image_w == per_mask_w

    #     factor = 1080 / max(per_image_h, per_image_w)
    #     resize_h, resize_w = int(round(per_image_h * factor)), int(
    #         round(per_image_w * factor))
    #     per_image = cv2.resize(per_image, (resize_w, resize_h))
    #     per_mask = cv2.resize(per_mask, (resize_w, resize_h))

    #     cv2.imencode('.jpg', per_image)[1].tofile(target_jpg_path)
    #     cv2.imencode('.png', per_mask)[1].tofile(target_png_path)

    # 准备多进程参数
    pool_args = [(file_name, file_path, png_label_name, png_label_path,
                  root_dataset_path, save_dataset_path)
                 for file_name, file_path, png_label_name, png_label_path in
                 all_image_path_list]

    # 使用多进程处理
    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_single_image, pool_args),
                 total=len(pool_args)))


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/salient_object_detection_dataset/AIM500'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AIM500'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/salient_object_detection_dataset/AM2K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/AM2K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/salient_object_detection_dataset/DIS5K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/DIS5K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/salient_object_detection_dataset/HRS10K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRS10K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/salient_object_detection_dataset/HRSOD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/HRSOD'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/salient_object_detection_dataset/MAGICK'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/MAGICK'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/salient_object_detection_dataset/UHRSD'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/UHRSD'
    preprocess_image(root_dataset_path, save_dataset_path)

    #######################################################################################
    root_dataset_path = r'/root/autodl-tmp/human_matting_dataset/Deep_Automatic_Portrait_Matting'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/Deep_Automatic_Portrait_Matting'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/human_matting_dataset/matting_human_half'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/matting_human_half'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/human_matting_dataset/P3M-500-NP'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-NP'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/human_matting_dataset/P3M-500-P'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M-500-P'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/human_matting_dataset/P3M10K'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/P3M10K'
    preprocess_image(root_dataset_path, save_dataset_path)

    root_dataset_path = r'/root/autodl-tmp/human_matting_dataset/RealWorldPortrait636'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/RealWorldPortrait636'
    preprocess_image(root_dataset_path, save_dataset_path)
