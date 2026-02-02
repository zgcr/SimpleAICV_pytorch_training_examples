import os
import shutil

from tqdm import tqdm

from multiprocessing import Pool


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
                per_png_label_name = file_name.split('.')[0] + '.json'
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

    # for item in tqdm(all_image_path_list):
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
        list(
            tqdm(pool.imap(copy_single_image, all_image_path_list),
                 total=len(all_image_path_list)))


if __name__ == '__main__':
    for i in range(0, 50):
        file_index = f"{i:06d}"

        root_dataset_path = f'/root/autodl-tmp/interactive_segmentation_dataset/sa_{file_index}_filter_part_object'
        save_dataset_path = f'/root/autodl-tmp/interactive_segmentation_dataset_new2/sa_{file_index}_filter_part_object'

        print(f"\n{'=' * 50}")
        print(f"🚀 开始处理文件：{root_dataset_path}")
        print(f"{'=' * 50}")

        preprocess_image(root_dataset_path, save_dataset_path)
