import os
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm

from multiprocessing import Pool


def check_image(args):
    """检查单个图片是否能被正常读取"""
    file_name, file_path = args

    try:
        file_size = os.path.getsize(file_path)
        # 小于100字节的jpg几乎不可能是有效图片
        if file_size < 100:
            print(f'{file_name}, 文件过小')
            return False, [file_name, file_path]

        try:
            with Image.open(file_path) as img:
                # 验证图片完整性，但不完全解码
                img.verify()
        except Exception as e:
            print(f'{file_name}, PIL验证失败: {str(e)}')
            return False, [file_name, file_path]

        # 使用指定方式读取图片
        per_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        # 检查是否成功读取
        if per_image is None:
            print(f'{file_name}, opencv读取失败')
            return False, [file_name, file_path]

        # 检查图片是否有有效的shape
        if per_image.shape[0] == 0 or per_image.shape[1] == 0:
            print(f'{file_name}, 图片尺寸无效')
            return False, [file_name, file_path]

        return True, [file_name, file_path]

    except Exception as e:
        print(f'{file_name}, opencv读取异常')
        return False, [file_name, file_path]


def preprocess_image(root_dataset_path, save_dataset_path):
    os.makedirs(save_dataset_path, exist_ok=True)

    all_image_path_list = []

    for root, dirs, files in os.walk(root_dataset_path):
        for file_name in files:
            if file_name.lower().endswith('.jpg'):
                file_path = os.path.join(root, file_name)

                if os.path.exists(file_path):
                    all_image_path_list.append([
                        file_name,
                        file_path,
                    ])

    print(f"1111", len(all_image_path_list), all_image_path_list[0])

    valid_count, invalid_count = 0, 0
    valid_image_path_list, invalid_image_path_list = [], []

    # 使用多进程加速处理
    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(check_image, all_image_path_list),
                 total=len(all_image_path_list)))

    # 统计结果
    for valid_flag, img_info in results:
        if valid_flag:
            valid_count += 1
            valid_image_path_list.append(img_info)
        else:
            invalid_count += 1
            invalid_image_path_list.append(img_info)

    print(len(all_image_path_list), len(valid_image_path_list),
          len(invalid_image_path_list))

    for file_name, file_path in tqdm(invalid_image_path_list):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"删除成功: {file_name}")
            else:
                print(f"文件不存在，跳过: {file_path}")
        except Exception as e:
            print(f"删除失败 {file_path}: {str(e)}")


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/objects365_2020'
    save_dataset_path = r'/root/autodl-tmp/objects365_2020'
    preprocess_image(root_dataset_path, save_dataset_path)
