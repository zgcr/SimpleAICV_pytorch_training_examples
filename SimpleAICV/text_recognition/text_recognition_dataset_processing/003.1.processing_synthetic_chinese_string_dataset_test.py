import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import collections
import imagesize
import json
import math
import os
import shutil
import numpy as np
import random

from tqdm import tqdm

from text_recognition_processing_utils import get_half_angle_of_symbols
from final_char_table import final_char_table

half_full_dict = {
    "，": ",",
    "；": ";",
    "：": ":",
    "？": "?",
    "（": "(",
    "）": ")",
    "！": "!",
}


def generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path):
    save_dataset_path = save_dataset_path + f'_testsubset'
    save_dataset_name = save_dataset_path.split("\\")[-1]

    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    root_dataset_label_char_path = os.path.join(root_dataset_path,
                                                'char_std_5990.txt')
    with open(root_dataset_label_char_path, 'r', encoding='UTF-8') as f:
        root_dataset_label_chars = f.readlines()

    root_dataset_idx_char_dict = collections.OrderedDict()
    for idx, per_char in enumerate(root_dataset_label_chars):
        per_char = per_char.rstrip('\n')
        root_dataset_idx_char_dict[idx] = per_char

    print('1111', len(root_dataset_idx_char_dict))

    root_dataset_label_dict = collections.OrderedDict()

    root_dataset_test_label_path = os.path.join(root_dataset_path,
                                                'data_test.txt')
    with open(root_dataset_test_label_path, 'r', encoding='UTF-8') as f:
        root_dataset_test_origin_labels = f.readlines()

    for per_image_origin_label in root_dataset_test_origin_labels:
        per_image_origin_label = per_image_origin_label.rstrip('\n')
        per_image_origin_label_image_name = per_image_origin_label.split(
            ' ')[0]
        per_image_origin_label_image_char = per_image_origin_label.split(
            ' ')[1:]
        per_image_origin_label_image_char = list(
            map(lambda x: root_dataset_idx_char_dict[int(x)],
                per_image_origin_label_image_char))
        per_image_origin_label_image_char = ''.join(
            per_image_origin_label_image_char)
        root_dataset_label_dict[
            per_image_origin_label_image_name] = per_image_origin_label_image_char

    print('2222', len(root_dataset_label_dict))

    for key, value in root_dataset_label_dict.items():
        print(key, value)
        break

    dataset_label_dict = collections.OrderedDict()
    for key, value in tqdm(root_dataset_label_dict.items()):
        per_image_origin_name = key
        per_image_label = value

        per_image_label_final_char = ""
        for per_char in per_image_label:
            per_half_angle_char = get_half_angle_of_symbols(per_char)
            if per_half_angle_char in half_full_dict.keys():
                per_half_angle_char = half_full_dict[per_half_angle_char]
            if per_half_angle_char not in final_char_table:
                per_half_angle_char = '㍿'
            per_image_label_final_char += per_half_angle_char

        if len(per_image_label_final_char) < 1:
            continue

        if len(per_image_label_final_char) > 80:
            continue

        origin_image_path = os.path.join(root_dataset_path, 'images',
                                         per_image_origin_name)

        if not os.path.exists(origin_image_path):
            continue

        text_image_w, text_image_h = imagesize.get(origin_image_path)

        if text_image_h < 8 or text_image_w < 8:
            continue

        save_image_dir_path = os.path.join(save_dataset_path, 'test')

        if not os.path.exists(save_image_dir_path):
            os.makedirs(save_image_dir_path)

        modified_image_name = f'{save_dataset_name}_{per_image_origin_name.split(".")[0]}.jpg'
        save_image_path = os.path.join(save_image_dir_path,
                                       modified_image_name)
        shutil.copyfile(origin_image_path, save_image_path)

        dataset_label_dict[modified_image_name] = per_image_label_final_char

    print("5555", len(dataset_label_dict))

    with open(os.path.join(save_dataset_path,
                           f'{save_dataset_name}_test.json'),
              'w',
              encoding='UTF-8') as json_f:
        json.dump(dataset_label_dict, json_f, ensure_ascii=False)


if __name__ == '__main__':
    root_dataset_path = r'D:\BaiduNetdiskDownload\text_dataset\chinese_ocr\synthetic_chinese_string_dataset'
    save_dataset_path = r'D:\BaiduNetdiskDownload\text_recognition_dataset\synthetic_chinese_string_dataset'
    generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path)
