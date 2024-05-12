import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import collections
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
                                                save_dataset_path,
                                                train_ratio):

    save_dataset_name = save_dataset_path.split("\\")[-1]

    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    root_dataset_label_path = os.path.join(root_dataset_path, 'labels.txt')
    with open(root_dataset_label_path, 'r', encoding='UTF-8') as f:
        root_dataset_origin_labels = f.readlines()

    root_dataset_label_dict = collections.OrderedDict()
    for per_image_origin_label in root_dataset_origin_labels:
        per_image_origin_label = per_image_origin_label.rstrip('\n')
        per_image_origin_label_image_name = per_image_origin_label.split(
            ' ')[0]
        per_image_origin_label_image_char = per_image_origin_label.split(
            ' ')[1:]
        per_image_origin_label_image_char = ' '.join(
            per_image_origin_label_image_char)
        root_dataset_label_dict[
            per_image_origin_label_image_name] = per_image_origin_label_image_char

    print('1111', len(root_dataset_label_dict))

    for key, value in root_dataset_label_dict.items():
        print(key, value)
        break

    root_dataset_image_path = os.path.join(root_dataset_path, 'images')

    all_image_label_name_path_list = []
    for per_image_name in tqdm(os.listdir(root_dataset_image_path)):
        per_image_name_prefix = per_image_name.split('.')[0]

        per_image_path = os.path.join(root_dataset_image_path, per_image_name)
        if not os.path.exists(per_image_path):
            continue

        per_image_label = root_dataset_label_dict[per_image_name]

        all_image_label_name_path_list.append([
            per_image_name, per_image_name_prefix, per_image_path,
            per_image_label
        ])

    print('2222', len(all_image_label_name_path_list),
          all_image_label_name_path_list[0])

    dataset_image_name_list = set()
    dataset_image_path_dict = collections.OrderedDict()
    dataset_label_dict = collections.OrderedDict()
    for per_image_name, per_image_name_prefix, per_image_path, per_image_label in tqdm(
            all_image_label_name_path_list):
        if not os.path.exists(per_image_path):
            continue

        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 -1)

        if per_image is None:
            continue

        if len(per_image.shape) != 3:
            continue

        text_image_h, text_image_w, _ = per_image.shape

        if text_image_h < 8 or text_image_w < 8:
            continue

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

        keep_image_name = f'{save_dataset_name}_{per_image_name.split(".")[0]}.jpg'
        save_image_path = os.path.join(save_dataset_path, 'total')

        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

        keep_image_path = os.path.join(save_image_path, keep_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(keep_image_path)

        dataset_image_name_list.add(keep_image_name)
        dataset_image_path_dict[keep_image_name] = keep_image_path
        dataset_label_dict[keep_image_name] = per_image_label_final_char

    dataset_image_name_list = list(dataset_image_name_list)

    print("2222", len(dataset_image_name_list), len(dataset_image_path_dict),
          len(dataset_label_dict), dataset_image_name_list[0])

    random.shuffle(dataset_image_name_list)
    train_dataset_image_name_list = dataset_image_name_list[
        0:int(len(dataset_image_name_list) * train_ratio)]
    test_dataset_image_name_list = dataset_image_name_list[
        int(len(dataset_image_name_list) * train_ratio):]
    print('3333', len(dataset_image_name_list),
          len(train_dataset_image_name_list),
          len(test_dataset_image_name_list))

    train_dataset_label_dict = collections.OrderedDict()
    save_train_image_path = os.path.join(save_dataset_path, 'train')
    if not os.path.exists(save_train_image_path):
        os.makedirs(save_train_image_path)
    for per_image_name in tqdm(train_dataset_image_name_list):
        per_image_path = dataset_image_path_dict[per_image_name]
        shutil.copy(per_image_path, save_train_image_path)

        per_image_label = dataset_label_dict[per_image_name]
        train_dataset_label_dict[per_image_name] = per_image_label
    print('4444', len(train_dataset_image_name_list),
          len(train_dataset_label_dict))

    with open(os.path.join(save_dataset_path,
                           f'{save_dataset_name}_train.json'),
              'w',
              encoding='UTF-8') as json_f:
        json.dump(train_dataset_label_dict, json_f, ensure_ascii=False)

    test_dataset_label_dict = collections.OrderedDict()
    save_test_image_path = os.path.join(save_dataset_path, 'test')
    if not os.path.exists(save_test_image_path):
        os.makedirs(save_test_image_path)
    for per_image_name in tqdm(test_dataset_image_name_list):
        per_image_path = dataset_image_path_dict[per_image_name]
        shutil.copy(per_image_path, save_test_image_path)

        per_image_label = dataset_label_dict[per_image_name]
        test_dataset_label_dict[per_image_name] = per_image_label
    print('5555', len(test_dataset_image_name_list),
          len(test_dataset_label_dict))

    with open(os.path.join(save_dataset_path,
                           f'{save_dataset_name}_test.json'),
              'w',
              encoding='UTF-8') as json_f:
        json.dump(test_dataset_label_dict, json_f, ensure_ascii=False)


if __name__ == '__main__':
    root_dataset_path = r'D:\BaiduNetdiskDownload\text_dataset\chinese_ocr\chinese_dataset'
    save_dataset_path = r'D:\BaiduNetdiskDownload\text_recognition_dataset\chinese_dataset'
    train_ratio = 0.8
    generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path, train_ratio)
