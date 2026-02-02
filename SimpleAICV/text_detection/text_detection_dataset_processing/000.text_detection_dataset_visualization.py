import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import collections
import json
import os
import shutil
import math, sys
import numpy as np
import random
import pyclipper
import glob

from tqdm import tqdm

from text_detection_processing_utils import get_half_angle_of_symbols, shrink_polygon_pyclipper
from shapely.geometry import Polygon

half_full_dict = {
    "，": ",",
    "；": ";",
    "：": ":",
    "？": "?",
    "（": "(",
    "）": ")",
    "！": "!",
}


def generate_standard_format_txt_for_each_image(root_image_path,
                                                root_label_path,
                                                save_image_path):
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    all_image_name_path_list = []
    for per_image_name in os.listdir(root_image_path):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(root_image_path, per_image_name)
            if not os.path.exists(per_image_path):
                continue
        all_image_name_path_list.append([per_image_name, per_image_path])

    print('1111', len(all_image_name_path_list), all_image_name_path_list[0])

    root_label = json.load(open(root_label_path, 'r', encoding='UTF-8'))

    print('2222', len(root_label))

    for key, value in root_label.items():
        print(key, value)
        break

    for per_image_name, per_image_path in tqdm(all_image_name_path_list):
        if not os.path.exists(per_image_path):
            continue

        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        per_image_label = root_label[per_image_name]

        for per_label in per_image_label:
            per_label_box = per_label['points']
            per_label_box_type = per_label['ignore']

            per_label_box = np.array(per_label_box, np.int32)
            per_label_box = per_label_box.reshape((-1, 1, 2))
            if not per_label_box_type:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.polylines(per_image,
                          pts=[per_label_box],
                          isClosed=True,
                          color=color,
                          thickness=3)

        keep_image_name = f'{per_image_name.split(".")[0]}.jpg'
        keep_image_path = os.path.join(save_image_path, keep_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(keep_image_path)


if __name__ == '__main__':
    # root_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2017RCTW_text_detection\test'
    # root_label_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2017RCTW_text_detection\ICDAR2017RCTW_text_detection_test.json'
    # save_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2017RCTW_text_detection\test_可视化'
    # generate_standard_format_txt_for_each_image(root_image_path,
    #                                             root_label_path,
    #                                             save_image_path)

    # root_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2017RCTW_text_detection\train'
    # root_label_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2017RCTW_text_detection\ICDAR2017RCTW_text_detection_train.json'
    # save_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2017RCTW_text_detection\train_可视化'
    # generate_standard_format_txt_for_each_image(root_image_path,
    #                                             root_label_path,
    #                                             save_image_path)

    # root_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ART_text_detection\test'
    # root_label_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ART_text_detection\ICDAR2019ART_text_detection_test.json'
    # save_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ART_text_detection\test_可视化'
    # generate_standard_format_txt_for_each_image(root_image_path,
    #                                             root_label_path,
    #                                             save_image_path)

    # root_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ART_text_detection\train'
    # root_label_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ART_text_detection\ICDAR2019ART_text_detection_train.json'
    # save_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ART_text_detection\train_可视化'
    # generate_standard_format_txt_for_each_image(root_image_path,
    #                                             root_label_path,
    #                                             save_image_path)

    # root_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019LSVT_text_detection\test'
    # root_label_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019LSVT_text_detection\ICDAR2019LSVT_text_detection_test.json'
    # save_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019LSVT_text_detection\test_可视化'
    # generate_standard_format_txt_for_each_image(root_image_path,
    #                                             root_label_path,
    #                                             save_image_path)

    # root_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019MLT_text_detection\train'
    # root_label_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019MLT_text_detection\ICDAR2019MLT_text_detection_train.json'
    # save_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019MLT_text_detection\train_可视化'
    # generate_standard_format_txt_for_each_image(root_image_path,
    #                                             root_label_path,
    #                                             save_image_path)

    root_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ReCTS_text_detection\test'
    root_label_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ReCTS_text_detection\ICDAR2019ReCTS_text_detection_test.json'
    save_image_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019ReCTS_text_detection\test_可视化'
    generate_standard_format_txt_for_each_image(root_image_path,
                                                root_label_path,
                                                save_image_path)
