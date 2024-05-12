import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import collections
import json
import math
import numpy as np

sys.setrecursionlimit(100000)

from tqdm import tqdm

from utils import get_text_line_image, get_half_angle_of_symbols
from final_char_table import final_char_table
from process_curve_line_utils import GtToBezierBoxes, PostProcess

half_full_dict = {
    "，": ",",
    "；": ";",
    "：": ":",
    "？": "?",
    "（": "(",
    "）": ")",
    "！": "!",
}

# 后处理
postProcess = PostProcess(hard_border_threshold=None,
                          box_score_threshold=0.5,
                          min_area_size=15,
                          max_box_num=1000,
                          rectangle_similarity=0.6,
                          min_box_size=3,
                          line_text_expand_ratio=1.2,
                          curve_text_expand_ratio=1.5)


def generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path,
                                                dataset_type):
    root_dataset_name = root_dataset_path.split('\\')[-1]

    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    save_image_path = os.path.join(save_dataset_path, dataset_type)

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    origin_image_path = os.path.join(root_dataset_path, dataset_type)

    origin_label_path = os.path.join(
        root_dataset_path, f'{root_dataset_name}_{dataset_type}.json')
    with open(origin_label_path, 'r', encoding='UTF-8') as json_f:
        origin_label_dict = json.load(json_f)

    all_image_name_list = []
    for per_image_name in tqdm(os.listdir(origin_image_path)):
        per_image_path = os.path.join(origin_image_path, per_image_name)
        if not os.path.exists(per_image_path):
            continue

        per_image_label = origin_label_dict[per_image_name]

        all_image_name_list.append(
            [per_image_name, per_image_path, per_image_label])

    print(len(all_image_name_list), all_image_name_list[0])

    text_count = 0
    label_dict = collections.OrderedDict()
    for per_image_name, per_image_path, per_image_label in tqdm(
            all_image_name_list):
        try:
            per_image_name_prefix = per_image_name.split(".")[0]
            per_image = cv2.imdecode(
                np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            if per_image is None:
                continue

            print('1111', per_image_name)

            origin_h, origin_w, _ = per_image.shape
            resize = 2000
            factor = resize / max(origin_h, origin_w)

            resize_h, resize_w = math.ceil(origin_h * factor), math.ceil(
                origin_w * factor)
            per_image = cv2.resize(per_image, (resize_w, resize_h))

            text_index = 0
            for per_box_annot in per_image_label:
                text_coords = per_box_annot['points']
                text_label = per_box_annot['label']
                text_ignore_flag = per_box_annot['ignore']

                if text_ignore_flag:
                    continue

                if len(text_coords) < 4:
                    continue

                if text_label is None:
                    continue

                per_text_final_label = ""
                for per_char in text_label:
                    per_half_angle_char = get_half_angle_of_symbols(per_char)
                    if per_half_angle_char in half_full_dict.keys():
                        per_half_angle_char = half_full_dict[
                            per_half_angle_char]
                    if per_half_angle_char not in final_char_table:
                        per_half_angle_char = '㍿'
                    per_text_final_label += per_half_angle_char

                per_text_final_label = per_text_final_label.replace(" ", "")

                if len(per_text_final_label) < 1:
                    continue

                if len(per_text_final_label) > 80:
                    continue

                set_flag = True
                all_str = set(per_text_final_label)
                for per_str in all_str:
                    if per_str != "㍿":
                        set_flag = False
                        break

                if set_flag:
                    continue

                x_coords, y_coords = [], []
                for per_coord in text_coords:
                    x_coords.append(math.ceil(per_coord[0] * factor))
                    y_coords.append(math.ceil(per_coord[1] * factor))

                ltrb = [
                    min(x_coords),
                    min(y_coords),
                    max(x_coords),
                    max(y_coords),
                ]

                image_h, image_w, _ = per_image.shape

                if ltrb[0] < 0 or ltrb[1] < 0 or ltrb[2] > image_w or ltrb[
                        3] > image_h:
                    continue

                if ltrb[3] - ltrb[1] < 1 or ltrb[2] - ltrb[0] < 1:
                    continue

                text_coords = np.array(text_coords, np.int32)
                text_coords = np.array(text_coords * factor, np.int32)
                text_coords = np.array(text_coords,
                                       dtype=np.float32).astype(np.int32)

                keep_image_name = f'{per_image_name_prefix}_text_line_{text_index}.jpg'
                keep_image_path = os.path.join(save_image_path,
                                               keep_image_name)

                if len(x_coords) <= 3:
                    continue
                elif len(x_coords) == 4:
                    text_image = get_text_line_image(x_coords, y_coords,
                                                     per_image)
                else:
                    text_image = GtToBezierBoxes(per_image, text_coords,
                                                 postProcess, keep_image_name)

                if text_image is None:
                    continue

                text_image_h, text_image_w = text_image.shape[
                    0], text_image.shape[1]

                if text_image_h < 8 or text_image_w < 8:
                    continue

                cv2.imencode('.jpg', text_image)[1].tofile(keep_image_path)

                label_dict[keep_image_name] = per_text_final_label

                text_index += 1
                text_count += 1
        except:
            continue

    print("2222", len(label_dict), text_count)

    save_dataset_name = save_dataset_path.split('\\')[-1]
    with open(os.path.join(save_dataset_path,
                           f'{save_dataset_name}_{dataset_type}.json'),
              'w',
              encoding='UTF-8') as json_f:
        json.dump(label_dict, json_f, ensure_ascii=False)


if __name__ == '__main__':
    root_dataset_path = r'D:\BaiduNetdiskDownload\text_detection_dataset\ICDAR2019LSVT_text_detection'
    save_dataset_path = r'D:\BaiduNetdiskDownload\text_recognition_dataset_curve_line\ICDAR2019LSVT_text_recognition'
    dataset_type = 'train'
    generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path,
                                                dataset_type)

    root_dataset_path = r'D:\BaiduNetdiskDownload\text_detection_dataset\ICDAR2019LSVT_text_detection'
    save_dataset_path = r'D:\BaiduNetdiskDownload\text_recognition_dataset_curve_line\ICDAR2019LSVT_text_recognition'
    dataset_type = 'test'
    generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path,
                                                dataset_type)
