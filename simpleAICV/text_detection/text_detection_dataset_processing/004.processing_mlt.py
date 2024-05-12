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


def generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path,
                                                train_ratio):
    save_dataset_name = save_dataset_path.split("\\")[-1]

    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    root_dataset_image_path = os.path.join(root_dataset_path, 'train_images')
    root_dataset_label_path = os.path.join(root_dataset_path, 'train_gts')

    all_image_label_name_path_list = []
    for per_image_name in os.listdir(root_dataset_image_path):
        per_image_name_prefix = per_image_name.split('.')[0]
        per_label_name = per_image_name_prefix + '.txt'
        per_image_path = os.path.join(root_dataset_image_path, per_image_name)
        per_label_path = os.path.join(root_dataset_label_path, per_label_name)
        if not os.path.exists(per_image_path) or not os.path.exists(
                per_label_path):
            continue
        all_image_label_name_path_list.append(
            [per_image_name, per_label_name, per_image_path, per_label_path])

    print('1111', len(all_image_label_name_path_list),
          all_image_label_name_path_list[0])

    dataset_image_name_list = set()
    dataset_image_path_dict = collections.OrderedDict()
    dataset_label_dict = collections.OrderedDict()
    for per_image_name, per_label_name, per_image_path, per_label_path in tqdm(
            all_image_label_name_path_list):
        if not os.path.exists(per_image_path) or not os.path.exists(
                per_label_path):
            continue

        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        if per_image is None:
            continue

        if len(per_image.shape) != 3:
            continue

        image_h, image_w, _ = per_image.shape

        if image_h < 100 or image_w < 100:
            continue

        with open(per_label_path, 'r', encoding='UTF-8') as f:
            per_image_origin_labels = f.readlines()

        illegal_language = False
        per_image_labels = []
        for per_label_line in per_image_origin_labels:
            per_label_line = per_label_line.rstrip('\n')
            per_label_coords = per_label_line.split(',')[0:8]
            per_label_coords = list(map(lambda x: int(x), per_label_coords))
            per_label_final_coords = [
                [per_label_coords[0], per_label_coords[1]],
                [per_label_coords[2], per_label_coords[3]],
                [per_label_coords[4], per_label_coords[5]],
                [per_label_coords[6], per_label_coords[7]],
            ]
            # language:{'Korean', 'Arabic', 'Japanese', 'Chinese', 'Latin', 'Mixed', 'Symbols', 'None', 'Bangla', 'Hindi'}
            per_label_language = per_label_line.split(',')[8]

            if per_label_language not in ['Chinese', 'Latin']:
                illegal_language = True
                break

            per_label_char = per_label_line.split(',')[9:]
            per_label_char = ','.join(per_label_char)[1:-1]
            per_label_char = per_label_char.replace(" ", "")

            per_label_final_char = ""
            for per_char in per_label_char:
                per_half_angle_char = get_half_angle_of_symbols(per_char)
                if per_half_angle_char in half_full_dict.keys():
                    per_half_angle_char = half_full_dict[per_half_angle_char]
                per_label_final_char += per_half_angle_char

            per_label_final_char = per_label_final_char.replace('###', '㍿')
            per_label_final_char = per_label_final_char.replace('#', '㍿')
            per_image_labels.append(
                [per_label_final_coords, per_label_final_char])

        if illegal_language:
            continue

        resize = 1920
        factor = resize / max(image_h, image_w)

        resize_h, resize_w = math.ceil(image_h * factor), math.ceil(image_w *
                                                                    factor)
        per_image = cv2.resize(per_image, (resize_w, resize_h))

        illegal_label_flag = False
        per_image_resized_image_label = []
        for per_box, per_label in per_image_labels:
            per_box = np.array(per_box) * factor
            per_box = per_box.tolist()

            if per_label == "" or per_label is None:
                illegal_label_flag = True
                break

            per_image_resized_image_label.append({
                'points': per_box,
                'label': per_label,
            })

        if illegal_label_flag:
            continue

        per_image_labels = per_image_resized_image_label
        image_h, image_w = resize_h, resize_w

        # 图像h,w的矩形和每个标注的多边形求交集区域多边形，保证每个多边形框不越界
        # 如果结果中出现多个多边形，说明标注不合法
        total_matrix = np.array([[0, 0], [image_w, 0], [image_w, image_h],
                                 [0, image_h]])
        illegal_out_borded_box = False
        filter_border_per_image_labels = []
        for per_line in per_image_labels:
            per_box = per_line['points']
            per_label = per_line['label']
            per_box = np.array(per_box)
            polygon_shape = Polygon(per_box)
            per_box = np.expand_dims(per_box, axis=0)

            pc = pyclipper.Pyclipper()
            pc.AddPath(total_matrix, pyclipper.PT_CLIP, True)
            try:
                pc.AddPaths(per_box, pyclipper.PT_SUBJECT, True)
            except Exception as e:
                print(per_image_name, '***', e)
                print('is_valid', polygon_shape.is_valid, 'is_simple',
                      polygon_shape.is_simple, 'is_empty',
                      polygon_shape.is_empty, 'is_closed',
                      polygon_shape.is_closed)
                continue
            result_box = pc.Execute(pyclipper.CT_INTERSECTION,
                                    pyclipper.PFT_EVENODD,
                                    pyclipper.PFT_EVENODD)

            if len(np.array(result_box, dtype=object).shape) != 3:
                illegal_out_borded_box = True
                break

            result_box = np.array(result_box)

            if result_box.shape[0] != 1:
                illegal_out_borded_box = True
                break

            result_box = result_box[0]
            per_box = result_box.tolist()
            per_gt_ignore = False
            if per_label == '㍿':
                per_gt_ignore = True

            filter_border_per_image_labels.append({
                'points': per_box,
                'label': per_label,
                'ignore': per_gt_ignore,
            })

        if illegal_out_borded_box:
            continue

        # 过滤一遍每个多边形框坐标，如果有超出h,w边界的框说明标注不合法
        illegal_coord = False
        for per_line in filter_border_per_image_labels:
            per_box = per_line['points']
            per_label = per_line['label']

            for per_coord in per_box:
                if per_coord[0] < 0 or per_coord[1] < 0 or per_coord[
                        0] > image_w or per_coord[1] > image_h:
                    illegal_coord = True
                    break

            if illegal_coord:
                break

        if illegal_coord:
            continue

        # 过滤所有多边形框，看有无自相交的不合法多边形，如果有说明多边形框不合法
        illegal_self_intersect_box = False
        for per_line in filter_border_per_image_labels:
            per_box = per_line['points']
            per_label = per_line['label']

            result_box = pyclipper.SimplifyPolygon(per_box)

            if len(np.array(result_box, dtype=object).shape) != 3:
                illegal_self_intersect_box = True
                break

            if np.array(result_box).shape[0] != 1 or np.array(
                    result_box).shape[2] != 2:
                illegal_self_intersect_box = True
                break

        if illegal_self_intersect_box:
            continue

        # 过滤所有多边形框，看有无面积小于等于9的多边形框，如果有说明标注不合法
        # 9即3x3个像素，目前认为当前单个字符在图像上要大于等于这个面积
        illegal_area_box = False
        for per_line in filter_border_per_image_labels:
            per_box = per_line['points']
            per_label = per_line['label']

            polygon_box = Polygon(per_box)
            polygon_box_area = polygon_box.area

            if polygon_box_area < 9:  # 3x3
                illegal_area_box = True
                break

        if illegal_area_box:
            continue

        # 和DBnet一样的收缩方法，判断收缩率为0.6时任意两个框是否不相交，如果相交，则多边形框标注不合法
        # 注意shrink_polygon_pyclipper函数中distance最后会再减一，这个表示框按0.8倍率收缩时再放大一个像素
        # 此时如果任意两个框还是不相交，则任意两个框之间距离至少有2个像素
        # 在训练代码中，即在制作probility_mask时，shrink_ratio = 0.6
        shrink_ratio = 0.6
        shrink_border_label = []
        shrink_bad_flag = False
        for per_line in filter_border_per_image_labels:
            per_box = per_line['points']
            per_label = per_line['label']

            shrinked_box = shrink_polygon_pyclipper(per_box,
                                                    shrink_ratio=shrink_ratio)
            if len(np.array(shrinked_box, dtype=object).shape
                   ) != 3 or np.array(shrinked_box).shape[0] != 1:
                shrink_bad_flag = True
                break
            per_box = np.array(shrinked_box)[0]
            shrink_border_label.append({
                'points': per_box,
                'label': per_label,
            })

        if shrink_bad_flag:
            continue

        # 判断收缩后有无两框相交的情况
        shrink_insection_flag = False
        for i in range(len(shrink_border_label)):
            for j in range(len(shrink_border_label)):
                if i == j:
                    continue

                box1 = shrink_border_label[i]['points']
                box2 = shrink_border_label[j]['points']
                box1 = np.array(box1)
                box2 = np.array(box2)
                box2 = np.expand_dims(box2, axis=0)

                pc = pyclipper.Pyclipper()
                pc.AddPath(box1, pyclipper.PT_CLIP, True)
                pc.AddPaths(box2, pyclipper.PT_SUBJECT, True)

                result = pc.Execute(pyclipper.CT_INTERSECTION,
                                    pyclipper.PFT_EVENODD,
                                    pyclipper.PFT_EVENODD)
                if result != []:
                    shrink_insection_flag = True
                    break

            if shrink_insection_flag:
                break

        if shrink_insection_flag:
            continue

        keep_image_name = f'{save_dataset_name}_{per_image_name.split(".")[0]}.jpg'
        save_image_path = os.path.join(save_dataset_path, 'total')

        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)

        keep_image_path = os.path.join(save_image_path, keep_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(keep_image_path)

        dataset_image_name_list.add(keep_image_name)
        dataset_image_path_dict[keep_image_name] = keep_image_path
        dataset_label_dict[keep_image_name] = filter_border_per_image_labels

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
    root_dataset_path = r'D:\BaiduNetdiskDownload\text_dataset\ICDAR2019MLT'
    save_dataset_path = r'D:\BaiduNetdiskDownload\text_dataset_处理后\ICDAR2019MLT_text_detection'
    train_ratio = 0.8
    generate_standard_format_txt_for_each_image(root_dataset_path,
                                                save_dataset_path, train_ratio)
