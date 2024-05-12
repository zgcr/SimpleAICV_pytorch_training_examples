import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import math
import numpy as np
import cv2
import os
from collections import OrderedDict
import json
import shutil
import collections
import pyclipper
from shapely.geometry import Polygon


def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def order_points(src):
    rect = np.zeros((4, 2), dtype="float32")
    src = sorted(src, key=lambda x: x[0])
    if src[0][1] <= src[1][1]:
        rect[0] = src[0]
        rect[3] = src[1]
    else:
        rect[0] = src[1]
        rect[3] = src[0]

    if src[2][1] <= src[3][1]:
        rect[1] = src[2]
        rect[2] = src[3]
    else:
        rect[1] = src[3]
        rect[2] = src[2]

    return rect


def get_min_area_point(polygon):
    polygon = np.array(polygon).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(polygon)
    points = cv2.boxPoints(rect)
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    return x_coords, y_coords, rect


def get_text_line_image(x_coords, y_coords, image):
    src = [[x, y] for x, y in zip(x_coords, y_coords)]
    src = order_points(np.array(src, dtype="float32"))

    height1 = dist([src[0][0], src[0][1]], [src[3][0], src[3][1]])
    height2 = dist([src[1][0], src[1][1]], [src[2][0], src[2][1]])
    text_line_height = int((height1 + height2) / 2.0)

    if height1 < 2 or height2 < 2:
        return None

    width1 = dist([src[0][0], src[0][1]], [src[1][0], src[1][1]])
    width2 = dist([src[2][0], src[2][1]], [src[3][0], src[3][1]])
    text_line_width = int((width1 + width2) / 2.0)

    if width1 < 2 or width2 < 2:
        return None

    dst = [[0, 0], [text_line_width - 1, 0],
           [text_line_width - 1, text_line_height - 1],
           [0, text_line_height - 1]]
    M = cv2.getPerspectiveTransform(np.array(src, dtype="float32"),
                                    np.array(dst, dtype="float32"))

    text_line_image = cv2.warpPerspective(image, M,
                                          (text_line_width, text_line_height))

    return text_line_image


def get_half_angle_of_symbols(char):
    inside_code = ord(char)
    if inside_code == 12288:  # 全角空格直接转换
        inside_code = 32
    elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
        inside_code -= 65248

    return chr(inside_code)


def shrink_polygon_pyclipper(polygon_box, shrink_ratio=0.8):
    polygon_shape = Polygon(polygon_box)
    distance = (polygon_shape.area *
                (1 - np.power(shrink_ratio, 2)) / polygon_shape.length) - 1
    subject = [tuple(l) for l in polygon_box]
    list_box = [list(coord) for coord in subject]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(list_box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked_polygon_box = padding.Execute(-distance)

    return shrinked_polygon_box
