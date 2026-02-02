import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import collections
import json
import math
import shutil
import numpy as np
import random

from tqdm import tqdm

from char_tables.common_standard_chinese_char_table import common_standard_chinese_char_first_table, common_standard_chinese_char_second_table, common_standard_chinese_char_third_table
from char_tables.num_and_alpha_char_table import num_char_table, alpha_char_table
from char_tables.total_char_table import total_char_table

half_full_dict = {
    "，": ",",
    "；": ";",
    "：": ":",
    "？": "?",
    "（": "(",
    "）": ")",
    "！": "!",
}


def combine_tables():
    total_table_list = []
    for per_char in num_char_table:
        if per_char in half_full_dict.keys():
            per_char = half_full_dict[per_char]
        if per_char not in total_table_list:
            total_table_list.append(per_char)
    print('1111', len(total_table_list))
    for per_char in alpha_char_table:
        if per_char in half_full_dict.keys():
            per_char = half_full_dict[per_char]
        if per_char not in total_table_list:
            total_table_list.append(per_char)
    print('2222', len(total_table_list))
    for per_char in total_char_table:
        if per_char in half_full_dict.keys():
            per_char = half_full_dict[per_char]
        if per_char not in total_table_list:
            total_table_list.append(per_char)
    print('3333', len(total_table_list))
    for per_char in common_standard_chinese_char_first_table:
        if per_char in half_full_dict.keys():
            per_char = half_full_dict[per_char]
        if per_char not in total_table_list:
            total_table_list.append(per_char)
    print('4444', len(total_table_list))
    for per_char in common_standard_chinese_char_second_table:
        if per_char in half_full_dict.keys():
            per_char = half_full_dict[per_char]
        if per_char not in total_table_list:
            total_table_list.append(per_char)
    print('5555', len(total_table_list))
    for per_char in common_standard_chinese_char_third_table:
        if per_char in half_full_dict.keys():
            per_char = half_full_dict[per_char]
        if per_char not in total_table_list:
            total_table_list.append(per_char)
    print('6666', len(total_table_list))

    print(total_table_list)


if __name__ == '__main__':
    combine_tables()
