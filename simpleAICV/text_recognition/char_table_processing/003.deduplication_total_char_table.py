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
    final_table_list = []
    for per_char in total_char_table:
        if per_char in half_full_dict.keys():
            per_char = half_full_dict[per_char]
        if per_char not in final_table_list:
            final_table_list.append(per_char)
    print('1111', len(final_table_list))

    print(final_table_list)


if __name__ == '__main__':
    combine_tables()
