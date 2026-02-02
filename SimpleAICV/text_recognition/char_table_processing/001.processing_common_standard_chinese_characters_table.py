import cv2
import collections
import json
import math
import os
import shutil
import numpy as np
import random

from tqdm import tqdm

half_full_dict = {
    "，": ",",
    "；": ";",
    "：": ":",
    "？": "?",
    "（": "(",
    "）": ")",
    "！": "!",
}


def convert_table_from_txt_to_py(root_txt_path):
    table_list = []
    with open(root_txt_path, 'r', encoding='utf-8') as f:
        all_chars = f.readlines()
        for per_char in all_chars:
            table_list.append(per_char.strip())

    print(len(table_list))
    print(table_list)


if __name__ == '__main__':
    root_txt_path = r'D:\BaiduNetdiskDownload\text_dataset\common_standard_chinese_characters_table\level-1.txt'
    convert_table_from_txt_to_py(root_txt_path)

    root_txt_path = r'D:\BaiduNetdiskDownload\text_dataset\common_standard_chinese_characters_table\level-2.txt'
    convert_table_from_txt_to_py(root_txt_path)

    root_txt_path = r'D:\BaiduNetdiskDownload\text_dataset\common_standard_chinese_characters_table\level-3.txt'
    convert_table_from_txt_to_py(root_txt_path)
