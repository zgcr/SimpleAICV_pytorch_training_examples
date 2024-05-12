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

from char_tables.final_char_table import final_char_table

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
    print('1111', len(final_char_table))


if __name__ == '__main__':
    combine_tables()
