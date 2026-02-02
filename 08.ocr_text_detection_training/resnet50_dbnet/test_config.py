import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import text_detection_dataset_path

from SimpleAICV.text_detection import models
from SimpleAICV.text_detection import losses
from SimpleAICV.text_detection import decode
from SimpleAICV.text_detection.datasets.text_detection_dataset import TextDetection
from SimpleAICV.text_detection.common import RandomRotate, MainDirectionRandomRotate, Resize, Normalize, DBNetTextDetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_dbnet'
    input_image_size = [1024, 1024]

    # load backbone pretrained model or not
    backbone_pretrained_path = ''
    model = models.__dict__[network](
        **{
            'backbone_pretrained_path': backbone_pretrained_path,
        })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['DBNetLoss'](**{
        'probability_weight': 1,
        'threshold_weight': 5,
        'binary_weight': 1,
        'negative_ratio': 3,
        'k': 50,
    })

    decoder = decode.__dict__['DBNetDecoder'](**{
        'use_morph_open': False,
        'hard_border_threshold': None,
        'box_score_threshold': 0.5,
        'min_area_size': 9,
        'max_box_num': 1000,
        'rectangle_similarity': 0.6,
        'min_box_size': 3,
        'line_text_expand_ratio': 1.2,
        'curve_text_expand_ratio': 1.5,
    })

    # 完整数据集必须在list中第0个位置
    val_dataset_name_list = [
        [
            'ICDAR2017RCTW_text_detection',
            'ICDAR2019ART_text_detection',
            'ICDAR2019LSVT_text_detection',
            'ICDAR2019MLT_text_detection',
        ],
    ]
    val_dataset_list = []
    for per_sub_val_dataset_name in val_dataset_name_list:
        per_sub_val_dataset = TextDetection(
            text_detection_dataset_path,
            set_name=per_sub_val_dataset_name,
            set_type='test',
            transform=transforms.Compose([
                Resize(resize=input_image_size[0]),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)

    test_collater = DBNetTextDetectionCollater(resize=input_image_size[0],
                                               min_box_size=3,
                                               min_max_threshold=[0.3, 0.7],
                                               shrink_ratio=0.6)

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8

    # 计算precision若insection_pred_ious大于precision_iou_threshold认为两个框重叠
    precision_iou_threshold = 0.5
    # 计算recall若insection_target_ious大于recall_iou_threshold认为两个框重叠
    recall_iou_threshold = 0.5
    # 一对多/多对一的惩罚系数,[0,1]区间
    punish_factor = 1.0
    # 一对多和多对一时如果有match_count_threshold个框与一个框的iou都大于阈值，则执行一对多或多对一逻辑
    match_count_threshold = 2
