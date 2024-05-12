import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import text_detection_dataset_path

from simpleAICV.text_detection import models
from simpleAICV.text_detection import losses
from simpleAICV.text_detection import decode
from simpleAICV.text_detection.datasets.text_detection_dataset import TextDetection
from simpleAICV.text_detection.common import RandomRotate, MainDirectionRandomRotate, Resize, Normalize, DBNetTextDetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'VANDBNet'
    input_image_size = [960, 960]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/van_finetune_on_imagenet1k_from_official_pretrain/van_b1-acc80.956.pth'
    model = models.__dict__[network](**{
        'backbone_type': 'van_b1_backbone',
        'backbone_pretrained_path': backbone_pretrained_path,
        'inter_planes': 256,
        'k': 50,
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['DBNetLoss'](**{
        'probability_weight': 1,
        'threshold_weight': 5,
        'binary_weight': 1,
        'negative_ratio': 3,
        'k': 50,
    })
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

    train_dataset = TextDetection(text_detection_dataset_path,
                                  set_name=[
                                      'ICDAR2017RCTW_text_detection',
                                      'ICDAR2019ART_text_detection',
                                      'ICDAR2019LSVT_text_detection',
                                      'ICDAR2019MLT_text_detection',
                                  ],
                                  set_type='train',
                                  transform=transforms.Compose([
                                      MainDirectionRandomRotate(
                                          angle=[0, 90, 180, 270],
                                          prob=[0.55, 0.15, 0.15, 0.15]),
                                      RandomRotate(angle=[-45, 45], prob=0.3),
                                      Resize(resize=input_image_size[0]),
                                      Normalize(),
                                  ]))

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

    train_collater = DBNetTextDetectionCollater(resize=input_image_size[0],
                                                min_box_size=3,
                                                min_max_threshold=[0.3, 0.7],
                                                shrink_ratio=0.6)
    test_collater = DBNetTextDetectionCollater(resize=input_image_size[0],
                                               min_box_size=3,
                                               min_max_threshold=[0.3, 0.7],
                                               shrink_ratio=0.6)

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8
    accumulation_steps = 4

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-3,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
            'min_lr': 1e-6,
        },
    )

    epochs = 100
    print_interval = 100
    save_interval = 20

    eval_epoch = []
    for i in range(epochs):
        if i % 20 == 0:
            eval_epoch.append(i)
    save_model_metric = 'f1'

    sync_bn = False
    use_amp = False
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
        'backend': 'aot_eager',
    }

    use_ema_model = False
    ema_model_decay = 0.9999

    # 计算precision若insection_pred_ious大于precision_iou_threshold认为两个框重叠
    precision_iou_threshold = 0.5
    # 计算recall若insection_target_ious大于recall_iou_threshold认为两个框重叠
    recall_iou_threshold = 0.5
    # 一对多/多对一的惩罚系数,[0,1]区间
    punish_factor = 1.0
    # 一对多和多对一时如果有match_count_threshold个框与一个框的iou都大于阈值，则执行一对多或多对一逻辑
    match_count_threshold = 2
