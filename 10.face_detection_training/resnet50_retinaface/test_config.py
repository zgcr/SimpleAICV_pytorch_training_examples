import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import face_detection_dataset_path

from SimpleAICV.face_detection import models
from SimpleAICV.face_detection import losses
from SimpleAICV.face_detection import decode
from SimpleAICV.face_detection.datasets.face_detection_dataset import FaceDetectionDataset
from SimpleAICV.face_detection.common import YoloStyleResize, Normalize, FaceDetectionCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_retinaface'
    num_classes = 1
    input_image_size = [1024, 1024]

    model = models.__dict__[network](**{
        'backbone_pretrained_path': '',
    })

    # load total pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    val_criterion = losses.__dict__['RetinaFaceLoss'](
        **{
            'anchor_sizes': [[8, 16, 32], [32, 64, 128], [128, 256, 512]],
            'strides': [8, 16, 32],
            'alpha': 0.25,
            'gamma': 2,
            'beta': 1.0 / 9.0,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'box_loss_type': 'SmoothL1',
        })
    decoder = decode.__dict__['RetinaFaceDecoder'](
        **{
            'anchor_sizes': [[8, 16, 32], [32, 64, 128], [128, 256, 512]],
            'strides': [8, 16, 32],
            'max_object_num': 200,
            'min_score_threshold': 0.3,
            'topn': 1000,
            'nms_type': 'python_nms',
            'nms_threshold': 0.3,
        })

    # 完整数据集必须在list中第0个位置
    val_dataset_name_list = [
        [
            'wider_face',
        ],
    ]

    val_dataset_list = []
    for per_sub_dataset_list in val_dataset_name_list:
        per_sub_val_dataset = FaceDetectionDataset(
            face_detection_dataset_path,
            set_name_list=per_sub_dataset_list,
            set_type='val',
            transform=transforms.Compose([
                YoloStyleResize(resize=input_image_size[0],
                                stride=32,
                                multi_scale=False,
                                multi_scale_range=[0.8, 1.0]),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)

    val_collater = FaceDetectionCollater(resize=input_image_size[0])

    eval_type = 'VOC'
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8
