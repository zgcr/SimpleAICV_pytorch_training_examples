import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import face_parsing_dataset_path

from simpleAICV.face_parsing import models
from simpleAICV.face_parsing import losses
from simpleAICV.face_parsing.datasets.face_parsing_dataset import FaceParsingDataset, FaceSynthetics_19_CLASSES
from simpleAICV.face_parsing.common import YoloStyleResize, Normalize, FaceParsingCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'convformerm36_pfan_face_parsing'
    # 包含背景类
    num_classes = 19
    input_image_size = [512, 512]

    model = models.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path,
                    model,
                    loading_new_input_size_position_encoding_weight=False)

    test_criterion = losses.__dict__['CELoss'](**{})

    val_dataset_name_list = [
        [
            'FaceSynthetics',
        ],
    ]
    val_dataset_list = []
    for per_sub_dataset_list in val_dataset_name_list:
        per_sub_val_dataset = FaceParsingDataset(
            face_parsing_dataset_path,
            set_name_list=per_sub_dataset_list,
            set_type='train',
            cats=FaceSynthetics_19_CLASSES,
            transform=transforms.Compose([
                YoloStyleResize(resize=input_image_size[0]),
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)

    val_collater = FaceParsingCollater(resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 8
