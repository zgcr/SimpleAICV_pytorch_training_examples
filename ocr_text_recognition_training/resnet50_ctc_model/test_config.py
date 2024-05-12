import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import text_recognition_dataset_path

from simpleAICV.text_recognition.models import CTCModel
from simpleAICV.text_recognition import losses
from simpleAICV.text_recognition.char_sets.final_char_table import final_char_table
from simpleAICV.text_recognition.char_sets.num_and_alpha_char_table import num_char_table, alpha_char_table
from simpleAICV.text_recognition.char_sets.common_standard_chinese_char_table import common_standard_chinese_char_first_table, common_standard_chinese_char_second_table, common_standard_chinese_char_third_table
from simpleAICV.text_recognition.datasets.text_recognition_dataset import CNENTextRecognition
from simpleAICV.text_recognition.common import RandomScale, RandomGaussianBlur, RandomBrightness, RandomRotate, Distort, Stretch, Perspective, Normalize, KeepRatioResizeTextRecognitionCollater, CTCTextLabelConverter, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'CTCModel'
    resize_h = 32
    str_max_length = 80

    all_char_table = final_char_table
    num_char_table = num_char_table
    alpha_char_table = alpha_char_table
    common_standard_chinese_char_first_table = common_standard_chinese_char_first_table
    common_standard_chinese_char_second_table = common_standard_chinese_char_second_table
    common_standard_chinese_char_third_table = common_standard_chinese_char_third_table

    # please make sure your converter type is the same as 'predictor'
    converter = CTCTextLabelConverter(chars_set_list=final_char_table,
                                      str_max_length=str_max_length,
                                      garbage_char='㍿')
    # all char + '[CTCblank]' = 12111 + 1 = 12112
    num_classes = converter.num_classes

    # load backbone pretrained model or not
    backbone_pretrained_path = ''
    model_config = {
        'backbone': {
            'name': 'resnet50backbone',
            'param': {
                'inplanes': 1,
                'pretrained_path': backbone_pretrained_path,
            }
        },
        'encoder': {
            'name': 'BiLSTMEncoder',
            'param': {},
        },
        'predictor': {
            'name': 'CTCEnhancePredictor',
            'param': {
                'hidden_planes': 512,
                'num_classes': num_classes + 1,
            }
        },
    }

    model = CTCModel(model_config)

    # load total pretrained model or not
    trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/ocr_text_recognition_training/resnet50_ctc_model/checkpoints/epoch_50.pth'
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['CTCLoss'](**{
        "blank_index": num_classes - 1,
        "use_focal_weight": False,
        "gamma": 2.0,
    })

    # 完整数据集必须在list中第0个位置
    val_dataset_name_list = [
        [
            'aistudio_baidu_street',
            'chinese_dataset',
            'synthetic_chinese_string_dataset_testsubset',
            'meta_self_learning_car',
            'meta_self_learning_document_testsubset',
            'meta_self_learning_hand',
            'meta_self_learning_street',
            'meta_self_learning_syn',
        ],
    ]
    val_dataset_list = []
    for per_sub_val_dataset_name in val_dataset_name_list:
        per_sub_val_dataset = CNENTextRecognition(
            text_recognition_dataset_path,
            set_name=per_sub_val_dataset_name,
            set_type='test',
            str_max_length=str_max_length,
            transform=transforms.Compose([
                Normalize(),
            ]))
        val_dataset_list.append(per_sub_val_dataset)

    test_collater = KeepRatioResizeTextRecognitionCollater(resize_h=resize_h)

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 8
