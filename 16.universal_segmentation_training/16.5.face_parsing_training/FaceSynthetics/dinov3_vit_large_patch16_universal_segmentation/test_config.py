import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from tools.path import face_parsing_dataset_path

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import segmentation_decode
from SimpleAICV.universal_segmentation.datasets.face_parsing_dataset import FaceParsingDataset, FaceSynthetics_19_CLASSES
from SimpleAICV.universal_segmentation.face_parsing_common import YoloStyleResize, RandomHorizontalFlip, Normalize, FaceParsingTestCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'dinov3_vit_large_patch16_universal_segmentation'
    query_num = 100
    # num_classes has background class
    num_classes = 19
    input_image_size = [512, 512]

    model = models.__dict__[network](**{
        'image_size': input_image_size[0],
        'query_num': query_num,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/universal_segmentation_train_face_parsing_on_FaceSynthetics/dinov3_vit_large_patch16_universal_segmentation_epoch_100.pth'
    load_state_dict(trained_model_path, model)

    decoder = segmentation_decode.__dict__['UniversalSegmentationDecoder'](
        **{
            'topk': 100,
            'min_score_threshold': 0.1,
            'mask_threshold': 0.5,
            'binary_mask': True,
        }).cuda()

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

    val_collater = FaceParsingTestCollater(resize=input_image_size[0])

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 8
