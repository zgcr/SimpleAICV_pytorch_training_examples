import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path
from simpleAICV.detection.models import yolov5
from simpleAICV.detection.decode import Yolov5Decoder
from simpleAICV.datasets.cocodataset import CocoDetection
from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'COCO'
    network = 'yolov5l'
    pretrained = False
    num_classes = 80
    input_image_size = 416

    model = yolov5.__dict__[network](**{
        'pretrained':
        pretrained,
        'anchor_sizes': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                         [59, 119], [116, 90], [156, 198], [373, 326]],
        'strides': [8, 16, 32],
        'num_classes':
        num_classes,
    })

    decoder = Yolov5Decoder()

    val_dataset = CocoDetection(COCO2017_path,
                                set_name='val2017',
                                transform=transforms.Compose([
                                    Normalize(),
                                    YoloStyleResize(
                                        resize=input_image_size,
                                        multi_scale=False,
                                        multi_scale_range=[0.5, 1.0]),
                                ]))

    seed = 0
    batch_size = 64
    num_workers = 16
    trained_model_path = ''
