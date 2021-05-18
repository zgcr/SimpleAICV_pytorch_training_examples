import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path
from simpleAICV.detection.models import centernet
from simpleAICV.detection.decode import CenterNetDecoder
from simpleAICV.datasets.cocodataset import CocoDetection
from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'COCO'
    network = 'resnet18_centernet'
    pretrained = False
    num_classes = 80
    input_image_size = 512

    model = centernet.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })

    decoder = CenterNetDecoder()

    val_dataset = CocoDetection(COCO2017_path,
                                set_name='val2017',
                                transform=transforms.Compose([
                                    Normalize(),
                                    YoloStyleResize(
                                        resize=input_image_size,
                                        multi_scale=False,
                                        multi_scale_range=[0.6, 1.4]),
                                ]))

    seed = 0
    batch_size = 64
    num_workers = 16
    trained_model_path = ''
