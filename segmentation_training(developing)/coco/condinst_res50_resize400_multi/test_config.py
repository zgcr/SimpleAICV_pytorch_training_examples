import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path
from simpleAICV.segmentation.models import condinst
from simpleAICV.segmentation.decode import CondInstDecoder
from simpleAICV.datasets.cocodataset import CocoSegmentation
from simpleAICV.segmentation.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'COCO'
    network = 'resnet50_condinst'
    pretrained = False
    num_classes = 80
    input_image_size = 400

    model = condinst.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })

    decoder = CondInstDecoder()

    val_dataset = CocoSegmentation(COCO2017_path,
                                   set_name='val2017',
                                   transform=transforms.Compose([
                                       Normalize(),
                                       RetinaStyleResize(
                                           resize=input_image_size,
                                           multi_scale=False,
                                           multi_scale_range=[0.8, 1.0]),
                                   ]))

    seed = 0
    # when testing,using nn.DataParallel mode,batch_size is total size
    batch_size = 8
    num_workers = 4
    trained_model_path = ''
