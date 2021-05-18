import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import COCO2017_path
from simpleAICV.detection.models import fcos
from simpleAICV.detection.decode import FCOSDecoder
from simpleAICV.datasets.cocodataset import CocoDetection
from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'COCO'
    network = 'resnet50_fcos'
    pretrained = False
    num_classes = 80
    input_image_size = 800

    model = fcos.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })

    decoder = FCOSDecoder()

    val_dataset = CocoDetection(COCO2017_path,
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
    batch_size = 16
    num_workers = 16
    trained_model_path = 'detection_training/coco/fcos_res50_resize800_multi_ciou/checkpoints/resnet50_fcos-epoch12-mAP0.37735844188817935.pth'
