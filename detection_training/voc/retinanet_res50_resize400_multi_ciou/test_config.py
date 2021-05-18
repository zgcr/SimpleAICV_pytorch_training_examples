import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import VOCdataset_path
from simpleAICV.detection.models import retinanet
from simpleAICV.detection.decode import RetinaDecoder
from simpleAICV.datasets.vocdataset import VocDetection
from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize

import torchvision.transforms as transforms


class config:
    dataset_name = 'VOC'
    network = 'resnet50_retinanet'
    pretrained = False
    num_classes = 20
    input_image_size = 400

    model = retinanet.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })
    decoder = RetinaDecoder()

    val_dataset = VocDetection(root_dir=VOCdataset_path,
                               image_sets=[('2007', 'test')],
                               transform=transforms.Compose([
                                   Normalize(),
                                   RetinaStyleResize(
                                       resize=input_image_size,
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
                               ]),
                               keep_difficult=False)

    seed = 0
    batch_size = 64
    num_workers = 16
    eval_iou_threshold = 0.5
    trained_model_path = 'detection_training/voc/retinanet_res50_resize400_multi_ciou/checkpoints/resnet50_retinanet-epoch12-mAP0.7711355083699912.pth'
