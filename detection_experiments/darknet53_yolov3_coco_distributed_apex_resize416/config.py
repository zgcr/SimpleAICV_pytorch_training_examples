import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import COCO2017_path
from public.detection.dataset.cocodataset_mosaic_multiscale import CocoDetection, RandomFlip, RandomTranslate, PadToSquare, Resize

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    log = './log'  # Path to save log
    checkpoint_path = './checkpoints'  # Path to store checkpoint model
    resume = './checkpoints/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    train_dataset_path = os.path.join(COCO2017_path, 'images/train2017')
    val_dataset_path = os.path.join(COCO2017_path, 'images/val2017')
    dataset_annotations_path = os.path.join(COCO2017_path, 'annotations')

    network = "darknet53_yolov3"
    pretrained = False
    num_classes = 80
    seed = 0
    input_image_size = 416
    use_mosaic = False
    mosaic_center_range = [0.5, 1.5]

    use_multi_scale = False
    multi_scale_range = [0.5, 1.5]
    stride = 32

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train2017",
                                  resize=input_image_size,
                                  use_mosaic=use_mosaic,
                                  mosaic_center_range=mosaic_center_range,
                                  transform=transforms.Compose([
                                      RandomFlip(flip_prob=0.5),
                                      RandomTranslate(translate_prob=0.5),
                                      PadToSquare(resize=input_image_size,
                                                  use_mosaic=use_mosaic),
                                  ]))
    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annotations_path,
                                set="val2017",
                                resize=input_image_size,
                                use_mosaic=False,
                                mosaic_center_range=mosaic_center_range,
                                transform=transforms.Compose([
                                    PadToSquare(resize=input_image_size,
                                                use_mosaic=False),
                                ]))

    epochs = 100
    per_node_batch_size = 8
    lr = 1e-5
    num_workers = 4
    print_interval = 1
    apex = True
    sync_bn = False