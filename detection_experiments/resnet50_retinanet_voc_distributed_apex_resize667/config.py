import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import VOCdataset_path
from public.detection.dataset.vocdataset import VocDetection, Resize, RandomFlip, RandomCrop, RandomTranslate

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    log = './log'  # Path to save log
    checkpoint_path = './checkpoints'  # Path to store checkpoint model
    resume = './checkpoints/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    dataset_path = VOCdataset_path

    network = "resnet50_retinanet"
    pretrained = False
    num_classes = 20
    seed = 0
    input_image_size = 667

    train_dataset = VocDetection(root_dir=dataset_path,
                                 image_sets=[('2007', 'trainval'),
                                             ('2012', 'trainval')],
                                 transform=transforms.Compose([
                                     RandomFlip(flip_prob=0.5),
                                     RandomCrop(crop_prob=0.5),
                                     RandomTranslate(translate_prob=0.5),
                                     Resize(resize=input_image_size),
                                 ]),
                                 keep_difficult=False)
    val_dataset = VocDetection(root_dir=dataset_path,
                               image_sets=[('2007', 'test')],
                               transform=transforms.Compose([
                                   Resize(resize=input_image_size),
                               ]),
                               keep_difficult=False)

    epochs = 20
    per_node_batch_size = 12
    lr = 1e-4
    num_workers = 4
    print_interval = 100
    apex = True
    sync_bn = False