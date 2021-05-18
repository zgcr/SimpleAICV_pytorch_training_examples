import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

from tools.path import CIFAR100_path

from simpleAICV.classification import backbones
from simpleAICV.classification import losses

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100


class config:
    val_dataset_path = CIFAR100_path

    network = 'resnet18cifar'
    pretrained = False
    num_classes = 100
    input_image_size = 32

    model = backbones.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })
    criterion = losses.__dict__['CELoss']()

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    val_dataset_init = {
        "root": val_dataset_path,
        "train": False,
        "download": True,
        "transform": val_transform
    }

    val_dataset = CIFAR100(**val_dataset_init)

    distributed = True
    seed = 0
    batch_size = 128
    num_workers = 8
    trained_model_path = ''
