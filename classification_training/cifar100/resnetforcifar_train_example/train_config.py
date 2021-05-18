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
    train_dataset_path = CIFAR100_path
    val_dataset_path = CIFAR100_path
    # download CIFAR100 from here:https://www.cs.toronto.edu/~kriz/cifar.html

    network = 'resnet18cifar'
    pretrained = False
    num_classes = 100
    input_image_size = 32

    model = backbones.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })
    criterion = losses.__dict__['CELoss']()

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_dataset_init = {
        "root": train_dataset_path,
        "train": True,
        "download": True,
        "transform": train_transform
    }
    val_dataset_init = {
        "root": val_dataset_path,
        "train": False,
        "download": True,
        "transform": val_transform
    }

    train_dataset = CIFAR100(**train_dataset_init)
    val_dataset = CIFAR100(**val_dataset_init)

    seed = 0
    # batch_size is total size in DataParallel mode
    # batch_size is per gpu node size in DistributedDataParallel mode
    batch_size = 128
    num_workers = 8

    # choose 'SGD' or 'AdamW'
    optimizer = 'SGD'
    # 'AdamW' doesn't need gamma and momentum variable
    gamma = 0.2
    momentum = 0.9
    # choose 'MultiStepLR' or 'CosineLR'
    # milestones only use in 'MultiStepLR'
    scheduler = 'MultiStepLR'
    lr = 0.1
    weight_decay = 5e-4
    milestones = [60, 120, 160]
    warm_up_epochs = 0

    epochs = 200
    accumulation_steps = 1
    print_interval = 50

    # only in DistributedDataParallel mode can use sync_bn
    distributed = True
    sync_bn = False
    apex = True
