import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import CIFAR100_path

import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config:
    log = "./log"  # Path to save log
    checkpoint_path = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"
    evaluate = None  # 测试模型，evaluate为模型地址
    train_dataset_path = CIFAR100_path
    val_dataset_path = CIFAR100_path
    # download CIFAR100 from here:https://www.cs.toronto.edu/~kriz/cifar.html

    seed = 0
    num_classes = 100

    milestones = [60, 120, 160]
    epochs = 200
    batch_size = 128
    accumulation_steps = 1
    lr = 0.1
    gamma = 0.2
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 4
    print_interval = 30
    apex = False

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
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

    loss_list = [
        {
            "loss_name": "CELoss",
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "ce_family",
            "loss_rate_decay": "lrdv3"
        },
        {
            "loss_name": "GKDLoss",
            "T": 1,
            "loss_rate": 0.1,
            "factor": 1,
            "loss_type": "gkd_family",
            "loss_rate_decay": "lrdv3"
        },
        {
            "loss_name": "CDLoss",
            "loss_rate": 6,
            "factor": 0.9,
            "loss_type": "fd_family",
            "loss_rate_decay": "lrdv3"
        },
    ]
