import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import ILSVRC2012_path

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config:

    log = "./log"  # Path to save log
    checkpoint_path = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"  # 从断点出重新加载模型，resume为模型地址
    evaluate = None  # 测试模型，evaluate为模型地址
    train_dataset_path = os.path.join(ILSVRC2012_path, 'train')
    val_dataset_path = os.path.join(ILSVRC2012_path, 'val')

    seed = 0
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    milestones = [30, 60, 90]
    epochs = 100
    batch_size = 256
    accumulation_steps = 1
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 8
    print_interval = 100
    apex = False

    train_dataset = datasets.ImageFolder(
        train_dataset_path,
        transforms.Compose([
            transforms.RandomResizedCrop(input_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    loss_list = [
        {
            "loss_name": "CELoss",
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "ce_family",
            "loss_rate_decay": "lrdv2"
        },
        {
            "loss_name": "GKDLoss",
            "T": 1,
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "gkd_family",
            "loss_rate_decay": "lrdv2"
        },
        {
            "loss_name": "CDLoss",
            "loss_rate": 6,
            "factor": 0.9,
            "loss_type": "fd_family",
            "loss_rate_decay": "lrdv2"
        },
    ]
