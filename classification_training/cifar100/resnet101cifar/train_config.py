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
from simpleAICV.classification.datasets.cifar100dataset import CIFAR100Dataset
from simpleAICV.classification.common import Opencv2PIL, TorchPad, TorchRandomHorizontalFlip, TorchRandomCrop, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet101cifar'
    num_classes = 100
    input_image_size = 32

    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['CELoss']()
    test_criterion = losses.__dict__['CELoss']()

    train_dataset = CIFAR100Dataset(
        root_dir=CIFAR100_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchPad(padding=4, fill=0, padding_mode='reflect'),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchRandomCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                  std=np.array([63.0, 62.1, 66.7]) / 255.0),
        ]))
    test_dataset = CIFAR100Dataset(
        root_dir=CIFAR100_path,
        set_name='test',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchMeanStdNormalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                  std=np.array([63.0, 62.1, 66.7]) / 255.0),
        ]))
    train_collater = ClassificationCollater()
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 128
    # num_workers is total workers
    num_workers = 16
    accumulation_steps = 1

    optimizer = (
        'SGD',
        {
            'lr': 0.1,
            'momentum': 0.9,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 5e-4,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.2,
            'milestones': [60, 120, 160],
        },
    )

    epochs = 200
    print_interval = 50

    sync_bn = False
    apex = True

    use_ema_model = False
    ema_model_decay = 0.9999