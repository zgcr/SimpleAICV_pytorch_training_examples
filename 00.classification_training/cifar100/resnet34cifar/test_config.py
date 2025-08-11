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
from simpleAICV.classification.common import Opencv2PIL, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet34cifar'
    num_classes = 100
    input_image_size = 32

    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['CELoss']()

    test_dataset = CIFAR100Dataset(
        root_dir=CIFAR100_path,
        set_name='test',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchMeanStdNormalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                  std=np.array([63.0, 62.1, 66.7]) / 255.0),
        ]))
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 16
