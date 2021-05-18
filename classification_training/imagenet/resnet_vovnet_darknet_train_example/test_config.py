import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.classification import backbones
from simpleAICV.classification import losses

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class config:
    val_dataset_path = os.path.join(ILSVRC2012_path, 'val')

    network = 'resnet18'
    pretrained = True
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })
    criterion = losses.__dict__['CELoss']()

    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    distributed = True
    seed = 0
    batch_size = 256
    num_workers = 16
    trained_model_path = ''
