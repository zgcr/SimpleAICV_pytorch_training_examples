import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.classification import backbones
from simpleAICV.classification import losses
from simpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from simpleAICV.classification.common import Opencv2PIL, TorchRandomResizedCrop, TorchRandomHorizontalFlip, RandAugment, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    '''
    for resnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    network = 'resnet50'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    criterion = losses.__dict__['CELoss']()

    train_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=input_image_size),
            TorchRandomHorizontalFlip(prob=0.5),
            RandAugment(N=2, M=10),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]))

    val_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=input_image_size * scale),
            TorchCenterCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]))
    collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 16

    # choose 'SGD' or 'AdamW'
    optimizer = (
        'SGD',
        {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
        },
    )

    # optimizer = (
    #     'AdamW',
    #     {
    #         'lr': 0.1,
    #         'weight_decay': 1e-4,
    #     },
    # )

    # scheduler = (
    #     'MultiStepLR',
    #     {
    #         'warm_up_epochs': 0,
    #         'gamma': 0.1,
    #         'milestones': [30, 60, 90],
    #     },
    # )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 5,
        },
    )

    epochs = 200
    print_interval = 100

    sync_bn = False
    apex = True