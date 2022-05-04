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
from simpleAICV.classification.common import Opencv2PIL, PIL2Opencv, Normalize, NormalizeTo255, TorchRandomResizedCrop, TorchRandomHorizontalFlip, TorchResize, TorchCenterCrop, PCAJitter, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    '''
    for resnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    network = 'RegNetY_200MF'
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
            PIL2Opencv(),
            Normalize(),
            PCAJitter(pca_std=0.1,
                      vals=[[0.2175, 0.0188, 0.0045]],
                      vecs=[[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]]),
            NormalizeTo255(),
            Opencv2PIL(),
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
    num_workers = 8

    # choose 'SGD' or 'AdamW'
    optimizer = (
        'SGD',
        {
            'lr': 0.2,
            'momentum': 0.9,
            'weight_decay': 5e-5,
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

    epochs = 100
    print_interval = 100

    sync_bn = False
    apex = True