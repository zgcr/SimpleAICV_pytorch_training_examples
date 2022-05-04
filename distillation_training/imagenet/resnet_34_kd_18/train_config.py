import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.distillation.distillmodel import KDModel
from simpleAICV.distillation import losses
from simpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from simpleAICV.classification.common import Opencv2PIL, TorchMeanStdNormalize, TorchRandomResizedCrop, TorchRandomHorizontalFlip, TorchResize, TorchCenterCrop, ClassificationCollater

import torch
import torchvision.transforms as transforms


class config:
    teacher = 'resnet34'
    student = 'resnet18'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    teacher_pretrained_model_path = os.path.join(
        BASE_DIR, 'pretrained_models/resnet/resnet34-acc73.930.pth')
    student_pretrained_model_path = ''
    freeze_teacher = True
    model = KDModel(teacher_type=teacher,
                    student_type=student,
                    teacher_pretrained_path=teacher_pretrained_model_path,
                    student_pretrained_path=student_pretrained_model_path,
                    freeze_teacher=freeze_teacher,
                    num_classes=num_classes)

    loss_list = ['CELoss', 'KDLoss']
    T = 1
    train_criterion = {}
    for loss_name in loss_list:
        if loss_name in ['KDLoss', 'DMLLoss']:
            train_criterion[loss_name] = losses.__dict__[loss_name](T)
        else:
            train_criterion[loss_name] = losses.__dict__[loss_name]()
    test_criterion = losses.__dict__['CELoss']()

    train_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=input_image_size),
            TorchRandomHorizontalFlip(prob=0.5),
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

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [30, 60, 90],
        },
    )

    # scheduler = (
    #     'CosineLR',
    #     {
    #         'warm_up_epochs': 0,
    #     },
    # )

    epochs = 100
    print_interval = 100

    sync_bn = False
    apex = True