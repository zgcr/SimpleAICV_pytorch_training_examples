import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleAICV.distillation.distillmodel import KDModel
from SimpleAICV.distillation import losses
from SimpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from SimpleAICV.classification.common import Opencv2PIL, TorchMeanStdNormalize, TorchRandomResizedCrop, TorchRandomHorizontalFlip, TorchResize, TorchCenterCrop, ClassificationCollater

import torch
import torchvision.transforms as transforms


class config:
    input_image_size = 224
    scale = 256 / 224

    teacher = 'resnet152'
    student = 'resnet50'
    teacher_pretrained_model_path = '/root/autodl-tmp/pretrained_models/resnet_train_from_scratch_on_imagenet1k/resnet152-acc77.834.pth'
    student_pretrained_model_path = '/root/autodl-tmp/pretrained_models/resnet_train_from_scratch_on_imagenet1k/resnet50-acc76.242.pth'
    freeze_teacher = True
    num_classes = 1000
    use_gradient_checkpoint = False
    model = KDModel(teacher_type=teacher,
                    student_type=student,
                    teacher_pretrained_path=teacher_pretrained_model_path,
                    student_pretrained_path=student_pretrained_model_path,
                    freeze_teacher=freeze_teacher,
                    num_classes=num_classes,
                    use_gradient_checkpoint=use_gradient_checkpoint)

    loss_list = ['CELoss', 'KDLoss']
    loss_ratio = {'CELoss': 1.0, 'KDLoss': 1.0}

    T = 1.0
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

    test_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=int(input_image_size * scale)),
            TorchCenterCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]))
    train_collater = ClassificationCollater()
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 64
    accumulation_steps = 1

    optimizer = (
        'SGD',
        {
            'lr': 0.1,
            'momentum': 0.9,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-4,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [30, 60, 90],
        },
    )

    epochs = 100
    print_interval = 100

    sync_bn = False
    use_amp = True
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }
