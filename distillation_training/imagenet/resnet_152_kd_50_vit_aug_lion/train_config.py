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
from simpleAICV.classification.common import Opencv2PIL, TorchRandomResizedCrop, TorchRandomHorizontalFlip, RandAugment, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, RandomErasing, ClassificationCollater, MixupCutmixClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    teacher = 'resnet152'
    student = 'resnet50'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    teacher_pretrained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/resnet_finetune_on_imagenet1k_from_imagenet21k_pretrain/resnet152-acc81.236.pth'
    student_pretrained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/resnet_finetune_on_imagenet1k_from_imagenet21k_pretrain/resnet50-acc79.484.pth'
    freeze_teacher = True
    model = KDModel(teacher_type=teacher,
                    student_type=student,
                    teacher_pretrained_path=teacher_pretrained_model_path,
                    student_pretrained_path=student_pretrained_model_path,
                    freeze_teacher=freeze_teacher,
                    num_classes=num_classes)

    loss_list = ['OneHotLabelCELoss', 'KDLoss']
    loss_ratio = {'OneHotLabelCELoss': 1.0, 'KDLoss': 1.0}

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
            RandAugment(magnitude=9,
                        num_layers=2,
                        resize=input_image_size,
                        mean=[0.485, 0.456, 0.406],
                        integer=True,
                        weight_idx=None,
                        magnitude_std=0.5,
                        magnitude_max=None),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
            RandomErasing(prob=0.25, mode='pixel', max_count=1),
        ]))

    test_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=input_image_size * scale),
            TorchCenterCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]))

    train_collater = MixupCutmixClassificationCollater(
        use_mixup=True,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        mixup_cutmix_prob=1.0,
        switch_to_cutmix_prob=0.5,
        mode='batch',
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000)
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 30
    accumulation_steps = 16

    optimizer = (
        'Lion',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-4,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 5,
            'min_lr': 1e-6,
        },
    )

    epochs = 300
    print_interval = 10

    sync_bn = False
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }
