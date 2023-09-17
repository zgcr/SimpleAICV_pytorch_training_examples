import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.contrastive_learning import models
from simpleAICV.contrastive_learning import losses
from simpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from simpleAICV.classification.common import Opencv2PIL, load_state_dict
from simpleAICV.contrastive_learning.common import DINOAugmentation, DINOPretrainCollater

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet18_dino_pretrain_model'
    head_planes = 65536
    global_crop_nums = 2
    local_crop_nums = 8

    teacher_model = models.__dict__[network](**{
        'head_planes': head_planes,
        'head_use_bn': False,
        'head_use_norm_last_layer': True,
    })

    student_model = models.__dict__[network](**{
        'head_planes': head_planes,
        'head_use_bn': False,
        'head_use_norm_last_layer': True,
    })

    # load pretrained model or not
    trained_teacher_model_path = ''
    load_state_dict(trained_teacher_model_path, teacher_model)

    # load pretrained model or not
    trained_student_model_path = ''
    load_state_dict(trained_student_model_path, student_model)

    train_criterion = losses.__dict__['DINOLoss'](
        head_planes,
        global_crop_nums=global_crop_nums,
        local_crop_nums=local_crop_nums,
        warmup_teacher_temp_epochs=0,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9)

    train_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            DINOAugmentation(global_resize=224,
                             local_resize=96,
                             global_crops_scale=(0.14, 1.0),
                             local_crops_scale=(0.05, 0.14),
                             local_crops_number=local_crop_nums,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]))
    train_collater = DINOPretrainCollater(
        global_and_local_crop_nums=global_crop_nums + local_crop_nums)

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 60

    optimizer = (
        'SGD',
        {   # lr = base_lr:0.03 * batch_size / 256
            'lr': 0.06,
            'momentum': 0.9,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-4,
            'no_weight_decay_layer_name_list': [],
        },
    )

    lr_scheduler = (
        'Cosine',
        {
            'warm_up_epochs': 10,
            'final_value': 1e-6,
        },
    )

    weight_decay_scheduler = (
        'Cosine',
        {
            'warm_up_epochs': 0,
            'final_value': 1e-4,
        },
    )

    momentum_teacher_scheduler = (
        'Cosine',
        {
            'warm_up_epochs': 0,
            # recommend setting a higher value with small batches:
            # for example use 0.9995 with batch size of 256
            'momentum': 0.9995,
            'final_value': 1,
        },
    )

    epochs = 400
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
