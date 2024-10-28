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
from simpleAICV.classification.common import Opencv2PIL, TorchRandomResizedCrop, TorchRandomHorizontalFlip, RandAugment, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, RandomErasing, ClassificationCollater, MixupCutmixClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    '''
    for resnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    network = 'van_b0'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    # drop_path_prob [0.1, 0.1, 0.1, 0.2] for [b0, b1, b2, b3]
    model = backbones.__dict__[network](**{
        'drop_path_prob': 0.1,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples/pretrained_models/van_convert_from_official_weights/van_b0_pytorch_official_weight_convert.pth'
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['OneHotLabelCELoss']()
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
    batch_size = 2048
    # num_workers is total workers
    num_workers = 64
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-3,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 5e-2,
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
    print_interval = 100

    sync_bn = False
    use_amp = False
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    use_ema_model = False
    ema_model_decay = 0.9999

    clip_grad_value = 1
    clip_max_norm = 1
