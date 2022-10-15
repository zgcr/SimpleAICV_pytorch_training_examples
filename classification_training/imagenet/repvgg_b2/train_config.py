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
from simpleAICV.classification.common import Opencv2PIL, TorchRandomResizedCrop, TorchRandomHorizontalFlip, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    '''
    for resnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    network = 'RepVGG_B2'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'deploy': False,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['CELoss']()
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
            TorchResize(resize=input_image_size * scale),
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
            'weight_decay': 1e-4,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
        },
    )

    epochs = 120
    print_interval = 100

    sync_bn = False
    apex = True

    use_ema_model = False
    ema_model_decay = 0.9999