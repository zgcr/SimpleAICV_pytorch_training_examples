import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.contrastive_learning import models
from simpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from simpleAICV.classification.common import Opencv2PIL, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_dino_pretrain_model'
    head_planes = 65536
    global_crop_nums = 2
    local_crop_nums = 8
    input_image_size = 224
    scale = 256 / 224

    model = models.__dict__[network](**{
        'head_planes': head_planes,
        'head_use_bn': False,
        'head_use_norm_last_layer': True,
    })

    # load pretrained model or not
    trained_model_path = '/root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/contrastive_learning_training/imagenet/dino_resnet50_epoch100/checkpoints/best_student.pth'
    load_state_dict(trained_model_path, model)

    seed = 0
    save_num = 1
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    show_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=input_image_size * scale),
            TorchCenterCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=mean, std=std),
        ]))