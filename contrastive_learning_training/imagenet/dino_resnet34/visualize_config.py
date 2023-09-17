import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

import resnet
from simpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from simpleAICV.classification.common import Opencv2PIL, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet34backbone'
    input_image_size = 224

    model = resnet.__dict__[network](**{})

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    seed = 0
    save_image_num = 10
    save_feature_map_level = 3
    save_max_channel_num = 256

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    show_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=input_image_size),
            TorchCenterCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=mean, std=std),
        ]))
