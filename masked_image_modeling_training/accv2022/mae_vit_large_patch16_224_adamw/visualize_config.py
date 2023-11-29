import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import accv2022_dataset_path, accv2022_broken_list_path

from simpleAICV.masked_image_modeling import models
from simpleAICV.classification.datasets.accv2022traindataset import ACCV2022TrainDataset
from simpleAICV.classification.common import Opencv2PIL, TorchRandomResizedCrop, TorchMeanStdNormalize, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'vit_large_patch16_224_mae_pretrain_model'
    input_image_size = 224

    model = models.__dict__[network](**{})

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    seed = 0
    save_num = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    show_dataset = ACCV2022TrainDataset(
        root_dir=accv2022_dataset_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=input_image_size, scale=(0.2, 1.0)),
            TorchMeanStdNormalize(mean=mean, std=std),
        ]),
        broken_list_path=accv2022_broken_list_path)
