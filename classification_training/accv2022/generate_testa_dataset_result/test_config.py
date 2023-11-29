import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import accv2022_dataset_path, accv2022_broken_list_path

from simpleAICV.classification import backbones
from accv2022testadataset import ACCV2022TestaDataset, Opencv2PIL, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    '''
    for resnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    network = 'vit_large_patch16'
    num_classes = 5000
    input_image_size = 224
    scale = 256 / 224
    set_name = 'testa'

    model = backbones.__dict__[network](**{
        'image_size': 224,
        'global_pool': True,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/vit_finetune_on_accv2022_from_mae_pretrain/vit_large_patch16-acc90.693.pth'
    load_state_dict(trained_model_path, model)

    test_dataset = ACCV2022TestaDataset(
        root_dir=accv2022_dataset_path,
        set_name=set_name,
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=input_image_size * scale),
            TorchCenterCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]),
        broken_list_path=accv2022_broken_list_path)
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 16
    # num_workers is total workers
    num_workers = 20
