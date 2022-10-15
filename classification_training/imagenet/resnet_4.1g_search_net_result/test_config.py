import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.classification import losses
from simpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from simpleAICV.classification.common import Opencv2PIL, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

from resnet import ResNet, BasicBlock, Bottleneck

import torch
import torchvision.transforms as transforms


class config:
    '''
    for resnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    network = 'resnet_4.1g'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    net_config = {
        'stem_width': 64,
        'depth': 18,
        'w_0': 32,
        'w_a': 17.77597968937125,
        'w_m': 1.9609275565633586,
    }

    model = ResNet(
        **{
            'resnet_config': net_config,
            'block': Bottleneck,
            'num_classes': num_classes,
        })

    # load pretrained model or not
    trained_model_path = ''
    # trained_model_path = os.path.join(BASE_DIR, '')
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['CELoss']()

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
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 16