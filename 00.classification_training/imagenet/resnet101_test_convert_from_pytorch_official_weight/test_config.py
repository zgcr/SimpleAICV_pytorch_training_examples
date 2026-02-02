import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleAICV.classification import backbones
from SimpleAICV.classification import losses
from SimpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from SimpleAICV.classification.common import Opencv2PIL, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, ClassificationCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    '''
    for resnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    network = 'resnet101'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    # trained_model_path = '/root/autodl-tmp/pretrained_models/resnet_convert_from_pytorch_official_weights/resnet101-63fe2227-acc1-77.374_pytorch_official_weight_convert.pth'
    trained_model_path = '/root/autodl-tmp/pretrained_models/resnet_convert_from_pytorch_official_weights/resnet101-cd907fc2-acc1-81.886_pytorch_official_weight_convert.pth'
    load_state_dict(trained_model_path, model)

    test_criterion = losses.__dict__['CELoss']()

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
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 16
