import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from SimpleAICV.classification import backbones
from SimpleAICV.classification.common import load_state_dict
from tools.utils import set_seed


class config:
    network = 'resnet50'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/resnet_finetune_on_imagenet1k_from_imagenet21k_pretrain/resnet50-acc80.110.pth'
    load_state_dict(trained_model_path, model)

    seed = 0


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_coco_images/000000001551.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_classify_result.jpg',
        help='output image path')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(origin_image)

    transform = transforms.Compose([
        transforms.Resize(int(config.input_image_size * config.scale)),
        transforms.CenterCrop(config.input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = torch.tensor(image).unsqueeze(0)

    model = config.model

    model.eval()

    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
        output = output.squeeze(0)

    # top5
    top5_probs, top5_indices = torch.topk(output, k=5, dim=0)
    for i in range(5):
        per_class = top5_indices[i].item()
        per_prob = top5_probs[i].item()
        print(f'Class: {per_class}, Prob: {per_prob:.4f}')

    max_prob, max_index = torch.max(output, dim=0)
    max_prob, max_index = max_prob.item(), max_index.item()

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    color = [random.randint(0, 255) for _ in range(3)]
    text = f'Class: {max_index}, Prob: {max_prob:.4f}'
    cv2.putText(origin_image,
                text, (30, 30),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                color=color,
                thickness=1)

    cv2.imencode('.jpg', origin_image)[1].tofile(args.output_image_path)

    return


if __name__ == '__main__':
    args = parse_args()
    inference(args)
