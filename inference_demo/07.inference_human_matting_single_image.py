import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.human_matting import models
from SimpleAICV.human_matting.common import load_state_dict
from tools.utils import set_seed


class config:
    input_image_size = 1024
    network = 'dinov3_vit_base_patch16_pfan_matting'

    model = models.__dict__[network](**{})

    trained_model_path = '/root/autodl-tmp/pretrained_models/pfan_matting_train_on_human_matting_dataset/dinov3_vit_base_patch16_pfan_matting_epoch_100.pth'
    load_state_dict(trained_model_path, model)

    seed = 0

    clip_threshold = 0.2


def preprocess_image(image, resize):
    origin_image = image.copy()
    h, w, _ = origin_image.shape

    image = cv2.resize(image, (resize, resize))

    # normalize
    image = image.astype(np.float32) / 255.

    return origin_image, image, [resize, resize], [h, w]


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_human_matting_images/P3M-500-NP_p_00355abf.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_human_matting_result.jpg',
        help='output image path')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = origin_image.copy()

    origin_image, resized_img, [resize_h,
                                resize_w], [origin_h,
                                            origin_w] = preprocess_image(
                                                image, config.input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    model = config.model

    model.eval()

    with torch.no_grad():
        outputs = model(resized_img)

    outputs = outputs[2]
    outputs = outputs[0]
    outputs = torch.squeeze(outputs, dim=0)
    outputs = outputs.numpy()
    outputs = outputs[:resize_h, :resize_w]

    outputs = cv2.resize(outputs, (origin_w, origin_h))
    outputs[outputs < config.clip_threshold] = 0
    outputs = (outputs * 255.).astype(np.uint8)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype(np.uint8)

    b_channel, g_channel, r_channel = cv2.split(origin_image)
    combine_image = cv2.merge([b_channel, g_channel, r_channel, outputs])

    cv2.imencode('.png', combine_image)[1].tofile(args.output_image_path)

    return


if __name__ == '__main__':
    args = parse_args()
    inference(args)
