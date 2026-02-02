import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import math
import numpy as np
import random

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.text_recognition.models import CTCModel
from SimpleAICV.text_recognition.char_sets.final_char_table import final_char_table
from SimpleAICV.text_recognition.common import CTCTextLabelConverter, load_state_dict
from tools.utils import set_seed


class config:
    network = 'CTCModel'
    resize_h = 32
    str_max_length = 80

    # please make sure your converter type is the same as 'predictor'
    converter = CTCTextLabelConverter(chars_set_list=final_char_table,
                                      str_max_length=str_max_length,
                                      garbage_char='㍿')
    # all char + '[CTCblank]' = 12111 + 1 = 12112
    num_classes = converter.num_classes

    model = CTCModel(backbone_type='convformerm36backbone',
                     backbone_pretrained_path='',
                     planes=512,
                     num_classes=num_classes + 1)

    # load total pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/ctc_model_train_on_ocr_text_recognition_dataset/convformerm36_ctc_model_epoch_50.pth'
    load_state_dict(trained_model_path, model)

    seed = 0


def preprocess_image(image, resize_h):
    origin_image = image.copy()
    h, w = origin_image.shape[0], origin_image.shape[1]

    origin_size = [h, w]

    resize_w = max(1, int(math.floor(resize_h * w / float(h))))

    image = cv2.resize(image, (resize_w, resize_h))

    padded_w = int(((resize_w // 32) + 1) * 32)

    padded_img = np.zeros((resize_h, resize_w + padded_w, 3), dtype=np.float32)
    padded_img[:resize_h, :resize_w, :] = image

    # normalize
    padded_img = padded_img.astype(np.float32) / 255.

    scaled_size = [resize_h, resize_w]

    return origin_image, padded_img, scaled_size, origin_size


def add_chinese_text(image, text, position, font_size=30, color=(0, 0, 255)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 转换颜色从BGR到RGB
    rgb_color = (color[2], color[1], color[0])

    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default(font_size)
    draw.text(position, text, font=font, fill=rgb_color)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_ocr_text_recognition_images/aistudio_baidu_street_img_100005.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_ocr_text_recognition_result.jpg',
        help='output image path')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = origin_image.copy()

    origin_image, resized_img, scaled_size, origin_size = preprocess_image(
        image, config.resize_h)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    model = config.model
    converter = config.converter

    model.eval()

    with torch.no_grad():
        outputs = model(resized_img)

    input_lengths = torch.IntTensor([outputs.shape[1]] * outputs.shape[0])

    _, pred_indexes = outputs.max(dim=2)
    pred_strs = converter.decode(pred_indexes.cpu().numpy(),
                                 input_lengths.cpu().numpy())
    pred_strs = pred_strs[0]

    print('1111', pred_strs)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    color = [random.randint(0, 255) for _ in range(3)]
    text = f'{pred_strs}'

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
