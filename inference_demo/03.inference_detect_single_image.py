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

from SimpleAICV.detection import models
from SimpleAICV.detection import decode
from SimpleAICV.detection.common import load_state_dict
from SimpleAICV.detection.datasets.cocodataset import COCO_CLASSES, COCO_CLASSES_COLOR
from tools.utils import set_seed


class config:
    network = 'dinov3_vit_base_patch16_fcos'
    num_classes = 80
    input_image_size = 1024

    model = models.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/fcos_train_on_coco/dinov3_vit_base_patch16_fcos-metric51.052.pth'
    load_state_dict(trained_model_path, model)

    decoder = decode.__dict__['FCOSDecoder'](**{
        'strides': [8, 16, 32, 64, 128],
        'max_object_num': 100,
        'min_score_threshold': 0.5,
        'topn': 1000,
        'nms_type': 'python_nms',
        'nms_threshold': 0.6,
    })

    seed = 0

    # 'retina_style', 'yolo_style'
    image_resize_type = 'yolo_style'

    classes_name = COCO_CLASSES
    classes_color = COCO_CLASSES_COLOR


def preprocess_image(image, resize, resize_type):
    assert resize_type in ['retina_style', 'yolo_style']

    origin_image = image.copy()
    h, w, _ = origin_image.shape

    if resize_type == 'retina_style':
        ratio = 1333. / 800
        scales = (resize, int(round(resize * ratio)))

        max_long_edge, max_short_edge = max(scales), min(scales)
        factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        factor = resize / max(h, w)

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
    image = cv2.resize(image, (resize_w, resize_h))

    pad_w = 0 if resize_w % 32 == 0 else 32 - resize_w % 32
    pad_h = 0 if resize_h % 32 == 0 else 32 - resize_h % 32

    padded_img = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                          dtype=np.float32)
    padded_img[:resize_h, :resize_w, :] = image
    scale = factor

    scaled_size = [resize_h, resize_w]

    # normalize
    padded_img = padded_img.astype(np.float32) / 255.

    padded_mask = None
    if 'detr' in config.network:
        padded_mask = np.ones((resize_h + pad_h, resize_w + pad_w), dtype=bool)
        padded_mask[:resize_h, :resize_w] = False

    return origin_image, padded_img, padded_mask, scale, scaled_size


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
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_detect_result.jpg',
        help='output image path')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = origin_image.copy()

    origin_image, resized_img, padded_mask, scale, scaled_size = preprocess_image(
        image, config.input_image_size, config.image_resize_type)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)
    if padded_mask is not None:
        padded_mask = torch.tensor(padded_mask).unsqueeze(0)
        scaled_size = [scaled_size]

    model = config.model
    decoder = config.decoder

    model.eval()

    with torch.no_grad():
        if 'detr' in config.network:
            outputs = model(resized_img, padded_mask)
            scores, classes, boxes = decoder(outputs, scaled_size)
        else:
            outputs = model(resized_img)
            scores, classes, boxes = decoder(outputs)

    boxes /= scale

    scores = scores.squeeze(0)
    classes = classes.squeeze(0)
    boxes = boxes.squeeze(0)

    scores = scores[classes > -1]
    boxes = boxes[classes > -1]
    classes = classes[classes > -1]

    # clip boxes
    origin_h, origin_w = origin_image.shape[0], origin_image.shape[1]
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], origin_w)
    boxes[:, 3] = np.minimum(boxes[:, 3], origin_h)

    classes_name = config.classes_name
    classes_color = config.classes_color

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    # draw all pred boxes
    for per_score, per_class_index, per_box in zip(scores, classes, boxes):
        per_score = per_score.astype(np.float32)
        per_class_index = per_class_index.astype(np.int32)
        per_box = per_box.astype(np.int32)

        class_name, class_color = classes_name[per_class_index], classes_color[
            per_class_index]

        left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                            per_box[3])
        cv2.rectangle(origin_image,
                      left_top,
                      right_bottom,
                      color=class_color,
                      thickness=2,
                      lineType=cv2.LINE_AA)

        text = f'{class_name}:{per_score:.3f}'
        text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
                             left_top[1] - text_size[1] - 3)
        cv2.rectangle(origin_image,
                      left_top,
                      fill_right_bottom,
                      color=class_color,
                      thickness=-1,
                      lineType=cv2.LINE_AA)
        cv2.putText(origin_image,
                    text, (left_top[0], left_top[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    cv2.imencode('.jpg', origin_image)[1].tofile(args.output_image_path)

    return


if __name__ == '__main__':
    args = parse_args()
    inference(args)
