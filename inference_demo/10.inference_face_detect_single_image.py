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

from SimpleAICV.face_detection import models
from SimpleAICV.face_detection import decode
from SimpleAICV.face_detection.common import load_state_dict
from SimpleAICV.face_detection.datasets.face_detection_dataset import FACE_CLASSES, FACE_CLASSES_COLOR
from tools.utils import set_seed


class config:
    network = 'resnet50_retinaface'
    num_classes = 1
    input_image_size = 1024

    model = models.__dict__[network](**{})

    # load total pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/retinaface_train_on_face_detection_dataset/resnet50_retinaface-metric66.224.pth'
    load_state_dict(trained_model_path, model)

    decoder = decode.__dict__['RetinaFaceDecoder'](
        **{
            'anchor_sizes': [[8, 16, 32], [32, 64, 128], [128, 256, 512]],
            'strides': [8, 16, 32],
            'max_object_num': 200,
            'min_score_threshold': 0.3,
            'topn': 1000,
            'nms_type': 'python_nms',
            'nms_threshold': 0.3,
        })

    seed = 0

    classes_name = FACE_CLASSES
    classes_color = FACE_CLASSES_COLOR


def preprocess_image(image, resize):
    origin_image = image.copy()
    h, w, _ = origin_image.shape

    factor = resize / max(h, w)

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
    image = cv2.resize(image, (resize_w, resize_h))

    padded_img = np.zeros((resize, resize, 3), dtype=np.float32)
    padded_img[:resize_h, :resize_w, :] = image
    scale = factor

    scaled_size = [resize_h, resize_w]

    # normalize
    padded_img = padded_img.astype(np.float32) / 255.

    return origin_image, padded_img, scale, scaled_size


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_face_detection_images/0_Parade_marchingband_1_104.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_face_detect_result.jpg',
        help='output image path')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = origin_image.copy()

    origin_image, resized_img, scale, scaled_size = preprocess_image(
        image, config.input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    model = config.model
    decoder = config.decoder

    model.eval()

    with torch.no_grad():
        outputs = model(resized_img)
        scores, classes, boxes = decoder(outputs)
        scores, classes, boxes = scores[0], classes[0], boxes[0]

    boxes /= scale

    scores = scores[classes > -1]
    boxes = boxes[classes > -1]
    classes = classes[classes > -1]

    # clip boxes
    origin_h, origin_w = origin_image.shape[0], origin_image.shape[1]
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], origin_w)
    boxes[:, 3] = np.minimum(boxes[:, 3], origin_h)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    # draw all pred boxes
    for per_score, per_box in zip(scores, boxes):
        per_box = per_box.astype(np.int32)
        per_score = per_score.astype(np.float32)
        class_color = FACE_CLASSES_COLOR[0]
        class_name = FACE_CLASSES[0]

        left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                            per_box[3])

        cv2.rectangle(origin_image,
                      left_top,
                      right_bottom,
                      color=class_color,
                      thickness=1,
                      lineType=cv2.LINE_AA)

        text = f'{per_score:.3f}'
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
