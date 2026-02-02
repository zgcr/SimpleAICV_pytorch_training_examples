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

from SimpleAICV.semantic_segmentation import models
from SimpleAICV.semantic_segmentation.common import load_state_dict
from SimpleAICV.semantic_segmentation.datasets.ade20kdataset import ADE20K_CLASSES, ADK20K_CLASSES_COLOR
from tools.utils import set_seed


class config:
    network = 'dinov3_vit_base_patch16_pfan_semantic_segmentation'
    input_image_size = 512
    # num_classes has background class
    num_classes = 151

    model = models.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/pfan_semantic_segmentation_train_on_ade20k/dinov3_vit_base_patch16_pfan_semantic_segmentation-metric45.964.pth'
    load_state_dict(trained_model_path, model)

    seed = 0

    classes_name = ADE20K_CLASSES
    classes_color = ADK20K_CLASSES_COLOR


def preprocess_image(image, resize):
    origin_image = image.copy()
    h, w, _ = origin_image.shape

    factor = resize / max(h, w)

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
    image = cv2.resize(image, (resize_w, resize_h))

    pad_w = 0 if resize_w % 32 == 0 else 32 - resize_w % 32
    pad_h = 0 if resize_h % 32 == 0 else 32 - resize_h % 32

    padded_img = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                          dtype=np.float32)
    padded_img[:resize_h, :resize_w, :] = image
    scale = factor

    # normalize
    padded_img = padded_img.astype(np.float32) / 255.

    return origin_image, padded_img, scale, [resize_h, resize_w]


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_ade20k_images/ADE_val_00000031.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_semantic_segment_result.jpg',
        help='output image path')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = origin_image.copy()

    origin_image, resized_img, scale, [resize_h, resize_w] = preprocess_image(
        image, config.input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    model = config.model

    model.eval()

    with torch.no_grad():
        outputs = model(resized_img)

    # pred shape:[b,c,h,w] -> [b,h,w,c]
    outputs = outputs.permute(0, 2, 3, 1).squeeze(0).contiguous()
    outputs = torch.argmax(outputs, dim=-1)
    outputs = outputs.numpy()
    outputs = outputs[:resize_h, :resize_w]
    origin_h, origin_w = origin_image.shape[0], origin_image.shape[1]
    outputs = cv2.resize(outputs, (origin_w, origin_h),
                         interpolation=cv2.INTER_NEAREST)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype(np.uint8)

    classes_name = config.classes_name
    classes_color = config.classes_color

    all_classes = np.unique(outputs)

    print('1212', all_classes)

    all_colors = []
    for per_class in all_classes:
        per_class = int(per_class)
        if per_class < 0 or per_class > 255:
            continue

        class_name, class_color = classes_name[per_class], classes_color[
            per_class]
        all_colors.append(class_color)
    all_classes = list(all_classes)

    print('1313', len(all_classes), len(all_colors))

    if len(all_classes) == 0:
        cv2.imencode('.jpg', origin_image)[1].tofile(args.output_image_path)

        return

    per_image_mask = np.zeros(
        (origin_image.shape[0], origin_image.shape[1], 3))
    per_image_contours = []
    for idx, per_class in enumerate(all_classes):
        if per_class < 0 or per_class > 255:
            continue

        per_class_mask = np.nonzero(outputs == per_class)
        per_image_mask[per_class_mask[0], per_class_mask[1]] = all_colors[idx]
        # get contours
        new_per_image_mask = np.zeros(
            (origin_image.shape[0], origin_image.shape[1]))
        new_per_image_mask[per_class_mask[0], per_class_mask[1]] = 255
        contours, _ = cv2.findContours(new_per_image_mask.astype(np.uint8),
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        per_image_contours.append(contours)

    print('1414', per_image_mask.shape, origin_image.shape)

    per_image_mask = per_image_mask.astype(np.uint8)
    per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

    all_classes_mask = np.nonzero(per_image_mask != 0)
    per_image_mask[all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
        origin_image[all_classes_mask[0], all_classes_mask[1]], 0.5,
        per_image_mask[all_classes_mask[0], all_classes_mask[1]], 1, 0)
    no_class_mask = np.nonzero(per_image_mask == 0)
    per_image_mask[no_class_mask[0],
                   no_class_mask[1]] = origin_image[no_class_mask[0],
                                                    no_class_mask[1]]
    for contours in per_image_contours:
        cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 2)

    cv2.imencode('.jpg', per_image_mask)[1].tofile(args.output_image_path)

    return


if __name__ == '__main__':
    args = parse_args()
    inference(args)
