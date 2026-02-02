import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.human_parsing import models
from SimpleAICV.human_parsing.common import load_state_dict
from SimpleAICV.human_parsing.datasets.human_parsing_dataset import CIHP_20_CLASSES, CLASSES_20_COLOR
from tools.utils import set_seed


class config:
    network = 'dinov3_vit_base_patch16_pfan_human_parsing'
    # 包含背景类
    num_classes = 20
    input_image_size = 512

    model = models.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/human_parsing_train_on_CIHP/dinov3_vit_base_patch16_pfan_human_parsing-metric58.247.pth'
    load_state_dict(trained_model_path, model)

    seed = 0

    classes_name = CIHP_20_CLASSES
    classes_color = CLASSES_20_COLOR


def preprocess_image(image, resize):
    # PIL image(RGB) to opencv image(RGB)
    image = np.asarray(image).astype(np.float32)

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


@torch.no_grad()
def predict(image):
    set_seed(config.seed)

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
        origin_image = origin_image.astype(np.float32)
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        origin_image = Image.fromarray(np.uint8(origin_image))

        return origin_image

    per_image_mask = np.zeros(
        (origin_image.shape[0], origin_image.shape[1], 3))
    for idx, per_class in enumerate(all_classes):
        if per_class < 0 or per_class > 255:
            continue

        per_class_mask = np.nonzero(outputs == per_class)
        per_image_mask[per_class_mask[0], per_class_mask[1]] = all_colors[idx]

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

    origin_image = origin_image.astype(np.float32)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    origin_image = Image.fromarray(np.uint8(origin_image))

    per_image_mask = per_image_mask.astype(np.float32)
    per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_BGR2RGB)
    per_image_mask = Image.fromarray(np.uint8(per_image_mask))

    return per_image_mask


title = '人体 parsing demo'
description = '选择一张图片进行人体parsing吧！'
inputs = gr.Image(type='pil')
outputs = gr.Image(type='pil')
gradio_demo = gr.Interface(
    fn=predict,
    title=title,
    description=description,
    inputs=inputs,
    outputs=outputs,
    examples=[
        'test_human_parsing_images/LIP_100034_483681.jpg',
        'test_human_parsing_images/LIP_100396_1228208.jpg',
        'test_human_parsing_images/LIP_100434_223573.jpg',
        'test_human_parsing_images/LIP_100678_1260457.jpg',
        'test_human_parsing_images/LIP_100909_1208726.jpg',
    ])
# local website: http://127.0.0.1:6006/
gradio_demo.launch(share=True,
                   server_name='0.0.0.0',
                   server_port=6006,
                   show_error=True)
