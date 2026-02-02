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

from SimpleAICV.text_detection import models
from SimpleAICV.text_detection import decode
from SimpleAICV.text_detection.common import load_state_dict
from tools.utils import set_seed


class config:
    network = 'convformerm36_dbnet'
    input_image_size = 1024

    # load backbone pretrained model or not
    model = models.__dict__[network](**{})

    # load total pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/dbnet_train_on_ocr_text_detection_dataset/convformerm36_dbnet_epoch_100.pth'
    load_state_dict(trained_model_path, model)

    decoder = decode.__dict__['DBNetDecoder'](**{
        'use_morph_open': False,
        'hard_border_threshold': None,
        'box_score_threshold': 0.5,
        'min_area_size': 9,
        'max_box_num': 1000,
        'rectangle_similarity': 0.6,
        'min_box_size': 3,
        'line_text_expand_ratio': 1.2,
        'curve_text_expand_ratio': 1.5,
    })

    seed = 0


def preprocess_image(image, resize):
    # PIL image(RGB) to opencv image(RGB)
    image = np.asarray(image).astype(np.float32)

    origin_image = image.copy()
    h, w, _ = origin_image.shape

    origin_size = [h, w]

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

    scaled_size = [resize_h, resize_w]

    return origin_image, padded_img, scale, scaled_size, origin_size


@torch.no_grad()
def predict(image):
    set_seed(config.seed)

    origin_image, resized_img, scale, scaled_size, origin_size = preprocess_image(
        image, config.input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)
    scaled_size = [scaled_size]
    origin_size = [origin_size]

    model = config.model
    decoder = config.decoder

    model.eval()

    with torch.no_grad():
        outputs = model(resized_img)

    batch_boxes, batch_scores = decoder(outputs, scaled_size)
    one_image_boxes, one_image_box_scores = batch_boxes[0], batch_scores[0]

    print('1111', origin_image.shape,
          resized_img.shape, outputs.shape, scale, scaled_size, origin_size,
          len(one_image_boxes), one_image_boxes[0].shape,
          len(one_image_box_scores), one_image_box_scores[0].shape)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype(np.uint8)

    masks_num = len(one_image_boxes)

    masks_class_color = []
    for _ in range(masks_num):
        masks_class_color.append(list(np.random.choice(range(256), size=3)))

    print('1212', masks_num, len(masks_class_color), masks_class_color[0])

    per_image_mask = np.zeros(
        (origin_image.shape[0], origin_image.shape[1], 3), dtype=np.float32)
    per_image_contours = []
    for i in range(masks_num):
        per_box = one_image_boxes[i]
        per_mask_score = one_image_box_scores[i]
        per_box = per_box / scale

        per_mask = np.zeros((origin_image.shape[0], origin_image.shape[1]),
                            dtype=np.float32)

        points = np.array(per_box, np.int32)
        points = points.reshape((-1, 1, 2))
        # 填充多边形
        cv2.fillPoly(per_mask, [points], 1)

        per_mask_color = np.array(
            (masks_class_color[i][0], masks_class_color[i][1],
             masks_class_color[i][2]))

        per_object_mask = np.nonzero(per_mask == 1.)
        per_image_mask[per_object_mask[0], per_object_mask[1]] = per_mask_color

        # get contours
        new_per_image_mask = np.zeros(
            (origin_image.shape[0], origin_image.shape[1]))
        new_per_image_mask[per_object_mask[0], per_object_mask[1]] = 255
        contours, _ = cv2.findContours(new_per_image_mask.astype(np.uint8),
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        per_image_contours.append(contours)

    per_image_mask = per_image_mask.astype(np.uint8)
    per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

    all_object_mask = np.nonzero(per_image_mask != 0)

    per_image_mask[all_object_mask[0], all_object_mask[1]] = cv2.addWeighted(
        origin_image[all_object_mask[0], all_object_mask[1]], 0.5,
        per_image_mask[all_object_mask[0], all_object_mask[1]], 1, 0)
    no_class_mask = np.nonzero(per_image_mask == 0)
    per_image_mask[no_class_mask[0],
                   no_class_mask[1]] = origin_image[no_class_mask[0],
                                                    no_class_mask[1]]
    for contours in per_image_contours:
        cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 1)

    per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_BGR2RGB)
    per_image_mask = Image.fromarray(np.uint8(per_image_mask))

    return per_image_mask


title = '文本分割demo'
description = '选择一张图片进行文本分割吧！'
inputs = gr.Image(type='pil')
outputs = gr.Image(type='pil')
gradio_demo = gr.Interface(
    fn=predict,
    title=title,
    description=description,
    inputs=inputs,
    outputs=outputs,
    examples=[
        'test_ocr_text_detection_images/ICDAR2017RCTW_text_detection_image_1075.jpg',
        'test_ocr_text_detection_images/ICDAR2017RCTW_text_detection_image_1132.jpg',
        'test_ocr_text_detection_images/ICDAR2017RCTW_text_detection_image_1236.jpg',
        'test_ocr_text_detection_images/ICDAR2019ReCTS_text_detection_train_ReCTS_000049.jpg',
        'test_ocr_text_detection_images/ICDAR2019ReCTS_text_detection_train_ReCTS_000099.jpg',
    ])
# local website: http://127.0.0.1:6006/
gradio_demo.launch(share=True,
                   server_name='0.0.0.0',
                   server_port=6006,
                   show_error=True)
