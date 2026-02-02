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

from SimpleAICV.universal_segmentation import models
from SimpleAICV.universal_segmentation import matting_decode
from SimpleAICV.universal_segmentation.human_matting_common import load_state_dict
from tools.utils import set_seed


class config:
    network = 'dinov3_vit_large_patch16_universal_matting'
    query_num = 100
    # num_classes has background class
    num_classes = 2
    input_image_size = 1024

    model = models.__dict__[network](**{
        'image_size': input_image_size,
        'query_num': query_num,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/universal_matting_train_human_matting_on_human_matting_dataset/dinov3_vit_large_patch16_universal_matting_epoch_50.pth'
    load_state_dict(trained_model_path, model)

    decoder = matting_decode.__dict__['UniversalMattingDecoder'](
        **{
            'topk': 100,
            'min_score_threshold': 0.1,
        }).cuda()

    seed = 0

    clip_threshold = 0.2

    # 'retina_style', 'yolo_style'
    image_resize_type = 'yolo_style'


def preprocess_image(image, resize, resize_type):
    assert resize_type in ['retina_style', 'yolo_style']

    # PIL image(RGB) to opencv image(RGB)
    image = np.asarray(image).astype(np.float32)

    origin_image = image.copy()
    h, w, _ = origin_image.shape

    origin_size = [h, w]

    if resize_type == 'retina_style':
        ratio = 1333. / 800
        scales = (resize, int(round(resize * ratio)))

        max_long_edge, max_short_edge = max(scales), min(scales)
        factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        factor = resize / max(h, w)

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
    image = cv2.resize(image, (resize_w, resize_h))

    padded_img = np.zeros((resize, resize, 3), dtype=np.float32)
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
        image, config.input_image_size, config.image_resize_type)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)
    scaled_size = [scaled_size]
    origin_size = [origin_size]

    model = config.model
    decoder = config.decoder

    model.eval()

    with torch.no_grad():
        outputs = model(resized_img)

    batch_masks, batch_scores, batch_classes = decoder(outputs, scaled_size,
                                                       origin_size)
    one_image_masks, one_image_classes, one_image_scores = batch_masks[
        0], batch_classes[0], batch_scores[0]

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype(np.uint8)

    b_channel, g_channel, r_channel = cv2.split(origin_image)

    print('1111', one_image_masks.shape, one_image_classes.shape,
          one_image_scores.shape, origin_image.shape)

    masks_num = one_image_masks.shape[0]

    combine_images = []
    for i in range(masks_num):
        per_object_outputs = one_image_masks[i]
        per_object_outputs[per_object_outputs < config.clip_threshold] = 0
        per_object_outputs = (per_object_outputs * 255.).astype(np.uint8)
        per_object_combine_image = cv2.merge(
            [r_channel, g_channel, b_channel, per_object_outputs])
        combine_images.append(per_object_combine_image)

    return combine_images


title = 'universal matting demo'
description = '选择一张图片进行抠图吧！'
inputs = gr.Image(type='pil')
outputs = gr.Gallery(label='Matting Results')
gradio_demo = gr.Interface(
    fn=predict,
    title=title,
    description=description,
    inputs=inputs,
    outputs=outputs,
    examples=[
        'test_human_matting_images/P3M-500-NP_p_0be67476.jpg',
        'test_human_matting_images/P3M-500-NP_p_094cb88b.jpg',
        'test_human_matting_images/P3M-500-NP_p_00355abf.jpg',
        'test_human_matting_images/P3M-500-NP_p_0644cc81.jpg',
        'test_human_matting_images/P3M-500-NP_p_0846ec6f.jpg',
    ])
# local website: http://127.0.0.1:6006/
gradio_demo.launch(share=True,
                   server_name='0.0.0.0',
                   server_port=6006,
                   show_error=True)
