import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import gradio as gr
import numpy as np
import math

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
    # PIL image(RGB) to opencv image(RGB)
    image = np.asarray(image).astype(np.float32)

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


@torch.no_grad()
def predict(image):
    set_seed(config.seed)

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

    return pred_strs


title = '文本识别demo'
description = '选择一张图片进行文本识别吧！'
inputs = gr.Image(type='pil')
outputs = gr.Textbox()
gradio_demo = gr.Interface(
    fn=predict,
    title=title,
    description=description,
    inputs=inputs,
    outputs=outputs,
    examples=[
        'test_ocr_text_recognition_images/aistudio_baidu_street_img_100005.jpg',
        'test_ocr_text_recognition_images/aistudio_baidu_street_img_100054.jpg',
        'test_ocr_text_recognition_images/aistudio_baidu_street_img_100087.jpg',
        'test_ocr_text_recognition_images/aistudio_baidu_street_img_100120.jpg',
    ])
# local website: http://127.0.0.1:6006/
gradio_demo.launch(share=True,
                   server_name='0.0.0.0',
                   server_port=6006,
                   show_error=True)
