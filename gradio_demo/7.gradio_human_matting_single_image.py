import os
import sys
import warnings

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import gradio as gr
import random
import numpy as np
from PIL import Image

import torch

from simpleAICV.human_matting import models
from simpleAICV.human_matting.common import load_state_dict

seed = 0
model_name = 'convformerm36_pfan_matting'

trained_model_path = '/root/autodl-tmp/pretrained_models/pfan_matting_train_on_human_matting_dataset/convformerm36_pfan_matting-metric0.985.pth'
input_image_size = 832
clip_threshold = 0.2

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

assert model_name in models.__dict__.keys(), 'Unsupported model!'
model = models.__dict__[model_name](**{})

if trained_model_path:
    load_state_dict(trained_model_path, model)
else:
    print('No pretrained model load!')
model.eval()


def preprocess_image(image, resize):
    # PIL image(RGB) to opencv image(RGB)
    image = np.asarray(image).astype(np.float32)

    origin_image = image.copy()
    h, w, _ = origin_image.shape

    image = cv2.resize(image, (resize, resize))

    # normalize
    image = image.astype(np.float32) / 255.

    return origin_image, image, [resize, resize], [h, w]


@torch.no_grad
def predict(image):
    origin_image, resized_img, [resize_h,
                                resize_w], [origin_h,
                                            origin_w] = preprocess_image(
                                                image, input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        outputs = model(resized_img)

    outputs = outputs[2]
    outputs = outputs[0]
    outputs = torch.squeeze(outputs, dim=0)
    outputs = outputs.numpy()
    outputs = outputs[:resize_h, :resize_w]

    outputs = cv2.resize(outputs, (origin_w, origin_h))
    outputs[outputs < clip_threshold] = 0
    outputs = (outputs * 255.).astype('uint8')
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype('uint8')

    b_channel, g_channel, r_channel = cv2.split(origin_image)
    combine_image = cv2.merge([r_channel, g_channel, b_channel, outputs])

    return combine_image


title = '人像matting demo'
description = '选择一张图片进行人像matting吧！'
inputs = gr.Image(type='pil')
outputs = gr.Image(type='pil')
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
