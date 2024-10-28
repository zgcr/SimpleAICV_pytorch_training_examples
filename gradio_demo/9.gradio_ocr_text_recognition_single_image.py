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
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from simpleAICV.text_recognition.models import CTCModel
from simpleAICV.text_recognition.char_sets.final_char_table import final_char_table
from simpleAICV.text_recognition.common import CTCTextLabelConverter, load_state_dict

seed = 0

trained_model_path = '/root/autodl-tmp/pretrained_models/ctc_model_train_on_ocr_text_recognition_dataset/convformerm36_ctc_model-metric99.453.pth'
resize_h = 32
str_max_length = 80

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# please make sure your converter type is the same as 'predictor'
converter = CTCTextLabelConverter(chars_set_list=final_char_table,
                                  str_max_length=str_max_length,
                                  garbage_char='㍿')
# all char + '[CTCblank]' = 12111 + 1 = 12112
num_classes = converter.num_classes

model_config = {
    'backbone': {
        'name': 'convformerm36backbone',
        'param': {}
    },
    'encoder': {
        'name': 'BiLSTMEncoder',
        'param': {},
    },
    'predictor': {
        'name': 'CTCPredictor',
        'param': {
            'hidden_planes': 512,
            'num_classes': num_classes + 1,
        }
    },
}
model = CTCModel(model_config)
if trained_model_path:
    load_state_dict(trained_model_path, model)
else:
    print('No pretrained model load!')
model.eval()


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


def predict(image):
    origin_image, resized_img, scaled_size, origin_size = preprocess_image(
        image, resize_h)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        outputs = model(resized_img)

    input_lengths = torch.IntTensor([outputs.shape[1]] * outputs.shape[0])

    _, pred_indexes = outputs.max(dim=2)
    pred_strs = converter.decode(pred_indexes.cpu().numpy(),
                                 input_lengths.cpu().numpy())

    return pred_strs


title = '文本识别'
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
