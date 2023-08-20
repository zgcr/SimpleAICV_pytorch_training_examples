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

from simpleAICV.semantic_segmentation import models
from simpleAICV.semantic_segmentation.common import load_state_dict

from simpleAICV.semantic_segmentation.datasets.cocosemanticsegmentationdataset import COCO_CLASSES, COCO_CLASSES_COLOR

seed = 0
model_name = 'u2net'
# ade20k class
model_num_classes = 80
trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/u2net_train_from_scratch_on_coco/u2net-metric66.529.pth'
input_image_size = 512
reduce_zero_label = True
classify_threshold = 0.3

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

assert model_name in models.__dict__.keys(), 'Unsupported model!'
model = models.__dict__[model_name](**{
    'num_classes': model_num_classes,
})
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

    scale_factor = min(resize / max(h, w), resize / min(h, w))
    resize_w, resize_h = int(round(w * scale_factor)), int(
        round(h * scale_factor))

    image = cv2.resize(image, (resize_w, resize_h))

    padded_img = np.zeros((resize, resize, 3), dtype=np.float32)
    padded_img[:resize_h, :resize_w, :] = image
    scale = scale_factor

    # normalize
    padded_img = padded_img.astype(np.float32) / 255.

    return origin_image, padded_img, scale, [resize_h, resize_w]


def predict(image):
    origin_image, resized_img, scale, [resize_h, resize_w] = preprocess_image(
        image, input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    outputs = model(resized_img)
    # pred shape:[b,c,h,w] -> [b,h,w,c]
    outputs = outputs.permute(0, 2, 3, 1).squeeze(0).contiguous()
    outputs = torch.argmax(outputs, axis=-1)
    outputs = outputs.numpy()
    outputs = outputs[:resize_h, :resize_w]
    origin_h, origin_w = origin_image.shape[0], origin_image.shape[1]
    outputs = cv2.resize(outputs, (origin_w, origin_h),
                         interpolation=cv2.INTER_NEAREST)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)

    all_classes = np.unique(outputs)
    for per_class in all_classes:
        per_class = int(per_class)
        if reduce_zero_label:
            if per_class < 0 or per_class > 255:
                continue
            if per_class != 255:
                class_name, class_color = COCO_CLASSES[
                    per_class], COCO_CLASSES_COLOR[per_class]
            else:
                class_name, class_color = 'background', (255, 255, 255)
        else:
            if per_class < 0 or per_class > 255:
                continue
            if per_class != 0:
                class_name, class_color = COCO_CLASSES[
                    per_class - 1], COCO_CLASSES_COLOR[per_class - 1]
            else:
                class_name, class_color = 'background', (255, 255, 255)

        class_color = np.array(
            (class_color[0], class_color[1], class_color[2]))
        per_mask = (outputs == per_class).astype(np.float32)
        per_mask = np.expand_dims(per_mask, axis=-1)
        per_mask = np.tile(per_mask, (1, 1, 3))
        mask_color = np.expand_dims(np.expand_dims(class_color, axis=0),
                                    axis=0)

        per_mask = per_mask * mask_color
        origin_image = 0.5 * per_mask + origin_image

    origin_image = origin_image.astype(np.float32)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    origin_image = Image.fromarray(np.uint8(origin_image))

    return origin_image


title = '语义分割'
description = '选择一张图片进行语义分割吧！'
inputs = gr.Image(type='pil')
outputs = gr.Image(type='pil')
gradio_demo = gr.Interface(fn=predict,
                           title=title,
                           description=description,
                           inputs=inputs,
                           outputs=outputs,
                           examples=[
                               'test_images/000000001551.jpg',
                               'test_images/000000010869.jpg',
                               'test_images/000000011379.jpg',
                               'test_images/000000015108.jpg',
                               'test_images/000000016656.jpg',
                           ])
gradio_demo.launch(share=True,
                   server_name='0.0.0.0',
                   server_port=6006,
                   show_error=True)