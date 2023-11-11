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

from simpleAICV.detection import models
from simpleAICV.detection import decode
from simpleAICV.detection.common import load_state_dict

from simpleAICV.detection.datasets.cocodataset import COCO_CLASSES, COCO_CLASSES_COLOR

seed = 0
model_name = 'resnet50_detr'
decoder_name = 'DETRDecoder'
# coco class
model_num_classes = 80
trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/detr_train_from_scratch_on_coco/resnet50_detr-yoloresize1024-metric36.941.pth'
input_image_size = 1024
# 'retina_style', 'yolo_style'
image_resize_type = 'yolo_style'
min_score_threshold = 0.5
classes_name = COCO_CLASSES
classes_color = COCO_CLASSES_COLOR

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# assert model_name in models.__dict__.keys(), 'Unsupported model!'
model = models.__dict__[model_name](**{
    'num_classes': model_num_classes,
})
if trained_model_path:
    load_state_dict(trained_model_path, model)
else:
    print('No pretrained model load!')
model.eval()

assert decoder_name in decode.__dict__.keys(), 'Unsupported decoder!'
decoder = decode.__dict__[decoder_name]()


def preprocess_image(image, resize, resize_type):
    assert resize_type in ['retina_style', 'yolo_style']

    # PIL image(RGB) to opencv image(RGB)
    image = np.asarray(image).astype(np.float32)

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
    if 'detr' in model_name:
        padded_mask = np.ones((resize_h + pad_h, resize_w + pad_w), dtype=bool)
        padded_mask[:resize_h, :resize_w] = False

    return origin_image, padded_img, padded_mask, scale, scaled_size


def predict(image):
    origin_image, resized_img, padded_mask, scale, scaled_size = preprocess_image(
        image, input_image_size, image_resize_type)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)
    if padded_mask is not None:
        padded_mask = torch.tensor(padded_mask).unsqueeze(0)
        scaled_size = [scaled_size]

    if 'detr' in model_name:
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

    boxes = boxes[scores > min_score_threshold]
    classes = classes[scores > min_score_threshold]
    scores = scores[scores > min_score_threshold]

    # clip boxes
    origin_h, origin_w = origin_image.shape[0], origin_image.shape[1]
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], origin_w)
    boxes[:, 3] = np.minimum(boxes[:, 3], origin_h)

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

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    origin_image = Image.fromarray(np.uint8(origin_image))

    return origin_image


title = '目标检测'
description = '选择一张图片进行目标检测吧！'
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