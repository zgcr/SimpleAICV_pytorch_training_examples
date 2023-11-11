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
    origin_image = origin_image.astype('uint8')

    all_classes = np.unique(outputs)
    print('1212', all_classes)

    if reduce_zero_label:
        all_colors = []
        for per_class in all_classes:
            per_class = int(per_class)
            if per_class < 0 or per_class > 255:
                continue
            if per_class != 255:
                class_name, class_color = COCO_CLASSES[
                    per_class], COCO_CLASSES_COLOR[per_class]
                all_colors.append(class_color)
        all_classes = list(all_classes)
        if 255 in all_classes:
            all_classes.remove(255)
    else:
        all_colors = []
        for per_class in all_classes:
            per_class = int(per_class)
            if per_class == 0:
                continue
            if per_class < 0 or per_class > 80:
                continue
            if per_class != 0:
                class_name, class_color = COCO_CLASSES[
                    per_class - 1], COCO_CLASSES_COLOR[per_class - 1]
                all_colors.append(class_color)
        all_classes = list(all_classes)
        if 0 in all_classes:
            all_classes.remove(0)
    print('1313', len(all_classes), len(all_colors))

    if len(all_classes) == 0:
        origin_image = origin_image.astype(np.float32)
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        origin_image = Image.fromarray(np.uint8(origin_image))

        return origin_image

    if reduce_zero_label:
        per_image_mask = np.zeros(
            (origin_image.shape[0], origin_image.shape[1], 3))
        per_image_contours = []
        for idx, per_class in enumerate(all_classes):
            if per_class < 0 or per_class > 255:
                continue
            # reduce_zero_label=True,label为255表示background
            if per_class != 255:
                per_class_mask = np.nonzero(outputs == per_class)
                per_image_mask[per_class_mask[0],
                               per_class_mask[1]] = all_colors[idx]
                # get contours
                new_per_image_mask = np.zeros(
                    (origin_image.shape[0], origin_image.shape[1]))
                new_per_image_mask[per_class_mask[0], per_class_mask[1]] = 255
                contours, _ = cv2.findContours(
                    new_per_image_mask.astype('uint8'), cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
                per_image_contours.append(contours)
    else:
        per_image_mask = np.zeros(
            (origin_image.shape[0], origin_image.shape[1], 3))
        per_image_contours = []
        for idx, per_class in enumerate(all_classes):
            if per_class < 0 or per_class > 255:
                continue
            # reduce_zero_label=False,label为0表示background
            if per_class != 0:
                per_class_mask = np.nonzero(outputs == per_class)
                per_image_mask[per_class_mask[0],
                               per_class_mask[1]] = all_colors[idx]
                # get contours
                new_per_image_mask = np.zeros(
                    (origin_image.shape[0], origin_image.shape[1]))
                new_per_image_mask[per_class_mask[0], per_class_mask[1]] = 255
                contours, _ = cv2.findContours(
                    new_per_image_mask.astype('uint8'), cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
                per_image_contours.append(contours)

    print('1414', per_image_mask.shape, origin_image.shape)

    per_image_mask = per_image_mask.astype('uint8')
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
        cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 1)

    origin_image = origin_image.astype(np.float32)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    origin_image = Image.fromarray(np.uint8(origin_image))

    per_image_mask = per_image_mask.astype(np.float32)
    per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_BGR2RGB)
    per_image_mask = Image.fromarray(np.uint8(per_image_mask))

    return per_image_mask


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