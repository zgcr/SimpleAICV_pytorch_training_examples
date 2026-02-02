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
from SimpleAICV.universal_segmentation import segmentation_decode
from SimpleAICV.universal_segmentation.instance_segmentation_common import load_state_dict
from SimpleAICV.universal_segmentation.datasets.cocodataset import COCO_CLASSES, COCO_CLASSES_COLOR
from tools.utils import set_seed


class config:
    network = 'dinov3_vit_large_patch16_universal_segmentation'
    query_num = 200
    # num_classes has background class
    num_classes = 81
    input_image_size = 1024

    model = models.__dict__[network](**{
        'image_size': input_image_size,
        'query_num': query_num,
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/universal_segmentation_train_instance_segmentation_on_coco/dinov3_vit_large_patch16_universal_segmentation_epoch_50.pth'
    load_state_dict(trained_model_path, model)

    decoder = segmentation_decode.__dict__['UniversalSegmentationDecoder'](
        **{
            'topk': 100,
            'min_score_threshold': 0.1,
            'mask_threshold': 0.5,
            'binary_mask': True,
        }).cuda()

    seed = 0

    # 'retina_style', 'yolo_style'
    image_resize_type = 'yolo_style'

    classes_name = COCO_CLASSES
    classes_color = COCO_CLASSES_COLOR


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

    print('1111', one_image_masks.shape, one_image_classes.shape,
          one_image_scores.shape, origin_image.shape)

    masks_num = one_image_masks.shape[0]

    masks_class_color = []
    for _ in range(masks_num):
        masks_class_color.append(list(np.random.choice(range(256), size=3)))

    print('1212', masks_num, len(masks_class_color), masks_class_color[0])

    per_image_mask = np.zeros(
        (origin_image.shape[0], origin_image.shape[1], 3))
    per_image_contours = []
    for i in range(masks_num):
        per_mask = one_image_masks[i, :, :]
        per_mask_score = one_image_scores[i]

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


title = 'universal segmentation demo'
description = '选择一张图片进行分割吧！'
inputs = gr.Image(type='pil')
outputs = gr.Image(type='pil')
gradio_demo = gr.Interface(fn=predict,
                           title=title,
                           description=description,
                           inputs=inputs,
                           outputs=outputs,
                           examples=[
                               'test_coco_images/000000001551.jpg',
                               'test_coco_images/000000010869.jpg',
                               'test_coco_images/000000011379.jpg',
                               'test_coco_images/000000015108.jpg',
                               'test_coco_images/000000016656.jpg',
                           ])
# local website: http://127.0.0.1:6006/
gradio_demo.launch(share=True,
                   server_name='0.0.0.0',
                   server_port=6006,
                   show_error=True)
