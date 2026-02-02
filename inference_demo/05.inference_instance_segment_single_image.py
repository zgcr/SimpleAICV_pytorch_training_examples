import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.instance_segmentation import models
from SimpleAICV.instance_segmentation import decode
from SimpleAICV.instance_segmentation.common import load_state_dict
from SimpleAICV.instance_segmentation.datasets.cocodataset import COCO_CLASSES, COCO_CLASSES_COLOR
from tools.utils import set_seed


class config:
    network = 'dinov3_vit_base_patch16_solov2'
    num_classes = 80
    input_image_size = 1024

    model = models.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/solov2_train_on_coco/dinov3_vit_base_patch16_solov2-metric43.591.pth'
    load_state_dict(trained_model_path, model)

    decoder = decode.__dict__['SOLOV2Decoder'](
        **{
            'strides': (8, 8, 16, 32, 32),
            'grid_nums': (40, 36, 24, 16, 12),
            'mask_feature_upsample_scale': 4,
            'max_mask_num': 100,
            'topn': 500,
            'min_score_threshold': 0.1,
            'keep_score_threshold': 0.3,
            'mask_threshold': 0.5,
            'update_threshold': 0.05,
        })

    seed = 0

    # 'retina_style', 'yolo_style'
    image_resize_type = 'yolo_style'

    classes_name = COCO_CLASSES
    classes_color = COCO_CLASSES_COLOR


def preprocess_image(image, resize, resize_type):
    assert resize_type in ['retina_style', 'yolo_style']

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


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_coco_images/000000001551.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_instance_segment_result.jpg',
        help='output image path')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = origin_image.copy()

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

    batch_masks, batch_labels, batch_scores = decoder(outputs, scaled_size,
                                                      origin_size)
    one_image_masks, one_image_labels, one_image_scores = batch_masks[
        0], batch_labels[0], batch_scores[0]

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype(np.uint8)

    print('1111', one_image_masks.shape, one_image_labels.shape,
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

    cv2.imencode('.jpg', per_image_mask)[1].tofile(args.output_image_path)

    return


if __name__ == '__main__':
    args = parse_args()
    inference(args)
