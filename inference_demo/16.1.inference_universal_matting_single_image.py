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


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_human_matting_images/P3M-500-NP_p_00355abf.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo',
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
    input_image_name_prefix = args.input_image_path.split('/')[-1].split(
        '.')[0]
    for i in range(masks_num):
        per_object_outputs = one_image_masks[i]
        per_object_outputs[per_object_outputs < config.clip_threshold] = 0
        per_object_outputs = (per_object_outputs * 255.).astype(np.uint8)
        per_object_combine_image = cv2.merge(
            [b_channel, g_channel, r_channel, per_object_outputs])

        per_object_combine_image_save_path = os.path.join(
            args.output_image_path,
            f'{input_image_name_prefix}_result_{i}.jpg')
        cv2.imencode('.png', per_object_combine_image)[1].tofile(
            per_object_combine_image_save_path)

    return


if __name__ == '__main__':
    args = parse_args()
    inference(args)
