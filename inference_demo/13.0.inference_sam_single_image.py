import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import ast
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleAICV.interactive_segmentation.models.segment_anything import sam
from SimpleAICV.interactive_segmentation.common import load_state_dict
from tools.utils import set_seed


class config:
    network = 'sam_h'
    input_image_size = 1024

    model = sam.__dict__[network](**{
        'image_size': input_image_size,
    })

    # load pretrained model or not
    trained_model_path = '/root/autodl-tmp/pretrained_models/sam_pytorch_official_weights/sam_vit_h_4b8939.pth'
    load_state_dict(trained_model_path, model)

    seed = 0


def preprocess_image(image, resize):
    origin_image = image.copy()
    h, w, _ = origin_image.shape

    origin_size = [h, w]

    factor = resize / max(h, w)

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
    image = cv2.resize(image, (resize_w, resize_h))

    # normalize
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    image = (image - mean) / std

    padded_img = np.zeros(
        (max(resize_h, resize_w), max(resize_h, resize_w), 3),
        dtype=np.float32)
    padded_img[:resize_h, :resize_w, :] = image

    scale = factor
    scaled_size = [resize_h, resize_w]

    return origin_image, padded_img, scale, scaled_size, origin_size


def preprocess_prompt_mask(mask, resize):
    h, w = mask.shape

    factor = resize / max(h, w)

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
    mask = cv2.resize(mask, (resize_w, resize_h),
                      interpolation=cv2.INTER_NEAREST)

    padded_mask = np.zeros((max(resize_h, resize_w), max(resize_h, resize_w)),
                           dtype=np.float32)
    padded_mask[:resize_h, :resize_w] = mask

    return padded_mask


def parse_prompt_points(points_str):
    """
    解析prompt points，格式: [[x, y, label], [x, y, label], ...]
    """
    points = ast.literal_eval(points_str)

    if not isinstance(points, list) or len(points) == 0:
        raise argparse.ArgumentTypeError("prompt points 必须是非空列表")

    for point in points:
        if not isinstance(point, list) or len(point) != 3:
            raise argparse.ArgumentTypeError("每个point必须是包含3个int值的列表")
        if not all(isinstance(v, int) for v in point):
            raise argparse.ArgumentTypeError("point中的所有值必须是整数")

    return points


def parse_prompt_box(box_str):
    """
    解析prompt box，格式: [x1, y1, x2, y2]
    """
    box = ast.literal_eval(box_str)

    if not isinstance(box, list) or len(box) != 4:
        raise argparse.ArgumentTypeError("prompt box 必须是包含4个int值的列表")

    if not all(isinstance(v, int) for v in box):
        raise argparse.ArgumentTypeError("box中的所有值必须是整数")

    return box


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument(
        '--input-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/gradio_demo/test_sam_images/truck.jpg',
        help='input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default=
        '/root/code/SimpleAICV_pytorch_training_examples/inference_demo/inference_sam_result.jpg',
        help='output image path')
    parser.add_argument(
        '--input-prompt-points',
        type=parse_prompt_points,
        default=None,
        help=
        'input prompt points, 格式: [[x,y,label],[x,y,label],...], 如: [[100,200,1],[150,250,0]]'
    )
    parser.add_argument(
        '--input-prompt-box',
        type=parse_prompt_box,
        default=None,
        help='input prompt box, 格式: [x1,y1,x2,y2], 例如: [100,200,300,400]')
    parser.add_argument('--input-prompt-mask-path',
                        type=str,
                        default=None,
                        help='input prompt mask file path')
    parser.add_argument('--mask-out-idx',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3],
                        help='mask out index')

    return parser.parse_args()


@torch.no_grad()
def inference(args):
    set_seed(config.seed)

    origin_image = cv2.imdecode(
        np.fromfile(args.input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    image = origin_image.copy()

    origin_image, resized_img, scale, scaled_size, origin_size = preprocess_image(
        image, config.input_image_size)
    resized_img = torch.tensor(resized_img).permute(2, 0, 1).unsqueeze(0)

    assert args.input_prompt_points is not None or args.input_prompt_box is not None

    input_prompt_points = None
    if args.input_prompt_points is not None:
        prompt_points = args.input_prompt_points
        prompt_points = np.array(prompt_points, dtype=np.float32)

        prompt_points, prompt_labels = prompt_points[:,
                                                     0:2], prompt_points[:,
                                                                         2:3]
        prompt_points = prompt_points * scale
        prompt_points = np.concatenate([prompt_points, prompt_labels], axis=1)
        input_prompt_points = torch.tensor(np.expand_dims(prompt_points,
                                                          axis=0),
                                           dtype=torch.float32)

    input_prompt_box = None
    if args.input_prompt_box is not None:
        prompt_box = args.input_prompt_box
        prompt_box = np.array(prompt_box, dtype=np.float32)
        prompt_box = prompt_box * scale
        input_prompt_box = torch.tensor(np.expand_dims(prompt_box, axis=0),
                                        dtype=torch.float32)

    input_prompt_mask = None
    if args.input_prompt_mask_path is not None:
        input_prompt_mask = np.array(Image.open(
            args.input_prompt_mask_path).convert('L'),
                                     dtype=np.uint8)
        input_prompt_mask[input_prompt_mask > 0] = 1

        input_prompt_mask = preprocess_prompt_mask(input_prompt_mask,
                                                   config.input_image_size)
        input_prompt_mask = torch.tensor(
            input_prompt_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_prompt_mask = F.interpolate(input_prompt_mask, (256, 256),
                                          mode='nearest')

    batch_prompts = {
        'prompt_point': input_prompt_points,
        'prompt_box': input_prompt_box,
        'prompt_mask': input_prompt_mask
    }

    mask_out_idx = [args.mask_out_idx]

    model = config.model

    model.eval()

    with torch.no_grad():
        mask_preds, iou_preds = model(resized_img,
                                      batch_prompts,
                                      mask_out_idxs=mask_out_idx)
        mask_preds, iou_preds = mask_preds[0][0], iou_preds[0][0]
        binary_mask_preds = mask_preds > 0.

    masks = binary_mask_preds.numpy().astype(np.float32)
    masks = masks[:scaled_size[0], :scaled_size[1]]

    iou_preds = iou_preds.numpy()

    masks = cv2.resize(masks, (origin_size[1], origin_size[0]),
                       interpolation=cv2.INTER_NEAREST)

    binary_mask = (masks.copy() * 255.).astype(np.uint8)

    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    origin_image = origin_image.astype(np.uint8)

    masks_class_color = list(np.random.choice(range(256), size=3))

    per_image_mask = np.zeros(
        (origin_image.shape[0], origin_image.shape[1], 3))

    per_image_contours = []
    per_mask = masks

    per_mask_color = np.array(
        (masks_class_color[0], masks_class_color[1], masks_class_color[2]))

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
