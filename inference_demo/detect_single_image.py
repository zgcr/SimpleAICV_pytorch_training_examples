import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import math
import argparse
import random
import numpy as np

import torch

from simpleAICV import datasets
from simpleAICV.detection import models
from simpleAICV.detection import decode
from simpleAICV.datasets.cocodataset import COCO_CLASSES, COCO_CLASSES_COLOR
from simpleAICV.datasets.vocdataset import VOC_CLASSES, VOC_CLASSES_COLOR

from tools.utils import compute_flops_and_params


def parse_args():
    parser = argparse.ArgumentParser(description='detect image')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--model', type=str, help='name of model')
    parser.add_argument('--decoder', type=str, help='name of decoder')
    parser.add_argument('--trained_dataset_name',
                        type=str,
                        help='name of trained dataset')
    parser.add_argument('--trained_num_classes',
                        type=int,
                        default=80,
                        help='model class num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=800,
                        help='input image size')
    parser.add_argument('--image_resize_style',
                        type=str,
                        help='image resize style')
    parser.add_argument('--min_score_threshold',
                        type=float,
                        default=0.5,
                        help='min score threshold')
    parser.add_argument('--trained_model_path',
                        type=str,
                        default='',
                        help='trained model path')
    parser.add_argument('--test_image_path', type=str, help='test image path')
    parser.add_argument("--save_image_path",
                        type=str,
                        help="save detected image path")
    parser.add_argument('--show_image',
                        default=False,
                        action='store_true',
                        help='show_image or not')
    parser.add_argument('--use_gpu',
                        default=False,
                        action='store_true',
                        help='use gpu to test or not')
    args = parser.parse_args()

    return args


def load_image_for_detection_inference(args, divisor=32):
    assert args.image_resize_style in ['retinastyle',
                                       'yolostyle'], 'wrong style!'

    img = cv2.imread(args.test_image_path)
    origin_img = img
    h, w, _ = img.shape

    # normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(
        np.float32) / np.float32(255.)

    if args.image_resize_style == 'retinastyle':
        factor = args.input_image_size / min(h, w)
        if max(h, w) * factor > args.input_image_size * np.float32(
                1333. / 800):
            factor = args.input_image_size * np.float32(1333. / 800) / max(
                h, w)

        resize_h, resize_w = math.ceil(h * factor), math.ceil(w * factor)
        img = cv2.resize(img, (resize_w, resize_h))

        pad_w = 0 if resize_w % divisor == 0 else divisor - resize_w % divisor
        pad_h = 0 if resize_h % divisor == 0 else divisor - resize_h % divisor

        padded_img = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                              dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = img
        scale = factor

    elif args.image_resize_style == 'yolostyle':
        max_size = max(h, w)
        factor = args.input_image_size / max_size
        resize_h, resize_w = math.ceil(h * factor), math.ceil(w * factor)

        img = cv2.resize(img, (resize_w, resize_h))

        pad_w = 0 if resize_w % divisor == 0 else divisor - resize_w % divisor
        pad_h = 0 if resize_h % divisor == 0 else divisor - resize_h % divisor

        padded_img = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                              dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = img
        scale = factor

    return padded_img, origin_img, scale


def inference():
    args = parse_args()
    print(f'args: {args}')

    assert args.trained_dataset_name in ['COCO', 'VOC'], 'Unsupported dataset!'
    assert args.model in models.__dict__.keys(), 'Unsupported model!'
    assert args.decoder in decode.__dict__.keys(), 'Unsupported decoder!'

    if args.use_gpu:
        # only use one Graphics card to inference
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        assert torch.cuda.is_available(), 'need gpu to train network!'
        torch.cuda.empty_cache()

    if args.seed:
        seed = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.use_gpu:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # for cudnn
            cudnn.enabled = True
            cudnn.deterministic = True
            cudnn.benchmark = False

    model = models.__dict__[args.model](
        **{
            'num_classes': args.trained_num_classes,
        })
    decoder = decode.__dict__[args.decoder]()

    if args.use_gpu:
        model = model.cuda()
        decoder = decoder.cuda()

    if args.trained_model_path:
        saved_model = torch.load(args.trained_model_path,
                                 map_location=torch.device('cpu'))
        model.load_state_dict(saved_model)

    model.eval()

    flops, params = compute_flops_and_params(args, model)
    print(f'model: {args.model}, flops: {flops}, params: {params}')

    resized_img, origin_img, scale = load_image_for_detection_inference(args)
    resized_img = torch.tensor(resized_img)

    if args.use_gpu:
        resized_img = resized_img.cuda()

    out_tuples = model(resized_img.permute(2, 0, 1).float().unsqueeze(0))
    scores, classes, boxes = decoder(*out_tuples)
    scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
    boxes /= scale

    scores = scores.squeeze(0)
    classes = classes.squeeze(0)
    boxes = boxes.squeeze(0)

    scores = scores[classes > -1]
    boxes = boxes[classes > -1]
    classes = classes[classes > -1]

    boxes = boxes[scores > args.min_score_threshold]
    classes = classes[scores > args.min_score_threshold]
    scores = scores[scores > args.min_score_threshold]

    # clip boxes
    origin_h, origin_w = origin_img.shape[0], origin_img.shape[1]
    boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)
    boxes[:, 2] = torch.clamp(boxes[:, 2], max=origin_w)
    boxes[:, 3] = torch.clamp(boxes[:, 3], max=origin_h)

    if args.trained_dataset_name == 'COCO':
        dataset_classes_name = COCO_CLASSES
        dataset_classes_color = COCO_CLASSES_COLOR
    else:
        dataset_classes_name = VOC_CLASSES
        dataset_classes_color = VOC_CLASSES_COLOR

    # draw all pred boxes
    for per_score, per_class_index, per_box in zip(scores, classes, boxes):
        per_score = per_score.numpy().astype(np.float32)
        per_class_index = per_class_index.numpy().astype(np.int32)
        per_box = per_box.numpy().astype(np.int32)

        class_name, class_color = dataset_classes_name[
            per_class_index], dataset_classes_color[per_class_index]

        left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                            per_box[3])
        cv2.rectangle(origin_img,
                      left_top,
                      right_bottom,
                      color=class_color,
                      thickness=2,
                      lineType=cv2.LINE_AA)

        text = f'{class_name}:{per_score:.3f}'
        text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
                             left_top[1] - text_size[1] - 3)
        cv2.rectangle(origin_img,
                      left_top,
                      fill_right_bottom,
                      color=class_color,
                      thickness=-1,
                      lineType=cv2.LINE_AA)
        cv2.putText(origin_img,
                    text, (left_top[0], left_top[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    if args.save_image_path:
        cv2.imwrite(os.path.join(args.save_image_path, 'detection_result.jpg'),
                    origin_img)

    if args.show_image:
        cv2.namedWindow("detection_result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('detection_result', origin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    inference()