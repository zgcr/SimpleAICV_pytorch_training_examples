import time
import random
import argparse
import json
import os
import sys
import cv2
import warnings
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from tqdm import tqdm
from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from public.detection.dataset.cocodataset import collater
from public.detection.models.retinanet import RetinaNet
from public.detection.models.fcos import FCOS
from public.detection.models.centernet import CenterNet
from public.detection.models.yolov3 import YOLOV3
from public.detection.models.decode import RetinaDecoder, FCOSDecoder, CenterNetDecoder, YOLOV3Decoder
from public.detection.dataset.cocodataset import CocoDetection, Resize
from pycocotools.cocoeval import COCOeval
from public.detection.dataset.cocodataset import COCO_CLASSES, coco_class_colors
from public.detection.dataset.vocdataset import VOC_CLASSES, voc_class_colors


def _retinanet(arch, pretrained_model_path, num_classes):
    model = RetinaNet(arch, num_classes=num_classes)

    pretrained_models = torch.load(pretrained_model_path,
                                   map_location=torch.device('cpu'))

    # only load state_dict()
    model.load_state_dict(pretrained_models, strict=False)

    return model


def _fcos(arch, pretrained_model_path, num_classes):
    model = FCOS(arch, num_classes=num_classes)

    pretrained_models = torch.load(pretrained_model_path,
                                   map_location=torch.device('cpu'))

    # only load state_dict()
    model.load_state_dict(pretrained_models, strict=False)

    return model


def _centernet(arch, pretrained_model_path, num_classes):
    model = CenterNet(arch, num_classes=num_classes)

    pretrained_models = torch.load(pretrained_model_path,
                                   map_location=torch.device('cpu'))

    # only load state_dict()
    model.load_state_dict(pretrained_models, strict=False)

    return model


def _yolov3(arch, pretrained_model_path, num_classes):
    model = YOLOV3(arch, num_classes=num_classes)

    pretrained_models = torch.load(pretrained_model_path,
                                   map_location=torch.device('cpu'))

    # only load state_dict()
    model.load_state_dict(pretrained_models, strict=False)

    return model


def test_model(args):
    print(args)
    if args.use_gpu:
        # use one Graphics card to test
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if not torch.cuda.is_available():
            raise Exception("need gpu to test network!")
        torch.cuda.empty_cache()

    if args.seed is not None:
        random.seed(args.seed)
        if args.use_gpu:
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = True

    if args.use_gpu:
        cudnn.benchmark = True
        cudnn.enabled = True

    if args.detector == "retinanet":
        model = _retinanet(args.backbone, args.pretrained_model_path,
                           args.num_classes)
        decoder = RetinaDecoder(image_w=args.input_image_size,
                                image_h=args.input_image_size,
                                min_score_threshold=args.min_score_threshold)
    elif args.detector == "fcos":
        model = _fcos(args.backbone, args.pretrained_model_path,
                      args.num_classes)
        decoder = FCOSDecoder(image_w=args.input_image_size,
                              image_h=args.input_image_size,
                              min_score_threshold=args.min_score_threshold)
    elif args.detector == "centernet":
        model = _centernet(args.backbone, args.pretrained_model_path,
                           args.num_classes)
        decoder = CenterNetDecoder(
            image_w=args.input_image_size,
            image_h=args.input_image_size,
            min_score_threshold=args.min_score_threshold)
    elif args.detector == "yolov3":
        model = _yolov3(args.backbone, args.pretrained_model_path,
                        args.num_classes)
        decoder = YOLOV3Decoder(image_w=args.input_image_size,
                                image_h=args.input_image_size,
                                min_score_threshold=args.min_score_threshold)
    else:
        print("unsupport detection model!")
        return

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(
        f"backbone:{args.backbone},detector: '{args.detector}', flops: {flops}, params: {params}"
    )

    model.eval()

    if args.use_gpu:
        model = model.cuda()
        decoder = decoder.cuda()
        model = nn.DataParallel(model)

    # load image and image preprocessing
    img = cv2.imread(args.test_image_path)
    origin_img = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    height, width, _ = img.shape
    max_image_size = max(height, width)
    resize_factor = args.input_image_size / max_image_size
    resize_height, resize_width = int(height * resize_factor), int(
        width * resize_factor)
    img = cv2.resize(img, (resize_width, resize_height))
    resized_img = np.zeros((args.input_image_size, args.input_image_size, 3))
    resized_img[0:resize_height, 0:resize_width] = img
    scale = resize_factor
    resized_img = torch.tensor(resized_img)

    print(resized_img.shape)

    if args.use_gpu:
        resized_img = resized_img.cuda()
    # inference image
    cls_heads, reg_heads, batch_anchors = model(
        resized_img.permute(2, 0, 1).float().unsqueeze(0))
    scores, classes, boxes = decoder(cls_heads, reg_heads, batch_anchors)
    scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
    # snap boxes to fit origin image
    boxes /= scale

    scores = scores.squeeze(0)
    classes = classes.squeeze(0)
    boxes = boxes.squeeze(0)

    # draw all boxes
    for per_score, per_class_index, per_box in zip(scores, classes, boxes):
        per_score = per_score.numpy()
        per_class_index = per_class_index.numpy().astype(np.int32)
        per_box = per_box.numpy().astype(np.int32)

        class_name = COCO_CLASSES[per_class_index]
        color = coco_class_colors[per_class_index]

        text = '{}:{:.3f}'.format(class_name, per_score)

        cv2.putText(origin_img,
                    text, (per_box[0], per_box[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    color=color,
                    thickness=2)
        cv2.rectangle(origin_img, (per_box[0], per_box[1]),
                      (per_box[2], per_box[3]),
                      color=color,
                      thickness=2)

    if args.save_detected_image:
        cv2.imwrite('detection_result.jpg', origin_img)

    cv2.imshow('detection_result', origin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Single Image Detection Testing')
    parser.add_argument('--backbone', type=str, help='name of backbone')
    parser.add_argument('--detector', type=str, help='name of detector')
    parser.add_argument('--num_classes',
                        type=int,
                        default=80,
                        help='model class num')
    parser.add_argument('--min_score_threshold',
                        type=float,
                        default=0.3,
                        help='min score threshold')
    parser.add_argument('--pretrained_model_path',
                        type=str,
                        help='pretrained model path')
    parser.add_argument("--use_gpu",
                        action="store_true",
                        help="use gpu to test or not")
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=667,
                        help='input image size')
    parser.add_argument('--test_image_path', type=str, help='test image path')
    parser.add_argument("--save_detected_image",
                        action="store_true",
                        help="save detected image or not")
    args = parser.parse_args()
    test_model(args)