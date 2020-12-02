import time
import random
import argparse
import json
import os
import sys
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
from torch.utils.data import DataLoader
from public.path import VOCdataset_path
from public.detection.dataset.vocdataset import collater
from public.detection.models.retinanet import RetinaNet
from public.detection.models.fcos import FCOS
from public.detection.models.centernet import CenterNet
from public.detection.models.yolov3 import YOLOV3
from public.detection.models.decode import RetinaDecoder, FCOSDecoder, CenterNetDecoder, YOLOV3Decoder
from public.detection.dataset.vocdataset import VocDetection, Normalize, Resize


def _retinanet(arch, use_pretrained_model, pretrained_model_path, num_classes):
    model = RetinaNet(arch, num_classes=num_classes)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def _fcos(arch, use_pretrained_model, pretrained_model_path, num_classes):
    model = FCOS(arch, num_classes=num_classes)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def _centernet(arch, use_pretrained_model, pretrained_model_path, num_classes):
    model = CenterNet(arch, num_classes=num_classes)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def _yolov3(arch, use_pretrained_model, pretrained_model_path, num_classes):
    model = YOLOV3(arch, num_classes=num_classes)
    if use_pretrained_model:
        pretrained_models = torch.load(pretrained_model_path,
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def compute_voc_ap(recall, precision, use_07_metric=True):
    if use_07_metric:
        # use voc 2007 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                # get max precision  for recall >= t
                p = np.max(precision[recall >= t])
            # average 11 recall point precision
            ap = ap + p / 11.
    else:
        # use voc>=2010 metric,average all different recall precision as ap
        # recall add first value 0. and last value 1.
        mrecall = np.concatenate(([0.], recall, [1.]))
        # precision add first value 0. and last value 0.
        mprecision = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mprecision.size - 1, 0, -1):
            mprecision[i - 1] = np.maximum(mprecision[i - 1], mprecision[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrecall[1:] != mrecall[:-1])[0]

        # sum (\Delta recall) * prec
        ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])

    return ap


def compute_ious(a, b):
    """
    :param a: [N,(x1,y1,x2,y2)]
    :param b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """

    a = np.expand_dims(a, axis=1)  # [N,1,4]
    b = np.expand_dims(b, axis=0)  # [1,M,4]

    overlap = np.maximum(0.0,
                         np.minimum(a[..., 2:], b[..., 2:]) -
                         np.maximum(a[..., :2], b[..., :2]))  # [N,M,(w,h)]

    overlap = np.prod(overlap, axis=-1)  # [N,M]

    area_a = np.prod(a[..., 2:] - a[..., :2], axis=-1)
    area_b = np.prod(b[..., 2:] - b[..., :2], axis=-1)

    iou = overlap / (area_a + area_b - overlap)

    return iou


def validate(val_dataset, model, decoder, args):
    model = model.module
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        all_ap, mAP = evaluate_voc(val_dataset,
                                   model,
                                   decoder,
                                   num_classes=args.num_classes,
                                   iou_thread=0.5)

    return all_ap, mAP


def evaluate_voc(val_dataset, model, decoder, num_classes, iou_thread=0.5):
    preds, gts = [], []
    indexes = []
    for index in range(len(val_dataset)):
        indexes.append(index)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collater)

    start_time = time.time()

    for data in tqdm(val_loader):
        images, gt_annots, scales = torch.tensor(data['img']), torch.tensor(
            data['annot']), torch.tensor(data['scale'])
        gt_bboxes, gt_classes = gt_annots[:, :, 0:4], gt_annots[:, :, 4]
        gt_bboxes /= scales.unsqueeze(-1).unsqueeze(-1)

        for per_img_gt_bboxes, per_img_gt_classes in zip(
                gt_bboxes, gt_classes):
            per_img_gt_bboxes = per_img_gt_bboxes[per_img_gt_classes > -1]
            per_img_gt_classes = per_img_gt_classes[per_img_gt_classes > -1]
            gts.append([per_img_gt_bboxes, per_img_gt_classes])

        if args.use_gpu:
            images = images.cuda().float()
        else:
            images = images.float()

        if args.detector == "retinanet":
            cls_heads, reg_heads, batch_anchors = model(images)
            scores, classes, boxes = decoder(cls_heads, reg_heads,
                                             batch_anchors)
        elif args.detector == "fcos":
            cls_heads, reg_heads, center_heads, batch_positions = model(images)
            scores, classes, boxes = decoder(cls_heads, reg_heads,
                                             center_heads, batch_positions)
        elif args.detector == "centernet":
            heatmap_output, offset_output, wh_output = model(images)
            scores, classes, boxes = decoder(heatmap_output, offset_output,
                                             wh_output)
        elif args.detector == "yolov3":
            obj_heads, reg_heads, cls_heads, batch_anchors = model(images)
            scores, classes, boxes = decoder(obj_heads, reg_heads, cls_heads,
                                             batch_anchors)

        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        scales = scales.unsqueeze(-1).unsqueeze(-1)
        boxes /= scales

        for per_img_scores, per_img_classes, per_img_boxes in zip(
                scores, classes, boxes):
            per_img_scores = per_img_scores[per_img_classes > -1]
            per_img_boxes = per_img_boxes[per_img_classes > -1]
            per_img_classes = per_img_classes[per_img_classes > -1]
            preds.append([per_img_boxes, per_img_classes, per_img_scores])

    print("all val sample decode done.")
    testing_time = (time.time() - start_time)
    per_image_testing_time = testing_time / len(val_dataset)

    print(f"per_image_testing_time:{per_image_testing_time:.3f}")

    all_ap = {}
    for class_index in tqdm(range(num_classes)):
        per_class_gt_boxes = [
            image[0][image[1] == class_index] for image in gts
        ]
        per_class_pred_boxes = [
            image[0][image[1] == class_index] for image in preds
        ]
        per_class_pred_scores = [
            image[2][image[1] == class_index] for image in preds
        ]

        fp = np.zeros((0, ))
        tp = np.zeros((0, ))
        scores = np.zeros((0, ))
        total_gts = 0

        # loop for each sample
        for per_image_gt_boxes, per_image_pred_boxes, per_image_pred_scores in zip(
                per_class_gt_boxes, per_class_pred_boxes,
                per_class_pred_scores):
            total_gts = total_gts + len(per_image_gt_boxes)
            # one gt can only be assigned to one predicted bbox
            assigned_gt = []
            # loop for each predicted bbox
            for index in range(len(per_image_pred_boxes)):
                scores = np.append(scores, per_image_pred_scores[index])
                if per_image_gt_boxes.shape[0] == 0:
                    # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(per_image_pred_boxes[index], axis=0)
                iou = compute_ious(per_image_gt_boxes, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = compute_voc_ap(recall, precision)
        all_ap[class_index] = ap

    mAP = 0.
    for _, class_mAP in all_ap.items():
        mAP += float(class_mAP)
    mAP /= num_classes

    return all_ap, mAP


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

    voc_val_dataset = VocDetection(root_dir=VOCdataset_path,
                                   image_sets=[('2007', 'test')],
                                   transform=transforms.Compose([
                                       Normalize(),
                                       Resize(resize=args.input_image_size),
                                   ]),
                                   keep_difficult=False)

    if args.detector == "retinanet":
        model = _retinanet(args.backbone, args.use_pretrained_model,
                           args.pretrained_model_path, args.num_classes)
        decoder = RetinaDecoder(image_w=args.input_image_size,
                                image_h=args.input_image_size,
                                min_score_threshold=args.min_score_threshold)
    elif args.detector == "fcos":
        model = _fcos(args.backbone, args.use_pretrained_model,
                      args.pretrained_model_path, args.num_classes)
        decoder = FCOSDecoder(image_w=args.input_image_size,
                              image_h=args.input_image_size,
                              min_score_threshold=args.min_score_threshold)
    elif args.detector == "centernet":
        model = _centernet(args.backbone, args.use_pretrained_model,
                           args.pretrained_model_path, args.num_classes)
        decoder = CenterNetDecoder(
            image_w=args.input_image_size,
            image_h=args.input_image_size,
            min_score_threshold=args.min_score_threshold)
    elif args.detector == "yolov3":
        model = _yolov3(args.backbone, args.use_pretrained_model,
                        args.pretrained_model_path, args.num_classes)
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

    if args.use_gpu:
        model = model.cuda()
        decoder = decoder.cuda()
        model = nn.DataParallel(model)

    print(f"start eval.")
    all_ap, mAP = validate(voc_val_dataset, model, decoder, args)
    print(f"eval done.")
    for class_index, class_AP in all_ap.items():
        print(f"class: {class_index},AP: {class_AP:.3f}")
    print(f"mAP: {mAP:.3f}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch VOC Detection Testing')
    parser.add_argument('--backbone', type=str, help='name of backbone')
    parser.add_argument('--detector', type=str, help='name of detector')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='inference batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='num workers')
    parser.add_argument('--num_classes',
                        type=int,
                        default=20,
                        help='model class num')
    parser.add_argument('--min_score_threshold',
                        type=float,
                        default=0.05,
                        help='min score threshold')
    parser.add_argument("--use_pretrained_model",
                        action="store_true",
                        help="use pretrained model or not")
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
    args = parser.parse_args()
    test_model(args)
