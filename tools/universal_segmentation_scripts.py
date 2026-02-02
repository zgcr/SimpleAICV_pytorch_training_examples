import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import collections
import numpy as np
import time
from tqdm import tqdm

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp.autocast_mode import autocast

from SimpleAICV.classification.common import get_amp_type, AverageMeter
from tools.scripts import all_reduce_operation_in_group_for_variables


def test_semantic_segmentation_dataset(test_loader, model, decoder, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    total_area_intersect = torch.zeros((config.num_classes, ),
                                       dtype=torch.float64).cuda()
    total_area_pred = torch.zeros((config.num_classes, ),
                                  dtype=torch.float64).cuda()
    total_area_gt = torch.zeros((config.num_classes, ),
                                dtype=torch.float64).cuda()
    total_area_union = torch.zeros((config.num_classes, ),
                                   dtype=torch.float64).cuda()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            images, masks, sizes, origin_sizes = data['image'], data[
                'mask'], data['size'], data['origin_size']
            if model_on_cuda:
                images, masks = images.cuda(), masks.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outs = model(images)
            batch_masks, batch_scores, batch_classes = decoder(
                outs, sizes, origin_sizes)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            for per_image_pred_masks, per_image_scores, per_image_classes, per_image_mask, per_image_size in zip(
                    batch_masks, batch_scores, batch_classes, masks, sizes):
                if len(per_image_pred_masks) == 0:
                    per_image_pred = np.zeros(
                        (int(per_image_size[0]), int(per_image_size[1])),
                        dtype=np.float32)
                    per_image_pred = torch.from_numpy(per_image_pred).cuda()
                else:
                    # [N,H,W]->[H,W,N]
                    per_image_pred_masks = per_image_pred_masks.transpose(
                        1, 2, 0)
                    per_image_pred_masks = cv2.resize(
                        per_image_pred_masks,
                        (int(per_image_size[1]), int(per_image_size[0])),
                        interpolation=cv2.INTER_NEAREST)

                    if len(per_image_pred_masks.shape) != 3:
                        per_image_pred_masks = np.expand_dims(
                            per_image_pred_masks, axis=-1)

                    # [H,W,N]->[N,H,W]
                    per_image_pred_masks = per_image_pred_masks.transpose(
                        2, 0, 1)

                    # per_image_classes前景类别从0开始, per_image_mask前景类别从1开始,需要+1才能对齐
                    per_image_classes = per_image_classes + 1

                    # 预测结果合并到一张mask上
                    per_image_pred = np.zeros(
                        (int(per_image_size[0]), int(per_image_size[1])),
                        dtype=np.float32)
                    for per_object_pred_mask, per_object_class in zip(
                            per_image_pred_masks, per_image_classes):
                        per_image_pred[per_object_pred_mask >
                                       0] = per_object_class
                    per_image_pred = torch.from_numpy(per_image_pred).cuda()

                per_image_mask = per_image_mask[0:int(per_image_size[0]),
                                                0:int(per_image_size[1])]

                # per_image_pred:[h,w,c] -> (-1)
                # per_image_mask:[h,w] -> (-1)
                per_image_pred, per_image_mask = per_image_pred.reshape(
                    -1), per_image_mask.reshape(-1)

                per_image_intersect = per_image_pred[per_image_pred ==
                                                     per_image_mask]

                per_image_intersect_area = torch.histc(
                    per_image_intersect.float(),
                    bins=(config.num_classes),
                    min=0,
                    max=config.num_classes - 1)
                per_image_pred_area = torch.histc(per_image_pred.float(),
                                                  bins=(config.num_classes),
                                                  min=0,
                                                  max=config.num_classes - 1)
                per_image_mask_area = torch.histc(per_image_mask.float(),
                                                  bins=(config.num_classes),
                                                  min=0,
                                                  max=config.num_classes - 1)

                per_image_union_area = per_image_pred_area + per_image_mask_area - per_image_intersect_area

                total_area_intersect += per_image_intersect_area.double()
                total_area_pred += +per_image_pred_area.double()
                total_area_gt += per_image_mask_area.double()
                total_area_union += per_image_union_area.double()

            end = time.time()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / (config.batch_size //
                                               config.gpus_num) * 1000
        per_image_inference_time = batch_time.avg / (config.batch_size //
                                                     config.gpus_num) * 1000

        result_dict = collections.OrderedDict()
        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        per_class_precisions = torch.zeros((config.num_classes, ),
                                           dtype=torch.float64).cuda()
        per_class_recalls = torch.zeros((config.num_classes, ),
                                        dtype=torch.float64).cuda()
        per_class_ious = torch.zeros((config.num_classes, ),
                                     dtype=torch.float64).cuda()
        per_class_dices = torch.zeros((config.num_classes, ),
                                      dtype=torch.float64).cuda()

        exist_num_class = 0.
        mean_precision, mean_recall, mean_iou, mean_dice = 0., 0., 0., 0.
        for i, (per_class_area_intersect, per_class_area_pred,
                per_class_area_gt, per_class_area_union) in enumerate(
                    zip(total_area_intersect, total_area_pred, total_area_gt,
                        total_area_union)):
            if per_class_area_gt == 0:
                continue

            exist_num_class += 1.

            if per_class_area_pred != 0:
                per_class_precisions[i] = (per_class_area_intersect /
                                           per_class_area_pred) * 100.
            mean_precision += per_class_precisions[i]

            if per_class_area_gt != 0:
                per_class_recalls[i] = (per_class_area_intersect /
                                        per_class_area_gt) * 100.
            mean_recall += per_class_recalls[i]

            if per_class_area_union != 0:
                per_class_ious[i] = (per_class_area_intersect /
                                     per_class_area_union) * 100.
            mean_iou += per_class_ious[i]

            if (per_class_area_pred + per_class_area_gt) != 0:
                per_class_dices[i] = 2. * (
                    per_class_area_intersect /
                    (per_class_area_pred + per_class_area_gt)) * 100.
            mean_dice += per_class_dices[i]

        if exist_num_class > 0:
            mean_precision = mean_precision / exist_num_class
            mean_recall = mean_recall / exist_num_class
            mean_iou = mean_iou / exist_num_class
            mean_dice = mean_dice / exist_num_class

        result_dict['exist_num_class'] = exist_num_class
        result_dict['mean_precision'] = mean_precision
        result_dict['mean_recall'] = mean_recall
        result_dict['mean_iou'] = mean_iou
        result_dict['mean_dice'] = mean_dice

    precision_dict = collections.OrderedDict()
    for i, per_precision in enumerate(per_class_precisions):
        precision_dict[f'class_{i}_precision'] = per_precision

    recall_dict = collections.OrderedDict()
    for i, per_recall in enumerate(per_class_recalls):
        recall_dict[f'class_{i}_recall'] = per_recall

    iou_dict = collections.OrderedDict()
    for i, per_iou in enumerate(per_class_ious):
        iou_dict[f'class_{i}_iou'] = per_iou

    dice_dict = collections.OrderedDict()
    for i, per_dice in enumerate(per_class_dices):
        dice_dict[f'class_{i}_dice'] = per_dice

    return result_dict


def evaluate_coco_instance_segmentation_dataset(test_loader, model, decoder,
                                                config):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    test_dataset = config.test_dataset
    ids = [idx for idx in range(len(test_dataset))]
    batch_size = int(config.batch_size // config.gpus_num)

    with torch.no_grad():
        results, image_ids = [], []
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for i, data in tqdm(enumerate(test_loader)):
            images = data['image']
            if model_on_cuda:
                images = images.cuda()

            scaled_size = data['size']
            origin_size = data['origin_size']

            per_batch_ids = ids[i * batch_size:(i + 1) * batch_size]

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            outs = model(images)
            batch_masks, batch_scores, batch_classes = decoder(
                outs, scaled_size, origin_size)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            for per_image_masks, per_image_classes, per_image_scores, index in zip(
                    batch_masks, batch_classes, batch_scores, per_batch_ids):

                for per_mask, per_class, per_score in zip(
                        per_image_masks, per_image_classes, per_image_scores):
                    rle = mask_util.encode(
                        np.array(per_mask[:, :, np.newaxis],
                                 order='F'))[0].copy()
                    rle['counts'] = rle['counts'].decode()
                    image_result = {
                        'image_id': test_dataset.image_ids[index],
                        'category_id':
                        test_dataset.coco_label_to_cat_id[per_class],
                        'score': float(per_score),
                        'segmentation': rle,
                    }
                    results.append(image_result)

                image_ids.append(test_dataset.image_ids[index])

                print('{}/{}'.format(index, len(test_dataset)), end='\r')

            end = time.time()

        variable_definitions = {
            0: 'IoU=0.50:0.95,area=all,maxDets=100,mAP',
            1: 'IoU=0.50,area=all,maxDets=100,mAP',
            2: 'IoU=0.75,area=all,maxDets=100,mAP',
            3: 'IoU=0.50:0.95,area=small,maxDets=100,mAP',
            4: 'IoU=0.50:0.95,area=medium,maxDets=100,mAP',
            5: 'IoU=0.50:0.95,area=large,maxDets=100,mAP',
            6: 'IoU=0.50:0.95,area=all,maxDets=1,mAR',
            7: 'IoU=0.50:0.95,area=all,maxDets=10,mAR',
            8: 'IoU=0.50:0.95,area=all,maxDets=100,mAR',
            9: 'IoU=0.50:0.95,area=small,maxDets=100,mAR',
            10: 'IoU=0.50:0.95,area=medium,maxDets=100,mAR',
            11: 'IoU=0.50:0.95,area=large,maxDets=100,mAR',
        }

        result_dict = collections.OrderedDict()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / batch_size * 1000
        per_image_inference_time = batch_time.avg / batch_size * 1000

        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        if len(results) == 0:
            for _, value in variable_definitions.items():
                result_dict[value] = 0
            return result_dict

        # load results in COCO evaluation tool
        coco_true = test_dataset.coco
        coco_pred = coco_true.loadRes(results)

        coco_eval = COCOeval(coco_true, coco_pred, 'segm')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        eval_result = coco_eval.stats

        for i, var in enumerate(eval_result):
            result_dict[variable_definitions[i]] = var * 100

        return result_dict


def test_instance_segmentation_dataset(test_loader, model, decoder, config):
    assert config.eval_type in ['COCO']

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    func_dict = {
        'COCO': evaluate_coco_instance_segmentation_dataset,
    }
    result_dict = func_dict[config.eval_type](test_loader, model, decoder,
                                              config)

    return result_dict


class EvalMeter:

    def __init__(self, config):
        self.thresh = config.thresh
        self.squared_beta = config.squared_beta
        self.thresh_num = len(self.thresh)

        self.precision_list = np.zeros(self.thresh_num, dtype=np.float32)
        self.recall_list = np.zeros(self.thresh_num, dtype=np.float32)
        self.miou_list = np.zeros(self.thresh_num, dtype=np.float32)
        self.sample_num = 0
        self.f_squared_beta_list = []

        self.f_squared_beta_average = 0
        self.f_squared_beta_max = 0
        self.miou_average = 0
        self.miou_max = 0
        self.precision_average = 0
        self.recall_average = 0
        self.precision_max = 0
        self.recall_max = 0

    def add_batch_result(self, preds, masks):
        # pred shape:[b,c,h,w] -> [b,h,w,c]
        # label shape:[b,h,w]
        preds = preds.permute(0, 2, 3, 1).contiguous()
        num_classes = preds.shape[3]
        assert num_classes == 1
        preds = torch.squeeze(preds, dim=-1)
        preds, masks = preds.cpu().numpy(), masks.cpu().numpy()

        for i in range(self.thresh_num):
            pred_foreground = preds > self.thresh[i]
            mask_foreground = masks > self.thresh[i]

            intersection = np.sum(np.sum(pred_foreground & mask_foreground,
                                         axis=1),
                                  axis=1)
            all_masks = np.sum(np.sum(mask_foreground, axis=1), axis=1)
            all_preds = np.sum(np.sum(pred_foreground, axis=1), axis=1)
            union = all_preds + all_masks - intersection

            self.precision_list[i] += np.sum(intersection / (all_preds + 1e-4))
            self.recall_list[i] += np.sum(intersection / (all_masks + 1e-4))
            self.miou_list[i] += np.sum(intersection / (union + 1e-4))

        self.sample_num = self.sample_num + masks.shape[0]

    def compute_all_metrics(self):
        self.precision_list = self.precision_list / self.sample_num
        self.recall_list = self.recall_list / self.sample_num
        self.miou_list = self.miou_list / self.sample_num
        self.f_squared_beta_list = (
            1 + self.squared_beta) * self.precision_list * self.recall_list / (
                self.squared_beta * self.precision_list + self.recall_list +
                1e-4)

        self.f_squared_beta_average = np.mean(self.f_squared_beta_list)
        self.f_squared_beta_max = np.max(self.f_squared_beta_list)
        self.miou_average = np.mean(self.miou_list)
        self.miou_max = np.max(self.miou_list)
        self.precision_average = np.mean(self.precision_list)
        self.precision_max = np.max(self.precision_list)
        self.recall_average = np.mean(self.recall_list)
        self.recall_max = np.max(self.recall_list)


def validate_salient_object_detection_segmentation_for_all_dataset(
        val_loader_list, model, decoder, config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.val_dataset_name_list, val_loader_list,
                        config.val_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_salient_object_detection_segmentation(
            per_sub_dataset_loader, model, decoder, config)
        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_salient_object_detection_segmentation(test_loader, model, decoder,
                                                   config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    eval_metric = EvalMeter(config)

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            images, masks, sizes, origin_sizes = data['image'], data[
                'mask'], data['size'], data['origin_size']
            if model_on_cuda:
                images, masks = images.cuda(), masks.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outs = model(images)
            batch_masks, batch_scores, batch_classes = decoder(
                outs, sizes, origin_sizes)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            mask_h, mask_w = masks.shape[1], masks.shape[2]

            outputs = []
            for per_image_preds, per_image_sizes in zip(batch_masks, sizes):
                if len(per_image_preds) == 0:
                    per_image_preds = np.zeros(
                        (int(per_image_sizes[0]), int(per_image_sizes[1])),
                        dtype=np.float32)
                else:
                    per_image_preds = np.squeeze(per_image_preds, axis=0)
                    per_image_preds = cv2.resize(
                        per_image_preds,
                        (int(per_image_sizes[1]), int(per_image_sizes[0])))

                padding_per_image_preds = np.zeros((mask_h, mask_w),
                                                   dtype=np.float32)
                padding_per_image_preds[
                    0:per_image_preds.shape[0],
                    0:per_image_preds.shape[1]] = per_image_preds
                padding_per_image_preds = torch.from_numpy(
                    padding_per_image_preds).float().cuda()
                padding_per_image_preds = padding_per_image_preds.unsqueeze(
                    0).unsqueeze(0)

                outputs.append(padding_per_image_preds)

            outputs = torch.cat(outputs, dim=0)

            eval_metric.add_batch_result(outputs, masks)

            end = time.time()

        eval_metric.compute_all_metrics()

        miou_average = eval_metric.miou_average
        miou_max = eval_metric.miou_max
        precision_average = eval_metric.precision_average
        precision_max = eval_metric.precision_max
        recall_average = eval_metric.recall_average
        recall_max = eval_metric.recall_max
        f_squared_beta_average = eval_metric.f_squared_beta_average
        f_squared_beta_max = eval_metric.f_squared_beta_max

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / (config.batch_size //
                                               config.gpus_num) * 1000
        per_image_inference_time = batch_time.avg / (config.batch_size //
                                                     config.gpus_num) * 1000

        result_dict = collections.OrderedDict()

        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        result_dict['f_squared_beta_average'] = f_squared_beta_average
        result_dict['f_squared_beta_max'] = f_squared_beta_max
        result_dict['mean_precision'] = precision_average
        result_dict['mean_recall'] = recall_average
        result_dict['max_precision'] = precision_max
        result_dict['max_recall'] = recall_max
        result_dict['miou_average'] = miou_average
        result_dict['miou_max'] = miou_max

    return result_dict


def validate_face_parsing_for_all_dataset(val_loader_list, model, decoder,
                                          config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.val_dataset_name_list, val_loader_list,
                        config.val_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_face_parsing(per_sub_dataset_loader,
                                                      model, decoder, config)
        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_face_parsing(test_loader, model, decoder, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    total_area_intersect = torch.zeros((config.num_classes, ),
                                       dtype=torch.float64).cuda()
    total_area_pred = torch.zeros((config.num_classes, ),
                                  dtype=torch.float64).cuda()
    total_area_gt = torch.zeros((config.num_classes, ),
                                dtype=torch.float64).cuda()
    total_area_union = torch.zeros((config.num_classes, ),
                                   dtype=torch.float64).cuda()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            images, masks, sizes, origin_sizes = data['image'], data[
                'mask'], data['size'], data['origin_size']
            if model_on_cuda:
                images, masks = images.cuda(), masks.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outs = model(images)
            batch_masks, batch_scores, batch_classes = decoder(
                outs, sizes, origin_sizes)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            for per_image_pred_masks, per_image_scores, per_image_classes, per_image_mask, per_image_size in zip(
                    batch_masks, batch_scores, batch_classes, masks, sizes):
                if len(per_image_pred_masks) == 0:
                    per_image_pred = np.zeros(
                        (int(per_image_size[0]), int(per_image_size[1])),
                        dtype=np.float32)
                    per_image_pred = torch.from_numpy(per_image_pred).cuda()
                else:
                    # [N,H,W]->[H,W,N]
                    per_image_pred_masks = per_image_pred_masks.transpose(
                        1, 2, 0)
                    per_image_pred_masks = cv2.resize(
                        per_image_pred_masks,
                        (int(per_image_size[1]), int(per_image_size[0])),
                        interpolation=cv2.INTER_NEAREST)

                    if len(per_image_pred_masks.shape) != 3:
                        per_image_pred_masks = np.expand_dims(
                            per_image_pred_masks, axis=-1)

                    # [H,W,N]->[N,H,W]
                    per_image_pred_masks = per_image_pred_masks.transpose(
                        2, 0, 1)

                    # per_image_classes前景类别从0开始, per_image_mask前景类别从1开始,需要+1才能对齐
                    per_image_classes = per_image_classes + 1

                    # 预测结果合并到一张mask上
                    per_image_pred = np.zeros(
                        (int(per_image_size[0]), int(per_image_size[1])),
                        dtype=np.float32)
                    for per_object_pred_mask, per_object_class in zip(
                            per_image_pred_masks, per_image_classes):
                        per_image_pred[per_object_pred_mask >
                                       0] = per_object_class
                    per_image_pred = torch.from_numpy(per_image_pred).cuda()

                per_image_mask = per_image_mask[0:int(per_image_size[0]),
                                                0:int(per_image_size[1])]

                # per_image_pred:[h,w] -> (-1)
                # per_image_mask:[h,w] -> (-1)
                per_image_pred, per_image_mask = per_image_pred.reshape(
                    -1), per_image_mask.reshape(-1)

                per_image_intersect = per_image_pred[per_image_pred ==
                                                     per_image_mask]

                per_image_intersect_area = torch.histc(
                    per_image_intersect.float(),
                    bins=(config.num_classes),
                    min=0,
                    max=config.num_classes - 1)
                per_image_pred_area = torch.histc(per_image_pred.float(),
                                                  bins=(config.num_classes),
                                                  min=0,
                                                  max=config.num_classes - 1)
                per_image_mask_area = torch.histc(per_image_mask.float(),
                                                  bins=(config.num_classes),
                                                  min=0,
                                                  max=config.num_classes - 1)

                per_image_union_area = per_image_pred_area + per_image_mask_area - per_image_intersect_area

                total_area_intersect += per_image_intersect_area.double()
                total_area_pred += +per_image_pred_area.double()
                total_area_gt += per_image_mask_area.double()
                total_area_union += per_image_union_area.double()

            end = time.time()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / (config.batch_size //
                                               config.gpus_num) * 1000
        per_image_inference_time = batch_time.avg / (config.batch_size //
                                                     config.gpus_num) * 1000

        result_dict = collections.OrderedDict()
        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        per_class_precisions = torch.zeros((config.num_classes, ),
                                           dtype=torch.float64).cuda()
        per_class_recalls = torch.zeros((config.num_classes, ),
                                        dtype=torch.float64).cuda()
        per_class_ious = torch.zeros((config.num_classes, ),
                                     dtype=torch.float64).cuda()
        per_class_dices = torch.zeros((config.num_classes, ),
                                      dtype=torch.float64).cuda()

        exist_num_class = 0.
        mean_precision, mean_recall, mean_iou, mean_dice = 0., 0., 0., 0.
        for i, (per_class_area_intersect, per_class_area_pred,
                per_class_area_gt, per_class_area_union) in enumerate(
                    zip(total_area_intersect, total_area_pred, total_area_gt,
                        total_area_union)):
            if per_class_area_gt == 0:
                continue

            exist_num_class += 1.

            if per_class_area_pred != 0:
                per_class_precisions[i] = (per_class_area_intersect /
                                           per_class_area_pred) * 100.
            mean_precision += per_class_precisions[i]

            if per_class_area_gt != 0:
                per_class_recalls[i] = (per_class_area_intersect /
                                        per_class_area_gt) * 100.
            mean_recall += per_class_recalls[i]

            if per_class_area_union != 0:
                per_class_ious[i] = (per_class_area_intersect /
                                     per_class_area_union) * 100.
            mean_iou += per_class_ious[i]

            if (per_class_area_pred + per_class_area_gt) != 0:
                per_class_dices[i] = 2. * (
                    per_class_area_intersect /
                    (per_class_area_pred + per_class_area_gt)) * 100.
            mean_dice += per_class_dices[i]

        if exist_num_class > 0:
            mean_precision = mean_precision / exist_num_class
            mean_recall = mean_recall / exist_num_class
            mean_iou = mean_iou / exist_num_class
            mean_dice = mean_dice / exist_num_class

        result_dict['exist_num_class'] = exist_num_class
        result_dict['mean_precision'] = mean_precision
        result_dict['mean_recall'] = mean_recall
        result_dict['mean_iou'] = mean_iou
        result_dict['mean_dice'] = mean_dice

    for i, per_precision in enumerate(per_class_precisions):
        result_dict[f'class_{i}_precision'] = per_precision

    for i, per_recall in enumerate(per_class_recalls):
        result_dict[f'class_{i}_recall'] = per_recall

    for i, per_iou in enumerate(per_class_ious):
        result_dict[f'class_{i}_iou'] = per_iou

    for i, per_dice in enumerate(per_class_dices):
        result_dict[f'class_{i}_dice'] = per_dice

    return result_dict


def validate_human_parsing_for_all_dataset(val_loader_list, model, decoder,
                                           config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.val_dataset_name_list, val_loader_list,
                        config.val_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_human_parsing(per_sub_dataset_loader,
                                                       model, decoder, config)
        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_human_parsing(test_loader, model, decoder, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    total_area_intersect = torch.zeros((config.num_classes, ),
                                       dtype=torch.float64).cuda()
    total_area_pred = torch.zeros((config.num_classes, ),
                                  dtype=torch.float64).cuda()
    total_area_gt = torch.zeros((config.num_classes, ),
                                dtype=torch.float64).cuda()
    total_area_union = torch.zeros((config.num_classes, ),
                                   dtype=torch.float64).cuda()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            images, masks, sizes, origin_sizes = data['image'], data[
                'mask'], data['size'], data['origin_size']
            if model_on_cuda:
                images, masks = images.cuda(), masks.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outs = model(images)
            batch_masks, batch_scores, batch_classes = decoder(
                outs, sizes, origin_sizes)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            for per_image_pred_masks, per_image_scores, per_image_classes, per_image_mask, per_image_size in zip(
                    batch_masks, batch_scores, batch_classes, masks, sizes):
                if len(per_image_pred_masks) == 0:
                    per_image_pred = np.zeros(
                        (int(per_image_size[0]), int(per_image_size[1])),
                        dtype=np.float32)
                    per_image_pred = torch.from_numpy(per_image_pred).cuda()
                else:
                    # [N,H,W]->[H,W,N]
                    per_image_pred_masks = per_image_pred_masks.transpose(
                        1, 2, 0)
                    per_image_pred_masks = cv2.resize(
                        per_image_pred_masks,
                        (int(per_image_size[1]), int(per_image_size[0])),
                        interpolation=cv2.INTER_NEAREST)

                    if len(per_image_pred_masks.shape) != 3:
                        per_image_pred_masks = np.expand_dims(
                            per_image_pred_masks, axis=-1)

                    # [H,W,N]->[N,H,W]
                    per_image_pred_masks = per_image_pred_masks.transpose(
                        2, 0, 1)

                    # per_image_classes前景类别从0开始, per_image_mask前景类别从1开始,需要+1才能对齐
                    per_image_classes = per_image_classes + 1

                    # 预测结果合并到一张mask上
                    per_image_pred = np.zeros(
                        (int(per_image_size[0]), int(per_image_size[1])),
                        dtype=np.float32)
                    for per_object_pred_mask, per_object_class in zip(
                            per_image_pred_masks, per_image_classes):
                        per_image_pred[per_object_pred_mask >
                                       0] = per_object_class
                    per_image_pred = torch.from_numpy(per_image_pred).cuda()

                per_image_mask = per_image_mask[0:int(per_image_size[0]),
                                                0:int(per_image_size[1])]

                # per_image_pred:[h,w] -> (-1)
                # per_image_mask:[h,w] -> (-1)
                per_image_pred, per_image_mask = per_image_pred.reshape(
                    -1), per_image_mask.reshape(-1)

                per_image_intersect = per_image_pred[per_image_pred ==
                                                     per_image_mask]

                per_image_intersect_area = torch.histc(
                    per_image_intersect.float(),
                    bins=(config.num_classes),
                    min=0,
                    max=config.num_classes - 1)
                per_image_pred_area = torch.histc(per_image_pred.float(),
                                                  bins=(config.num_classes),
                                                  min=0,
                                                  max=config.num_classes - 1)
                per_image_mask_area = torch.histc(per_image_mask.float(),
                                                  bins=(config.num_classes),
                                                  min=0,
                                                  max=config.num_classes - 1)

                per_image_union_area = per_image_pred_area + per_image_mask_area - per_image_intersect_area

                total_area_intersect += per_image_intersect_area.double()
                total_area_pred += +per_image_pred_area.double()
                total_area_gt += per_image_mask_area.double()
                total_area_union += per_image_union_area.double()

            end = time.time()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / (config.batch_size //
                                               config.gpus_num) * 1000
        per_image_inference_time = batch_time.avg / (config.batch_size //
                                                     config.gpus_num) * 1000

        result_dict = collections.OrderedDict()
        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        per_class_precisions = torch.zeros((config.num_classes, ),
                                           dtype=torch.float64).cuda()
        per_class_recalls = torch.zeros((config.num_classes, ),
                                        dtype=torch.float64).cuda()
        per_class_ious = torch.zeros((config.num_classes, ),
                                     dtype=torch.float64).cuda()
        per_class_dices = torch.zeros((config.num_classes, ),
                                      dtype=torch.float64).cuda()

        exist_num_class = 0.
        mean_precision, mean_recall, mean_iou, mean_dice = 0., 0., 0., 0.
        for i, (per_class_area_intersect, per_class_area_pred,
                per_class_area_gt, per_class_area_union) in enumerate(
                    zip(total_area_intersect, total_area_pred, total_area_gt,
                        total_area_union)):
            if per_class_area_gt == 0:
                continue

            exist_num_class += 1.

            if per_class_area_pred != 0:
                per_class_precisions[i] = (per_class_area_intersect /
                                           per_class_area_pred) * 100.
            mean_precision += per_class_precisions[i]

            if per_class_area_gt != 0:
                per_class_recalls[i] = (per_class_area_intersect /
                                        per_class_area_gt) * 100.
            mean_recall += per_class_recalls[i]

            if per_class_area_union != 0:
                per_class_ious[i] = (per_class_area_intersect /
                                     per_class_area_union) * 100.
            mean_iou += per_class_ious[i]

            if (per_class_area_pred + per_class_area_gt) != 0:
                per_class_dices[i] = 2. * (
                    per_class_area_intersect /
                    (per_class_area_pred + per_class_area_gt)) * 100.
            mean_dice += per_class_dices[i]

        if exist_num_class > 0:
            mean_precision = mean_precision / exist_num_class
            mean_recall = mean_recall / exist_num_class
            mean_iou = mean_iou / exist_num_class
            mean_dice = mean_dice / exist_num_class

        result_dict['exist_num_class'] = exist_num_class
        result_dict['mean_precision'] = mean_precision
        result_dict['mean_recall'] = mean_recall
        result_dict['mean_iou'] = mean_iou
        result_dict['mean_dice'] = mean_dice

    for i, per_precision in enumerate(per_class_precisions):
        result_dict[f'class_{i}_precision'] = per_precision

    for i, per_recall in enumerate(per_class_recalls):
        result_dict[f'class_{i}_recall'] = per_recall

    for i, per_iou in enumerate(per_class_ious):
        result_dict[f'class_{i}_iou'] = per_iou

    for i, per_dice in enumerate(per_class_dices):
        result_dict[f'class_{i}_dice'] = per_dice

    return result_dict


def train_universal_segmentation(train_loader, model, criterion, optimizer,
                                 scheduler, epoch, logger, config):
    '''
    train universal segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    amp_type = get_amp_type(model)

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, masks, labels = data['image'], data['mask'], data['label']
        images = images.cuda()

        skip_batch_flag = False

        if torch.any(torch.isinf(images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)):
            skip_batch_flag = True

        if config.use_amp:
            with autocast(device_type="cuda", dtype=amp_type):
                mask_preds, class_preds = model(images)
                loss_value = criterion(mask_preds, class_preds, masks, labels)
        else:
            mask_preds, class_preds = model(images)
            loss_value = criterion(mask_preds, class_preds, masks, labels)

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            print(f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
            skip_batch_flag = True

        loss = loss / config.accumulation_steps
        for key, value in loss_value.items():
            loss_value[key] = value / config.accumulation_steps

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                config.scaler.scale(loss).backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    config.scaler.scale(loss).backward()
        else:
            if iter_index % config.accumulation_steps == 0:
                loss.backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    loss.backward()

        if hasattr(config, 'skip_inf_nan_grad') and config.skip_inf_nan_grad:
            grad_inf_nan_flag = False
            for _, param in model.named_parameters():
                per_weight_grad = param.grad
                if per_weight_grad is not None:
                    if torch.any(torch.isinf(per_weight_grad)) or torch.any(
                            torch.isnan(per_weight_grad)):
                        grad_inf_nan_flag = True
            if grad_inf_nan_flag:
                print(f'GPU id:{local_rank},nan grad or inf grad!')
                skip_batch_flag = True

        [skip_batch_flag] = all_reduce_operation_in_group_for_variables(
            variables=[skip_batch_flag],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)

        if skip_batch_flag:
            log_info = f'skip this batch!'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None
            optimizer.zero_grad()
            continue

        torch.distributed.barrier(device_ids=[local_rank])

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):
                    config.scaler.unscale_(optimizer)

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                optimizer.step()
                optimizer.zero_grad()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
            for key, value in loss_value.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value],
                    operator=torch.distributed.ReduceOp.SUM,
                    group=config.group)
                loss_value[key] = value / float(config.gpus_num)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss
