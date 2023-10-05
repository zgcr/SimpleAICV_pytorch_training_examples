import cv2

import os

import collections
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval

from simpleAICV.classification.common import AverageMeter, AccMeter
from simpleAICV.diffusion_model.metrics.compute_fid_is_score import calculate_frechet_distance, compute_inception_score

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from simpleAICV.image_inpainting.metrics.inception import InpaintingImagePathDataset, inpainting_calculate_frechet_distance


def all_reduce_operation_in_group_for_variables(variables, operator, group):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables


def all_reduce_operation_in_group_for_tensors(tensors, operator, group):
    for i in range(len(tensors)):
        torch.distributed.all_reduce(tensors[i], op=operator, group=group)

    return tensors


def test_classification(test_loader, model, criterion, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AccMeter()

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            images, labels = data['image'], data['label']
            if model_on_cuda:
                images, labels = images.cuda(), labels.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            loss = criterion(outputs, labels)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss = loss / float(config.gpus_num)

            losses.update(loss, images.size(0))

            _, topk_indexes = torch.topk(outputs,
                                         k=5,
                                         dim=1,
                                         largest=True,
                                         sorted=True)
            correct_mask = topk_indexes.eq(
                labels.unsqueeze(-1).expand_as(topk_indexes)).float()
            correct_mask = correct_mask.cpu().numpy()

            acc1_correct_num, acc5_correct_num, sample_num = correct_mask[:, :1].sum(
            ), correct_mask[:, :5].sum(), images.size(0)
            acc1_correct_num, acc5_correct_num, sample_num = float(
                acc1_correct_num), float(acc5_correct_num), float(sample_num)

            # please keep same variable on different gpus has same data type for all reduce operation
            [acc1_correct_num, acc5_correct_num,
             sample_num] = all_reduce_operation_in_group_for_variables(
                 variables=[acc1_correct_num, acc5_correct_num, sample_num],
                 operator=torch.distributed.ReduceOp.SUM,
                 group=config.group)

            accs.update(acc1_correct_num, acc5_correct_num, sample_num)

            end = time.time()

    # top1(%)、top5(%)
    accs.compute()
    acc1 = accs.acc1 * 100
    acc5 = accs.acc5 * 100

    # avg_loss
    avg_loss = losses.avg

    # per image data load time(ms) and inference time(ms)
    per_image_load_time = data_time.avg / (config.batch_size //
                                           config.gpus_num) * 1000
    per_image_inference_time = batch_time.avg / (config.batch_size //
                                                 config.gpus_num) * 1000

    return acc1, acc5, avg_loss, per_image_load_time, per_image_inference_time


def train_classification(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train classification model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, labels = data['image'], data['label']
        images, labels = images.cuda(), labels.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(labels)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(labels)):
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
        else:
            if iter_index % config.accumulation_steps == 0:
                outputs = model(images)
                loss = criterion(outputs, labels)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        loss = loss / config.accumulation_steps

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

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
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
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def test_distill_classification(test_loader, model, criterion, config):
    teacher_model = model.module.teacher
    student_model = model.module.student

    tea_acc1, tea_acc5, tea_test_loss, _, _ = test_classification(
        test_loader, teacher_model, criterion, config)

    stu_acc1, stu_acc5, stu_test_loss, _, _ = test_classification(
        test_loader, student_model, criterion, config)

    return tea_acc1, tea_acc5, tea_test_loss, stu_acc1, stu_acc5, stu_test_loss


def train_distill_classification(train_loader, model, criterion, optimizer,
                                 scheduler, epoch, logger, config):
    '''
    train distill classification model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()
    if config.freeze_teacher:
        model.module.teacher.eval()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, labels = data['image'], data['label']
        images, labels = images.cuda(), labels.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(labels)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(labels)):
            continue

        if iter_index % config.accumulation_steps == 0:
            tea_outputs, stu_outputs = model(images)
            loss_value = {}
            for loss_name in criterion.keys():
                if loss_name in ['CELoss', 'OneHotLabelCELoss']:
                    if not config.freeze_teacher:
                        temp_loss = criterion[loss_name](
                            tea_outputs, labels) * config.loss_ratio[loss_name]
                        loss_value['tea_' + loss_name] = temp_loss

                    temp_loss = criterion[loss_name](
                        stu_outputs, labels) * config.loss_ratio[loss_name]
                    loss_value['stu_' + loss_name] = temp_loss
                else:
                    temp_loss = criterion[loss_name](
                        stu_outputs,
                        tea_outputs) * config.loss_ratio[loss_name]
                    loss_value[loss_name] = temp_loss
        else:
            # not reduce gradient while iter_index % config.accumulation_steps != 0
            with model.no_sync():
                tea_outputs, stu_outputs = model(images)
                loss_value = {}
                for loss_name in criterion.keys():
                    if loss_name in ['CELoss', 'OneHotLabelCELoss']:
                        if not config.freeze_teacher:
                            temp_loss = criterion[loss_name](
                                tea_outputs,
                                labels) * config.loss_ratio[loss_name]
                            loss_value['tea_' + loss_name] = temp_loss

                        temp_loss = criterion[loss_name](
                            stu_outputs, labels) * config.loss_ratio[loss_name]
                        loss_value['stu_' + loss_name] = temp_loss
                    else:
                        temp_loss = criterion[loss_name](
                            stu_outputs,
                            tea_outputs) * config.loss_ratio[loss_name]
                        loss_value[loss_name] = temp_loss

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        loss = loss / config.accumulation_steps
        for key, value in loss_value.items():
            loss_value[key] = value / config.accumulation_steps

        if iter_index % config.accumulation_steps == 0:
            loss.backward()
        else:
            # not reduce gradient while iter_index % config.accumulation_steps != 0
            with model.no_sync():
                loss.backward()

        if iter_index % config.accumulation_steps == 0:
            if hasattr(config, 'clip_max_norm') and config.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config.clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

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
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def compute_voc_ap(recall, precision, use_07_metric=False):
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
    '''
    :param a: [N,(x1,y1,x2,y2)]
    :param b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    '''

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


def evaluate_voc_detection(test_loader, model, criterion, decoder, config):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    batch_size = int(config.batch_size // config.gpus_num)

    with torch.no_grad():
        preds, gts = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for _, data in tqdm(enumerate(test_loader)):
            images, annots, scales, sizes = data['image'], data[
                'annots'], data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()

            if 'detr' in config.network:
                masks = data['mask']
                scaled_sizes = data['scaled_size']
                if model_on_cuda:
                    masks = masks.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            if 'detr' in config.network:
                outs_tuple = model(images, masks)
            else:
                outs_tuple = model(images)

            loss_value = criterion(outs_tuple, annots)
            loss = sum(loss_value.values())
            losses.update(loss, images.size(0))

            if 'detr' in config.network:
                pred_scores, pred_classes, pred_boxes = decoder(
                    outs_tuple, scaled_sizes)
            else:
                pred_scores, pred_classes, pred_boxes = decoder(outs_tuple)

            pred_boxes /= np.expand_dims(np.expand_dims(scales, axis=-1),
                                         axis=-1)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            annots = annots.cpu().numpy()
            gt_bboxes, gt_classes = annots[:, :, 0:4], annots[:, :, 4]
            gt_bboxes /= np.expand_dims(np.expand_dims(scales, axis=-1),
                                        axis=-1)

            for per_image_pred_scores, per_image_pred_classes, per_image_pred_boxes, per_image_gt_bboxes, per_image_gt_classes, per_image_size in zip(
                    pred_scores, pred_classes, pred_boxes, gt_bboxes,
                    gt_classes, sizes):
                per_image_pred_scores = per_image_pred_scores[
                    per_image_pred_classes > -1]
                per_image_pred_boxes = per_image_pred_boxes[
                    per_image_pred_classes > -1]
                per_image_pred_classes = per_image_pred_classes[
                    per_image_pred_classes > -1]

                # clip boxes
                per_image_pred_boxes[:, 0] = np.maximum(
                    per_image_pred_boxes[:, 0], 0)
                per_image_pred_boxes[:, 1] = np.maximum(
                    per_image_pred_boxes[:, 1], 0)
                per_image_pred_boxes[:, 2] = np.minimum(
                    per_image_pred_boxes[:, 2], per_image_size[1])
                per_image_pred_boxes[:, 3] = np.minimum(
                    per_image_pred_boxes[:, 3], per_image_size[0])

                preds.append([
                    per_image_pred_boxes, per_image_pred_classes,
                    per_image_pred_scores
                ])

                per_image_gt_bboxes = per_image_gt_bboxes[
                    per_image_gt_classes > -1]
                per_image_gt_classes = per_image_gt_classes[
                    per_image_gt_classes > -1]

                gts.append([per_image_gt_bboxes, per_image_gt_classes])

            end = time.time()

        test_loss = losses.avg

        result_dict = collections.OrderedDict()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / batch_size * 1000
        per_image_inference_time = batch_time.avg / batch_size * 1000

        result_dict['test_loss'] = test_loss
        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        all_iou_threshold_map = collections.OrderedDict()
        all_iou_threshold_per_class_ap = collections.OrderedDict()
        for per_iou_threshold in tqdm(config.eval_voc_iou_threshold_list):
            per_iou_threshold_all_class_ap = collections.OrderedDict()
            for class_index in range(config.num_classes):
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
                        scores = np.append(scores,
                                           per_image_pred_scores[index])
                        if per_image_gt_boxes.shape[0] == 0:
                            # if no gts found for the predicted bbox, assign the bbox to fp
                            fp = np.append(fp, 1)
                            tp = np.append(tp, 0)
                            continue
                        pred_box = np.expand_dims(per_image_pred_boxes[index],
                                                  axis=0)
                        iou = compute_ious(per_image_gt_boxes, pred_box)
                        gt_for_box = np.argmax(iou, axis=0)
                        max_overlap = iou[gt_for_box, 0]
                        if max_overlap >= per_iou_threshold and gt_for_box not in assigned_gt:
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
                ap = compute_voc_ap(recall, precision, use_07_metric=False)
                per_iou_threshold_all_class_ap[class_index] = ap * 100

            per_iou_threshold_map = 0.
            for _, per_iou_threshold_per_class_ap in per_iou_threshold_all_class_ap.items(
            ):
                per_iou_threshold_map += float(per_iou_threshold_per_class_ap)
            per_iou_threshold_map /= config.num_classes

            all_iou_threshold_map[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,mAP'] = per_iou_threshold_map
            all_iou_threshold_per_class_ap[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,per_class_ap'] = per_iou_threshold_all_class_ap

        for key, value in all_iou_threshold_map.items():
            result_dict[key] = value
        for key, value in all_iou_threshold_per_class_ap.items():
            result_dict[key] = value

        return result_dict


def evaluate_coco_detection(test_loader, model, criterion, decoder, config):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    test_dataset = config.test_dataset
    ids = [idx for idx in range(len(test_dataset))]
    batch_size = int(config.batch_size // config.gpus_num)

    with torch.no_grad():
        results, image_ids = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for i, data in tqdm(enumerate(test_loader)):
            images, annots, scales, sizes = data['image'], data[
                'annots'], data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()

            if 'detr' in config.network:
                masks = data['mask']
                scaled_sizes = data['scaled_size']
                if model_on_cuda:
                    masks = masks.cuda()

            per_batch_ids = ids[i * batch_size:(i + 1) * batch_size]

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            if 'detr' in config.network:
                outs_tuple = model(images, masks)
            else:
                outs_tuple = model(images)

            loss_value = criterion(outs_tuple, annots)
            loss = sum(loss_value.values())
            losses.update(loss, images.size(0))

            if 'detr' in config.network:
                scores, classes, boxes = decoder(outs_tuple, scaled_sizes)
            else:
                scores, classes, boxes = decoder(outs_tuple)

            boxes /= np.expand_dims(np.expand_dims(scales, axis=-1), axis=-1)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            for per_image_scores, per_image_classes, per_image_boxes, index, per_image_size in zip(
                    scores, classes, boxes, per_batch_ids, sizes):
                # clip boxes
                per_image_boxes[:, 0] = np.maximum(per_image_boxes[:, 0], 0)
                per_image_boxes[:, 1] = np.maximum(per_image_boxes[:, 1], 0)
                per_image_boxes[:, 2] = np.minimum(per_image_boxes[:, 2],
                                                   per_image_size[1])
                per_image_boxes[:, 3] = np.minimum(per_image_boxes[:, 3],
                                                   per_image_size[0])

                # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
                per_image_boxes[:, 2:] -= per_image_boxes[:, :2]

                for object_score, object_class, object_box in zip(
                        per_image_scores, per_image_classes, per_image_boxes):
                    object_score = float(object_score)
                    object_class = int(object_class)
                    object_box = object_box.tolist()
                    if object_class == -1:
                        break

                    image_result = {
                        'image_id':
                        test_dataset.image_ids[index],
                        'category_id':
                        test_dataset.coco_label_to_cat_id[object_class],
                        'score':
                        object_score,
                        'bbox':
                        object_box,
                    }
                    results.append(image_result)

                image_ids.append(test_dataset.image_ids[index])

                print('{}/{}'.format(index, len(test_dataset)), end='\r')

            end = time.time()

        test_loss = losses.avg

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

        result_dict['test_loss'] = test_loss
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

        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        eval_result = coco_eval.stats

        for i, var in enumerate(eval_result):
            result_dict[variable_definitions[i]] = var * 100

        return result_dict


def test_detection(test_loader, model, criterion, decoder, config):
    assert config.eval_type in ['COCO', 'VOC']

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    func_dict = {
        'COCO': evaluate_coco_detection,
        'VOC': evaluate_voc_detection,
    }
    result_dict = func_dict[config.eval_type](test_loader, model, criterion,
                                              decoder, config)

    return result_dict


def train_detection(train_loader, model, criterion, optimizer, scheduler,
                    epoch, logger, config):
    '''
    train detection model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, targets = data['image'], data['annots']
        images, targets = images.cuda(), targets.cuda()

        if 'detr' in config.network:
            targets = data['scaled_annots']
            targets = targets.cuda()

            masks = data['mask']
            masks = masks.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(targets)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(targets)):
            continue

        if torch.sum(images) == 0:
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    if 'detr' in config.network:
                        outs_tuple = model(images, masks)
                    else:
                        outs_tuple = model(images)
                    loss_value = criterion(outs_tuple, targets)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        if 'detr' in config.network:
                            outs_tuple = model(images, masks)
                        else:
                            outs_tuple = model(images)
                        loss_value = criterion(outs_tuple, targets)
        else:
            if iter_index % config.accumulation_steps == 0:
                if 'detr' in config.network:
                    outs_tuple = model(images, masks)
                else:
                    outs_tuple = model(images)
                loss_value = criterion(outs_tuple, targets)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    if 'detr' in config.network:
                        outs_tuple = model(images, masks)
                    else:
                        outs_tuple = model(images)
                    loss_value = criterion(outs_tuple, targets)

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

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

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
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
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def test_semantic_segmentation(test_loader, model, criterion, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

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
            images, masks, scales, sizes = data['image'], data['mask'], data[
                'scale'], data['size']
            if model_on_cuda:
                images, masks = images.cuda(), masks.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            loss = criterion(outputs, masks)
            losses.update(loss, images.size(0))

            # pred shape:[b,c,h,w] -> [b,h,w,c]
            outputs = outputs.permute(0, 2, 3, 1).contiguous()
            pixel_preds = torch.argmax(outputs, axis=-1)

            for per_image_pred, per_image_mask, per_image_size in zip(
                    pixel_preds, masks, sizes):
                per_image_pred = per_image_pred[0:int(per_image_size[0]),
                                                0:int(per_image_size[1])]
                per_image_mask = per_image_mask[0:int(per_image_size[0]),
                                                0:int(per_image_size[1])]
                # per_image_pred:[h,w,c] -> (-1)
                # per_image_mask:[h,w] -> (-1)
                per_image_pred, per_image_mask = per_image_pred.reshape(
                    -1), per_image_mask.reshape(-1)

                if config.ignore_index:
                    per_image_filter_mask = (per_image_mask !=
                                             config.ignore_index)
                    per_image_pred = per_image_pred[per_image_filter_mask]
                    per_image_mask = per_image_mask[per_image_filter_mask]

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

        # avg_loss
        test_loss = losses.avg

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / (config.batch_size //
                                               config.gpus_num) * 1000
        per_image_inference_time = batch_time.avg / (config.batch_size //
                                                     config.gpus_num) * 1000

        result_dict = collections.OrderedDict()
        result_dict['test_loss'] = test_loss
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


def train_semantic_segmentation(train_loader, model, criterion, optimizer,
                                scheduler, epoch, logger, config):
    '''
    train semantic segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, masks = data['image'], data['mask']
        images, masks = images.cuda(), masks.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(masks)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(masks)):
            continue

        if torch.sum(images) == 0:
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    outputs = model(images)
                    loss_value = {}
                    for loss_name in criterion.keys():
                        temp_loss = criterion[loss_name](outputs, masks)
                        temp_loss = config.loss_ratio[loss_name] * temp_loss
                        loss_value[loss_name] = temp_loss

                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        outputs = model(images)
                        loss_value = {}
                        for loss_name in criterion.keys():
                            temp_loss = criterion[loss_name](outputs, masks)
                            temp_loss = config.loss_ratio[loss_name] * temp_loss
                            loss_value[loss_name] = temp_loss
        else:
            if iter_index % config.accumulation_steps == 0:
                outputs = model(images)
                loss_value = {}
                for loss_name in criterion.keys():
                    temp_loss = criterion[loss_name](outputs, masks)
                    temp_loss = config.loss_ratio[loss_name] * temp_loss
                    loss_value[loss_name] = temp_loss
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    outputs = model(images)
                    loss_value = {}
                    for loss_name in criterion.keys():
                        temp_loss = criterion[loss_name](outputs, masks)
                        temp_loss = config.loss_ratio[loss_name] * temp_loss
                        loss_value[loss_name] = temp_loss

        loss = 0.
        for key, value in loss_value.items():
            loss += value

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

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

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
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
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def evaluate_coco_instance_segmentation(test_loader, model, criterion, decoder,
                                        config):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    test_dataset = config.test_dataset
    ids = [idx for idx in range(len(test_dataset))]
    batch_size = int(config.batch_size // config.gpus_num)

    with torch.no_grad():
        results, image_ids = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for i, data in tqdm(enumerate(test_loader)):
            images = data['image']
            if model_on_cuda:
                images = images.cuda()

            gt_bboxes = data['box']
            gt_masks = data['mask']

            scaled_size = data['size']
            origin_size = data['origin_size']

            per_batch_ids = ids[i * batch_size:(i + 1) * batch_size]

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            outs_tuple = model(images)

            loss_value = criterion(outs_tuple, gt_bboxes, gt_masks)
            loss = sum(loss_value.values())
            losses.update(loss, images.size(0))

            batch_masks, batch_labels, batch_scores = decoder(
                outs_tuple, scaled_size, origin_size)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            for per_image_masks, per_image_labels, per_image_scores, index in zip(
                    batch_masks, batch_labels, batch_scores, per_batch_ids):

                for per_mask, per_label, per_score in zip(
                        per_image_masks, per_image_labels, per_image_scores):
                    rle = mask_util.encode(
                        np.array(per_mask[:, :, np.newaxis],
                                 order='F'))[0].copy()
                    rle['counts'] = rle['counts'].decode()
                    image_result = {
                        'image_id': test_dataset.image_ids[index],
                        'category_id':
                        test_dataset.coco_label_to_cat_id[per_label],
                        'score': float(per_score),
                        'segmentation': rle,
                    }
                    results.append(image_result)

                image_ids.append(test_dataset.image_ids[index])

                print('{}/{}'.format(index, len(test_dataset)), end='\r')

            end = time.time()

        test_loss = losses.avg

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

        result_dict['test_loss'] = test_loss
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


def test_instance_segmentation(test_loader, model, criterion, decoder, config):
    assert config.eval_type in ['COCO']

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    func_dict = {
        'COCO': evaluate_coco_instance_segmentation,
    }
    result_dict = func_dict[config.eval_type](test_loader, model, criterion,
                                              decoder, config)

    return result_dict


def train_instance_segmentation(train_loader, model, criterion, optimizer,
                                scheduler, epoch, logger, config):
    '''
    train instance segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images = data['image']
        images = images.cuda()

        gt_bboxes = data['box']
        gt_masks = data['mask']

        if torch.any(torch.isinf(images)):
            continue

        if torch.any(torch.isnan(images)):
            continue

        if torch.sum(images) == 0:
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    outs_tuple = model(images)
                    loss_value = criterion(outs_tuple, gt_bboxes, gt_masks)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        outs_tuple = model(images)
                        loss_value = criterion(outs_tuple, gt_bboxes, gt_masks)
        else:
            if iter_index % config.accumulation_steps == 0:
                outs_tuple = model(images)
                loss_value = criterion(outs_tuple, gt_bboxes, gt_masks)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    outs_tuple = model(images)
                    loss_value = criterion(outs_tuple, gt_bboxes, gt_masks)

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

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

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
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
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def train_mae_self_supervised_learning(train_loader, model, criterion,
                                       optimizer, scheduler, epoch, logger,
                                       config):
    '''
    train mae self supervised model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, labels = data['image'], data['label']
        images, labels = images.cuda(), labels.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(labels)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(labels)):
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    outputs, masks = model(images)
                    loss = criterion(outputs, labels, masks)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        outputs, masks = model(images)
                        loss = criterion(outputs, labels, masks)
        else:
            if iter_index % config.accumulation_steps == 0:
                outputs, masks = model(images)
                loss = criterion(outputs, labels, masks)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    outputs, masks = model(images)
                    loss = criterion(outputs, labels, masks)

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        loss = loss / config.accumulation_steps

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

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
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
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def train_dino_self_supervised_learning(train_loader, teacher_model,
                                        student_model, criterion,
                                        student_optimizer, lr_scheduler,
                                        weight_decay_scheduler,
                                        momentum_teacher_scheduler, epoch,
                                        logger, config):
    '''
    train dino self supervised model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    student_model.train()
    teacher_model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1

    for _, data in enumerate(train_loader):
        images = data['image']
        images = [image.cuda() for image in images]

        currnet_epoch = iter_index / iters + (epoch - 1)

        if config.use_amp:
            with autocast():
                # only the 2 global views pass through the teacher
                teacher_outputs = teacher_model(
                    images[0:config.global_crop_nums])
                student_outputs = student_model(images)
                loss = criterion(student_outputs, teacher_outputs,
                                 currnet_epoch)
        else:
            # only the 2 global views pass through the teacher
            teacher_outputs = teacher_model(images[0:config.global_crop_nums])
            student_outputs = student_model(images)
            loss = criterion(student_outputs, teacher_outputs, currnet_epoch)

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            student_optimizer.zero_grad()
            continue

        if config.use_amp:
            config.scaler.scale(loss).backward()
        else:
            loss.backward()

        if config.use_amp:
            if hasattr(config, 'clip_max_norm') and config.clip_max_norm > 0:
                config.scaler.unscale_(student_optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(),
                                               config.clip_max_norm)
            config.scaler.step(student_optimizer)
            config.scaler.update()
            student_optimizer.zero_grad()
        else:
            if hasattr(config, 'clip_max_norm') and config.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(),
                                               config.clip_max_norm)
            student_optimizer.step()
            student_optimizer.zero_grad()

        # EMA update for the teacher
        with torch.no_grad():
            momentum = momentum_teacher_scheduler.current_value
            for param_student, param_teacher in zip(
                    student_model.module.parameters(),
                    teacher_model.module.parameters()):
                param_teacher.data = param_teacher.data * momentum + (
                    1 - momentum) * param_student.detach().data

        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)

        losses.update(loss, images[0].size(0))

        lr_scheduler.step(student_optimizer, iter_index / iters + (epoch - 1))
        weight_decay_scheduler.step(student_optimizer,
                                    iter_index / iters + (epoch - 1))
        momentum_teacher_scheduler.step(student_optimizer,
                                        iter_index / iters + (epoch - 1))

        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {lr_scheduler.current_value:.6f}, weight_decay: {weight_decay_scheduler.current_value:.6f}, momentum_teacher: {momentum_teacher_scheduler.current_value:.6f}, loss: {loss:.4f}'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg

    return avg_loss


def generate_diffusion_model_images(test_loader, model, sampler, config):
    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    local_rank = torch.distributed.get_rank()
    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for batch_idx, data in tqdm(enumerate(test_loader)):
            images = data['image']
            if model_on_cuda:
                images = images.cuda()

            labels = None
            if 'label' in data.keys(
            ) and config.num_classes and config.use_condition_label:
                labels = data['label']
                if model_on_cuda:
                    labels = labels.cuda()

            if torch.any(torch.isinf(images)):
                continue

            if torch.any(torch.isnan(images)):
                continue

            torch.cuda.synchronize()

            input_images, input_masks = None, None
            if config.use_input_images:
                input_images = images

            _, outputs = sampler(model,
                                 images.shape,
                                 class_label=labels,
                                 input_images=input_images,
                                 input_masks=input_masks,
                                 return_intermediates=True)

            torch.cuda.synchronize()

            mean = np.expand_dims(np.expand_dims(config.mean, axis=0), axis=0)
            std = np.expand_dims(np.expand_dims(config.std, axis=0), axis=0)

            for image_idx, (per_image,
                            per_output) in enumerate(zip(images, outputs)):
                per_image = per_image.cpu().numpy()
                per_image = per_image.transpose(1, 2, 0)
                per_image = (per_image * std + mean) * 255.

                per_output = per_output.transpose(1, 2, 0)
                per_output = (per_output * std + mean) * 255.

                per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
                per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

                per_output = np.ascontiguousarray(per_output, dtype=np.uint8)
                per_output = cv2.cvtColor(per_output, cv2.COLOR_RGB2BGR)

                save_image_name = f'image_{local_rank}_{batch_idx}_{image_idx}.jpg'
                save_image_path = os.path.join(config.save_test_image_dir,
                                               save_image_name)
                cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

                save_output_name = f'output_{local_rank}_{batch_idx}_{image_idx}.jpg'
                save_output_path = os.path.join(config.save_generate_image_dir,
                                                save_output_name)
                cv2.imencode('.jpg', per_output)[1].tofile(save_output_path)

    torch.distributed.barrier()

    test_images_path_list = []
    for per_image_name in os.listdir(config.save_test_image_dir):
        per_image_path = os.path.join(config.save_test_image_dir,
                                      per_image_name)
        test_images_path_list.append(per_image_path)

    generate_images_path_list = []
    for per_image_name in os.listdir(config.save_generate_image_dir):
        per_image_path = os.path.join(config.save_generate_image_dir,
                                      per_image_name)
        generate_images_path_list.append(per_image_path)

    test_image_num = len(test_images_path_list)
    generate_image_num = len(generate_images_path_list)

    return test_images_path_list, generate_images_path_list, test_image_num, generate_image_num


def compute_diffusion_model_metric(test_images_dataloader,
                                   generate_images_dataloader, test_image_num,
                                   generate_image_num, fid_model, config):

    for param in fid_model.parameters():
        param.requires_grad = False

    # switch to evaluate mode
    fid_model.eval()

    test_images_pred = np.empty((test_image_num, 2048))
    with torch.no_grad():
        test_images_start_idx = 0
        model_on_cuda = next(fid_model.parameters()).is_cuda
        for data in tqdm(test_images_dataloader):
            if model_on_cuda:
                data = data.cuda()

            preds = fid_model(data)
            per_batch_pred_features = preds[0].squeeze(-1).squeeze(
                -1).cpu().numpy()

            test_images_pred[test_images_start_idx:test_images_start_idx +
                             per_batch_pred_features.
                             shape[0]] = per_batch_pred_features
            test_images_start_idx = test_images_start_idx + per_batch_pred_features.shape[
                0]

    generate_images_pred = np.empty((generate_image_num, 2048))
    generate_images_cls_pred = np.empty((generate_image_num, 1008))

    with torch.no_grad():
        generate_images_start_idx = 0
        model_on_cuda = next(fid_model.parameters()).is_cuda
        for data in tqdm(generate_images_dataloader):
            if model_on_cuda:
                data = data.cuda()

            preds = fid_model(data)

            per_batch_pred_features = preds[0].squeeze(-1).squeeze(
                -1).cpu().numpy()
            per_batch_pred_probs = preds[1].cpu().numpy()

            generate_images_pred[
                generate_images_start_idx:generate_images_start_idx +
                per_batch_pred_features.shape[0]] = per_batch_pred_features

            generate_images_cls_pred[
                generate_images_start_idx:generate_images_start_idx +
                per_batch_pred_probs.shape[0]] = per_batch_pred_probs

            generate_images_start_idx = generate_images_start_idx + per_batch_pred_features.shape[
                0]

    mu1 = np.mean(test_images_pred, axis=0)
    sigma1 = np.cov(test_images_pred, rowvar=False)

    mu2 = np.mean(generate_images_pred, axis=0)
    sigma2 = np.cov(generate_images_pred, rowvar=False)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    is_score_mean, is_score_std = compute_inception_score(
        generate_images_cls_pred, config.is_data_split_num)

    return fid_value, is_score_mean, is_score_std


def train_diffusion_model(train_loader, model, criterion, trainer, optimizer,
                          scheduler, epoch, logger, config):
    '''
    train diffusion model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images = data['image']
        images = images.cuda()

        labels = None
        if 'label' in data.keys() and config.num_classes:
            labels = data['label']
            labels = labels.cuda()

        if torch.any(torch.isinf(images)):
            continue

        if torch.any(torch.isnan(images)):
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    pred_noise, noise = trainer(model,
                                                images,
                                                class_label=labels)
                    loss = criterion(pred_noise, noise)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        pred_noise, noise = trainer(model,
                                                    images,
                                                    class_label=labels)
                        loss = criterion(pred_noise, noise)
        else:
            if iter_index % config.accumulation_steps == 0:
                pred_noise, noise = trainer(model, images, class_label=labels)
                loss = criterion(pred_noise, noise)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    pred_noise, noise = trainer(model,
                                                images,
                                                class_label=labels)
                    loss = criterion(pred_noise, noise)

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        loss = loss / config.accumulation_steps

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

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
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
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def generate_inpainting_images_for_all_dataset(generator_model, config):
    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    all_dataset_images_num_dict = collections.OrderedDict()
    all_dataset_images_path_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset) in enumerate(
            zip(config.test_dataset_name_list, config.test_dataset_list)):
        per_sub_sampler = torch.utils.data.distributed.DistributedSampler(
            per_sub_dataset, shuffle=False)
        per_sub_dataset_loader = DataLoader(per_sub_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=num_workers,
                                            collate_fn=config.test_collater,
                                            sampler=per_sub_sampler)

        per_sub_test_images_path_list, per_sub_test_image_num = generate_inpainting_images_for_per_sub_dataset(
            per_sub_dataset_name, per_sub_dataset_loader, generator_model,
            config)

        all_dataset_images_num_dict[
            per_sub_dataset_name] = per_sub_test_image_num
        all_dataset_images_path_dict[
            per_sub_dataset_name] = per_sub_test_images_path_list

        torch.cuda.empty_cache()

    return all_dataset_images_num_dict, all_dataset_images_path_dict


def generate_inpainting_images_for_per_sub_dataset(per_sub_dataset_name,
                                                   per_sub_dataset_loader,
                                                   generator_model, config):
    local_rank = torch.distributed.get_rank()

    per_sub_save_image_dir = os.path.join(config.save_image_dir,
                                          per_sub_dataset_name)
    if local_rank == 0:
        os.makedirs(per_sub_save_image_dir
                    ) if not os.path.exists(per_sub_save_image_dir) else None

    torch.distributed.barrier()

    # switch to evaluate mode
    generator_model.eval()

    with torch.no_grad():
        model_on_cuda = next(generator_model.parameters()).is_cuda
        for batch_idx, data in tqdm(enumerate(per_sub_dataset_loader)):
            images, masks = data['image'], data['mask']
            if model_on_cuda:
                images, masks = images.cuda(), masks.cuda()

            if torch.any(torch.isinf(images)):
                continue

            if torch.any(torch.isnan(images)):
                continue

            torch.cuda.synchronize()

            masked_images = (images * (1 - masks).float()) + masks
            preds = generator_model(masked_images, masks)
            composition_images = (1 - masks) * images + masks * preds

            torch.cuda.synchronize()

            for image_idx, (per_image, per_mask,
                            per_composition_image) in enumerate(
                                zip(images, masks, composition_images)):
                per_image = per_image.cpu().numpy()
                # to hwc
                per_image = per_image.transpose(1, 2, 0)
                per_image = np.clip(per_image, -1., 1.)
                # RGB image [0,255]
                per_image = ((per_image + 1.) / 2.) * 255.
                per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
                per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

                per_mask = per_mask.cpu().numpy()
                # to hwc
                per_mask = per_mask.transpose(1, 2, 0).squeeze(axis=-1)
                per_mask = np.clip(per_mask, 0., 1.)
                # gray image [0,255]
                per_mask = per_mask * 255.
                per_mask = np.ascontiguousarray(per_mask, dtype=np.uint8)

                per_composition_image = per_composition_image.cpu().numpy()
                # to hwc
                per_composition_image = per_composition_image.transpose(
                    1, 2, 0)
                per_composition_image = np.clip(per_composition_image, -1., 1.)
                # RGB image [0,255]
                per_composition_image = (
                    (per_composition_image + 1.) / 2.) * 255.
                per_composition_image = np.ascontiguousarray(
                    per_composition_image, dtype=np.uint8)
                per_composition_image = cv2.cvtColor(per_composition_image,
                                                     cv2.COLOR_RGB2BGR)

                save_image_name = f'{per_sub_dataset_name}_{local_rank}_{batch_idx}_{image_idx}_image.jpg'
                save_image_path = os.path.join(per_sub_save_image_dir,
                                               save_image_name)
                cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

                save_mask_name = f'{per_sub_dataset_name}_{local_rank}_{batch_idx}_{image_idx}_mask.jpg'
                save_mask_path = os.path.join(per_sub_save_image_dir,
                                              save_mask_name)
                cv2.imencode('.jpg', per_mask)[1].tofile(save_mask_path)

                save_composition_image_name = f'{per_sub_dataset_name}_{local_rank}_{batch_idx}_{image_idx}_composition.jpg'
                save_composition_image_path = os.path.join(
                    per_sub_save_image_dir, save_composition_image_name)
                cv2.imencode('.jpg', per_composition_image)[1].tofile(
                    save_composition_image_path)

    torch.distributed.barrier()

    per_sub_test_images_path_list = []
    for per_image_name in os.listdir(per_sub_save_image_dir):
        if '_image.jpg' in per_image_name:
            per_image_path = os.path.join(per_sub_save_image_dir,
                                          per_image_name)

            per_mask_name = per_image_name.replace('_image.jpg', '_mask.jpg')
            per_mask_path = os.path.join(per_sub_save_image_dir, per_mask_name)

            per_composition_image_name = per_image_name.replace(
                '_image.jpg', '_composition.jpg')
            per_composition_image_path = os.path.join(
                per_sub_save_image_dir, per_composition_image_name)

            assert os.path.exists(per_image_path) and os.path.exists(
                per_mask_path) and os.path.exists(per_composition_image_path)

            per_sub_test_images_path_list.append(
                [per_image_path, per_mask_path, per_composition_image_path])

    per_sub_test_image_num = len(per_sub_test_images_path_list)

    return per_sub_test_images_path_list, per_sub_test_image_num


def compute_image_inpainting_model_metric_for_all_dataset(
        all_dataset_images_path_dict, fid_model, config):
    assert config.fid_model_batch_size % config.gpus_num == 0, 'config.fid_model_batch_size is not divisible by config.gpus_num!'
    assert config.fid_model_num_workers % config.gpus_num == 0, 'config.fid_model_num_workers is not divisible by config.gpus_num!'
    fid_model_batch_size = int(config.fid_model_batch_size // config.gpus_num)
    fid_model_num_workers = int(config.fid_model_num_workers //
                                config.gpus_num)

    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name,
                per_sub_test_images_path_list) in enumerate(
                    all_dataset_images_path_dict.items()):
        per_sub_dataset_image_num = len(per_sub_test_images_path_list)
        per_sub_test_images_dataset = InpaintingImagePathDataset(
            per_sub_test_images_path_list,
            transform=transforms.Compose([
                transforms.Resize(
                    [config.input_image_size, config.input_image_size]),
                transforms.ToTensor(),
            ]))
        per_sub_test_images_dataloader = DataLoader(
            per_sub_test_images_dataset,
            batch_size=fid_model_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=fid_model_num_workers)

        per_sub_dataset_result_dict = compute_image_inpainting_model_metric_for_per_sub_dataset(
            per_sub_test_images_dataloader, per_sub_dataset_image_num,
            fid_model, config)
        result_dict[per_sub_dataset_name] = per_sub_dataset_result_dict

    return result_dict


def compute_image_inpainting_model_metric_for_per_sub_dataset(
        per_sub_test_images_dataloader, per_sub_dataset_image_num, fid_model,
        config):
    for param in fid_model.parameters():
        param.requires_grad = False

    # switch to evaluate mode
    fid_model.eval()

    mae, psnr, ssim = 0., 0., 0.
    all_images_pred = np.empty((per_sub_dataset_image_num, 2048))
    all_composition_images_pred = np.empty((per_sub_dataset_image_num, 2048))
    with torch.no_grad():
        test_images_start_idx = 0
        model_on_cuda = next(fid_model.parameters()).is_cuda
        for images, composition_images in tqdm(per_sub_test_images_dataloader):
            if model_on_cuda:
                images = images.cuda()
                composition_images = composition_images.cuda()

            images_pred = fid_model(images)
            per_batch_images_pred_feature = images_pred[0].squeeze(-1).squeeze(
                -1).cpu().numpy()
            all_images_pred[test_images_start_idx:test_images_start_idx +
                            per_batch_images_pred_feature.
                            shape[0]] = per_batch_images_pred_feature

            composition_images_pred = fid_model(composition_images)
            per_batch_composition_images_pred_feature = composition_images_pred[
                0].squeeze(-1).squeeze(-1).cpu().numpy()
            all_composition_images_pred[
                test_images_start_idx:test_images_start_idx +
                per_batch_composition_images_pred_feature.
                shape[0]] = per_batch_composition_images_pred_feature

            test_images_start_idx = test_images_start_idx + per_batch_images_pred_feature.shape[
                0]

            for per_image, per_composition_image in zip(
                    images, composition_images):
                per_image = per_image * 255.
                per_image = per_image.permute(1, 2,
                                              0).cpu().numpy().astype(np.uint8)
                per_composition_image = per_composition_image * 255.
                per_composition_image = per_composition_image.permute(
                    1, 2, 0).cpu().numpy().astype(np.uint8)

                mae += np.sum(
                    np.abs(
                        per_image.astype(np.float32) -
                        per_composition_image.astype(np.float32))) / np.sum(
                            per_image.astype(np.float32) +
                            per_composition_image.astype(np.float32))
                psnr += compare_psnr(per_image, per_composition_image)
                ssim += compare_ssim(per_image,
                                     per_composition_image,
                                     channel_axis=-1)

    mae_value = mae / per_sub_dataset_image_num
    psnr_value = psnr / per_sub_dataset_image_num
    ssim_value = ssim / per_sub_dataset_image_num

    mu1 = np.mean(all_images_pred, axis=0)
    sigma1 = np.cov(all_images_pred, rowvar=False)

    mu2 = np.mean(all_composition_images_pred, axis=0)
    sigma2 = np.cov(all_composition_images_pred, rowvar=False)

    fid_value = inpainting_calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    per_sub_dataset_result_dict = {
        'mae': mae_value,
        'psnr': psnr_value,
        'ssim': ssim_value,
        'fid': fid_value,
    }

    return per_sub_dataset_result_dict


def train_image_inpainting_aot_gan_model(
        train_loader, generator_model, discriminator_model,
        reconstruction_criterion, adversarial_criterion, generator_optimizer,
        discriminator_optimizer, generator_scheduler, discriminator_scheduler,
        epoch, logger, config):
    '''
    train image inpainting aot gan model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    generator_model.train()
    discriminator_model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1

    for _, data in enumerate(train_loader):
        images, masks = data['image'], data['mask']
        images, masks = images.cuda(), masks.cuda()
        masked_images = (images * (1 - masks).float()) + masks

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(masks)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(masks)):
            continue

        if torch.sum(images) == 0:
            continue

        if config.use_amp:
            with autocast():
                preds = generator_model(masked_images, masks)
                composition_images = (1 - masks) * images + masks * preds

                composition_images_detach = composition_images.detach()

                generator_fake = discriminator_model(composition_images)
                discriminator_fake = discriminator_model(
                    composition_images_detach)
                discriminator_real = discriminator_model(images)

                loss_value = {}
                # reconstruction losses
                for loss_name in reconstruction_criterion.keys():
                    temp_loss = reconstruction_criterion[loss_name](preds,
                                                                    images)
                    temp_loss = config.reconstruction_loss_ratio[
                        loss_name] * temp_loss
                    loss_value[loss_name] = temp_loss

                # adversarial loss
                for loss_name in adversarial_criterion.keys():
                    discriminator_loss, generator_loss = adversarial_criterion[
                        loss_name](generator_fake, discriminator_fake,
                                   discriminator_real, masks)
                    generator_loss = config.adversarial_loss_ratio[
                        loss_name] * generator_loss
                    loss_value['generator_loss'] = generator_loss
                    loss_value['discriminator_loss'] = discriminator_loss

        else:
            preds = generator_model(masked_images, masks)
            composition_images = (1 - masks) * images + masks * preds

            composition_images_detach = composition_images.detach()

            generator_fake = discriminator_model(composition_images)
            discriminator_fake = discriminator_model(composition_images_detach)
            discriminator_real = discriminator_model(images)

            loss_value = {}
            # reconstruction losses
            for loss_name in reconstruction_criterion.keys():
                temp_loss = reconstruction_criterion[loss_name](preds, images)
                temp_loss = config.reconstruction_loss_ratio[
                    loss_name] * temp_loss
                loss_value[loss_name] = temp_loss

            # adversarial loss
            for loss_name in adversarial_criterion.keys():
                discriminator_loss, generator_loss = adversarial_criterion[
                    loss_name](generator_fake, discriminator_fake,
                               discriminator_real, masks)
                generator_loss = config.adversarial_loss_ratio[
                    loss_name] * generator_loss
                loss_value['generator_loss'] = generator_loss
                loss_value['discriminator_loss'] = discriminator_loss

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if inf_nan_flag:
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            continue

        generator_loss = 0.
        for key, value in loss_value.items():
            if key in config.generator_loss_list:
                generator_loss += value
        discriminator_loss = loss_value['discriminator_loss']

        loss = 0.
        for key, value in loss_value.items():
            loss += value

        if config.use_amp:
            config.scaler.scale(generator_loss).backward()
            config.scaler.scale(discriminator_loss).backward()
        else:
            generator_loss.backward()
            discriminator_loss.backward()

        if config.use_amp:
            if hasattr(config, 'generator_clip_max_norm'
                       ) and config.generator_clip_max_norm > 0:
                config.scaler.unscale_(generator_optimizer)
                torch.nn.utils.clip_grad_norm_(generator_model.parameters(),
                                               config.generator_clip_max_norm)

            if hasattr(config, 'discriminator_clip_max_norm'
                       ) and config.discriminator_clip_max_norm > 0:
                config.scaler.unscale_(discriminator_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    discriminator_model.parameters(),
                    config.discriminator_clip_max_norm)

            config.scaler.step(generator_optimizer)
            config.scaler.update()
            generator_optimizer.zero_grad()

            config.scaler.step(discriminator_optimizer)
            config.scaler.update()
            discriminator_optimizer.zero_grad()
        else:
            if hasattr(config, 'generator_clip_max_norm'
                       ) and config.generator_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(generator_model.parameters(),
                                               config.generator_clip_max_norm)

            if hasattr(config, 'discriminator_clip_max_norm'
                       ) and config.discriminator_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    discriminator_model.parameters(),
                    config.discriminator_clip_max_norm)

            generator_optimizer.step()
            generator_optimizer.zero_grad()

            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()

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

        generator_scheduler.step(generator_optimizer,
                                 iter_index / iters + (epoch - 1))
        discriminator_scheduler.step(discriminator_optimizer,
                                     iter_index / iters + (epoch - 1))

        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], generator_lr: {generator_scheduler.current_lr:.6f}, discriminator_lr: {discriminator_scheduler.current_lr:.6f}, loss: {loss:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value:.4f}, '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg

    return avg_loss
