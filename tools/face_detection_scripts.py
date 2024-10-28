import collections
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from simpleAICV.face_detection.common import AverageMeter
from tools.scripts import all_reduce_operation_in_group_for_variables, compute_voc_ap, compute_ious


def cal_precision_recall(gts, preds, image_size, per_iou_threshold):
    per_class_gt_boxes1 = [image[0][image[1] == 0] for image in gts
                           ]  #[[[a,b,c,d]],  [[a,b,c,d],[a,b,c,d]], ...]
    per_class_pred_boxes1 = [image[0][image[1] == 0] for image in preds]

    fp = [0, 0, 0, 0, 0, 0, 0, 0]
    tp = [0, 0, 0, 0, 0, 0, 0, 0]
    total_gts = [0, 0, 0, 0, 0, 0, 0, 0]
    for per_image_gt_boxes, per_image_pred_boxes, per_image_size in zip(
            per_class_gt_boxes1, per_class_pred_boxes1, image_size):  #单图遍历
        total_gts[0] = total_gts[0] + len(per_image_gt_boxes)
        # one gt can only be assigned to one predicted bbox
        # loop for each predicted bbox
        assigned_gt = []

        for index in range(len(per_image_pred_boxes)):  #遍历单图box
            if per_image_gt_boxes.shape[0] == 0:
                # if no gts found for the predicted bbox, assign the bbox to fp
                fp[0] += 1
                continue
            pred_box = np.expand_dims(per_image_pred_boxes[index], axis=0)
            iou = compute_ious(per_image_gt_boxes, pred_box)
            gt_for_box = np.argmax(iou, axis=0)
            max_overlap = iou[gt_for_box, 0]
            if max_overlap >= per_iou_threshold and gt_for_box not in assigned_gt:
                assigned_gt.append(gt_for_box)
                tp[0] += 1
            else:
                fp[0] += 1
        #计算分段
        assigned_pred = []
        for index in range(len(per_image_gt_boxes)):  #遍历单图box
            if per_image_gt_boxes.shape[0] != 0:
                gt_box = np.expand_dims(per_image_gt_boxes[index], axis=0)
                gt_box_w = gt_box[0][2] - gt_box[0][0]
                gt_box_h = gt_box[0][3] - gt_box[0][1]
                gt_box_scale = max(gt_box_w / per_image_size[1],
                                   gt_box_h / per_image_size[0])

                if per_image_pred_boxes.shape[0] == 0:
                    max_overlap = 0
                else:
                    iou = compute_ious(per_image_pred_boxes, gt_box)
                    pred_for_box = np.argmax(iou, axis=0)
                    max_overlap = iou[pred_for_box, 0]
                if max_overlap >= per_iou_threshold and pred_for_box not in assigned_pred:
                    assigned_pred.append(pred_for_box)
                    if gt_box_scale < 1 / 100:
                        total_gts[1] += 1
                        tp[1] += 1
                    if gt_box_scale < 1 / 20:
                        total_gts[2] += 1
                        tp[2] += 1
                    if gt_box_scale < 1 / 10:
                        total_gts[3] += 1
                        tp[3] += 1
                    if gt_box_scale < 1 / 5:
                        total_gts[4] += 1
                        tp[4] += 1
                    elif gt_box_scale < 1 / 4:
                        total_gts[5] += 1
                        tp[5] += 1
                    elif gt_box_scale < 1 / 3:
                        total_gts[6] += 1
                        tp[6] += 1
                    else:
                        total_gts[7] += 1
                        tp[7] += 1
                else:
                    if gt_box_scale < 1 / 100:
                        total_gts[1] += 1
                        fp[1] += 1
                    if gt_box_scale < 1 / 20:
                        total_gts[2] += 1
                        fp[2] += 1
                    if gt_box_scale < 1 / 10:
                        total_gts[3] += 1
                        fp[3] += 1
                    if gt_box_scale < 1 / 5:
                        total_gts[4] += 1
                        fp[4] += 1
                    elif gt_box_scale < 1 / 4:
                        total_gts[5] += 1
                        fp[5] += 1
                    elif gt_box_scale < 1 / 3:
                        total_gts[6] += 1
                        fp[6] += 1
                    else:
                        total_gts[7] += 1
                        fp[7] += 1
    recall, precision = [], []
    for i in range(len(total_gts)):
        if total_gts[i] < 1:
            recall.append(None)
            precision.append(None)
        else:
            recall.append(tp[i] / total_gts[i])
            precision.append(tp[i] / (tp[i] + fp[i]))

    return precision, recall, total_gts


def evaluate_voc_detection(val_loader, model, criterion, decoder, config):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    batch_size = int(config.batch_size // config.gpus_num)
    with torch.no_grad():
        preds, gts, image_size = [], [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for _, data in tqdm(enumerate(val_loader)):
            images, annots, scales, sizes = data['image'], data[
                'annots'], data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()
            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()
            outs_tuple = model(images)

            loss_value = criterion(outs_tuple, annots)
            loss = sum(loss_value.values())
            losses.update(loss, images.size(0))

            pred_scores, pred_classes, pred_boxes = decoder(outs_tuple)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            annots = annots.cpu().numpy()
            gt_bboxes, gt_classes = (annots[:, :,
                                            0:4]).copy(), (annots[:, :,
                                                                  4]).copy()

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
                per_image_pred_boxes[:,
                                     2] = np.minimum(per_image_pred_boxes[:,
                                                                          2],
                                                     per_image_size[1])  #w
                per_image_pred_boxes[:,
                                     3] = np.minimum(per_image_pred_boxes[:,
                                                                          3],
                                                     per_image_size[0])  #h

                preds.append([
                    per_image_pred_boxes, per_image_pred_classes,
                    per_image_pred_scores
                ])

                per_image_gt_bboxes = per_image_gt_bboxes[per_image_gt_classes
                                                          > -1]
                per_image_gt_classes = per_image_gt_classes[
                    per_image_gt_classes > -1]  #[0,0]

                gts.append([per_image_gt_bboxes, per_image_gt_classes])
                image_size.append(per_image_size)

            end = time.time()

        #gts:[[[a,b,c,d]],[0]],   [[[a,b,c,d],[a,b,c,d]],[0,0]],....
        result_dict = collections.OrderedDict()
        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / batch_size * 1000
        per_image_inference_time = batch_time.avg / batch_size * 1000

        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        all_iou_threshold_map = collections.OrderedDict()
        all_iou_threshold_per_class_ap = collections.OrderedDict()
        all_iou_all_precision = collections.OrderedDict()
        all_iou_all_recall = collections.OrderedDict()
        all_iou_recall1 = collections.OrderedDict()
        all_iou_recall2 = collections.OrderedDict()
        all_iou_recall3 = collections.OrderedDict()
        all_iou_recall4 = collections.OrderedDict()
        all_iou_recall5 = collections.OrderedDict()
        all_iou_recall6 = collections.OrderedDict()
        all_iou_recall7 = collections.OrderedDict()
        #计算recall precision
        #iou阈值遍历
        for per_iou_threshold in tqdm(config.eval_voc_iou_threshold_list):
            precisionl, recalll, total_gts = cal_precision_recall(
                gts, preds, image_size, per_iou_threshold)
            all_iou_all_precision[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,precision'] = precisionl[
                    0]
            all_iou_all_recall[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,recall'] = recalll[
                    0]

            all_iou_recall1['范围:0<area<1/100,框总数:'] = total_gts[1]
            all_iou_recall1[
                f'IoU={per_iou_threshold:.2f},0<area<1/100,maxDets=100,recall'] = recalll[
                    1]

            all_iou_recall2['范围:1/100<area<1/20,框总数:'] = total_gts[2]
            all_iou_recall2[
                f'IoU={per_iou_threshold:.2f},1/100<area<1/20,maxDets=100,recall'] = recalll[
                    2]

            all_iou_recall3['范围:1/20<area<1/10,框总数:'] = total_gts[3]
            all_iou_recall3[
                f'IoU={per_iou_threshold:.2f},1/20<area<1/10,maxDets=100,recall'] = recalll[
                    3]

            all_iou_recall4['范围:1/10<area<1/5,框总数:'] = total_gts[4]
            all_iou_recall4[
                f'IoU={per_iou_threshold:.2f},1/10<area<1/5,maxDets=100,recall'] = recalll[
                    4]

            all_iou_recall5['范围:1/5<area<1/4,框总数:'] = total_gts[5]
            all_iou_recall5[
                f'IoU={per_iou_threshold:.2f},1/5<area<1/4,maxDets=100,recall'] = recalll[
                    5]

            all_iou_recall6['范围:1/4<area<1/3,框总数:'] = total_gts[6]
            all_iou_recall6[
                f'IoU={per_iou_threshold:.2f},1/4<area<1/3,maxDets=100,recall'] = recalll[
                    6]

            all_iou_recall7['范围:1/3<area<1,框总数:'] = total_gts[7]
            all_iou_recall7[
                f'IoU={per_iou_threshold:.2f},1/3<area<1,maxDets=100,recall'] = recalll[
                    7]
        #计算map
        for per_iou_threshold in tqdm(
                config.eval_voc_iou_threshold_list):  #iou阈值遍历
            per_iou_threshold_all_class_ap = collections.OrderedDict()
            # per_iou_threshold_all_class_precision= collections.OrderedDict()
            # per_iou_threshold_all_class_recall= collections.OrderedDict()
            for class_index in range(config.num_classes):
                per_class_gt_boxes = [
                    image[0][image[1] == class_index] for image in gts
                ]  #[[[a,b,c,d]],  [[a,b,c,d],[a,b,c,d]], ...]
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
                        per_class_pred_scores):  #单图遍历
                    total_gts = total_gts + len(per_image_gt_boxes)
                    # one gt can only be assigned to one predicted bbox
                    assigned_gt = []
                    # loop for each predicted bbox
                    for index in range(len(per_image_pred_boxes)):  #遍历单图box
                        scores = np.append(scores,
                                           per_image_pred_scores[index])
                        if per_image_gt_boxes.shape[0] == 0:
                            # if no gts found for the predicted bbox, assign the bbox to fp
                            fp = np.append(fp, 1)
                            tp = np.append(tp, 0)
                            continue
                        pred_box = np.expand_dims(per_image_pred_boxes[index],
                                                  axis=0)
                        iou = compute_ious(per_image_gt_boxes, pred_box)  #？？
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
        for key, value in all_iou_all_precision.items():
            result_dict[key] = value
        for key, value in all_iou_all_recall.items():
            result_dict[key] = value
        for key, value in all_iou_recall1.items():
            result_dict[key] = value
        for key, value in all_iou_recall2.items():
            result_dict[key] = value
        for key, value in all_iou_recall3.items():
            result_dict[key] = value
        for key, value in all_iou_recall4.items():
            result_dict[key] = value
        for key, value in all_iou_recall5.items():
            result_dict[key] = value
        for key, value in all_iou_recall6.items():
            result_dict[key] = value
        for key, value in all_iou_recall7.items():
            result_dict[key] = value

        return result_dict


def validate_face_detection(val_loader, model, criterion, decoder, config):
    assert config.eval_type in ['VOC']

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    func_dict = {
        'VOC': evaluate_voc_detection,
    }
    result_dict = func_dict[config.eval_type](val_loader, model, criterion,
                                              decoder, config)

    return result_dict


def validate_face_detection_for_all_dataset(val_loader_list, model, criterion,
                                            decoder, config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.val_dataset_name_list, val_loader_list,
                        config.val_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_face_detection(
            per_sub_dataset_loader, model, criterion, decoder, config)
        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def train_face_detection(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train detection model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, targets = data['image'], data['annots']
        images, targets = images.cuda(), targets.cuda()

        skip_batch_flag = False

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(targets)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(targets)):
            skip_batch_flag = True

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    outs_tuple = model(images)
                    loss_value = criterion(outs_tuple, targets)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        outs_tuple = model(images)
                        loss_value = criterion(outs_tuple, targets)
        else:
            if iter_index % config.accumulation_steps == 0:
                outs_tuple = model(images)
                loss_value = criterion(outs_tuple, targets)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
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
                    if torch.any(torch.isnan(per_weight_grad)) or torch.any(
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
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        torch.distributed.barrier()

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
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss
