import collections
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from simpleAICV.salient_object_detection.common import AverageMeter
from tools.scripts import all_reduce_operation_in_group_for_variables


def validate_face_parsing_for_all_dataset(val_loader_list, model, criterion,
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
                                                      model, criterion, config)
        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_face_parsing(test_loader, model, criterion, config):
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
            images, masks, sizes = data['image'], data['mask'], data['size']
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

    for i, per_precision in enumerate(per_class_precisions):
        result_dict[f'class_{i}_precision'] = per_precision

    for i, per_recall in enumerate(per_class_recalls):
        result_dict[f'class_{i}_recall'] = per_recall

    for i, per_iou in enumerate(per_class_ious):
        result_dict[f'class_{i}_iou'] = per_iou

    for i, per_dice in enumerate(per_class_dices):
        result_dict[f'class_{i}_dice'] = per_dice

    return result_dict


def train_face_parsing(train_loader, model, criterion, optimizer, scheduler,
                       epoch, logger, config):
    '''
    train face parsing model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, masks = data['image'], data['mask']
        images, masks = images.cuda(), masks.cuda()

        skip_batch_flag = False

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(masks)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(masks)):
            skip_batch_flag = True

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
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss
