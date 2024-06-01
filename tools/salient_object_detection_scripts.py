import cv2
import os

import collections
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from scipy.ndimage import gaussian_filter

from simpleAICV.salient_object_detection.common import AverageMeter
from tools.scripts import all_reduce_operation_in_group_for_variables


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
        val_loader_list, model, criterion, config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.val_dataset_name_list, val_loader_list,
                        config.val_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_salient_object_detection_segmentation(
            per_sub_dataset_loader, model, criterion, config)
        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_salient_object_detection_segmentation(test_loader, model,
                                                   criterion, config):
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
            images, masks, sizes = data['image'], data['mask'], data['size']

            if model_on_cuda:
                images, masks = images.cuda(), masks.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

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


def train_salient_object_detection_segmentation(train_loader, model, criterion,
                                                optimizer, scheduler, epoch,
                                                logger, config):
    '''
    train salient object detection segmentation model for one epoch
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
