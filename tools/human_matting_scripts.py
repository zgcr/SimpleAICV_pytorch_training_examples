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

from simpleAICV.human_matting.common import AverageMeter
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

        self.sad = 0
        self.mae = 0
        self.mse = 0
        self.grad = 0
        self.conn = 0

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

        nan_inf_count = 0
        for per_pred, per_mask in zip(preds, masks):
            if np.any(np.isinf(per_pred)) or np.any(np.isnan(per_pred)):
                nan_inf_count += 1
                print('per image pred nan or inf pred!')
                continue

            self.sad += np.sum(np.abs(per_mask - per_pred)) / 1000
            self.mae += np.sum(np.abs(per_mask - per_pred)) / (
                per_mask.shape[0] * per_mask.shape[1])
            self.mse += np.sum((per_mask - per_pred)**
                               2) / (per_mask.shape[0] * per_mask.shape[1])
            self.grad += self.cal_gradient(per_pred, per_mask)
            self.conn += self.cal_conn(per_pred, per_mask)

        self.sample_num = self.sample_num + masks.shape[0] - nan_inf_count

    def cal_gradient(self, per_pred, per_mask):
        pd = per_pred
        gt = per_mask

        pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
        pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
        gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
        gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
        pd_mag = np.sqrt(pd_x**2 + pd_y**2)
        gt_mag = np.sqrt(gt_x**2 + gt_y**2)

        error_map = np.square(pd_mag - gt_mag)
        per_image_grad = np.sum(error_map) / 10

        return per_image_grad

    def cal_conn(self, per_pred, per_mask):
        pred = per_pred
        true = per_mask
        step = 0.1
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(true)
        for i in range(1, len(thresh_steps)):
            true_thresh = true >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (true_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(true)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        true_diff = true - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        true_phi = 1 - true_diff * (true_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        per_image_conn = np.sum(np.abs(true_phi - pred_phi)) / 1000

        return per_image_conn

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

        self.sad = self.sad / self.sample_num
        self.mae = self.mae / self.sample_num
        self.mse = self.mse / self.sample_num
        self.grad = self.grad / self.sample_num
        self.conn = self.conn / self.sample_num


def validate_human_matting_for_all_dataset(val_loader_list, model, criterion,
                                           config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.val_dataset_name_list, val_loader_list,
                        config.val_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_human_matting(per_sub_dataset_loader,
                                                       model, criterion,
                                                       config)
        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_human_matting(test_loader, model, criterion, config):
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
            outputs = outputs[2]

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

        sad = eval_metric.sad
        mae = eval_metric.mae
        mse = eval_metric.mse
        grad = eval_metric.grad
        conn = eval_metric.conn

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

        result_dict['sad'] = sad
        result_dict['mae'] = mae
        result_dict['mse'] = mse
        result_dict['grad'] = grad
        result_dict['conn'] = conn

    return result_dict


def train_human_matting(train_loader, model, criterion, optimizer, scheduler,
                        epoch, logger, config):
    '''
    train human matting model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, masks, trimaps, fg_maps, bg_maps = data['image'], data[
            'mask'], data['trimap'], data['fg_map'], data['bg_map']

        images, masks, trimaps, fg_maps, bg_maps = images.cuda(), masks.cuda(
        ), trimaps.cuda(), fg_maps.cuda(), bg_maps.cuda()

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
                    global_preds, local_preds, fused_preds = outputs

                    loss_value = {}
                    for loss_name in criterion.keys():
                        if loss_name in [
                                'GlobalTrimapCELoss',
                                'GloabelTrimapIouLoss',
                        ]:
                            temp_loss = criterion[loss_name](global_preds,
                                                             trimaps)
                        elif loss_name in [
                                'LocalAlphaLoss',
                                'LocalLaplacianLoss',
                        ]:
                            temp_loss = criterion[loss_name](local_preds,
                                                             masks, trimaps)
                        elif loss_name in [
                                'FusionAlphaLoss',
                                'FusionLaplacianLoss',
                        ]:
                            temp_loss = criterion[loss_name](fused_preds,
                                                             masks)
                        elif loss_name in [
                                'CompositionLoss',
                        ]:
                            temp_loss = criterion[loss_name](images, masks,
                                                             fg_maps, bg_maps,
                                                             fused_preds)

                        temp_loss = config.loss_ratio[loss_name] * temp_loss
                        loss_value[loss_name] = temp_loss
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        outputs = model(images)
                        global_preds, local_preds, fused_preds = outputs

                        loss_value = {}
                        for loss_name in criterion.keys():
                            if loss_name in [
                                    'GlobalTrimapCELoss',
                                    'GloabelTrimapIouLoss',
                            ]:
                                temp_loss = criterion[loss_name](global_preds,
                                                                 trimaps)
                            elif loss_name in [
                                    'LocalAlphaLoss',
                                    'LocalLaplacianLoss',
                            ]:
                                temp_loss = criterion[loss_name](local_preds,
                                                                 masks,
                                                                 trimaps)
                            elif loss_name in [
                                    'FusionAlphaLoss',
                                    'FusionLaplacianLoss',
                            ]:
                                temp_loss = criterion[loss_name](fused_preds,
                                                                 masks)
                            elif loss_name in [
                                    'CompositionLoss',
                            ]:
                                temp_loss = criterion[loss_name](images, masks,
                                                                 fg_maps,
                                                                 bg_maps,
                                                                 fused_preds)

                            temp_loss = config.loss_ratio[loss_name] * temp_loss
                            loss_value[loss_name] = temp_loss
        else:
            if iter_index % config.accumulation_steps == 0:
                outputs = model(images)
                global_preds, local_preds, fused_preds = outputs

                loss_value = {}
                for loss_name in criterion.keys():
                    if loss_name in [
                            'GlobalTrimapCELoss',
                            'GloabelTrimapIouLoss',
                    ]:
                        temp_loss = criterion[loss_name](global_preds, trimaps)
                    elif loss_name in [
                            'LocalAlphaLoss',
                            'LocalLaplacianLoss',
                    ]:
                        temp_loss = criterion[loss_name](local_preds, masks,
                                                         trimaps)
                    elif loss_name in [
                            'FusionAlphaLoss',
                            'FusionLaplacianLoss',
                    ]:
                        temp_loss = criterion[loss_name](fused_preds, masks)
                    elif loss_name in [
                            'CompositionLoss',
                    ]:
                        temp_loss = criterion[loss_name](images, masks,
                                                         fg_maps, bg_maps,
                                                         fused_preds)

                    temp_loss = config.loss_ratio[loss_name] * temp_loss
                    loss_value[loss_name] = temp_loss
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    outputs = model(images)
                    global_preds, local_preds, fused_preds = outputs

                    loss_value = {}
                    for loss_name in criterion.keys():
                        if loss_name in [
                                'GlobalTrimapCELoss',
                                'GloabelTrimapIouLoss',
                        ]:
                            temp_loss = criterion[loss_name](global_preds,
                                                             trimaps)
                        elif loss_name in [
                                'LocalAlphaLoss',
                                'LocalLaplacianLoss',
                        ]:
                            temp_loss = criterion[loss_name](local_preds,
                                                             masks, trimaps)
                        elif loss_name in [
                                'FusionAlphaLoss',
                                'FusionLaplacianLoss',
                        ]:
                            temp_loss = criterion[loss_name](fused_preds,
                                                             masks)
                        elif loss_name in [
                                'CompositionLoss',
                        ]:
                            temp_loss = criterion[loss_name](images, masks,
                                                             fg_maps, bg_maps,
                                                             fused_preds)

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
