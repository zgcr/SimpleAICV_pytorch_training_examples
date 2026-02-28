import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp.autocast_mode import autocast

from SimpleAICV.classification.common import get_amp_type, AverageMeter
from tools.scripts import all_reduce_operation_in_group_for_variables
from tools.interactive_segmentation_scripts import get_decoder_iters_prompt_points_and_prompt_mask


def train_sam_matting(train_loader, model, criterion, optimizer, scheduler,
                      epoch, logger, config):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    if config.frozen_image_encoder:
        model.module.image_encoder.eval()
    if config.frozen_prompt_encoder:
        model.module.prompt_encoder.eval()
    if config.frozen_mask_decoder:
        model.module.mask_decoder.eval()

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
        images, masks = data['image'], data['mask']
        images, masks = images.cuda(), masks.cuda()

        trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
            'bg_map']
        trimaps, fg_maps, bg_maps = trimaps.cuda(), fg_maps.cuda(
        ), bg_maps.cuda()

        prompt_points, prompt_boxs, prompt_masks = data['prompt_point'], data[
            'prompt_box'], data['prompt_mask']

        prompts = {}
        prompts['prompt_point'] = None
        prompts['prompt_box'] = None
        prompts['prompt_mask'] = None

        use_prompt_points_prob = config.prompt_probs['prompt_point']
        use_prompt_boxs_prob = config.prompt_probs['prompt_box']
        use_prompt_masks_prob = config.prompt_probs['prompt_mask']
        assert 0.0 <= use_prompt_points_prob <= 1.0
        assert 0.0 <= use_prompt_boxs_prob <= 1.0
        assert 0.0 <= use_prompt_masks_prob <= 1.0

        decoder_iters = config.decoder_iters

        if config.use_single_prompt:
            assert sum(config.prompt_probs.values()) == 1.
            use_prompt_prob = np.random.uniform(0, 1)

            if 0. < use_prompt_prob < use_prompt_points_prob:
                prompts['prompt_point'] = prompt_points.cuda()
                prompts['prompt_box'] = None
                prompts['prompt_mask'] = None
            elif use_prompt_points_prob < use_prompt_prob < (
                    use_prompt_points_prob + use_prompt_boxs_prob):
                prompts['prompt_point'] = None
                prompts['prompt_box'] = prompt_boxs.cuda()
                prompts['prompt_mask'] = None
            elif (use_prompt_points_prob +
                  use_prompt_boxs_prob) < use_prompt_prob < 1.:
                prompts['prompt_point'] = None
                prompts['prompt_box'] = None
                prompts['prompt_mask'] = prompt_masks.cuda()
                decoder_iters = 0
        else:
            assert sum(config.prompt_probs.values()) <= 3.
            use_prompt_point_prob = np.random.uniform(0, 1)
            use_prompt_box_prob = np.random.uniform(0, 1)
            use_prompt_mask_prob = np.random.uniform(0, 1)

            if use_prompt_point_prob < use_prompt_points_prob:
                prompts['prompt_point'] = prompt_points.cuda()

            if use_prompt_box_prob < use_prompt_boxs_prob:
                prompts['prompt_box'] = prompt_boxs.cuda()

            if prompts['prompt_point'] is None and prompts[
                    'prompt_box'] is None:
                prompts['prompt_point'] = prompt_points.cuda()
                prompts['prompt_box'] = prompt_boxs.cuda()

            if use_prompt_mask_prob < use_prompt_masks_prob:
                prompts['prompt_mask'] = prompt_masks.cuda()
                decoder_iters = 0

        skip_batch_flag = False

        if torch.any(torch.isinf(images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)):
            skip_batch_flag = True

        if config.use_amp:
            with autocast(device_type="cuda", dtype=amp_type):
                batch_image_embeddings = model.module.forward_image_encoder(
                    images)
        else:
            batch_image_embeddings = model.module.forward_image_encoder(images)

        all_iter_global_preds, all_iter_local_preds, all_iter_fused_preds, all_iter_iou_preds = [], [], [], []
        # 先执行第一次decoder forward
        if config.use_amp:
            with autocast(device_type="cuda", dtype=amp_type):
                global_preds, local_preds, fused_preds, iou_preds = model.module.forward_prompt_encoder_mask_decoder(
                    batch_image_embeddings,
                    prompts,
                    mask_out_idxs=config.mask_out_idxs)
        else:
            global_preds, local_preds, fused_preds, iou_preds = model.module.forward_prompt_encoder_mask_decoder(
                batch_image_embeddings,
                prompts,
                mask_out_idxs=config.mask_out_idxs)
        all_iter_global_preds.append(global_preds)
        all_iter_local_preds.append(local_preds)
        all_iter_fused_preds.append(fused_preds)
        all_iter_iou_preds.append(iou_preds)

        # 再执行decoder_iters次decoder forward
        for _ in range(decoder_iters):
            prompts = get_decoder_iters_prompt_points_and_prompt_mask(
                fused_preds, iou_preds, masks, prompts, config)

            if config.use_amp:
                with autocast(device_type="cuda", dtype=amp_type):
                    global_preds, local_preds, fused_preds, iou_preds = model.module.forward_prompt_encoder_mask_decoder(
                        batch_image_embeddings,
                        prompts,
                        mask_out_idxs=config.mask_out_idxs)
            else:
                global_preds, local_preds, fused_preds, iou_preds = model.module.forward_prompt_encoder_mask_decoder(
                    batch_image_embeddings,
                    prompts,
                    mask_out_idxs=config.mask_out_idxs)
            all_iter_global_preds.append(global_preds)
            all_iter_local_preds.append(local_preds)
            all_iter_fused_preds.append(fused_preds)
            all_iter_iou_preds.append(iou_preds)

        # 所有次decoder forward结果收集到all_iter_global_preds/all_iter_local_preds/all_iter_fused_preds/all_iter_iou_preds中计算loss
        if config.use_amp:
            with autocast(device_type="cuda", dtype=amp_type):
                loss_value = criterion(images, [
                    all_iter_global_preds,
                    all_iter_local_preds,
                    all_iter_fused_preds,
                    all_iter_iou_preds,
                ], [masks, trimaps, fg_maps, bg_maps])
        else:
            loss_value = criterion(images, [
                all_iter_global_preds,
                all_iter_local_preds,
                all_iter_fused_preds,
                all_iter_iou_preds,
            ], [masks, trimaps, fg_maps, bg_maps])

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
                # 手动 allreduce 梯度
                for param in model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.AVG)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    config.scaler.scale(loss).backward()
        else:
            if iter_index % config.accumulation_steps == 0:
                loss.backward()
                # 手动 allreduce 梯度
                for param in model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad, op=torch.distributed.ReduceOp.AVG)
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
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>6d}, {accumulation_iters:0>6d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss
