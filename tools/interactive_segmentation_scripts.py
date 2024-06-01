import collections
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.cuda.amp import autocast

from simpleAICV.interactive_segmentation.common import AverageMeter
from tools.scripts import all_reduce_operation_in_group_for_variables


class EvalMeter:

    def __init__(self):
        self.precision_list = 0.
        self.recall_list = 0.
        self.iou_list = 0.
        self.sample_num = 0

        self.precision_average = 0.
        self.recall_average = 0.
        self.iou_average = 0.

    def add_batch_result(self, preds, masks):
        preds = torch.cat(preds, dim=0)

        assert len(torch.unique(preds)) <= 2
        assert len(torch.unique(masks)) <= 2

        # preds shape:[b,1,h,w]
        # masks shape:[b,1,h,w]
        assert preds.shape[1] == 1
        assert masks.shape[1] == 1

        preds = preds.squeeze(1)
        masks = masks.squeeze(1)

        preds, masks = preds.cpu().numpy(), masks.cpu().numpy()

        intersection = np.sum(np.sum(preds & masks, axis=-1), axis=-1)
        all_masks = np.sum(np.sum(masks, axis=-1), axis=-1)
        all_preds = np.sum(np.sum(preds, axis=-1), axis=-1)
        union = all_preds + all_masks - intersection

        self.precision_list += np.sum(intersection / (all_preds + 1e-4))
        self.recall_list += np.sum(intersection / (all_masks + 1e-4))
        self.iou_list += np.sum(intersection / (union + 1e-4))

        self.sample_num = self.sample_num + masks.shape[0]

    def compute_all_metrics(self):
        self.precision_average = self.precision_list / self.sample_num
        self.recall_average = self.recall_list / self.sample_num
        self.iou_average = self.iou_list / self.sample_num


def train_sam(train_loader, model, criterion, optimizer, scheduler, epoch,
              logger, config):
    '''
    train semantic segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    if config.frozen_image_encoder:
        model.module.image_encoder.eval()
    if config.frozen_prompt_encoder:
        model.module.prompt_encoder.eval()
    if config.frozen_mask_decoder:
        model.module.mask_decoder.eval()

    local_rank = config.local_rank
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        batch_images, batch_masks, batch_prompts = data['batch_image'], data[
            'batch_mask'], data['batch_prompt']
        batch_images, batch_masks = batch_images.cuda(), batch_masks.cuda()

        prompt_points_prob = config.train_prompt_probs['prompt_point']
        prompt_boxs_prob = config.train_prompt_probs['prompt_box']
        prompt_masks_prob = config.train_prompt_probs['prompt_mask']
        assert 0.0 <= prompt_points_prob <= 1.0
        assert 0.0 <= prompt_boxs_prob <= 1.0
        assert 0.0 <= prompt_masks_prob <= 1.0

        input_batch_prompts = []
        for _ in range(len(batch_prompts)):
            input_batch_prompts.append({})

        assert len(input_batch_prompts) == len(batch_prompts)

        if config.use_single_prompt:
            assert sum(config.train_prompt_probs.values()) == 1.
            use_prompt_prob = np.random.uniform(0, 1)
            for image_idx, per_image_prompt in enumerate(batch_prompts):
                if 0. < use_prompt_prob < prompt_points_prob:
                    input_batch_prompts[image_idx][
                        'prompt_point'] = per_image_prompt[
                            'prompt_point'].cuda()
                    input_batch_prompts[image_idx]['prompt_box'] = None
                    input_batch_prompts[image_idx]['prompt_mask'] = None
                elif prompt_points_prob < use_prompt_prob < (
                        prompt_points_prob + prompt_boxs_prob):
                    input_batch_prompts[image_idx]['prompt_point'] = None
                    input_batch_prompts[image_idx][
                        'prompt_box'] = per_image_prompt['prompt_box'].cuda()
                    input_batch_prompts[image_idx]['prompt_mask'] = None
                elif (prompt_points_prob +
                      prompt_boxs_prob) < use_prompt_prob < 1.:
                    input_batch_prompts[image_idx][
                        'prompt_point'] = per_image_prompt[
                            'prompt_point'].cuda()
                    input_batch_prompts[image_idx][
                        'prompt_box'] = per_image_prompt['prompt_box'].cuda()
                    input_batch_prompts[image_idx][
                        'prompt_mask'] = per_image_prompt['prompt_mask'].cuda(
                        )
        else:
            assert sum(config.train_prompt_probs.values()) <= 3.
            use_prompt_point_prob = np.random.uniform(0, 1)
            use_prompt_box_prob = np.random.uniform(0, 1)
            use_prompt_mask_prob = np.random.uniform(0, 1)
            for image_idx, per_image_prompt in enumerate(batch_prompts):
                if use_prompt_point_prob < prompt_points_prob:
                    input_batch_prompts[image_idx][
                        'prompt_point'] = per_image_prompt[
                            'prompt_point'].cuda()
                else:
                    input_batch_prompts[image_idx]['prompt_point'] = None

                if use_prompt_box_prob < prompt_boxs_prob:
                    input_batch_prompts[image_idx][
                        'prompt_box'] = per_image_prompt['prompt_box'].cuda()
                else:
                    input_batch_prompts[image_idx]['prompt_box'] = None

                if input_batch_prompts[image_idx][
                        'prompt_point'] is None and input_batch_prompts[
                            image_idx]['prompt_box'] is None:
                    input_batch_prompts[image_idx][
                        'prompt_point'] = per_image_prompt[
                            'prompt_point'].cuda()
                    input_batch_prompts[image_idx][
                        'prompt_box'] = per_image_prompt['prompt_box'].cuda()

                if use_prompt_mask_prob < prompt_masks_prob:
                    input_batch_prompts[image_idx][
                        'prompt_mask'] = per_image_prompt['prompt_mask'].cuda(
                        )
                else:
                    input_batch_prompts[image_idx]['prompt_mask'] = None

        assert config.sigmoid_out is False
        assert config.binary_mask_out is False

        if torch.any(torch.isinf(batch_images)) or torch.any(
                torch.isinf(batch_masks)):
            continue

        if torch.any(torch.isnan(batch_images)) or torch.any(
                torch.isnan(batch_masks)):
            continue

        if torch.sum(batch_images) == 0:
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    batch_mask_outputs, batch_iou_outputs = model(
                        batch_images, batch_prompts, config.mask_out_idxs)
                    loss_value = criterion(
                        [batch_mask_outputs, batch_iou_outputs], batch_masks)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        batch_mask_outputs, batch_iou_outputs = model(
                            batch_images, batch_prompts, config.mask_out_idxs)
                        loss_value = criterion(
                            [batch_mask_outputs, batch_iou_outputs],
                            batch_masks)
        else:
            if iter_index % config.accumulation_steps == 0:
                batch_mask_outputs, batch_iou_outputs = model(
                    batch_images, batch_prompts, config.mask_out_idxs)
                loss_value = criterion([batch_mask_outputs, batch_iou_outputs],
                                       batch_masks)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    batch_mask_outputs, batch_iou_outputs = model(
                        batch_images, batch_prompts, config.mask_out_idxs)
                    loss_value = criterion(
                        [batch_mask_outputs, batch_iou_outputs], batch_masks)

        assert batch_mask_outputs[0].shape[1] == len(config.mask_out_idxs)

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
            losses.update(loss, batch_images.size(0))

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


def test_sam(test_loader, model, criterion, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    eval_metric = EvalMeter()

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            batch_images, batch_masks, batch_prompts = data[
                'batch_image'], data['batch_mask'], data['batch_prompt']

            assert len(torch.unique(batch_masks)) <= 2
            batch_masks = batch_masks.bool()

            if model_on_cuda:
                batch_images, batch_masks = batch_images.cuda(
                ), batch_masks.cuda()

            test_prompt_points_type = config.test_prompt_type['prompt_point']
            test_prompt_boxes_type = config.test_prompt_type['prompt_box']
            test_prompt_masks_type = config.test_prompt_type['prompt_mask']
            assert isinstance(test_prompt_points_type, bool)
            assert isinstance(test_prompt_boxes_type, bool)
            assert isinstance(test_prompt_masks_type, bool)

            assert (test_prompt_points_type is True
                    or test_prompt_boxes_type is True) is True

            input_batch_prompts = []
            for _ in range(len(batch_prompts)):
                input_batch_prompts.append({})

            assert len(input_batch_prompts) == len(batch_prompts)

            for image_idx, per_image_prompt in enumerate(batch_prompts):
                if test_prompt_points_type:
                    input_batch_prompts[image_idx][
                        'prompt_point'] = per_image_prompt[
                            'prompt_point'].cuda()
                else:
                    input_batch_prompts[image_idx]['prompt_point'] = None

                if test_prompt_boxes_type:
                    input_batch_prompts[image_idx][
                        'prompt_box'] = per_image_prompt['prompt_box'].cuda()
                else:
                    input_batch_prompts[image_idx]['prompt_box'] = None

                if test_prompt_masks_type:
                    input_batch_prompts[image_idx][
                        'prompt_mask'] = per_image_prompt['prompt_mask'].cuda(
                        )
                else:
                    input_batch_prompts[image_idx]['prompt_mask'] = None

            assert config.sigmoid_out is False
            assert config.binary_mask_out is True

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            batch_mask_outputs, batch_iou_outputs = model(
                batch_images, batch_prompts, config.mask_out_idxs)

            assert batch_mask_outputs[0].shape[1] == 1

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            for per_image_outputs in batch_mask_outputs:
                assert len(torch.unique(per_image_outputs)) <= 2

            eval_metric.add_batch_result(batch_mask_outputs, batch_masks)

            end = time.time()

        eval_metric.compute_all_metrics()

        precision_average = eval_metric.precision_average
        recall_average = eval_metric.recall_average
        iou_average = eval_metric.iou_average

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / (config.batch_size //
                                               config.gpus_num) * 1000
        per_image_inference_time = batch_time.avg / (config.batch_size //
                                                     config.gpus_num) * 1000

        result_dict = collections.OrderedDict()
        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        result_dict['mean_precision'] = precision_average
        result_dict['mean_recall'] = recall_average
        result_dict['mean_iou'] = iou_average

    return result_dict
