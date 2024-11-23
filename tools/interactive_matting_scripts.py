import cv2
import collections
import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F

from torch.cuda.amp import autocast

from simpleAICV.classification.common import AverageMeter
from tools.interactive_segmentation_scripts import get_and_combine_additional_prompt_points_and_masks_with_gt
from tools.scripts import all_reduce_operation_in_group_for_variables


class MattingEvalMeter:

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

    def add_batch_result(self, preds, masks, pred_ious):
        for per_pred, per_mask, per_pred_iou in zip(preds, masks, pred_ious):
            max_iou_idx = np.argmax(per_pred_iou, axis=0)
            per_pred = per_pred[max_iou_idx]

            for i in range(self.thresh_num):
                pred_foreground = per_pred > self.thresh[i]
                mask_foreground = per_mask > self.thresh[i]

                intersection = np.sum(np.sum(pred_foreground & mask_foreground,
                                             axis=-1),
                                      axis=-1)
                all_masks = np.sum(np.sum(mask_foreground, axis=-1), axis=-1)
                all_preds = np.sum(np.sum(pred_foreground, axis=-1), axis=-1)
                union = all_preds + all_masks - intersection

                self.precision_list[i] += np.sum(intersection /
                                                 (all_preds + 1e-4))
                self.recall_list[i] += np.sum(intersection /
                                              (all_masks + 1e-4))
                self.miou_list[i] += np.sum(intersection / (union + 1e-4))

        nan_inf_count = 0
        for per_pred, per_mask, per_pred_iou in zip(preds, masks, pred_ious):
            max_iou_idx = np.argmax(per_pred_iou, axis=0)
            per_pred = per_pred[max_iou_idx]

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

        self.sample_num = self.sample_num + len(masks) - nan_inf_count

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


def validate_matting_for_all_dataset(test_loader_list, model, config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.test_dataset_name_list, test_loader_list,
                        config.test_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_matting(per_sub_dataset_loader, model,
                                                 config)
        result_dict[f'{index}_' + per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_matting(test_loader, model, config):
    eval_metric = MattingEvalMeter(config)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            batch_images, batch_masks = data['image'], data['mask']
            input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
                'prompt_point'], data['prompt_box'], data['prompt_mask']
            trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
                'bg_map']

            sizes = data['size']

            if model_on_cuda:
                batch_images, batch_masks = batch_images.cuda(
                ), batch_masks.cuda()
                trimaps, fg_maps, bg_maps = trimaps.cuda(), fg_maps.cuda(
                ), bg_maps.cuda()

            batch_prompts = {}
            batch_prompts['prompt_point'] = input_prompt_points
            batch_prompts['prompt_box'] = input_prompt_boxs
            batch_prompts['prompt_mask'] = input_prompt_masks

            test_prompt_points_type = config.test_prompt_type['prompt_point']
            test_prompt_boxes_type = config.test_prompt_type['prompt_box']
            assert isinstance(test_prompt_points_type, bool)
            assert isinstance(test_prompt_boxes_type, bool)

            assert (test_prompt_points_type is True
                    or test_prompt_boxes_type is True) is True

            assert len(input_prompt_points) == len(input_prompt_boxs) == len(
                input_prompt_masks)

            loop_mask_iters = config.loop_mask_iters

            if test_prompt_points_type:
                batch_prompts['prompt_point'] = batch_prompts[
                    'prompt_point'].cuda()
                batch_prompts['prompt_box'] = None
                batch_prompts['prompt_mask'] = None
            elif test_prompt_boxes_type:
                batch_prompts['prompt_point'] = None
                batch_prompts['prompt_box'] = batch_prompts['prompt_box'].cuda(
                )
                batch_prompts['prompt_mask'] = None

            torch.cuda.synchronize()

            for iter_i in range(loop_mask_iters):
                if iter_i > 0:
                    iter_prompt_mask = batch_masks_fused_preds
                    iter_prompt_mask = F.interpolate(
                        iter_prompt_mask, (config.input_image_size // 4,
                                           config.input_image_size // 4),
                        mode="nearest")
                    batch_prompts['prompt_mask'] = iter_prompt_mask

                _, _, batch_masks_fused_preds, batch_iou_preds = model(
                    batch_images, batch_prompts, config.mask_out_idxs)

            if len(batch_masks_fused_preds.shape) == 5:
                batch_masks_fused_preds = torch.squeeze(
                    batch_masks_fused_preds, dim=2)
            batch_masks_fused_preds = batch_masks_fused_preds.cpu().numpy()
            batch_masks = torch.squeeze(batch_masks, dim=1).cpu().numpy()
            batch_iou_preds = batch_iou_preds.cpu().numpy()

            batch_pred_list, batch_mask_list, batch_iou_preds_list = [], [], []
            for image_idx in range(len(batch_masks_fused_preds)):
                per_image_size = sizes[image_idx]
                per_image_fused_preds = batch_masks_fused_preds[image_idx]
                per_image_fused_preds = per_image_fused_preds[:, :int(
                    per_image_size[0]), :int(per_image_size[1])]

                per_image_mask = batch_masks[image_idx]
                per_image_mask = per_image_mask[:int(per_image_size[0]), :int(
                    per_image_size[1])]

                per_image_iou_pred = batch_iou_preds[image_idx]

                batch_pred_list.append(per_image_fused_preds)
                batch_mask_list.append(per_image_mask)
                batch_iou_preds_list.append(per_image_iou_pred)

            eval_metric.add_batch_result(batch_pred_list, batch_mask_list,
                                         batch_iou_preds_list)

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

        result_dict = collections.OrderedDict()
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

    local_rank = config.local_rank
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1

    for _, data in enumerate(train_loader):
        batch_images, batch_masks = data['image'], data['mask']
        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']
        trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
            'bg_map']

        batch_images, batch_masks = batch_images.cuda(), batch_masks.cuda()
        trimaps, fg_maps, bg_maps = trimaps.cuda(), fg_maps.cuda(
        ), bg_maps.cuda()

        batch_prompts = {}
        batch_prompts['prompt_point'] = input_prompt_points
        batch_prompts['prompt_box'] = input_prompt_boxs
        batch_prompts['prompt_mask'] = input_prompt_masks

        prompt_points_prob = config.train_prompt_probs['prompt_point']
        prompt_boxs_prob = config.train_prompt_probs['prompt_box']
        prompt_masks_prob = config.train_prompt_probs['prompt_mask']
        assert 0.0 <= prompt_points_prob <= 1.0
        assert 0.0 <= prompt_boxs_prob <= 1.0
        assert 0.0 <= prompt_masks_prob <= 1.0

        current_decoder_point_iters = config.decoder_point_iters

        if config.use_single_prompt:
            assert sum(config.train_prompt_probs.values()) == 1.
            use_prompt_prob = np.random.uniform(0, 1)

            if 0. < use_prompt_prob < prompt_points_prob:
                batch_prompts['prompt_point'] = batch_prompts[
                    'prompt_point'].cuda()
                batch_prompts['prompt_box'] = None
                batch_prompts['prompt_mask'] = None
            elif prompt_points_prob < use_prompt_prob < (prompt_points_prob +
                                                         prompt_boxs_prob):
                batch_prompts['prompt_point'] = None
                batch_prompts['prompt_box'] = batch_prompts['prompt_box'].cuda(
                )
                batch_prompts['prompt_mask'] = None
                current_decoder_point_iters = 1
            elif (prompt_points_prob +
                  prompt_boxs_prob) < use_prompt_prob < 1.:
                batch_prompts['prompt_point'] = None
                batch_prompts['prompt_box'] = None
                batch_prompts['prompt_mask'] = batch_prompts[
                    'prompt_mask'].cuda()
                current_decoder_point_iters = 1
        else:
            assert sum(config.train_prompt_probs.values()) <= 3.
            use_prompt_point_prob = np.random.uniform(0, 1)
            use_prompt_box_prob = np.random.uniform(0, 1)
            use_prompt_mask_prob = np.random.uniform(0, 1)

            if use_prompt_point_prob < prompt_points_prob:
                batch_prompts['prompt_point'] = batch_prompts[
                    'prompt_point'].cuda()
            else:
                batch_prompts['prompt_point'] = None
                current_decoder_point_iters = 1

            if use_prompt_box_prob < prompt_boxs_prob:
                batch_prompts['prompt_box'] = batch_prompts['prompt_box'].cuda(
                )
            else:
                batch_prompts['prompt_box'] = None

            if batch_prompts['prompt_point'] is None and batch_prompts[
                    'prompt_box'] is None:
                batch_prompts['prompt_point'] = input_prompt_points.cuda()
                batch_prompts['prompt_box'] = input_prompt_boxs.cuda()

            if use_prompt_mask_prob < prompt_masks_prob:
                batch_prompts['prompt_mask'] = batch_prompts[
                    'prompt_mask'].cuda()
            else:
                batch_prompts['prompt_mask'] = None
                current_decoder_point_iters = 1

        skip_batch_flag = False

        if torch.any(torch.isinf(batch_images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(batch_images)):
            skip_batch_flag = True

        for iter_i in range(current_decoder_point_iters):
            if iter_i > 0:
                batch_prompts['prompt_point'], batch_prompts[
                    'prompt_mask'] = get_and_combine_additional_prompt_points_and_masks_with_gt(
                        batch_prompts['prompt_point'], batch_masks,
                        batch_masks_fused_preds, config)

            if config.use_amp:
                with autocast():
                    batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = model(
                        batch_images, batch_prompts, config.mask_out_idxs)
                    loss_value = criterion(batch_images, [
                        batch_masks_global_preds,
                        batch_masks_local_preds,
                        batch_masks_fused_preds,
                        batch_iou_preds,
                    ], [
                        batch_masks,
                        trimaps,
                        fg_maps,
                        bg_maps,
                    ])
            else:
                batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = model(
                    batch_images, batch_prompts, config.mask_out_idxs)
                loss_value = criterion(batch_images, [
                    batch_masks_global_preds,
                    batch_masks_local_preds,
                    batch_masks_fused_preds,
                    batch_iou_preds,
                ], [
                    batch_masks,
                    trimaps,
                    fg_maps,
                    bg_maps,
                ])

            loss = sum(loss_value.values())

            inf_nan_flag = False
            for key, value in loss_value.items():
                if torch.any(torch.isinf(value)) or torch.any(
                        torch.isnan(value)):
                    inf_nan_flag = True

            if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                inf_nan_flag = True

            if loss == 0. or inf_nan_flag:
                print(
                    f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
                skip_batch_flag = True

            if config.use_amp:
                config.scaler.scale(loss).backward()
            else:
                loss.backward()

            if hasattr(config,
                       'skip_inf_nan_grad') and config.skip_inf_nan_grad:
                grad_inf_nan_flag = False
                for _, param in model.named_parameters():
                    per_weight_grad = param.grad
                    if per_weight_grad is not None:
                        if torch.any(
                                torch.isnan(per_weight_grad)) or torch.any(
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

            if config.use_amp:
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

        scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(iter_index), int(
            iters)
        if iter_index % int(config.print_interval) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>6d}, {accumulation_iters:0>6d}], lr: {scheduler.current_lr:.6f}, loss: {loss:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value:.4f}, '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg

    return avg_loss
