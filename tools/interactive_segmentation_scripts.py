import collections
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.cuda.amp import autocast

from simpleAICV.classification.common import AverageMeter
from tools.scripts import all_reduce_operation_in_group_for_variables


class SegmentationEvalMeter:

    def __init__(self):
        self.precision_list = 0.
        self.recall_list = 0.
        self.iou_list = 0.
        self.sample_num = 0

        self.precision_average = 0.
        self.recall_average = 0.
        self.iou_average = 0.

    def add_batch_result(self, preds, preds_iou, masks):
        for per_pred, per_pred_iou, per_mask in zip(preds, preds_iou, masks):
            max_iou_idx = per_pred_iou.argmax(dim=0)
            per_pred = per_pred[max_iou_idx]

            intersection = np.sum(np.sum(per_pred * per_mask, axis=-1),
                                  axis=-1)
            all_masks = np.sum(np.sum(per_mask, axis=-1), axis=-1)
            all_preds = np.sum(np.sum(per_pred, axis=-1), axis=-1)

            union = all_preds + all_masks - intersection

            self.precision_list += np.sum(intersection / (all_preds + 1e-4))
            self.recall_list += np.sum(intersection / (all_masks + 1e-4))
            self.iou_list += np.sum(intersection / (union + 1e-4))

        self.sample_num = self.sample_num + len(masks)

    def compute_all_metrics(self):
        self.precision_average = self.precision_list / self.sample_num
        self.recall_average = self.recall_list / self.sample_num
        self.iou_average = self.iou_list / self.sample_num


def validate_segmentation_for_all_dataset(test_loader_list, model, config):
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader,
                per_sub_dataset) in enumerate(
                    zip(config.test_dataset_name_list, test_loader_list,
                        config.test_dataset_list)):

        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = validate_segmentation(per_sub_dataset_loader,
                                                      model, config)
        result_dict[f'{index}_' + per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def validate_segmentation(test_loader, model, config):
    eval_metric = SegmentationEvalMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            batch_images, batch_masks, input_prompts_points, input_prompts_boxs, input_prompts_masks = data[
                'image'], data['mask'], data['prompt_point'], data[
                    'prompt_box'], data['prompt_mask']
            if model_on_cuda:
                batch_images, batch_masks = batch_images.cuda(
                ), batch_masks.cuda()

            batch_prompts = {}
            batch_prompts['prompt_point'] = input_prompts_points
            batch_prompts['prompt_box'] = input_prompts_boxs
            batch_prompts['prompt_mask'] = input_prompts_masks

            test_prompt_points_type = config.test_prompt_type['prompt_point']
            test_prompt_boxes_type = config.test_prompt_type['prompt_box']
            assert isinstance(test_prompt_points_type, bool)
            assert isinstance(test_prompt_boxes_type, bool)

            assert (test_prompt_points_type is True
                    or test_prompt_boxes_type is True) is True

            assert len(input_prompts_points) == len(input_prompts_boxs) == len(
                input_prompts_masks)

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

            assert config.sigmoid_out == False
            assert config.binary_mask_out == False

            torch.cuda.synchronize()

            for iter_i in range(loop_mask_iters):
                if iter_i > 0:
                    max_iou_idx = batch_iou_outputs.argmax(dim=1)
                    batch_range = torch.arange(
                        batch_mask_outputs.shape[0],
                        device=batch_mask_outputs.device)
                    iter_prompt_mask = batch_mask_outputs[
                        batch_range, max_iou_idx].unsqueeze(1)
                    iter_prompt_mask = F.interpolate(
                        iter_prompt_mask, (config.input_image_size // 4,
                                           config.input_image_size // 4),
                        mode="nearest")
                    batch_prompts['prompt_mask'] = iter_prompt_mask

                batch_mask_outputs, batch_iou_outputs = model(
                    batch_images, batch_prompts, config.mask_out_idxs)
                batch_mask_outputs = torch.cat(batch_mask_outputs, dim=0)
                batch_iou_outputs = torch.cat(batch_iou_outputs, dim=0)

            batch_mask_outputs = (
                batch_mask_outputs
                > config.mask_threshold).float().cpu().numpy()
            batch_masks = batch_masks.squeeze(dim=1).float().cpu().numpy()

            eval_metric.add_batch_result(batch_mask_outputs, batch_iou_outputs,
                                         batch_masks)

        eval_metric.compute_all_metrics()

        precision_average = eval_metric.precision_average
        recall_average = eval_metric.recall_average
        iou_average = eval_metric.iou_average

        result_dict = collections.OrderedDict()
        result_dict['mean_precision'] = precision_average
        result_dict['mean_recall'] = recall_average
        result_dict['mean_iou'] = iou_average

    return result_dict


def train_distill_sam_encoder(train_loader, model, criterion, optimizer,
                              scheduler, epoch, logger, config):
    '''
    train distill sam model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()
    if config.freeze_teacher:
        model.module.teacher.eval()

    local_rank = config.local_rank
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        batch_images = data['image']
        batch_images = batch_images.cuda()

        skip_batch_flag = False

        if torch.any(torch.isinf(batch_images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(batch_images)):
            skip_batch_flag = True

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    tea_outputs, stu_outputs = model(batch_images)
                    loss_value = criterion(tea_outputs, stu_outputs)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        tea_outputs, stu_outputs = model(batch_images)
                        loss_value = criterion(tea_outputs, stu_outputs)
        else:
            if iter_index % config.accumulation_steps == 0:
                tea_outputs, stu_outputs = model(batch_images)
                loss_value = criterion(tea_outputs, stu_outputs)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    tea_outputs, stu_outputs = model(batch_images)
                    loss_value = criterion(tea_outputs, stu_outputs)

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


def sample_point_in_mask(mask, gt, num_samples=1, slic_labels=None):
    with torch.no_grad():
        if len(mask.shape) == 4:
            mask = mask[:, 0]
            gt = gt[:, 0]

        device = mask.device
        sample_list = []
        label_list = []
        fp = (mask != gt) * (gt == 0)
        fn = (mask != gt) * (gt == 1)
        fp_fn = fp | fn

        label_map = -2 * torch.ones_like(mask, dtype=torch.int32)
        label_map[fp] = 0
        label_map[fn] = 1

        _, h, w = mask.shape
        y_axis = torch.arange(h, device=device)[:, None].expand(h, w)
        x_axis = torch.arange(w, device=device)[None, :].expand(h, w)
        grid_points = torch.stack([x_axis, y_axis], dim=-1)

        if slic_labels is not None:
            slic_labels = torch.from_numpy(slic_labels).to(device)

        # TODO parallelize
        for cur_fp_fn, cur_label_map in zip(fp_fn, label_map):
            h, w = cur_fp_fn.shape
            if slic_labels is not None:
                cur_slic_labels = slic_labels.clone()
                cur_slic_labels[~cur_fp_fn] = -1
                unique, counts = torch.unique(cur_slic_labels,
                                              return_counts=True)
                # ignore the region with label -1 (the first item)
                unique = unique[1:]
                counts = counts[1:]

                freq = (counts / counts.sum()).tolist()
                candidate_points, candidate_labels = [], []
                for u, c in zip(unique, counts):
                    # only keep one pixel in each super pixel group
                    keep_one = torch.randint(c, (1, ))
                    candidate_points.append(
                        grid_points[cur_slic_labels == u][keep_one])
                    candidate_labels.append(
                        cur_label_map[cur_slic_labels == u][keep_one])
                if len(candidate_points) < num_samples:
                    sample_list.append(
                        torch.zeros(num_samples, 2, device=device))
                    label_list.append(-torch.ones(num_samples, device=device) *
                                      2)  # to ignore
                else:
                    selected = np.random.choices(range(len(candidate_points)),
                                                 freq,
                                                 k=num_samples)
                    sample_list.append(
                        torch.cat([candidate_points[i] for i in selected],
                                  dim=0))
                    label_list.append(
                        torch.cat([candidate_labels[i] for i in selected],
                                  dim=0))
            else:
                # TODO update the criteria for aborting sampling
                if cur_fp_fn.sum() < num_samples * 10:
                    sample_list.append(
                        torch.zeros(num_samples, 2, device=device))
                    label_list.append(
                        -2 *
                        torch.ones(num_samples, device=device))  # to ignore
                else:
                    candidate_points = grid_points[cur_fp_fn]
                    candidate_labels = cur_label_map[cur_fp_fn]
                    selected = torch.randint(candidate_points.shape[0],
                                             (num_samples, ))
                    sample_list.append(candidate_points[selected])
                    label_list.append(candidate_labels[selected])

    return torch.stack(sample_list, dim=0), torch.stack(label_list, dim=0)


def get_and_combine_additional_prompt_points_and_masks_with_teacher_out(
        input_prompts_points, tea_outputs, stu_outputs, config):
    with torch.no_grad():
        combined_input_prompts_points = []
        combined_input_prompts_masks = []
        masks_t, ious_t = tea_outputs
        masks_s, ious_s = stu_outputs
        batch_size = len(masks_t)
        for idx_t in range(batch_size):
            per_image_input_prompts_points = input_prompts_points[idx_t]
            # [4,h,w]
            mask_s, iou_s = masks_s[idx_t], ious_s[idx_t]
            # [4,h,w]
            mask_t, iou_t = masks_t[idx_t], ious_t[idx_t]

            origin_mask_s = mask_s.detach()

            mask_s = (mask_s.detach() > config.mask_threshold)
            mask_t = (mask_t.detach() > config.mask_threshold)

            max_iou_idx = iou_t.argmax(dim=0)
            mask_s = mask_s[max_iou_idx].unsqueeze(0)
            mask_t = mask_t[max_iou_idx].unsqueeze(0)

            choose_prompt_mask = origin_mask_s[max_iou_idx].unsqueeze(0)
            combined_input_prompts_masks.append(choose_prompt_mask)
            point, label = sample_point_in_mask(mask_s, mask_t,
                                                config.get_point_num_per_iter)
            del mask_s, mask_t, origin_mask_s
            point = torch.cat((point.squeeze(0), label), dim=-1)

            if per_image_input_prompts_points is not None:
                per_image_input_prompts_points = torch.cat(
                    [input_prompts_points[idx_t], point])
            combined_input_prompts_points.append(
                per_image_input_prompts_points)
        combined_input_prompts_points = torch.stack(
            combined_input_prompts_points, dim=0)
        combined_input_prompts_masks = torch.stack(
            combined_input_prompts_masks, dim=0)
        combined_input_prompts_masks = F.interpolate(
            combined_input_prompts_masks,
            (config.input_image_size // 4, config.input_image_size // 4),
            mode="nearest")

    return combined_input_prompts_points, combined_input_prompts_masks


# 新取点和上一轮mask都作为迭代提示
def get_and_combine_additional_prompt_points_and_masks_with_gt(
        input_prompts_points, gt_masks, pred_masks, config):
    with torch.no_grad():
        combined_input_prompts_points = []
        combined_input_prompts_masks = []
        masks_gt = gt_masks
        masks_pred = pred_masks
        batch_size = len(masks_gt)
        for idx_t in range(batch_size):
            per_image_input_prompts_points = input_prompts_points[idx_t]
            # [1,h,w]
            mask_gt = masks_gt[idx_t]
            # [output_nums,h,w]
            mask_pred = masks_pred[idx_t]
            origin_mask_pred = mask_pred.detach()

            mask_gt = (mask_gt.detach() > config.mask_threshold)
            mask_pred = (mask_pred.detach() > config.mask_threshold)

            intersection = (mask_pred * mask_gt).sum(dim=(1, 2))
            union = mask_pred.sum(dim=(1, 2)) + mask_gt.sum(
                dim=(1, 2)) - intersection
            ious = intersection / union

            max_iou_idx = ious.argmax(dim=0)
            mask_pred = mask_pred[max_iou_idx].unsqueeze(0)

            choose_prompt_mask = origin_mask_pred[max_iou_idx].unsqueeze(0)
            combined_input_prompts_masks.append(choose_prompt_mask)

            point, label = sample_point_in_mask(mask_pred, mask_gt,
                                                config.get_point_num_per_iter)
            del mask_pred, mask_gt, origin_mask_pred
            point = torch.cat((point.squeeze(0), label), dim=-1)

            if per_image_input_prompts_points is not None:
                per_image_input_prompts_points = torch.cat(
                    [input_prompts_points[idx_t], point])
            combined_input_prompts_points.append(
                per_image_input_prompts_points)
        combined_input_prompts_points = torch.stack(
            combined_input_prompts_points, dim=0)
        combined_input_prompts_masks = torch.stack(
            combined_input_prompts_masks, dim=0)

        combined_input_prompts_masks = F.interpolate(
            combined_input_prompts_masks,
            (config.input_image_size // 4, config.input_image_size // 4),
            mode="nearest")

    return combined_input_prompts_points, combined_input_prompts_masks


def train_distill_sam_model(train_loader, model, criterion, optimizer,
                            scheduler, epoch, logger, config):
    '''
    train distill sam model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()
    if config.freeze_teacher:
        model.module.teacher.eval()

    if config.frozen_student_image_encoder:
        model.module.student.image_encoder.eval()
    if config.frozen_student_prompt_encoder:
        model.module.student.prompt_encoder.eval()
    if config.frozen_student_mask_decoder:
        model.module.student.mask_decoder.eval()

    local_rank = config.local_rank
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1

    for _, data in enumerate(train_loader):
        batch_images, batch_masks = data['image'], data['mask']
        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        batch_images, batch_masks = batch_images.cuda(), batch_masks.cuda()

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

        assert config.sigmoid_out is False
        assert config.binary_mask_out is False

        skip_batch_flag = False

        if torch.any(torch.isinf(batch_images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(batch_images)):
            skip_batch_flag = True

        for iter_i in range(current_decoder_point_iters):
            if iter_i > 0:
                batch_prompts['prompt_point'], batch_prompts[
                    'prompt_mask'] = get_and_combine_additional_prompt_points_and_masks_with_teacher_out(
                        batch_prompts['prompt_point'], tea_outputs,
                        stu_outputs, config)

            if config.use_amp:
                with autocast():
                    tea_outputs, stu_outputs = model(batch_images,
                                                     batch_prompts,
                                                     config.mask_out_idxs)
                    loss_value = criterion(tea_outputs, stu_outputs,
                                           batch_masks)
            else:
                tea_outputs, stu_outputs = model(batch_images, batch_prompts,
                                                 config.mask_out_idxs)
                loss_value = criterion(tea_outputs, stu_outputs, batch_masks)

            assert tea_outputs[0].shape[1] == len(config.mask_out_idxs)
            assert stu_outputs[0].shape[1] == len(config.mask_out_idxs)

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


def train_sam_segmentation(train_loader, model, criterion, optimizer,
                           scheduler, epoch, logger, config):
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

        batch_images, batch_masks = batch_images.cuda(), batch_masks.cuda()

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

        assert config.sigmoid_out is False
        assert config.binary_mask_out is False

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
                        batch_mask_outputs, config)

            if config.use_amp:
                with autocast():
                    batch_mask_outputs, batch_iou_outputs = model(
                        batch_images, batch_prompts, config.mask_out_idxs)
                    loss_value = criterion(
                        [batch_mask_outputs, batch_iou_outputs], batch_masks)
            else:
                batch_mask_outputs, batch_iou_outputs = model(
                    batch_images, batch_prompts, config.mask_out_idxs)
                loss_value = criterion([batch_mask_outputs, batch_iou_outputs],
                                       batch_masks)

            assert batch_mask_outputs.shape[1] == len(config.mask_out_idxs)

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
