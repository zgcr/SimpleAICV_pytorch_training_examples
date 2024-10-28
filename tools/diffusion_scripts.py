import cv2

import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from simpleAICV.classification.common import AverageMeter
from simpleAICV.diffusion_model.metrics.compute_fid_is_score import calculate_frechet_distance, compute_inception_score
from tools.scripts import all_reduce_operation_in_group_for_variables


def generate_diffusion_model_images(test_loader, model, sampler, config):
    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    local_rank = config.local_rank
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

    local_rank = config.local_rank
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

        skip_batch_flag = False

        if torch.any(torch.isinf(images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)):
            skip_batch_flag = True

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
            print(f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
            skip_batch_flag = True

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
