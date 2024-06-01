import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import re
import time
import math

import torch
from torch.utils.data import DataLoader

from tools.optimizers import Lion
from tools.scripts import train_image_inpainting_aot_gan_model
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         build_training_mode)


def build_optimizer(config, model, model_type):
    assert model_type in [
        'generator_model',
        'discriminator_model',
    ]
    if model_type == 'generator_model':
        optimizer_name = config.generator_optimizer[0]
        optimizer_parameters = config.generator_optimizer[1]

    elif model_type == 'discriminator_model':
        optimizer_name = config.discriminator_optimizer[0]
        optimizer_parameters = config.discriminator_optimizer[1]

    assert optimizer_name in ['SGD', 'AdamW', 'Lion'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    weight_decay = optimizer_parameters['weight_decay']

    # if global_weight_decay = False,set 1d parms weight decay = 0.
    global_weight_decay = optimizer_parameters['global_weight_decay']

    # if global_weight_decay = True,no_weight_decay_layer_name_list can't be set.
    no_weight_decay_layer_name_list = []
    if 'no_weight_decay_layer_name_list' in optimizer_parameters.keys(
    ) and isinstance(optimizer_parameters['no_weight_decay_layer_name_list'],
                     list):
        no_weight_decay_layer_name_list = optimizer_parameters[
            'no_weight_decay_layer_name_list']

    # training trick only for VIT
    if 'lr_layer_decay' and 'lr_layer_decay_block' and 'block_name' in optimizer_parameters.keys(
    ):
        lr_layer_decay = optimizer_parameters['lr_layer_decay']
        lr_layer_decay_block = optimizer_parameters['lr_layer_decay_block']
        block_name = optimizer_parameters['block_name']

        num_layers = len(lr_layer_decay_block) + 1
        lr_layer_scales = list(lr_layer_decay**(num_layers - i)
                               for i in range(num_layers + 1))

        layer_scale_id_0_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
        ]

        param_layer_name_list = []
        param_layer_weight_dict = {}
        param_layer_decay_dict, param_layer_lr_dict = {}, {}
        param_layer_lr_scale_dict = {}

        not_group_layer_name_list = []
        not_group_layer_weight_dict = {}
        not_group_layer_decay_dict, not_group_layer_lr_dict = {}, {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            in_not_group_layer = False
            if block_name in name:
                not_group_layer_name_list.append(name)
                not_group_layer_weight_dict[name] = param
                in_not_group_layer = True
            else:
                param_layer_name_list.append(name)
                param_layer_weight_dict[name] = param

            if in_not_group_layer is False:
                if any(per_layer_scale_id_0_name in name
                       for per_layer_scale_id_0_name in
                       layer_scale_id_0_name_list):
                    param_layer_lr_scale_dict[name] = lr_layer_scales[0]
                else:
                    param_layer_lr_scale_dict[name] = 1.

            if global_weight_decay is False:
                if param.ndim == 1 or any(no_weight_decay_layer_name in name
                                          for no_weight_decay_layer_name in
                                          no_weight_decay_layer_name_list):
                    if in_not_group_layer:
                        not_group_layer_decay_dict[name] = 0.
                    else:
                        param_layer_decay_dict[name] = 0.
                else:
                    per_layer_weight_decay = weight_decay
                    if 'sub_layer_weight_decay' in optimizer_parameters.keys(
                    ) and isinstance(
                            optimizer_parameters['sub_layer_weight_decay'],
                            dict):
                        for per_sub_layer_name_prefix, per_sub_layer_weight_decay in optimizer_parameters[
                                'sub_layer_weight_decay'].items():
                            if per_sub_layer_name_prefix in name:
                                per_layer_weight_decay = per_sub_layer_weight_decay
                                break

                    if in_not_group_layer:
                        not_group_layer_decay_dict[
                            name] = per_layer_weight_decay
                    else:
                        param_layer_decay_dict[name] = per_layer_weight_decay
            else:
                if in_not_group_layer:
                    not_group_layer_decay_dict[name] = weight_decay
                else:
                    param_layer_decay_dict[name] = weight_decay

            per_layer_lr = lr
            if 'sub_layer_lr' in optimizer_parameters.keys() and isinstance(
                    optimizer_parameters['sub_layer_lr'], dict):
                for per_sub_layer_name_prefix, per_sub_layer_lr in optimizer_parameters[
                        'sub_layer_lr'].items():
                    if per_sub_layer_name_prefix in name:
                        per_layer_lr = per_sub_layer_lr
                        break
            if in_not_group_layer:
                not_group_layer_lr_dict[name] = per_layer_lr
            else:
                param_layer_lr_dict[name] = per_layer_lr

        assert len(param_layer_name_list) == len(
            param_layer_weight_dict) == len(param_layer_decay_dict) == len(
                param_layer_lr_dict) == len(param_layer_lr_scale_dict)

        assert len(not_group_layer_name_list) == len(
            not_group_layer_weight_dict) == len(
                not_group_layer_decay_dict) == len(not_group_layer_lr_dict)

        per_group_weight_nums = len(not_group_layer_name_list) // len(
            lr_layer_decay_block)
        for layer_id in range(0, len(lr_layer_decay_block)):
            for per_group_id in range(per_group_weight_nums):
                per_group_layer_names = not_group_layer_name_list[
                    layer_id * per_group_weight_nums + per_group_id]

                if not isinstance(per_group_layer_names, list):
                    per_layer_name = per_group_layer_names
                    param_layer_name_list.append(per_layer_name)
                    param_layer_weight_dict[
                        per_layer_name] = not_group_layer_weight_dict[
                            per_layer_name]
                    param_layer_decay_dict[
                        per_layer_name] = not_group_layer_decay_dict[
                            per_layer_name]
                    param_layer_lr_dict[
                        per_layer_name] = not_group_layer_lr_dict[
                            per_layer_name]
                    param_layer_lr_scale_dict[
                        per_layer_name] = lr_layer_scales[layer_id + 1]
                else:
                    for per_layer_name in per_group_layer_names:
                        param_layer_name_list.append(per_layer_name)
                        param_layer_weight_dict[
                            per_layer_name] = not_group_layer_weight_dict[
                                per_layer_name]
                        param_layer_decay_dict[
                            per_layer_name] = not_group_layer_decay_dict[
                                per_layer_name]
                        param_layer_lr_dict[
                            per_layer_name] = not_group_layer_lr_dict[
                                per_layer_name]
                        param_layer_lr_scale_dict[
                            per_layer_name] = lr_layer_scales[layer_id + 1]

        assert len(param_layer_name_list) == len(
            param_layer_weight_dict) == len(param_layer_decay_dict) == len(
                param_layer_lr_dict) == len(param_layer_lr_scale_dict)

        unique_decays = list(set(param_layer_decay_dict.values()))
        unique_lrs = list(set(param_layer_lr_dict.values()))
        unique_lr_scales = list(set(param_layer_lr_scale_dict.values()))

        lr_weight_decay_combination = []
        for per_decay in unique_decays:
            for per_lr in unique_lrs:
                for per_lr_scale in unique_lr_scales:
                    lr_weight_decay_combination.append(
                        [per_decay, per_lr, per_lr_scale])

        model_params_weight_decay_list = []
        model_layer_weight_decay_list = []
        for per_decay, per_lr, per_lr_scale in lr_weight_decay_combination:
            per_decay_lr_lrscale_param_list, per_decay_lr_lrscale_name_list = [], []
            for per_layer_name in param_layer_name_list:
                per_layer_weight = param_layer_weight_dict[per_layer_name]
                per_layer_weight_decay = param_layer_decay_dict[per_layer_name]
                per_layer_lr = param_layer_lr_dict[per_layer_name]
                per_layer_lr_scale = param_layer_lr_scale_dict[per_layer_name]

                if per_layer_weight_decay == per_decay and per_layer_lr == per_lr and per_layer_lr_scale == per_lr_scale:
                    per_decay_lr_lrscale_param_list.append(per_layer_weight)
                    per_decay_lr_lrscale_name_list.append(per_layer_name)

            assert len(per_decay_lr_lrscale_param_list) == len(
                per_decay_lr_lrscale_name_list)

            if len(per_decay_lr_lrscale_param_list) > 0:
                model_params_weight_decay_list.append({
                    'params':
                    per_decay_lr_lrscale_param_list,
                    'weight_decay':
                    per_decay,
                    'lr':
                    per_lr * per_lr_scale,
                })
                model_layer_weight_decay_list.append({
                    'name': per_decay_lr_lrscale_name_list,
                    'weight_decay': per_decay,
                    'lr': per_lr,
                    'lr_scale': per_lr_scale,
                })

        assert len(model_params_weight_decay_list) == len(
            model_layer_weight_decay_list)

    else:
        param_layer_name_list = []
        param_layer_weight_dict = {}
        param_layer_decay_dict, param_layer_lr_dict = {}, {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_layer_name_list.append(name)
            param_layer_weight_dict[name] = param

            if global_weight_decay is False:
                if param.ndim == 1 or any(no_weight_decay_layer_name in name
                                          for no_weight_decay_layer_name in
                                          no_weight_decay_layer_name_list):
                    param_layer_decay_dict[name] = 0.
                else:
                    per_layer_weight_decay = weight_decay
                    if 'sub_layer_weight_decay' in optimizer_parameters.keys(
                    ) and isinstance(
                            optimizer_parameters['sub_layer_weight_decay'],
                            dict):
                        for per_sub_layer_name_prefix, per_sub_layer_weight_decay in optimizer_parameters[
                                'sub_layer_weight_decay'].items():
                            if per_sub_layer_name_prefix in name:
                                per_layer_weight_decay = per_sub_layer_weight_decay
                                break
                    param_layer_decay_dict[name] = per_layer_weight_decay
            else:
                param_layer_decay_dict[name] = weight_decay

            per_layer_lr = lr
            if 'sub_layer_lr' in optimizer_parameters.keys() and isinstance(
                    optimizer_parameters['sub_layer_lr'], dict):
                for per_sub_layer_name_prefix, per_sub_layer_lr in optimizer_parameters[
                        'sub_layer_lr'].items():
                    if per_sub_layer_name_prefix in name:
                        per_layer_lr = per_sub_layer_lr
                        break
            param_layer_lr_dict[name] = per_layer_lr

        assert len(param_layer_name_list) == len(
            param_layer_weight_dict) == len(param_layer_decay_dict) == len(
                param_layer_lr_dict)

        unique_decays = list(set(param_layer_decay_dict.values()))
        unique_lrs = list(set(param_layer_lr_dict.values()))

        lr_weight_decay_combination = []
        for per_decay in unique_decays:
            for per_lr in unique_lrs:
                lr_weight_decay_combination.append([per_decay, per_lr])

        model_params_weight_decay_list = []
        model_layer_weight_decay_list = []
        for per_decay, per_lr in lr_weight_decay_combination:
            per_decay_lr_param_list, per_decay_lr_name_list = [], []
            for per_layer_name in param_layer_name_list:
                per_layer_weight = param_layer_weight_dict[per_layer_name]
                per_layer_weight_decay = param_layer_decay_dict[per_layer_name]
                per_layer_lr = param_layer_lr_dict[per_layer_name]

                if per_layer_weight_decay == per_decay and per_layer_lr == per_lr:
                    per_decay_lr_param_list.append(per_layer_weight)
                    per_decay_lr_name_list.append(per_layer_name)

            assert len(per_decay_lr_param_list) == len(per_decay_lr_name_list)

            if len(per_decay_lr_param_list) > 0:
                model_params_weight_decay_list.append({
                    'params': per_decay_lr_param_list,
                    'weight_decay': per_decay,
                    'lr': per_lr,
                })
                model_layer_weight_decay_list.append({
                    'name': per_decay_lr_name_list,
                    'weight_decay': per_decay,
                    'lr': per_lr,
                })

        assert len(model_params_weight_decay_list) == len(
            model_layer_weight_decay_list)

    if optimizer_name == 'SGD':
        momentum = optimizer_parameters['momentum']
        nesterov = False if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        return torch.optim.SGD(
            model_params_weight_decay_list,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov), model_layer_weight_decay_list
    elif optimizer_name == 'AdamW':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta1']
        beta2 = 0.999 if 'beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta2']
        return torch.optim.AdamW(model_params_weight_decay_list,
                                 lr=lr,
                                 betas=(beta1,
                                        beta2)), model_layer_weight_decay_list
    elif optimizer_name == 'Lion':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta1']
        beta2 = 0.99 if 'beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta2']
        return Lion(model_params_weight_decay_list,
                    lr=lr,
                    betas=(beta1, beta2)), model_layer_weight_decay_list


class Scheduler:

    def __init__(self, config, optimizer, model_type):
        assert model_type in [
            'generator_model',
            'discriminator_model',
        ]
        if model_type == 'generator_model':
            self.scheduler_name = config.generator_scheduler[0]
            self.scheduler_parameters = config.generator_scheduler[1]
            self.optimizer_parameters = config.generator_optimizer[1]

        elif model_type == 'discriminator_model':
            self.scheduler_name = config.discriminator_scheduler[0]
            self.scheduler_parameters = config.discriminator_scheduler[1]
            self.optimizer_parameters = config.discriminator_optimizer[1]

        self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
        self.epochs = config.epochs

        self.lr = self.optimizer_parameters['lr']
        self.current_lr = self.lr

        self.init_param_groups_lr = [
            param_group["lr"] for param_group in optimizer.param_groups
        ]

        assert self.scheduler_name in ['MultiStepLR', 'CosineLR',
                                       'PolyLR'], 'Unsupported scheduler!'
        assert self.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
        assert self.epochs > 0, 'Illegal epochs!'

    def step(self, optimizer, epoch):
        if self.scheduler_name == 'MultiStepLR':
            gamma = self.scheduler_parameters['gamma']
            milestones = self.scheduler_parameters['milestones']
        elif self.scheduler_name == 'CosineLR':
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']
        elif self.scheduler_name == 'PolyLR':
            power = self.scheduler_parameters['power']
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']

        assert len(self.init_param_groups_lr) == len(optimizer.param_groups)

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group_init_lr = self.init_param_groups_lr[idx]

            if self.scheduler_name == 'MultiStepLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else gamma**len(
                    [m
                     for m in milestones if m <= epoch]) * param_group_init_lr
            elif self.scheduler_name == 'CosineLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else 0.5 * (
                    math.cos((epoch - self.warm_up_epochs) /
                             (self.epochs - self.warm_up_epochs) * math.pi) +
                    1) * (param_group_init_lr - min_lr) + min_lr
            elif self.scheduler_name == 'PolyLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else (
                    (1 - (epoch - self.warm_up_epochs) /
                     (self.epochs - self.warm_up_epochs))**
                    power) * (param_group_init_lr - min_lr) + min_lr

            param_group["lr"] = param_group_current_lr

        if self.scheduler_name == 'MultiStepLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else gamma**len(
                [m for m in milestones if m <= epoch]) * self.lr
        elif self.scheduler_name == 'CosineLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else 0.5 * (
                math.cos((epoch - self.warm_up_epochs) /
                         (self.epochs - self.warm_up_epochs) * math.pi) +
                1) * (self.lr - min_lr) + min_lr
        elif self.scheduler_name == 'PolyLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else (
                (1 - (epoch - self.warm_up_epochs) /
                 (self.epochs - self.warm_up_epochs))**
                power) * (self.lr - min_lr) + min_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Image Inpainting Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    if local_rank == 0:
        os.makedirs(
            checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
        os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    torch.distributed.barrier()

    logger = get_logger('train', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=config.seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        config.train_dataset, shuffle=True)
    train_loader = DataLoader(config.train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=config.train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in [
                    'generator_model',
                    'discriminator_model',
                    'reconstruction_criterion',
                    'adversarial_criterion',
                    'loss_name',
            ]:
                log_info = f'{key}: {value}'
                logger.info(log_info) if local_rank == 0 else None

    generator_model = config.generator_model.cuda()
    discriminator_model = config.discriminator_model.cuda()
    reconstruction_criterion = config.reconstruction_criterion
    adversarial_criterion = config.adversarial_criterion

    for name in reconstruction_criterion.keys():
        reconstruction_criterion[name] = reconstruction_criterion[name].cuda()
    for name in adversarial_criterion.keys():
        adversarial_criterion[name] = adversarial_criterion[name].cuda()

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'----------------generator model parameters----------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, param in generator_model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    log_info = f'------------------generator model buffers------------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, buffer in generator_model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'----------------discriminator model parameters----------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, param in discriminator_model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    log_info = f'------------------discriminator model buffers------------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, buffer in discriminator_model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    generator_optimizer, generator_model_layer_weight_decay_list = build_optimizer(
        config, generator_model, model_type='generator_model')

    log_info = f'-------------generator model layers weight decay---------------'
    logger.info(log_info) if local_rank == 0 else None
    for per_layer_list in generator_model_layer_weight_decay_list:
        layer_name_list, layer_lr, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['lr'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, lr: {layer_lr}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(log_info) if local_rank == 0 else None

    discriminator_optimizer, discriminator_model_layer_weight_decay_list = build_optimizer(
        config, discriminator_model, model_type='discriminator_model')

    log_info = f'-------------discriminator model layers weight decay---------------'
    logger.info(log_info) if local_rank == 0 else None
    for per_layer_list in discriminator_model_layer_weight_decay_list:
        layer_name_list, layer_lr, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['lr'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, lr: {layer_lr}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(log_info) if local_rank == 0 else None

    generator_scheduler = Scheduler(config,
                                    generator_optimizer,
                                    model_type='generator_model')
    discriminator_scheduler = Scheduler(config,
                                        discriminator_optimizer,
                                        model_type='discriminator_model')

    generator_model, _, config.scaler = build_training_mode(
        config, generator_model)
    discriminator_model, _, config.scaler = build_training_mode(
        config, discriminator_model)

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        generator_model.load_state_dict(
            checkpoint['generator_model_state_dict'])
        discriminator_model.load_state_dict(
            checkpoint['discriminator_model_state_dict'])
        generator_optimizer.load_state_dict(
            checkpoint['generator_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(
            checkpoint['discriminator_optimizer_state_dict'])
        generator_scheduler.load_state_dict(
            checkpoint['generator_scheduler_state_dict'])
        discriminator_scheduler.load_state_dict(
            checkpoint['discriminator_scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time

        best_loss, train_loss, generator_lr, discriminator_lr = checkpoint[
            'best_loss'], checkpoint['train_loss'], checkpoint[
                'generator_lr'], checkpoint['discriminator_lr']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, generator_lr: {generator_lr:.6f}, discriminator_lr: {discriminator_lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

    # use torch 2.0 compile function
    config.compile_support = False
    log_info = f'using torch version:{torch.__version__}'
    logger.info(log_info) if local_rank == 0 else None
    if re.match(r'2.\d*.\d*', torch.__version__):
        config.compile_support = True
        log_info = f'this torch version support torch.compile function.'
        logger.info(log_info) if local_rank == 0 else None
    elif re.match(r'1.\d*.\d*', torch.__version__):
        log_info = f'this torch version unsupport torch.compile function.'
        logger.info(log_info) if local_rank == 0 else None
    else:
        log_info = f'unsupport torch version:{torch.__version__}'
        logger.info(log_info) if local_rank == 0 else None
        return

    config.use_compile = (config.compile_support and config.use_compile)
    if config.use_compile:
        # _orig_mod
        generator_model = torch.compile(generator_model,
                                        **config.compile_params)
        discriminator_model = torch.compile(discriminator_model,
                                            **config.compile_params)

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} generator_lr: {generator_scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None
        log_info = f'epoch {epoch:0>3d} discriminator_lr: {discriminator_scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        train_loss = train_image_inpainting_aot_gan_model(
            train_loader, generator_model, discriminator_model,
            reconstruction_criterion, adversarial_criterion,
            generator_optimizer, discriminator_optimizer, generator_scheduler,
            discriminator_scheduler, epoch, logger, config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        if epoch in config.save_epochs or epoch == config.epochs:
            if config.use_compile:
                save_best_generator_model = generator_model._orig_mod.module.state_dict(
                )
                save_best_discriminator_model = discriminator_model._orig_mod.module.state_dict(
                )
            else:
                save_best_generator_model = generator_model.module.state_dict()
                save_best_discriminator_model = discriminator_model.module.state_dict(
                )

            torch.save(
                save_best_generator_model,
                os.path.join(checkpoint_dir,
                             f'epoch_{epoch}_generator_model.pth'))
            torch.save(
                save_best_discriminator_model,
                os.path.join(checkpoint_dir,
                             f'epoch_{epoch}_discriminator_model.pth'))

        train_time += (time.time() - per_epoch_start_time) / 3600

        if local_rank == 0:
            # save best acc1 model and each epoch checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                if config.use_compile:
                    save_best_generator_model = generator_model._orig_mod.module.state_dict(
                    )
                    save_best_discriminator_model = discriminator_model._orig_mod.module.state_dict(
                    )
                else:
                    save_best_generator_model = generator_model.module.state_dict(
                    )
                    save_best_discriminator_model = discriminator_model.module.state_dict(
                    )

                torch.save(
                    save_best_generator_model,
                    os.path.join(checkpoint_dir, 'best_generator_model.pth'))
                torch.save(
                    save_best_discriminator_model,
                    os.path.join(checkpoint_dir,
                                 'best_discriminator_model.pth'))

            if config.use_compile:
                save_checkpoint_generator_model = generator_model._orig_mod.state_dict(
                )
                save_checkpoint_discriminator_model = discriminator_model._orig_mod.state_dict(
                )
            else:
                save_checkpoint_generator_model = generator_model.state_dict()
                save_checkpoint_discriminator_model = discriminator_model.state_dict(
                )

            torch.save(
                {
                    'epoch':
                    epoch,
                    'time':
                    train_time,
                    'best_loss':
                    best_loss,
                    'train_loss':
                    train_loss,
                    'generator_lr':
                    generator_scheduler.current_lr,
                    'discriminator_lr':
                    discriminator_scheduler.current_lr,
                    'generator_model_state_dict':
                    save_checkpoint_generator_model,
                    'discriminator_model_state_dict':
                    save_checkpoint_discriminator_model,
                    'generator_optimizer_state_dict':
                    generator_optimizer.state_dict(),
                    'discriminator_optimizer_state_dict':
                    discriminator_optimizer.state_dict(),
                    'generator_scheduler_state_dict':
                    generator_scheduler.state_dict(),
                    'discriminator_scheduler_state_dict':
                    discriminator_scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

    if local_rank == 0:
        if os.path.exists(
                os.path.join(checkpoint_dir, 'best_generator_model.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_generator_model.pth'),
                os.path.join(checkpoint_dir,
                             f'total_loss{best_loss:.3f}_generator_model.pth'))
        if os.path.exists(
                os.path.join(checkpoint_dir, 'best_discriminator_model.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_discriminator_model.pth'),
                os.path.join(
                    checkpoint_dir,
                    f'total_loss{best_loss:.3f}_discriminator_model.pth'))

    log_info = f'train done. train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == "__main__":
    main()
