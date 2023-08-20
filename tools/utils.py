import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import logging
import logging.handlers
import copy
import math
import numpy as np
import os
import random
import time

from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.cuda.amp import GradScaler

from tools.optimizers import Lion


def parse_args_example():
    '''
    args backup
    '''
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--string-variable',
                        type=str,
                        default='string',
                        help='explain variable')
    parser.add_argument('--float-variable',
                        type=float,
                        default=0.01,
                        help='explain variable')
    parser.add_argument('--int-variable',
                        type=int,
                        default=10,
                        help='explain variable')
    parser.add_argument('--list-variable',
                        type=list,
                        default=[1, 10, 100],
                        help='explain variable')
    # store_true即命令行有这个参数时，值为True,没有这个参数时，默认值为False
    # store_false即命令行有这个参数时，值为False,没有这个参数时，默认值为True
    parser.add_argument('--bool-variable',
                        default=False,
                        action='store_true',
                        help='explain variable')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='LOCAL_PROCESS_RANK in DistributedDataParallel model')

    return parser.parse_args()


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_name = os.path.join(log_dir, '{}.info.log'.format(name))
    file_handler = logging.handlers.TimedRotatingFileHandler(file_name,
                                                             when='W0',
                                                             encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_seed_init_fn(worker_id, num_workers, local_rank, seed):
    # worker_seed_init_fn function will be called at the beginning of each epoch
    # for each epoch the same worker has same seed value,so we add the current time to the seed
    worker_seed = num_workers * local_rank + worker_id + seed + int(
        time.time())
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_macs_and_params(config, model):
    assert isinstance(config.input_image_size, int) == True or isinstance(
        config.input_image_size,
        list) == True, 'Illegal input_image_size type!'

    if isinstance(config.input_image_size, int):
        macs_input = torch.randn(1, 3, config.input_image_size,
                                 config.input_image_size).cpu()
    elif isinstance(config.input_image_size, list):
        macs_input = torch.randn(1, 3, config.input_image_size[0],
                                 config.input_image_size[1]).cpu()

    model = model.cpu()

    macs, params = profile(model, inputs=(macs_input, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')

    return macs, params


class EmaModel(nn.Module):
    """ Model Exponential Moving Average V2
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/utils/model_ema.py
    decay=0.9999 means that when updating the model weights, we keep 99.99% of the previous model weights and only update 0.01% of the new weights at each iteration.
    ema_model_weights = decay * ema_model_weights + (1 - decay) * model_weights

    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    """

    def __init__(self, model, decay=0.9999):
        super(EmaModel, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.update_fn = lambda e, m: self.decay * e + (1. - self.decay) * m

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema_model.state_dict().values(),
                                      model.state_dict().values()):
                assert ema_v.shape == model_v.shape, 'wrong ema model!'
                ema_v.copy_(self.update_fn(ema_v, model_v))


def build_training_mode(config, model):
    ema_model, scaler = None, None
    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    local_rank = torch.distributed.get_rank()
    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        ema_model = EmaModel(model, decay=config.ema_model_decay)
        ema_model.ema_model = nn.parallel.DistributedDataParallel(
            ema_model.ema_model,
            device_ids=[local_rank],
            output_device=local_rank)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

    if hasattr(config, 'use_amp') and config.use_amp:
        scaler = GradScaler()

    return model, ema_model, scaler


class Scheduler:

    def __init__(self, config, optimizer):
        self.scheduler_name = config.scheduler[0]
        self.scheduler_parameters = config.scheduler[1]
        self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
        self.epochs = config.epochs
        self.optimizer_parameters = config.optimizer[1]
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


def build_optimizer(config, model):
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
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