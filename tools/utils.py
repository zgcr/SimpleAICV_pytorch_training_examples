import argparse
import logging
import logging.handlers
import copy
import math
import numpy as np
import os
import random
import time

import apex
from apex import amp
from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.ops


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

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='W0',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

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
                ema_v.copy_(self.update_fn(ema_v, model_v))


def build_training_mode(config, model, optimizer):
    '''
    Choose model training mode:nn.DataParallel/nn.parallel.DistributedDataParallel,use apex or not
    apex only used in mode:nn.parallel.DistributedDataParallel
    '''
    ema_model = None
    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if config.apex:
        if config.sync_bn:
            model = apex.parallel.convert_syncbn_model(model).cuda()

        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch.nn, 'Sigmoid')
        amp.register_float_function(torch.nn, 'Softmax')
        amp.register_float_function(torch.nn, 'Parameter')
        amp.register_float_function(torch.nn.functional, 'sigmoid')
        amp.register_float_function(torch.nn.functional, 'softmax')
        amp.register_float_function(torchvision.ops, 'deform_conv2d')

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        if hasattr(config, 'use_ema_model') and config.use_ema_model:
            ema_model = EmaModel(model, decay=config.ema_model_decay)
            ema_model.ema_model = apex.parallel.DistributedDataParallel(
                ema_model.ema_model, delay_allreduce=True)
        model = apex.parallel.DistributedDataParallel(model,
                                                      delay_allreduce=True)
    else:
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

    return model, ema_model


class Scheduler:

    def __init__(self, config):
        self.scheduler_name = config.scheduler[0]
        self.scheduler_parameters = config.scheduler[1]
        self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
        self.epochs = config.epochs
        self.optimizer_parameters = config.optimizer[1]
        self.lr = self.optimizer_parameters['lr']
        self.current_lr = self.lr

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

        if self.scheduler_name == 'MultiStepLR':
            self.current_lr = (
                (epoch + 1)
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else gamma**len(
                [m for m in milestones if m <= epoch]) * self.lr
        elif self.scheduler_name == 'CosineLR':
            self.current_lr = (
                epoch + 1
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else 0.5 * (
                math.cos((epoch - self.warm_up_epochs) /
                         (self.epochs - self.warm_up_epochs) * math.pi) +
                1) * (self.lr - min_lr) + min_lr
        elif self.scheduler_name == 'PolyLR':
            self.current_lr = (
                epoch + 1
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else (
                (1 - (epoch - self.warm_up_epochs) /
                 (self.epochs - self.warm_up_epochs))**
                power) * (self.lr - min_lr) + min_lr

        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = self.current_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = self.current_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def build_optimizer(config, model):
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    global_weight_decay = optimizer_parameters['global_weight_decay']
    weight_decay = optimizer_parameters['weight_decay']

    if global_weight_decay:
        decay_weight_list, decay_weight_name_list = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            decay_weight_list.append(param)
            decay_weight_name_list.append(name)

        model_params_weight_decay_list = [
            {
                'params': decay_weight_list,
                'weight_decay': weight_decay
            },
        ]
        model_layer_weight_decay_list = [
            {
                'name': decay_weight_name_list,
                'weight_decay': weight_decay
            },
        ]

    else:
        no_weight_decay_layer_name_list = optimizer_parameters[
            'no_weight_decay_layer_name_list']
        decay_weight_list, no_decay_weight_list = [], []
        decay_weight_name_list, no_decay_weight_name_list = [], []

        if 'lr_layer_decay' and 'lr_layer_decay_block' and 'block_name' in optimizer_parameters.keys(
        ):
            lr_layer_decay = optimizer_parameters['lr_layer_decay']
            lr_layer_decay_block = optimizer_parameters['lr_layer_decay_block']
            block_name = optimizer_parameters['block_name']

            num_layers = len(lr_layer_decay_block)
            lr_layer_scales = list(lr_layer_decay**(num_layers - i)
                                   for i in range(num_layers))

            decay_weight_list, no_decay_weight_list = [], []
            decay_weight_name_list, no_decay_weight_name_list = [], []

            not_group_layer_scale_weight_list = []
            not_group_layer_scale_weight_name_list = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if param.ndim <= 1 or name.endswith(".bias") or any(
                        no_weight_decay_layer_name in name
                        for no_weight_decay_layer_name in
                        no_weight_decay_layer_name_list):
                    no_decay_weight_list.append(param)
                    no_decay_weight_name_list.append(name)
                    continue

                if block_name in name:
                    not_group_layer_scale_weight_list.append(param)
                    not_group_layer_scale_weight_name_list.append(name)
                else:
                    decay_weight_list.append(param)
                    decay_weight_name_list.append(name)

            model_params_weight_decay_list = [
                {
                    "lr_scale": 1.,
                    'params': no_decay_weight_list,
                    'weight_decay': 0.
                },
                {
                    "lr_scale": 1.,
                    'params': decay_weight_list,
                    'weight_decay': weight_decay
                },
            ]

            model_layer_weight_decay_list = [
                {
                    "lr_scale": 1.,
                    'name': no_decay_weight_name_list,
                    'weight_decay': 0.,
                },
                {
                    "lr_scale": 1.,
                    'name': decay_weight_name_list,
                    'weight_decay': weight_decay,
                },
            ]

            assert len(not_group_layer_scale_weight_list) == len(
                not_group_layer_scale_weight_name_list)
            assert len(
                not_group_layer_scale_weight_name_list) % num_layers == 0

            per_group_weight_nums = len(
                not_group_layer_scale_weight_name_list) // num_layers
            for layer_id in range(0, num_layers):
                per_group_decay_weight_list,per_group_decay_weight_name_list=[],[]
                for per_group_id in range(per_group_weight_nums):
                    per_group_decay_weight_list.append(
                        not_group_layer_scale_weight_list[
                            layer_id * per_group_weight_nums + per_group_id])
                    per_group_decay_weight_name_list.append(
                        not_group_layer_scale_weight_name_list[
                            layer_id * per_group_weight_nums + per_group_id])
                model_params_weight_decay_list.append({
                    "lr_scale":
                    lr_layer_scales[layer_id],
                    'params':
                    per_group_decay_weight_list,
                    'weight_decay':
                    weight_decay,
                })
                model_layer_weight_decay_list.append({
                    "lr_scale":
                    lr_layer_scales[layer_id],
                    'name':
                    per_group_decay_weight_name_list,
                    'weight_decay':
                    weight_decay,
                })

        else:
            decay_weight_list, no_decay_weight_list = [], []
            decay_weight_name_list, no_decay_weight_name_list = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if param.ndim <= 1 or name.endswith(".bias") or any(
                        no_weight_decay_layer_name in name
                        for no_weight_decay_layer_name in
                        no_weight_decay_layer_name_list):
                    no_decay_weight_list.append(param)
                    no_decay_weight_name_list.append(name)
                else:
                    decay_weight_list.append(param)
                    decay_weight_name_list.append(name)

            model_params_weight_decay_list = [
                {
                    'params': no_decay_weight_list,
                    'weight_decay': 0.
                },
                {
                    'params': decay_weight_list,
                    'weight_decay': weight_decay
                },
            ]

            model_layer_weight_decay_list = [
                {
                    'name': no_decay_weight_name_list,
                    'weight_decay': 0.
                },
                {
                    'name': decay_weight_name_list,
                    'weight_decay': weight_decay
                },
            ]

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
