import argparse
import logging
import logging.handlers
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
    parser.add_argument('--bool-variable',
                        default=False,
                        action='store_true',
                        help='explain variable')

    return parser.parse_args()


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
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


def compute_flops_and_params(config, model):
    flops_input = torch.randn(1, 3, config.input_image_size,
                              config.input_image_size)

    model_on_cuda = next(model.parameters()).is_cuda
    if model_on_cuda:
        flops_input = flops_input.cuda()

    flops, params = profile(model, inputs=(flops_input, ), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')

    return flops, params


def build_optimizer(config, model):
    assert config.optimizer in ['SGD', 'AdamW'], 'Unsupported optimizer!'

    if config.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(),
                               lr=config.lr,
                               momentum=config.momentum,
                               weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.weight_decay)


def build_scheduler(config, optimizer):
    '''
    The value of config.warm_up_epochs is zero or an integer larger than 0
    '''
    assert config.scheduler in ['MultiStepLR',
                                'CosineLR'], 'Unsupported scheduler!'
    assert config.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
    if config.warm_up_epochs > 0:
        if config.scheduler == 'MultiStepLR':
            lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        elif config.scheduler == 'CosineLR':
            lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos(
                    (epoch - config.warm_up_epochs) /
                    (config.epochs - config.warm_up_epochs) * math.pi) + 1)
    elif config.warm_up_epochs == 0:
        if config.scheduler == 'MultiStepLR':
            lr_func = lambda epoch: config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        elif config.scheduler == 'CosineLR':
            lr_func = lambda epoch: 0.5 * (math.cos(
                (epoch - config.warm_up_epochs) /
                (config.epochs - config.warm_up_epochs) * math.pi) + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


def build_training_mode(config, model, optimizer):
    '''
    Choose model training mode:nn.DataParallel/nn.parallel.DistributedDataParallel,use apex or not
    '''
    if config.distributed:
        if config.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        if config.apex:
            amp.register_float_function(torch, 'sigmoid')
            amp.register_float_function(torch, 'softmax')
            amp.register_float_function(torchvision.ops, 'deform_conv2d')
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            model = apex.parallel.DistributedDataParallel(model,
                                                          delay_allreduce=True)
            if config.sync_bn:
                model = apex.parallel.convert_syncbn_model(model).cuda()
        else:
            local_rank = torch.distributed.get_rank()
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank)
    else:
        if config.apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model = nn.DataParallel(model)

    return model
