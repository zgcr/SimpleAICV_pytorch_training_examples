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


def build_optimizer(config, model):
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW'], 'Unsupported optimizer!'

    if optimizer_name == 'SGD':
        lr = optimizer_parameters['lr']
        momentum = optimizer_parameters['momentum']
        weight_decay = optimizer_parameters['weight_decay']
        return torch.optim.SGD(model.parameters(),
                               lr=lr,
                               momentum=momentum,
                               weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        lr = optimizer_parameters['lr']
        weight_decay = optimizer_parameters['weight_decay']
        return torch.optim.AdamW(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)


def build_scheduler(config, optimizer):
    '''
    The value of config.warm_up_epochs is zero or an integer larger than 0
    '''
    scheduler_name = config.scheduler[0]
    scheduler_parameters = config.scheduler[1]
    warm_up_epochs = scheduler_parameters['warm_up_epochs']
    epochs = config.epochs
    assert scheduler_name in ['MultiStepLR',
                              'CosineLR'], 'Unsupported scheduler!'
    assert warm_up_epochs >= 0, 'Illegal warm_up_epochs!'

    if scheduler_name == 'MultiStepLR':
        gamma = scheduler_parameters['gamma']
        milestones = scheduler_parameters['milestones']

    if warm_up_epochs > 0:
        if scheduler_name == 'MultiStepLR':
            lr_func = lambda epoch: (
                (epoch + 1
                 )) / warm_up_epochs if epoch < warm_up_epochs else gamma**len(
                     [m for m in milestones if m <= epoch])
        elif scheduler_name == 'CosineLR':
            lr_func = lambda epoch: (
                epoch + 1
            ) / warm_up_epochs if epoch < warm_up_epochs else 0.5 * (math.cos(
                (epoch - warm_up_epochs) /
                (epochs - warm_up_epochs) * math.pi) + 1)

    elif warm_up_epochs == 0:
        if scheduler_name == 'MultiStepLR':
            lr_func = lambda epoch: gamma**len(
                [m for m in milestones if m <= epoch])
        elif scheduler_name == 'CosineLR':
            lr_func = lambda epoch: 0.5 * (math.cos(
                (epoch - warm_up_epochs) /
                (epochs - warm_up_epochs) * math.pi) + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


def build_training_mode(config, model, optimizer):
    '''
    Choose model training mode:nn.DataParallel/nn.parallel.DistributedDataParallel,use apex or not
    apex only used in mode:nn.parallel.DistributedDataParallel
    '''
    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if config.apex:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch.nn, 'Sigmoid')
        amp.register_float_function(torch.nn, 'Softmax')
        amp.register_float_function(torch.nn.functional, 'sigmoid')
        amp.register_float_function(torch.nn.functional, 'softmax')
        amp.register_float_function(torchvision.ops, 'deform_conv2d')

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = apex.parallel.DistributedDataParallel(model,
                                                      delay_allreduce=True)
        if config.sync_bn:
            model = apex.parallel.convert_syncbn_model(model).cuda()
    else:
        local_rank = torch.distributed.get_rank()
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)

    return model