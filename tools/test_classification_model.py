import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.scripts import validate_classification
from tools.utils import get_logger, set_seed, compute_flops_and_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Model Testing')
    parser.add_argument('--work-dir',
                        type=str,
                        help='path for get testing config')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='LOCAL_PROCESS_RANK in DistributedDataParallel model')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from test_config import config
    log_dir = os.path.join(args.work_dir, 'log')

    set_seed(config.seed)

    local_rank = args.local_rank
    # start init process
    if config.distributed:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        torch.cuda.set_device(local_rank)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        config.val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(config.val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=False,
                            num_workers=config.num_workers,
                            sampler=val_sampler)

    if (config.distributed and local_rank == 0) or not config.distributed:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    global logger
    logger = get_logger('test', log_dir)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in ['model', 'criterion']:
                log_info = f'{key}: {value}'
                logger.info(log_info) if (
                    config.distributed
                    and local_rank == 0) or not config.distributed else None

    gpus_type, gpus_num = torch.cuda.get_device_name(
    ), torch.cuda.device_count()
    log_info = f'gpus_type: {gpus_type}, gpus_num: {gpus_num}'
    logger.info(log_info) if (config.distributed and local_rank
                              == 0) or not config.distributed else None

    model = config.model
    criterion = config.criterion

    if config.trained_model_path:
        saved_model = torch.load(os.path.join(BASE_DIR,
                                              config.trained_model_path),
                                 map_location=torch.device('cpu'))
        model.load_state_dict(saved_model)

    flops, params = compute_flops_and_params(config, model)
    log_info = f'model: {config.network}, flops: {flops}, params: {params}'
    logger.info(log_info) if (config.distributed and local_rank
                              == 0) or not config.distributed else None

    model = model.cuda()
    criterion = criterion.cuda()
    if config.distributed:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
    else:
        model = nn.DataParallel(model)

    top1, top5, loss, per_image_load_time, per_image_inference_time = validate_classification(
        val_loader, model, criterion, config)
    log_info = f'top1: {top1:.3f}%, top5: {top5:.3f}%, loss: {loss:.4f}, per_image_load_time: {per_image_load_time:.3f}ms, per_image_inference_time: {per_image_inference_time:.3f}ms'
    logger.info(log_info) if (config.distributed and local_rank
                              == 0) or not config.distributed else None

    return


if __name__ == '__main__':
    main()
