import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.scripts import test_detection
from tools.utils import get_logger, set_seed, compute_macs_and_params


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Detection Testing')
    parser.add_argument('--work-dir',
                        type=str,
                        help='path for get testing config')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from test_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    if local_rank == 0:
        os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    torch.distributed.barrier()

    logger = get_logger('test', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    test_loader = DataLoader(config.test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=config.test_collater)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in ['model']:
                log_info = f'{key}: {value}'
                logger.info(log_info) if local_rank == 0 else None

    model = config.model
    test_criterion = config.test_criterion
    decoder = config.decoder

    macs, params = compute_macs_and_params(config, model)
    log_info = f'model: {config.network}, macs: {macs}, params: {params}'
    logger.info(log_info) if local_rank == 0 else None

    model = model.cuda()
    test_criterion = test_criterion.cuda()

    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

    result_dict = test_detection(test_loader, model, test_criterion, decoder,
                                 config)
    log_info = f'eval type: {config.eval_type}\n'
    for key, value in result_dict.items():
        log_info += f'{key}: {value}\n'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == '__main__':
    main()
