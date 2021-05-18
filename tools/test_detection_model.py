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

from simpleAICV.detection.common import DetectionCollater

from tools.scripts import validate_detection
from tools.utils import get_logger, set_seed, compute_flops_and_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Detection Model Testing')
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

    set_seed(config.seed)

    collater = DetectionCollater()
    val_loader = DataLoader(config.val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            collate_fn=collater.next)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = get_logger('test', log_dir)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in [
                    'model', 'criterion', 'decoder', 'train_dataset',
                    'val_dataset'
            ]:
                log_info = f'{key}: {value}'
                logger.info(log_info)

    gpus_type, gpus_num = torch.cuda.get_device_name(
    ), torch.cuda.device_count()
    log_info = f'gpus_type: {gpus_type}, gpus_num: {gpus_num}'
    logger.info(log_info)

    model = config.model
    decoder = config.decoder

    if config.trained_model_path:
        saved_model = torch.load(os.path.join(BASE_DIR,
                                              config.trained_model_path),
                                 map_location=torch.device('cpu'))
        model.load_state_dict(saved_model)

    flops, params = compute_flops_and_params(config, model)
    log_info = f'model: {config.network}, flops: {flops}, params: {params}'
    logger.info(log_info)

    model = model.cuda()
    decoder = decoder.cuda()
    model = nn.DataParallel(model)

    result_dict = validate_detection(config.val_dataset, val_loader, model,
                                     decoder, config)
    log_info = f'eval_result: '
    if result_dict:
        for key, value in result_dict.items():
            log_info += f'{key}: {value} ,'
    else:
        log_info += f', no target detected in testset images!'
    logger.info(log_info)

    return


if __name__ == '__main__':
    main()
