import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import collections
import numpy as np
import os
import random
import csv

from tqdm import tqdm
from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader


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


def test_classification(test_loader, model, config):
    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    # switch to evaluate mode
    model.eval()

    test_results = collections.OrderedDict()
    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            paths, images = data['path'], data['image']
            if model_on_cuda:
                images = images.cuda()

            torch.cuda.synchronize()

            outputs = model(images)
            torch.cuda.synchronize()

            _, topk_indexes = torch.topk(outputs,
                                         k=1,
                                         dim=1,
                                         largest=True,
                                         sorted=True)
            topk_indexes = torch.squeeze(topk_indexes, dim=-1)

            for per_image_path, per_image_pred_index in zip(
                    paths, topk_indexes):
                image_name = per_image_path.split('/')[-1]
                written_index = f'{per_image_pred_index:0>4d}'
                test_results[image_name] = written_index

    return test_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Testing')
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
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    torch.distributed.barrier()

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

    model = config.model

    macs, params = compute_macs_and_params(config, model)
    print(f'model: {config.network}, macs: {macs}, params: {params}')

    model = model.cuda()

    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

    test_results = test_classification(test_loader, model, config)

    if local_rank == 0:
        with open(f"{config.set_name}_pred_results.csv", "w",
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for per_image_name, per_image_pred in test_results.items():
                writer.writerow([str(per_image_name), str(per_image_pred)])

    return


if __name__ == '__main__':
    main()
