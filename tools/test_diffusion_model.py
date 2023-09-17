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

import torchvision.transforms as transforms

from simpleAICV.diffusion_model.metrics.compute_fid_is_score import ImagePathDataset

from tools.scripts import generate_diffusion_model_images, compute_diffusion_model_metric
from tools.utils import get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Diffusion Model Testing')
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

    if not config.save_image_dir:
        config.save_image_dir = os.path.join(args.work_dir, 'images')
    else:
        config.save_image_dir = config.save_image_dir

    config.save_test_image_dir = os.path.join(config.save_image_dir,
                                              'test_images')
    config.save_generate_image_dir = os.path.join(config.save_image_dir,
                                                  'generate_images')

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    if local_rank == 0:
        os.makedirs(log_dir) if not os.path.exists(log_dir) else None
        os.makedirs(config.save_image_dir) if not os.path.exists(
            config.save_image_dir) else None
        os.makedirs(config.save_generate_image_dir) if not os.path.exists(
            config.save_generate_image_dir) else None
        os.makedirs(config.save_test_image_dir) if not os.path.exists(
            config.save_test_image_dir) else None

    torch.distributed.barrier()

    logger = get_logger('test', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        config.test_dataset, shuffle=False)
    test_loader = DataLoader(config.test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=config.test_collater,
                             sampler=test_sampler)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in ['model', 'fid_model']:
                log_info = f'{key}: {value}'
                logger.info(log_info) if local_rank == 0 else None

    model = config.model
    sampler = config.sampler

    model = model.cuda()
    sampler = sampler.cuda()

    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

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

    test_image_num, generate_image_num = len(test_images_path_list), len(
        generate_images_path_list)

    if len(test_images_path_list) != len(
            config.test_dataset) or len(generate_images_path_list) != len(
                config.test_dataset):
        test_images_path_list, generate_images_path_list, test_image_num, generate_image_num = generate_diffusion_model_images(
            test_loader, model, sampler, config)

        torch.cuda.empty_cache()
    else:
        log_info = f'skip generate images!!!'
        logger.info(log_info) if local_rank == 0 else None

    test_images_dataset = ImagePathDataset(test_images_path_list,
                                           transform=transforms.Compose([
                                               transforms.Resize([
                                                   config.input_image_size,
                                                   config.input_image_size
                                               ]),
                                               transforms.ToTensor(),
                                           ]))

    generate_images_dataset = ImagePathDataset(generate_images_path_list,
                                               transform=transforms.Compose([
                                                   transforms.Resize([
                                                       config.input_image_size,
                                                       config.input_image_size
                                                   ]),
                                                   transforms.ToTensor(),
                                               ]))

    assert config.fid_model_batch_size % config.gpus_num == 0, 'config.fid_model_batch_size is not divisible by config.gpus_num!'
    assert config.fid_model_num_workers % config.gpus_num == 0, 'config.fid_model_num_workers is not divisible by config.gpus_num!'
    fid_model_batch_size = int(config.fid_model_batch_size // config.gpus_num)
    fid_model_num_workers = int(config.fid_model_num_workers //
                                config.gpus_num)

    test_images_dataloader = DataLoader(test_images_dataset,
                                        batch_size=fid_model_batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=fid_model_num_workers)
    generate_images_dataloader = DataLoader(generate_images_dataset,
                                            batch_size=fid_model_batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=fid_model_num_workers)

    fid_model = config.fid_model

    fid_model = fid_model.cuda()

    fid_model = nn.parallel.DistributedDataParallel(fid_model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)

    fid_value, is_score_mean, is_score_std = compute_diffusion_model_metric(
        test_images_dataloader, generate_images_dataloader, test_image_num,
        generate_image_num, fid_model, config)

    torch.cuda.empty_cache()

    log_info = f'fid: {fid_value:.3f}, is_score: {is_score_mean:.3f}/{is_score_std:.3f}, test_image_num: {test_image_num}, generate_image_num: {generate_image_num}'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == '__main__':
    main()
