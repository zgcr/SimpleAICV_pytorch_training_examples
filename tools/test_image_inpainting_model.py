import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import collections

import torch
import torch.nn as nn

from tools.scripts import generate_inpainting_images_for_all_dataset, compute_image_inpainting_model_metric_for_all_dataset
from tools.utils import get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Image Inpainting Testing')
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

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    if local_rank == 0:
        os.makedirs(log_dir) if not os.path.exists(log_dir) else None
        os.makedirs(config.save_image_dir) if not os.path.exists(
            config.save_image_dir) else None

    torch.distributed.barrier()

    logger = get_logger('test', log_dir)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in [
                    'generator_model',
                    'fid_model',
                    'test_dataset_list',
            ]:
                log_info = f'{key}: {value}'
                logger.info(log_info) if local_rank == 0 else None

    generator_model = config.generator_model.cuda()
    generator_model = nn.parallel.DistributedDataParallel(
        generator_model, device_ids=[local_rank], output_device=local_rank)

    generate_done_flag = True
    if len(os.listdir(config.save_image_dir)) != len(
            config.test_dataset_name_list):
        generate_done_flag = False
    else:
        for per_sub_dataset_name in os.listdir(config.save_image_dir):
            if per_sub_dataset_name not in config.test_dataset_name_list:
                generate_done_flag = False
                break
            per_sub_save_image_dir = os.path.join(config.save_image_dir,
                                                  per_sub_dataset_name)
            if len(per_sub_save_image_dir) <= 0:
                generate_done_flag = False
                break

        if generate_done_flag:
            all_dataset_images_num_dict, all_dataset_images_path_dict = collections.OrderedDict(
            ), collections.OrderedDict()
            for per_sub_dataset_name in os.listdir(config.save_image_dir):
                per_sub_save_image_dir = os.path.join(config.save_image_dir,
                                                      per_sub_dataset_name)
                per_sub_test_images_path_list = []
                for per_image_name in os.listdir(per_sub_save_image_dir):
                    if '_image.jpg' in per_image_name:
                        per_image_path = os.path.join(per_sub_save_image_dir,
                                                      per_image_name)

                        per_mask_name = per_image_name.replace(
                            '_image.jpg', '_mask.jpg')
                        per_mask_path = os.path.join(per_sub_save_image_dir,
                                                     per_mask_name)

                        per_composition_image_name = per_image_name.replace(
                            '_image.jpg', '_composition.jpg')
                        per_composition_image_path = os.path.join(
                            per_sub_save_image_dir, per_composition_image_name)

                        assert os.path.exists(
                            per_image_path) and os.path.exists(
                                per_mask_path) and os.path.exists(
                                    per_composition_image_path)

                        per_sub_test_images_path_list.append([
                            per_image_path, per_mask_path,
                            per_composition_image_path
                        ])

                per_sub_test_image_num = len(per_sub_test_images_path_list)

                all_dataset_images_num_dict[
                    per_sub_dataset_name] = per_sub_test_image_num
                all_dataset_images_path_dict[
                    per_sub_dataset_name] = per_sub_test_images_path_list

    if not generate_done_flag:
        all_dataset_images_num_dict, all_dataset_images_path_dict = generate_inpainting_images_for_all_dataset(
            generator_model, config)

    torch.cuda.empty_cache()

    log_info = f'test_images_num:\n'
    for per_sub_dataset_name, per_sub_dataset_images_num in all_dataset_images_num_dict.items(
    ):
        log_info += f'{per_sub_dataset_name} images num: {per_sub_dataset_images_num}\n'
    logger.info(log_info) if local_rank == 0 else None

    fid_model = config.fid_model.cuda()
    fid_model = nn.parallel.DistributedDataParallel(fid_model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)

    result_dict = compute_image_inpainting_model_metric_for_all_dataset(
        all_dataset_images_path_dict, fid_model, config)

    torch.cuda.empty_cache()

    log_info = f'test_result:\n'
    for per_sub_dataset_name, per_sub_dataset_result_dict in result_dict.items(
    ):
        log_info += f'{per_sub_dataset_name}\n'
        for per_metric, per_metric_value in per_sub_dataset_result_dict.items(
        ):
            log_info += f'{per_metric}: {per_metric_value}\n'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == '__main__':
    main()
