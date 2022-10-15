import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

import copy
import collections
import json
import numpy as np

from thop import profile
from thop import clever_format

from simpleAICV.classification import losses
from simpleAICV.classification.datasets.ilsvrc2012dataset import ILSVRC2012Dataset
from simpleAICV.classification.common import Opencv2PIL, TorchRandomResizedCrop, TorchRandomHorizontalFlip, TorchResize, TorchCenterCrop, TorchMeanStdNormalize, ClassificationCollater

from resnet import ResNet, get_resnet_config, BasicBlock, Bottleneck

import torch
import torchvision.transforms as transforms


def generate_candidate_nets_config(sample_net_config):
    candidate_nets_config_json_path = './candidate_nets_config.json'

    if os.path.exists(candidate_nets_config_json_path):
        with open(candidate_nets_config_json_path, 'r',
                  encoding='utf-8') as load_f:
            candidate_nets_config = json.load(load_f)
        return candidate_nets_config

    base_net_param_range_config = sample_net_config[
        'base_net_param_range_config']
    target_glops = sample_net_config['target_glops'] * 1e9
    target_gflops_eps = sample_net_config['target_gflops_eps']
    sample_candidate_net_num = sample_net_config['sample_candidate_net_num']
    num_classes = sample_net_config['num_classes']
    input_image_size = sample_net_config['input_image_size']
    seed = sample_net_config['seed']

    np.random.seed(seed)
    candidate_nets_config = collections.OrderedDict()
    net_num_idx = 0

    while net_num_idx < sample_candidate_net_num:
        new_net_config = {}
        for param_name, param_range in base_net_param_range_config.items():
            param_sample_range, param_sample_type = param_range[
                0], param_range[1]
            assert param_sample_type in [None, 'int', 'float',
                                         'log'], 'Illegal sample type!'
            if not param_sample_type:
                param_sample_num = param_sample_range
            elif param_sample_type == 'int':
                param_sample_num = np.random.randint(
                    param_sample_range[0], int(param_sample_range[1] + 1))
                if param_name == 'w_0':
                    param_sample_num = int(param_sample_num // 8 * 8)

            elif param_sample_type == 'float':
                param_sample_num = np.random.uniform(param_sample_range[0],
                                                     param_sample_range[1])

            elif param_sample_type == 'log':
                param_sample_num = np.random.uniform(
                    np.log(param_sample_range[0]),
                    np.log(param_sample_range[1]))
                param_sample_num = np.exp(param_sample_num)

            new_net_config[param_name] = param_sample_num

        stem_width, all_stage_width, all_stage_depth = get_resnet_config(
            new_net_config, q=8)

        if len(all_stage_width) != 4 or len(all_stage_depth) != 4:
            continue

        net = ResNet(
            **{
                'resnet_config': new_net_config,
                'block': Bottleneck,
                'num_classes': num_classes,
            })
        macs, params = profile(net,
                               inputs=(torch.randn(1, 3, input_image_size,
                                                   input_image_size), ),
                               verbose=False)
        show_macs, show_params = clever_format([macs, params], '%.3f')
        valid_model_flag = (macs <= (1. + target_gflops_eps) * target_glops
                            and macs >=
                            (1. - target_gflops_eps) * target_glops)

        print(
            f'waiting_for_checked_net_config: {new_net_config}, channels: {all_stage_width}, depths: {all_stage_depth}, macs: {show_macs}, params: {show_params}\n'
        )
        if valid_model_flag:
            candidate_nets_config[net_num_idx] = {
                'net_num_idx': net_num_idx,
                'net_config': new_net_config,
                'macs': show_macs,
                'params': show_params,
            }
            print(
                f'valid_candidate_net_idx:{net_num_idx}, net_config: {new_net_config}, channels: {all_stage_width}, depths: {all_stage_depth}, macs: {show_macs}, params: {show_params}\n'
            )
            net_num_idx += 1

    with open(candidate_nets_config_json_path, 'w',
              encoding='utf-8') as load_f:
        json.dump(candidate_nets_config, load_f)

    return candidate_nets_config


class config:
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224
    seed = 0

    sample_net_config = {
        'base_net_param_range_config': {
            # param:(Fixed param or param range,sample type None(not sample)/'int'/'log'/'float')
            'stem_width': (64, None),
            'depth': ([12, 20], 'int'),
            'w_0': ([32, 96], 'int'),
            'w_a': ([16, 96], 'log'),
            'w_m': ([1.2, 2.5], 'float'),
        },
        'target_glops': 4.1,
        'target_gflops_eps': 0.05,
        'sample_candidate_net_num': 16,
        'num_classes': num_classes,
        'input_image_size': input_image_size,
        'seed': seed,
    }

    candidate_nets_config = generate_candidate_nets_config(sample_net_config)

    train_criterion = losses.__dict__['CELoss']()
    test_criterion = losses.__dict__['CELoss']()

    train_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=input_image_size),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]))

    test_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=input_image_size * scale),
            TorchCenterCrop(resize=input_image_size),
            TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]))

    train_collater = ClassificationCollater()
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 16
    accumulation_steps = 1

    optimizer = (
        'SGD',
        {
            'lr': 0.1,
            'momentum': 0.9,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-4,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
        },
    )

    epochs = 25
    print_interval = 100

    sync_bn = False
    apex = True

    use_ema_model = False
    ema_model_decay = 0.9999