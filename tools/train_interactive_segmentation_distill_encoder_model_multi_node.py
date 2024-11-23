import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import re
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from tools.interactive_segmentation_scripts import train_distill_sam_encoder
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         build_optimizer, Scheduler, EmaModel)


def build_training_mode(config, model):
    ema_model, scaler = None, None
    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    local_rank = config.local_rank
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank,
                                                find_unused_parameters=True)

    if hasattr(config, 'use_amp') and config.use_amp:
        scaler = GradScaler()

    return model, ema_model, scaler


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Interactive Segmentation Distill Encoder Training'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank

    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

    # total_rank多机多卡判断
    total_rank = torch.distributed.get_rank()
    config.total_rank = total_rank

    # 获取多机多卡训练时所有节点上的卡总数
    per_node_process_nums = int(os.environ['LOCAL_WORLD_SIZE'])
    per_node_gpus_num = torch.cuda.device_count()
    per_node_per_process_gpus_num = int(per_node_gpus_num /
                                        per_node_process_nums)

    world_size = torch.distributed.get_world_size()
    config.gpus_num = int(per_node_per_process_gpus_num * world_size)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    if local_rank == 0:
        os.makedirs(
            checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
        os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    torch.distributed.barrier()

    logger = get_logger('train', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=config.seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        config.train_dataset, shuffle=True)
    train_loader = DataLoader(config.train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=config.train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in ['model']:
                log_info = f'{key}: {value}'
                logger.info(log_info) if local_rank == 0 else None

    model = config.model.cuda()
    train_criterion = config.train_criterion.cuda()

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'--------------------parameters--------------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, param in model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    log_info = f'--------------------buffers--------------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, buffer in model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    optimizer, model_layer_weight_decay_list = build_optimizer(config, model)

    log_info = f'-------------layers weight decay---------------'
    logger.info(log_info) if local_rank == 0 else None
    for per_layer_list in model_layer_weight_decay_list:
        layer_name_list, layer_lr, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['lr'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, lr: {layer_lr}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(log_info) if local_rank == 0 else None

    scheduler = Scheduler(config, optimizer)
    model, _, config.scaler = build_training_mode(config, model)

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time

        best_loss, train_loss, lr = checkpoint['best_loss'], checkpoint[
            'train_loss'], checkpoint['lr']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, lr: {lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

    # use torch 2.0 compile function
    config.compile_support = False
    log_info = f'using torch version:{torch.__version__}'
    logger.info(log_info) if local_rank == 0 else None
    if re.match(r'2.\d*.\d*', torch.__version__):
        config.compile_support = True
        log_info = f'this torch version support torch.compile function.'
        logger.info(log_info) if local_rank == 0 else None
    elif re.match(r'1.\d*.\d*', torch.__version__):
        log_info = f'this torch version unsupport torch.compile function.'
        logger.info(log_info) if local_rank == 0 else None
    else:
        log_info = f'unsupport torch version:{torch.__version__}'
        logger.info(log_info) if local_rank == 0 else None
        return

    config.use_compile = (config.compile_support and config.use_compile)
    if config.use_compile:
        # _orig_mod
        model = torch.compile(model, **config.compile_params)

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        train_loss = train_distill_sam_encoder(train_loader, model,
                                               train_criterion, optimizer,
                                               scheduler, epoch, logger,
                                               config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        if epoch % config.save_interval == 0:
            if local_rank == 0:
                if config.use_compile:
                    save_student_model = model._orig_mod.module.student.state_dict(
                    )
                else:
                    save_student_model = model.module.student.state_dict()
                torch.save(
                    save_student_model,
                    os.path.join(checkpoint_dir,
                                 f'student_model_epoch_{epoch}.pth'))

        if local_rank == 0:
            # save best acc1 model and each epoch checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                if config.use_compile:
                    save_best_student_model = model._orig_mod.module.student.state_dict(
                    )
                else:
                    save_best_student_model = model.module.student.state_dict()

                torch.save(save_best_student_model,
                           os.path.join(checkpoint_dir, 'best_student.pth'))

            if config.use_compile:
                save_checkpoint_model = model._orig_mod.state_dict()
            else:
                save_checkpoint_model = model.state_dict()

            torch.save(
                {
                    'epoch': epoch,
                    'time': train_time,
                    'best_loss': best_loss,
                    'train_loss': train_loss,
                    'lr': scheduler.current_lr,
                    'model_state_dict': save_checkpoint_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

    if local_rank == 0:
        if os.path.exists(os.path.join(checkpoint_dir, 'best_student.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_student.pth'),
                os.path.join(checkpoint_dir,
                             f'student-epoch{epoch}-loss{best_loss:.3f}.pth'))

    log_info = f'train done. train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == '__main__':
    main()
