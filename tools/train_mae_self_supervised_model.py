import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import time

import torch
from torch.utils.data import DataLoader

from tools.scripts import train_mae_self_supervised_learning
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         build_optimizer, Scheduler, build_training_mode)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Self Supervised Learning Training')
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
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
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

    for i, per_layer_list in enumerate(model_layer_weight_decay_list):
        if i == 0:
            log_info = f'-----------no weight decay layers--------------'
        elif i == 1:
            log_info = f'-------------weight decay layers---------------'
        logger.info(log_info) if local_rank == 0 else None

        layer_name_list, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(log_info) if local_rank == 0 else None

    scheduler = Scheduler(config)
    model, config.ema_model = build_training_mode(config, model, optimizer)

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

        if 'ema_model_state_dict' in checkpoint.keys():
            config.ema_model.ema_model.load_state_dict(
                checkpoint['ema_model_state_dict'])

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        train_loss = train_mae_self_supervised_learning(
            train_loader, model, train_criterion, optimizer, scheduler, epoch,
            logger, config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        if local_rank == 0:
            # save best acc1 model and each epoch checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                if config.use_ema_model:
                    torch.save(config.ema_model.ema_model.module.state_dict(),
                               os.path.join(checkpoint_dir, 'best.pth'))
                    torch.save(
                        config.ema_model.ema_model.module.encoder.state_dict(),
                        os.path.join(checkpoint_dir, 'best_encoder.pth'))
                else:
                    torch.save(model.module.state_dict(),
                               os.path.join(checkpoint_dir, 'best.pth'))
                    torch.save(
                        model.module.encoder.state_dict(),
                        os.path.join(checkpoint_dir, 'best_encoder.pth'))

            if config.use_ema_model:
                torch.save(
                    {
                        'epoch':
                        epoch,
                        'time':
                        train_time,
                        'best_loss':
                        best_loss,
                        'train_loss':
                        train_loss,
                        'lr':
                        scheduler.current_lr,
                        'model_state_dict':
                        model.state_dict(),
                        'optimizer_state_dict':
                        optimizer.state_dict(),
                        'ema_model_state_dict':
                        config.ema_model.ema_model.state_dict(),
                        'scheduler_state_dict':
                        scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'latest.pth'))
            else:
                torch.save(
                    {
                        'epoch': epoch,
                        'time': train_time,
                        'best_loss': best_loss,
                        'train_loss': train_loss,
                        'lr': scheduler.current_lr,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(checkpoint_dir, 'latest.pth'))

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

    if local_rank == 0:
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir,
                             f'{config.network}-loss{best_loss:.3f}.pth'))
        if os.path.exists(os.path.join(checkpoint_dir, 'best_encoder.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_encoder.pth'),
                os.path.join(
                    checkpoint_dir,
                    f'{config.network}-loss{best_loss:.3f}_encoder.pth'))

    log_info = f'train done. model: {config.network}, train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == '__main__':
    main()
