'''
Emerging Properties in Self-Supervised Vision Transformers
https://github.com/facebookresearch/dino
'''
import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import math
import time

import torch
from torch.utils.data import DataLoader

from tools.scripts import train_dino_self_supervised_learning
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         build_optimizer, build_training_mode)


class Scheduler:

    def __init__(self, config, scheduler_name='lr'):
        assert scheduler_name in ['lr', 'weight_decay', 'momentum_teacher']
        self.scheduler_name = scheduler_name
        if self.scheduler_name == 'lr':
            self.scheduler_policy_name = config.lr_scheduler[0]
            self.scheduler_parameters = config.lr_scheduler[1]
            self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
            self.epochs = config.epochs
            self.optimizer_parameters = config.optimizer[1]
            self.value = self.optimizer_parameters['lr']
            self.current_value = self.value

        elif self.scheduler_name == 'weight_decay':
            self.scheduler_policy_name = config.weight_decay_scheduler[0]
            self.scheduler_parameters = config.weight_decay_scheduler[1]
            self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
            self.epochs = config.epochs
            self.optimizer_parameters = config.optimizer[1]
            self.value = self.optimizer_parameters['weight_decay']
            self.current_value = self.value

        elif self.scheduler_name == 'momentum_teacher':
            self.scheduler_policy_name = config.momentum_teacher_scheduler[0]
            self.scheduler_parameters = config.momentum_teacher_scheduler[1]
            self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
            self.epochs = config.epochs
            self.value = self.scheduler_parameters['momentum']
            self.current_value = self.value

        assert self.scheduler_policy_name in ['MultiStep', 'Cosine',
                                              'Poly'], 'Unsupported scheduler!'
        assert self.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
        assert self.epochs > 0, 'Illegal epochs!'

    def step(self, optimizer, epoch):
        if self.scheduler_policy_name == 'MultiStep':
            gamma = self.scheduler_parameters['gamma']
            milestones = self.scheduler_parameters['milestones']
        elif self.scheduler_policy_name == 'Cosine':
            final_value = 0. if 'final_value' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['final_value']
        elif self.scheduler_policy_name == 'Poly':
            power = self.scheduler_parameters['power']
            final_value = 0. if 'final_value' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['final_value']

        if self.scheduler_policy_name == 'MultiStep':
            self.current_value = (
                epoch
            ) / self.warm_up_epochs * self.value if epoch < self.warm_up_epochs else gamma**len(
                [m for m in milestones if m <= epoch]) * self.value
        elif self.scheduler_policy_name == 'Cosine':
            self.current_value = (
                epoch
            ) / self.warm_up_epochs * self.value if epoch < self.warm_up_epochs else 0.5 * (
                math.cos((epoch - self.warm_up_epochs) /
                         (self.epochs - self.warm_up_epochs) * math.pi) +
                1) * (self.value - final_value) + final_value
        elif self.scheduler_policy_name == 'Poly':
            self.current_value = (
                epoch
            ) / self.warm_up_epochs * self.value if epoch < self.warm_up_epochs else (
                (1 - (epoch - self.warm_up_epochs) /
                 (self.epochs - self.warm_up_epochs))**
                power) * (self.value - final_value) + final_value

        if self.scheduler_name == 'lr':
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group[
                        "lr"] = self.current_value * param_group["lr_scale"]
                else:
                    param_group["lr"] = self.current_value
        elif self.scheduler_name == 'weight_decay':
            for param_group in optimizer.param_groups:
                if param_group["weight_decay"] == 0:
                    continue
                else:
                    param_group["weight_decay"] = self.current_value

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


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
            if key not in [
                    'teacher_model',
                    'student_model',
            ]:
                log_info = f'{key}: {value}'
                logger.info(log_info) if local_rank == 0 else None

    teacher_model = config.teacher_model.cuda()
    student_model = config.student_model.cuda()
    train_criterion = config.train_criterion.cuda()

    teacher_optimizer, _ = build_optimizer(config, teacher_model)

    student_optimizer, student_model_layer_weight_decay_list = build_optimizer(
        config, student_model)

    for i, per_layer_list in enumerate(student_model_layer_weight_decay_list):
        if i == 0:
            log_info = f'--------student no weight decay layers--------'
        elif i == 1:
            log_info = f'--------student weight decay layers--------'
        logger.info(log_info) if local_rank == 0 else None

        layer_name_list, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'student model. name: {name}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(log_info) if local_rank == 0 else None

    lr_scheduler = Scheduler(config, scheduler_name='lr')
    weight_decay_scheduler = Scheduler(config, scheduler_name='weight_decay')
    momentum_teacher_scheduler = Scheduler(config,
                                           scheduler_name='momentum_teacher')
    teacher_model, _ = build_training_mode(config, teacher_model,
                                           teacher_optimizer)
    student_model, _ = build_training_mode(config, student_model,
                                           student_optimizer)

    for param in teacher_model.parameters():
        param.requires_grad = False

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'------------teacher-parameters------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, param in teacher_model.named_parameters():
        log_info = f'teacher model. name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    log_info = f'------------teacher-buffers------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, buffer in teacher_model.named_buffers():
        log_info = f'teacher model. name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'------------student-parameters------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, param in student_model.named_parameters():
        log_info = f'student model. name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    log_info = f'------------student-buffers------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, buffer in student_model.named_buffers():
        log_info = f'student model. name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        teacher_model.load_state_dict(checkpoint['teacher_model_state_dict'])
        student_model.load_state_dict(checkpoint['student_model_state_dict'])
        student_optimizer.load_state_dict(
            checkpoint['student_optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        weight_decay_scheduler.load_state_dict(
            checkpoint['weight_decay_scheduler_state_dict'])
        momentum_teacher_scheduler.load_state_dict(
            checkpoint['momentum_teacher_scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time

        best_loss, train_loss, lr, weight_decay, momentum_teacher = checkpoint[
            'best_loss'], checkpoint['train_loss'], checkpoint[
                'lr'], checkpoint['weight_decay'], checkpoint[
                    'momentum_teacher']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, lr: {lr:.6f}, weight_decay: {weight_decay:.6f}, momentum_teacher: {momentum_teacher:.6f}'
        logger.info(log_info) if local_rank == 0 else None

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {lr_scheduler.current_value:.6f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        train_loss = train_dino_self_supervised_learning(
            train_loader, teacher_model, student_model, train_criterion,
            student_optimizer, lr_scheduler, weight_decay_scheduler,
            momentum_teacher_scheduler, epoch, logger, config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        if local_rank == 0:
            # save best acc1 model and each epoch checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(teacher_model.module.state_dict(),
                           os.path.join(checkpoint_dir, 'best_teacher.pth'))
                torch.save(student_model.module.state_dict(),
                           os.path.join(checkpoint_dir, 'best_student.pth'))

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
                    lr_scheduler.current_value,
                    'weight_decay':
                    weight_decay_scheduler.current_value,
                    'momentum_teacher':
                    momentum_teacher_scheduler.current_value,
                    'teacher_model_state_dict':
                    teacher_model.state_dict(),
                    'student_model_state_dict':
                    student_model.state_dict(),
                    'student_optimizer_state_dict':
                    student_optimizer.state_dict(),
                    'lr_scheduler_state_dict':
                    lr_scheduler.state_dict(),
                    'weight_decay_scheduler_state_dict':
                    weight_decay_scheduler.state_dict(),
                    'momentum_teacher_scheduler_state_dict':
                    momentum_teacher_scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

    if local_rank == 0:
        if os.path.exists(os.path.join(checkpoint_dir, 'best_teacher.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_teacher.pth'),
                os.path.join(
                    checkpoint_dir,
                    f'{config.network}-teacher-loss{best_loss:.3f}.pth'))
        if os.path.exists(os.path.join(checkpoint_dir, 'best_student.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best_student.pth'),
                os.path.join(
                    checkpoint_dir,
                    f'{config.network}-student-loss{best_loss:.3f}.pth'))

    log_info = f'train done. model: {config.network}, train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == '__main__':
    main()
