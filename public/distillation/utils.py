import os
import torch
import logging
from logging.handlers import TimedRotatingFileHandler


def get_logger(name, log_dir='log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    error_handler = TimedRotatingFileHandler(error_name,
                                             when='D',
                                             encoding='utf-8')
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_loss_alpha(alpha,
                      epoch,
                      factor=0.9,
                      loss_type="ce_family",
                      loss_rate_decay="lrdv1"):
    """动态调整蒸馏的比例

    loss_type: 损失函数的类型
        "ce_family": loss输入为student的pred以及label
        "kd_family": loss输入为student的pred、teacher的pred
        "gkd_family": loss输入为student的pred、teacher的pred以及label
        "fd_family": loss输入为student的feature、teacher的feature
    loss_rate_decay: 衰减策略
        "lrdv1": 一开始就有ce或者kd
        "lrdv2": 前30epoch没有ce或者kd
    """
    if loss_rate_decay not in [
            "lrdv0", "lrdv1", "lrdv2", "lrdv3", "lrdv4", "lrdv5"
    ]:
        raise Exception("loss_rate_decay error")

    if loss_type not in ["ce_family", "kd_family", "gkd_family", "fd_family"]:
        raise Exception("loss type error")
    if loss_rate_decay == "lrdv0":
        return alpha

    elif loss_rate_decay == "lrdv1":
        return alpha * (factor**(epoch // 30))
    elif loss_rate_decay == "lrdv2":
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 30 else alpha * (factor**(epoch // 30))
        else:
            return alpha * (factor**(epoch // 30))
    elif loss_rate_decay == "lrdv3":
        if epoch >= 160:
            exponent = 2
        elif epoch >= 60:
            exponent = 1
        else:
            exponent = 0
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 60 else alpha * (factor**exponent)
        else:
            return alpha * (factor**exponent)
    elif loss_rate_decay == "lrdv5":
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 60 else alpha
        else:
            if epoch >= 160:
                return alpha * (factor**3)
            elif epoch >= 120:
                return alpha * (factor**2)
            elif epoch >= 60:
                return alpha * (factor**1)
            else:
                return alpha
