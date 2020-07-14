import os
import random
import shutil
import argparse
import time
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from apex import amp
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config
from public.distillation import losses
from public.distillation.models.resnet import ChannelDistillResNet1834
from public.distillation.utils import AverageMeter, DataPrefetcher, get_logger, accuracy, adjust_loss_alpha


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet distillation Training')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=Config.momentum,
                        help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='batch size')
    parser.add_argument('--milestones',
                        type=list,
                        default=Config.milestones,
                        help='optimizer milestones')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=Config.accumulation_steps,
                        help='gradient accumulation steps')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=Config.input_image_size,
                        help='input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')

    return parser.parse_args()


def train(train_loader, net, criterion, optimizer, scheduler, epoch, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_total = AverageMeter()

    loss_ams = [AverageMeter()] * len(criterion)
    loss_alphas = []
    for loss_item in Config.loss_list:
        loss_rate = loss_item["loss_rate"]
        factor = loss_item["factor"]
        loss_type = loss_item["loss_type"]
        loss_rate_decay = loss_item["loss_rate_decay"]
        loss_alphas.append(
            adjust_loss_alpha(loss_rate, epoch, factor, loss_type,
                              loss_rate_decay))

    # switch to train mode
    net.train()

    iters = len(train_loader.dataset) // args.batch_size
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1
    while inputs is not None:
        inputs, labels = inputs.float().cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        stu_outputs, tea_outputs = net(inputs)
        loss = 0
        loss_detail = []
        for i, loss_item in enumerate(Config.loss_list):
            loss_type = loss_item["loss_type"]
            if loss_type == "ce_family":
                tmp_loss = loss_alphas[i] * criterion[i](stu_outputs[-1],
                                                         labels)
            elif loss_type == "kd_family":
                tmp_loss = loss_alphas[i] * criterion[i](stu_outputs[-1],
                                                         tea_outputs[-1])
            elif loss_type == "gkd_family":
                tmp_loss = loss_alphas[i] * criterion[i](
                    stu_outputs[-1], tea_outputs[-1], labels)
            elif loss_type == "fd_family":
                tmp_loss = loss_alphas[i] * criterion[i](stu_outputs[:-1],
                                                         tea_outputs[:-1])

            loss_detail.append(tmp_loss.item())
            loss_ams[i].update(tmp_loss.item(), inputs.size(0))
            loss += tmp_loss

        loss = loss / args.accumulation_steps

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if iter_index % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        acc1, acc5 = accuracy(stu_outputs[-1], labels, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        loss_total.update(loss.item(), inputs.size(0))

        inputs, labels = prefetcher.next()

        loss_log = ""
        if iter_index % args.print_interval == 0:
            loss_log += f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%, loss_total: {loss.item():.2f}, "

            for i, loss_item in enumerate(Config.loss_list):
                loss_name = loss_item["loss_name"]
                loss_log += f"{loss_name}: {loss_detail[i]:2f}, alpha: {loss_alphas[i]:2f}, "

            logger.info(loss_log)

        iter_index += 1

    scheduler.step()

    return top1.avg, top5.avg, loss_total.avg


def validate(val_loader, net):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()

    prefetcher = DataPrefetcher(val_loader)
    inputs, labels = prefetcher.next()
    with torch.no_grad():
        while inputs is not None:
            inputs = inputs.float().cuda()
            labels = labels.cuda()

            stu_outputs, _ = net(inputs)

            acc1, acc5 = accuracy(stu_outputs[-1], labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            inputs, labels = prefetcher.next()

    return top1.avg, top5.avg


def main(logger, args):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    config = {
        key: value
        for key, value in Config.__dict__.items() if not key.startswith("__")
    }
    logger.info(f"args: {config}")

    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(Config.val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    logger.info('finish loading data')

    # network
    net = ChannelDistillResNet1834(args.num_classes)
    net = net.cuda()

    # loss and optimizer
    criterion = []
    for loss_item in Config.loss_list:
        loss_name = loss_item["loss_name"]
        loss_type = loss_item["loss_type"]
        if "kd" in loss_type:
            criterion.append(losses.__dict__[loss_name](loss_item["T"]).cuda())
        else:
            criterion.append(losses.__dict__[loss_name]().cuda())

    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)

    if args.apex:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    net = nn.DataParallel(net)

    # only evaluate
    if args.evaluate:
        # load best model
        if not os.path.isfile(args.evaluate):
            raise Exception(
                f"{args.evaluate} is not a file, please check it again")
        logger.info("start evaluating")
        logger.info(f"start resuming model from {args.evaluate}")
        checkpoint = torch.load(args.evaluate,
                                map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["model_state_dict"])
        prec1, prec5 = validate(val_loader, net)
        logger.info(
            f"epoch {checkpoint['epoch']:0>3d}, top1 acc: {prec1:.2f}%, top5 acc: {prec5:.2f}%"
        )
        return

    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
        start_epoch += checkpoint["epoch"]
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(
            f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, lr: {checkpoint['lr']:.6f}, "
            f"top1_acc: {checkpoint['acc']}%, loss {checkpoint['loss']}%")

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info("start training")
    best_acc = 0.
    for epoch in range(start_epoch, args.epochs + 1):
        prec1, prec5, loss = train(train_loader, net, criterion, optimizer,
                                   scheduler, epoch, logger)
        logger.info(
            f"train: epoch {epoch:0>3d}, top1 acc: {prec1:.2f}%, top5 acc: {prec5:.2f}%"
        )

        prec1, prec5 = validate(val_loader, net)
        logger.info(
            f"val: epoch {epoch:0>3d}, top1 acc: {prec1:.2f}%, top5 acc: {prec5:.2f}%"
        )

        # remember best prec@1 and save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "acc": prec1,
                "loss": loss,
                "lr": scheduler.get_lr()[0],
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, os.path.join(args.checkpoints, "latest.pth"))
        if prec1 > best_acc:
            shutil.copyfile(os.path.join(args.checkpoints, "latest.pth"),
                            os.path.join(args.checkpoints, "best.pth"))
            best_acc = prec1

    logger.info(f"finish training, best_model_acc: {best_acc:.4f}")
    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")


if __name__ == "__main__":
    args = parse_args()
    logger = get_logger(__name__, args.log)
    main(logger, args)
