import sys
import os
import argparse
import random
import shutil
import time
import warnings
import json

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import numpy as np
from thop import profile
from thop import clever_format
from tqdm import tqdm
import apex
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from config import Config
from public.detection.dataset.cocodataset import COCODataPrefetcher, collater
from public.detection.models.loss import RetinaLoss
from public.detection.models.decode import RetinaDecoder
import retinanet
from public.imagenet.utils import get_logger
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch COCO Detection Distributed Training')
    parser.add_argument('--network',
                        type=str,
                        default=Config.network,
                        help='name of network')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--per_node_batch_size',
                        type=int,
                        default=Config.per_node_batch_size,
                        help='per_node batch size')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='load pretrained model params or not')
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
    parser.add_argument('--sync_bn',
                        type=bool,
                        default=Config.sync_bn,
                        help='use sync bn or not')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='LOCAL_PROCESS_RANK')

    return parser.parse_args()


def compute_voc_ap(recall, precision, use_07_metric=True):
    if use_07_metric:
        # use voc 2007 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                # get max precision  for recall >= t
                p = np.max(precision[recall >= t])
            # average 11 recall point precision
            ap = ap + p / 11.
    else:
        # use voc>=2010 metric,average all different recall precision as ap
        # recall add first value 0. and last value 1.
        mrecall = np.concatenate(([0.], recall, [1.]))
        # precision add first value 0. and last value 0.
        mprecision = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mprecision.size - 1, 0, -1):
            mprecision[i - 1] = np.maximum(mprecision[i - 1], mprecision[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrecall[1:] != mrecall[:-1])[0]

        # sum (\Delta recall) * prec
        ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])

    return ap


def compute_ious(a, b):
    """
    :param a: [N,(x1,y1,x2,y2)]
    :param b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """

    a = np.expand_dims(a, axis=1)  # [N,1,4]
    b = np.expand_dims(b, axis=0)  # [1,M,4]

    overlap = np.maximum(0.0,
                         np.minimum(a[..., 2:], b[..., 2:]) -
                         np.maximum(a[..., :2], b[..., :2]))  # [N,M,(w,h)]

    overlap = np.prod(overlap, axis=-1)  # [N,M]

    area_a = np.prod(a[..., 2:] - a[..., :2], axis=-1)
    area_b = np.prod(b[..., 2:] - b[..., :2], axis=-1)

    iou = overlap / (area_a + area_b - overlap)

    return iou


def validate(val_dataset, model, decoder):
    model = model.module
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        all_ap, mAP = evaluate_voc(val_dataset,
                                   model,
                                   decoder,
                                   num_classes=20,
                                   iou_thread=0.5)

    return all_ap, mAP


def evaluate_voc(val_dataset, model, decoder, num_classes=20, iou_thread=0.5):
    preds, gts = [], []
    for index in tqdm(range(len(val_dataset))):
        data = val_dataset[index]
        img, gt_annot, scale = data['img'], data['annot'], data['scale']

        gt_bboxes, gt_classes = gt_annot[:, 0:4], gt_annot[:, 4]
        gt_bboxes /= scale

        gts.append([gt_bboxes, gt_classes])

        cls_heads, reg_heads, batch_anchors = model(img.cuda().permute(
            2, 0, 1).float().unsqueeze(dim=0))
        preds_scores, preds_classes, preds_boxes = decoder(
            cls_heads, reg_heads, batch_anchors)
        preds_scores, preds_classes, preds_boxes = preds_scores.cpu(
        ), preds_classes.cpu(), preds_boxes.cpu()
        preds_boxes /= scale

        # make sure decode batch_size=1
        # preds_scores shape:[1,max_detection_num]
        # preds_classes shape:[1,max_detection_num]
        # preds_bboxes shape[1,max_detection_num,4]
        assert preds_scores.shape[0] == 1

        preds_scores = preds_scores.squeeze(0)
        preds_classes = preds_classes.squeeze(0)
        preds_boxes = preds_boxes.squeeze(0)

        preds_scores = preds_scores[preds_classes > -1]
        preds_boxes = preds_boxes[preds_classes > -1]
        preds_classes = preds_classes[preds_classes > -1]

        preds.append([preds_boxes, preds_classes, preds_scores])

    print("all val sample decode done.")

    all_ap = {}
    for class_index in tqdm(range(num_classes)):
        per_class_gt_boxes = [
            image[0][image[1] == class_index] for image in gts
        ]
        per_class_pred_boxes = [
            image[0][image[1] == class_index] for image in preds
        ]
        per_class_pred_scores = [
            image[2][image[1] == class_index] for image in preds
        ]

        fp = np.zeros((0, ))
        tp = np.zeros((0, ))
        scores = np.zeros((0, ))
        total_gts = 0

        # loop for each sample
        for per_image_gt_boxes, per_image_pred_boxes, per_image_pred_scores in zip(
                per_class_gt_boxes, per_class_pred_boxes,
                per_class_pred_scores):
            total_gts = total_gts + len(per_image_gt_boxes)
            # one gt can only be assigned to one predicted bbox
            assigned_gt = []
            # loop for each predicted bbox
            for index in range(len(per_image_pred_boxes)):
                scores = np.append(scores, per_image_pred_scores[index])
                if per_image_gt_boxes.shape[0] == 0:
                    # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(per_image_pred_boxes[index], axis=0)
                iou = compute_ious(per_image_gt_boxes, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = compute_voc_ap(recall, precision)
        all_ap[class_index] = ap

    mAP = 0.
    for _, class_mAP in all_ap.items():
        mAP += float(class_mAP)
    mAP /= num_classes

    return all_ap, mAP


def main():
    args = parse_args()
    global local_rank
    local_rank = args.local_rank
    if local_rank == 0:
        global logger
        logger = get_logger(__name__, args.log)

    torch.cuda.empty_cache()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    global gpus_num
    gpus_num = torch.cuda.device_count()
    if local_rank == 0:
        logger.info(f'use {gpus_num} gpus')
        logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    if local_rank == 0:
        logger.info('start loading data')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        Config.train_dataset, shuffle=True)
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.per_node_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=collater,
                              sampler=train_sampler)
    if local_rank == 0:
        logger.info('finish loading data')

    model = retinanet.__dict__[args.network](**{
        "pretrained": args.pretrained,
        "num_classes": args.num_classes,
    })

    for name, param in model.named_parameters():
        if local_rank == 0:
            logger.info(f"{name},{param.requires_grad}")

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    if local_rank == 0:
        logger.info(
            f"model: '{args.network}', flops: {flops}, params: {params}")

    criterion = RetinaLoss(image_w=args.input_image_size,
                           image_h=args.input_image_size,
                           alpha=0.25,
                           gamma=1.5).cuda()
    decoder = RetinaDecoder(image_w=args.input_image_size,
                            image_h=args.input_image_size).cuda()

    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=3,
                                                           verbose=True)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.apex:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = apex.parallel.DistributedDataParallel(model,
                                                      delay_allreduce=True)
        if args.sync_bn:
            model = apex.parallel.convert_syncbn_model(model)
    else:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            if local_rank == 0:
                logger.exception(
                    '{} is not a file, please check it again'.format(
                        args.resume))
            sys.exit(-1)
        if local_rank == 0:
            logger.info('start only evaluating')
            logger.info(f"start resuming model from {args.evaluate}")
        checkpoint = torch.load(args.evaluate,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        if local_rank == 0:
            logger.info(f"start eval.")
            all_ap, mAP = validate(Config.val_dataset, model, decoder)
            logger.info(f"eval done.")
            for class_index, class_AP in all_ap.items():
                logger.info(f"class: {class_index},AP: {class_AP:.3f}")
            logger.info(f"mAP: {mAP:.3f}")

        return

    best_map = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        if local_rank == 0:
            logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if local_rank == 0:
            logger.info(
                f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, best_map: {checkpoint['best_map']}, "
                f"loss: {checkpoint['loss']:3f}, cls_loss: {checkpoint['cls_loss']:2f}, reg_loss: {checkpoint['reg_loss']:2f}"
            )

    if local_rank == 0:
        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)

    if local_rank == 0:
        logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        cls_losses, reg_losses, losses = train(train_loader, model, criterion,
                                               optimizer, scheduler, epoch,
                                               args)
        if local_rank == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, cls_loss: {cls_losses:.2f}, reg_loss: {reg_losses:.2f}, loss: {losses:.2f}"
            )

        if epoch % 5 == 0 or epoch == args.epochs:
            if local_rank == 0:
                logger.info(f"start eval.")
                all_ap, mAP = validate(Config.val_dataset, model, decoder)
                logger.info(f"eval done.")
                for class_index, class_AP in all_ap.items():
                    logger.info(f"class: {class_index},AP: {class_AP:.3f}")
                logger.info(f"mAP: {mAP:.3f}")
                if mAP > best_map:
                    torch.save(model.module.state_dict(),
                               os.path.join(args.checkpoints, "best.pth"))
                    best_map = mAP
        if local_rank == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'best_map': best_map,
                    'cls_loss': cls_losses,
                    'reg_loss': reg_losses,
                    'loss': losses,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.checkpoints, 'latest.pth'))

    if local_rank == 0:
        logger.info(f"finish training, best_map: {best_map:.3f}")
    training_time = (time.time() - start_time) / 3600
    if local_rank == 0:
        logger.info(
            f"finish training, total training time: {training_time:.2f} hours")


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    cls_losses, reg_losses, losses = [], [], []

    # switch to train mode
    model.train()

    iters = len(train_loader.dataset) // (args.per_node_batch_size * gpus_num)
    prefetcher = COCODataPrefetcher(train_loader)
    images, annotations = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, annotations = images.cuda().float(), annotations.cuda()
        cls_heads, reg_heads, batch_anchors = model(images)
        cls_loss, reg_loss = criterion(cls_heads, reg_heads, batch_anchors,
                                       annotations)
        loss = cls_loss + reg_loss
        if cls_loss == 0.0 or reg_loss == 0.0:
            optimizer.zero_grad()
            continue

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()

        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        losses.append(loss.item())

        images, annotations = prefetcher.next()

        if local_rank == 0 and iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>5d}, {iters:0>5d}], cls_loss: {cls_loss.item():.2f}, reg_loss: {reg_loss.item():.2f}, loss_total: {loss.item():.2f}"
            )

        iter_index += 1

    scheduler.step(np.mean(losses))

    return np.mean(cls_losses), np.mean(reg_losses), np.mean(losses)


if __name__ == '__main__':
    main()
