import collections
import numpy as np
import time
from tqdm import tqdm

from apex import amp
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval

from simpleAICV.classification.common import ClassificationDataPrefetcher, AverageMeter, AccMeter
from simpleAICV.detection.common import DetectionDataPrefetcher
from simpleAICV.instance_segmentation.common import SegmentationDataPrefetcher


def all_reduce_operation_in_group_for_variables(variables, operator, group):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables


def validate_classification(val_loader, model, criterion, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AccMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(val_loader)):
            images, labels = data['image'], data['label']
            if model_on_cuda:
                images, labels = images.cuda(), labels.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            loss = criterion(outputs, labels)

            torch.distributed.barrier()
            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss = loss / float(config.gpus_num)

            losses.update(loss, images.size(0))

            _, topk_indexes = torch.topk(outputs,
                                         k=5,
                                         dim=1,
                                         largest=True,
                                         sorted=True)
            correct_mask = topk_indexes.eq(
                labels.unsqueeze(-1).expand_as(topk_indexes)).float()
            correct_mask = correct_mask.cpu().numpy()

            acc1_correct_num, acc5_correct_num, sample_num = correct_mask[:, :1].sum(
            ), correct_mask[:, :5].sum(), images.size(0)
            acc1_correct_num, acc5_correct_num, sample_num = float(
                acc1_correct_num), float(acc5_correct_num), float(sample_num)

            # please keep same variable on different gpus has same data type for all reduce operation
            torch.distributed.barrier()
            [acc1_correct_num, acc5_correct_num,
             sample_num] = all_reduce_operation_in_group_for_variables(
                 variables=[acc1_correct_num, acc5_correct_num, sample_num],
                 operator=torch.distributed.ReduceOp.SUM,
                 group=config.group)

            accs.update(acc1_correct_num, acc5_correct_num, sample_num)

            end = time.time()

    # top1(%)、top5(%)
    accs.compute()
    acc1 = accs.acc1 * 100
    acc5 = accs.acc5 * 100

    # avg_loss
    avg_loss = losses.avg

    # per image data load time(ms) and inference time(ms)
    per_image_load_time = data_time.avg / (config.batch_size //
                                           config.gpus_num) * 1000
    per_image_inference_time = batch_time.avg / (config.batch_size //
                                                 config.gpus_num) * 1000

    return acc1, acc5, avg_loss, per_image_load_time, per_image_inference_time


def train_classification(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train classification model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size

    prefetcher = ClassificationDataPrefetcher(train_loader)
    images, labels = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, labels = images.cuda(), labels.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(labels)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(labels)):
            continue

        outputs = model(images)
        loss = criterion(outputs, labels)

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            optimizer.zero_grad()
            continue

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.barrier()
        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)

        losses.update(loss, images.size(0))

        images, labels = prefetcher.next()

        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, loss: {loss:.4f}'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    scheduler.step()

    return losses.avg


def validate_distill_classification(val_loader, model, criterion, config):
    teacher_model = model.module.teacher
    student_model = model.module.student

    tea_acc1, tea_acc5, tea_test_loss, _, _ = validate_classification(
        val_loader, teacher_model, criterion, config)

    stu_acc1, stu_acc5, stu_test_loss, _, _ = validate_classification(
        val_loader, student_model, criterion, config)

    return tea_acc1, tea_acc5, tea_test_loss, stu_acc1, stu_acc5, stu_test_loss


def train_distill_classification(train_loader, model, criterion, optimizer,
                                 scheduler, epoch, logger, config):
    '''
    distill classification model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()
    if config.freeze_teacher:
        model.module.teacher.eval()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size

    prefetcher = ClassificationDataPrefetcher(train_loader)
    images, labels = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, labels = images.cuda(), labels.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(labels)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(labels)):
            continue

        tea_outputs, stu_outputs = model(images)

        loss = 0
        loss_value = {}
        for loss_name in criterion.keys():
            if loss_name in ['CELoss']:
                if not config.freeze_teacher:
                    temp_loss = criterion[loss_name](tea_outputs, labels)
                    loss_value['tea_' + loss_name] = temp_loss
                    loss += temp_loss
                temp_loss = criterion[loss_name](stu_outputs, labels)
                loss_value['stu_' + loss_name] = temp_loss
                loss += temp_loss
            elif loss_name in ['DKDLoss']:
                temp_loss = criterion[loss_name](stu_outputs, tea_outputs,
                                                 labels)
                loss_value[loss_name] = temp_loss
                loss += temp_loss
            else:
                temp_loss = criterion[loss_name](stu_outputs, tea_outputs)
                loss_value[loss_name] = temp_loss
                loss += temp_loss

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            optimizer.zero_grad()
            continue

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.barrier()

        for key, value in loss_value.items():
            [value] = all_reduce_operation_in_group_for_variables(
                variables=[value],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss_value[key] = value / float(config.gpus_num)

        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)

        losses.update(loss, images.size(0))

        images, labels = prefetcher.next()

        log_info = ''
        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, loss: {loss:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value:.4f} '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    scheduler.step()

    return losses.avg


def compute_voc_ap(recall, precision, use_07_metric=False):
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
    '''
    :param a: [N,(x1,y1,x2,y2)]
    :param b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    '''

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


def evaluate_voc_detection(val_loader, model, criterion, decoder, config):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    batch_size = int(config.batch_size // config.gpus_num)

    with torch.no_grad():
        preds, gts = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for _, data in tqdm(enumerate(val_loader)):
            images, annots, scales, sizes = data['image'], data[
                'annots'], data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            outs_tuple = model(images)

            pred_scores, pred_classes, pred_boxes = decoder(outs_tuple)

            pred_boxes /= np.expand_dims(np.expand_dims(scales, axis=-1),
                                         axis=-1)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            annots = annots.cpu().numpy()
            gt_bboxes, gt_classes = annots[:, :, 0:4], annots[:, :, 4]
            gt_bboxes /= np.expand_dims(np.expand_dims(scales, axis=-1),
                                        axis=-1)

            for per_image_pred_scores, per_image_pred_classes, per_image_pred_boxes, per_image_gt_bboxes, per_image_gt_classes, per_image_size in zip(
                    pred_scores, pred_classes, pred_boxes, gt_bboxes,
                    gt_classes, sizes):
                per_image_pred_scores = per_image_pred_scores[
                    per_image_pred_classes > -1]
                per_image_pred_boxes = per_image_pred_boxes[
                    per_image_pred_classes > -1]
                per_image_pred_classes = per_image_pred_classes[
                    per_image_pred_classes > -1]

                # clip boxes
                per_image_pred_boxes[:, 0] = np.maximum(
                    per_image_pred_boxes[:, 0], 0)
                per_image_pred_boxes[:, 1] = np.maximum(
                    per_image_pred_boxes[:, 1], 0)
                per_image_pred_boxes[:, 2] = np.minimum(
                    per_image_pred_boxes[:, 2], per_image_size[1])
                per_image_pred_boxes[:, 3] = np.minimum(
                    per_image_pred_boxes[:, 3], per_image_size[0])

                preds.append([
                    per_image_pred_boxes, per_image_pred_classes,
                    per_image_pred_scores
                ])

                per_image_gt_bboxes = per_image_gt_bboxes[
                    per_image_gt_classes > -1]
                per_image_gt_classes = per_image_gt_classes[
                    per_image_gt_classes > -1]

                gts.append([per_image_gt_bboxes, per_image_gt_classes])

            end = time.time()

        result_dict = collections.OrderedDict()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / batch_size * 1000
        per_image_inference_time = batch_time.avg / batch_size * 1000

        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        all_iou_threshold_map = collections.OrderedDict()
        all_iou_threshold_per_class_ap = collections.OrderedDict()
        for per_iou_threshold in tqdm(config.eval_voc_iou_threshold_list):
            per_iou_threshold_all_class_ap = collections.OrderedDict()
            for class_index in range(config.num_classes):
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
                        scores = np.append(scores,
                                           per_image_pred_scores[index])
                        if per_image_gt_boxes.shape[0] == 0:
                            # if no gts found for the predicted bbox, assign the bbox to fp
                            fp = np.append(fp, 1)
                            tp = np.append(tp, 0)
                            continue
                        pred_box = np.expand_dims(per_image_pred_boxes[index],
                                                  axis=0)
                        iou = compute_ious(per_image_gt_boxes, pred_box)
                        gt_for_box = np.argmax(iou, axis=0)
                        max_overlap = iou[gt_for_box, 0]
                        if max_overlap >= per_iou_threshold and gt_for_box not in assigned_gt:
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
                ap = compute_voc_ap(recall, precision, use_07_metric=False)
                per_iou_threshold_all_class_ap[class_index] = ap * 100

            per_iou_threshold_map = 0.
            for _, per_iou_threshold_per_class_ap in per_iou_threshold_all_class_ap.items(
            ):
                per_iou_threshold_map += float(per_iou_threshold_per_class_ap)
            per_iou_threshold_map /= config.num_classes

            all_iou_threshold_map[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,mAP'] = per_iou_threshold_map
            all_iou_threshold_per_class_ap[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,per_class_ap'] = per_iou_threshold_all_class_ap

        for key, value in all_iou_threshold_map.items():
            result_dict[key] = value
        for key, value in all_iou_threshold_per_class_ap.items():
            result_dict[key] = value

        return result_dict


def evaluate_coco_detection(val_loader, model, criterion, decoder, config):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    val_dataset = config.val_dataset
    ids = [idx for idx in range(len(val_dataset))]
    batch_size = int(config.batch_size // config.gpus_num)

    with torch.no_grad():
        results, image_ids = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for i, data in tqdm(enumerate(val_loader)):
            images, annots, scales, sizes = data['image'], data[
                'annots'], data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()

            per_batch_ids = ids[i * batch_size:(i + 1) * batch_size]

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            outs_tuple = model(images)

            scores, classes, boxes = decoder(outs_tuple)

            boxes /= np.expand_dims(np.expand_dims(scales, axis=-1), axis=-1)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            for per_image_scores, per_image_classes, per_image_boxes, index, per_image_size in zip(
                    scores, classes, boxes, per_batch_ids, sizes):
                # clip boxes
                per_image_boxes[:, 0] = np.maximum(per_image_boxes[:, 0], 0)
                per_image_boxes[:, 1] = np.maximum(per_image_boxes[:, 1], 0)
                per_image_boxes[:, 2] = np.minimum(per_image_boxes[:, 2],
                                                   per_image_size[1])
                per_image_boxes[:, 3] = np.minimum(per_image_boxes[:, 3],
                                                   per_image_size[0])

                # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
                per_image_boxes[:, 2:] -= per_image_boxes[:, :2]

                for object_score, object_class, object_box in zip(
                        per_image_scores, per_image_classes, per_image_boxes):
                    object_score = float(object_score)
                    object_class = int(object_class)
                    object_box = object_box.tolist()
                    if object_class == -1:
                        break

                    image_result = {
                        'image_id':
                        val_dataset.image_ids[index],
                        'category_id':
                        val_dataset.coco_label_to_cat_id[object_class],
                        'score':
                        object_score,
                        'bbox':
                        object_box,
                    }
                    results.append(image_result)

                image_ids.append(val_dataset.image_ids[index])

                print('{}/{}'.format(index, len(val_dataset)), end='\r')

            end = time.time()

        variable_definitions = {
            0: 'IoU=0.50:0.95,area=all,maxDets=100,mAP',
            1: 'IoU=0.50,area=all,maxDets=100,mAP',
            2: 'IoU=0.75,area=all,maxDets=100,mAP',
            3: 'IoU=0.50:0.95,area=small,maxDets=100,mAP',
            4: 'IoU=0.50:0.95,area=medium,maxDets=100,mAP',
            5: 'IoU=0.50:0.95,area=large,maxDets=100,mAP',
            6: 'IoU=0.50:0.95,area=all,maxDets=1,mAR',
            7: 'IoU=0.50:0.95,area=all,maxDets=10,mAR',
            8: 'IoU=0.50:0.95,area=all,maxDets=100,mAR',
            9: 'IoU=0.50:0.95,area=small,maxDets=100,mAR',
            10: 'IoU=0.50:0.95,area=medium,maxDets=100,mAR',
            11: 'IoU=0.50:0.95,area=large,maxDets=100,mAR',
        }

        result_dict = collections.OrderedDict()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / batch_size * 1000
        per_image_inference_time = batch_time.avg / batch_size * 1000

        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict[
            'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        if len(results) == 0:
            for _, value in variable_definitions.items():
                result_dict[value] = 0
            return result_dict

        # load results in COCO evaluation tool
        coco_true = val_dataset.coco
        coco_pred = coco_true.loadRes(results)

        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        eval_result = coco_eval.stats

        for i, var in enumerate(eval_result):
            result_dict[variable_definitions[i]] = var * 100

        return result_dict


def validate_detection(val_loader, model, criterion, decoder, config):
    assert config.eval_type in ['COCO', 'VOC']

    func_dict = {
        'COCO': evaluate_coco_detection,
        'VOC': evaluate_voc_detection,
    }
    result_dict = func_dict[config.eval_type](val_loader, model, criterion,
                                              decoder, config)

    return result_dict


def train_detection(train_loader, model, criterion, optimizer, scheduler,
                    epoch, logger, config):
    '''
    train classification model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size

    prefetcher = DetectionDataPrefetcher(train_loader)
    images, targets = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, targets = images.cuda(), targets.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(targets)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(targets)):
            continue

        if torch.sum(images) == 0:
            continue

        outs_tuple = model(images)
        loss_dict = criterion(outs_tuple, targets)

        loss = sum(loss_dict.values())

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            optimizer.zero_grad()
            continue

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.barrier()
        for key, value in loss_dict.items():
            [value] = all_reduce_operation_in_group_for_variables(
                variables=[value],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss_dict[key] = value / float(config.gpus_num)

        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)

        losses.update(loss, images.size(0))

        images, targets = prefetcher.next()

        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, total_loss: {loss:.4f}'
            for key, value in loss_dict.items():
                log_info += f', {key}: {value:.4f}'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    scheduler.step()

    return losses.avg


def evaluate_coco_segmentation(val_dataset,
                               val_loader,
                               model,
                               decoder,
                               config,
                               mask_threshold=0.5):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    ids = [idx for idx in range(len(val_dataset))]

    results, image_ids = [], []
    model_on_cuda = next(model.parameters()).is_cuda
    end = time.time()
    for i, data in tqdm(enumerate(val_loader)):
        if model_on_cuda:
            images, scales, origin_hws = data['image'].cuda(
            ), data['scale'], data['origin_hw']
        else:
            images, scales, origin_hws = data['image'], data['scale'], data[
                'origin_hw']

        per_batch_ids = ids[i * config.batch_size:(i + 1) * config.batch_size]

        torch.cuda.synchronize()
        data_time.update(time.time() - end, images.size(0))
        end = time.time()

        outs_tuple = model(images)
        scores, classes, masks, boxes = decoder(*outs_tuple)

        scores, classes, masks, boxes = scores.cpu(), classes.cpu(), masks.cpu(
        ), boxes.cpu()
        scales = scales.unsqueeze(-1).unsqueeze(-1)
        boxes /= scales

        torch.cuda.synchronize()
        batch_time.update(time.time() - end, images.size(0))

        for per_image_scores, per_image_classes, per_image_boxes, index, per_image_masks, per_image_scale, per_image_origin_hw in zip(
                scores, classes, boxes, per_batch_ids, masks, scales,
                origin_hws):
            # clip boxes
            per_image_boxes[:, 0] = torch.clamp(per_image_boxes[:, 0], min=0)
            per_image_boxes[:, 1] = torch.clamp(per_image_boxes[:, 1], min=0)
            per_image_boxes[:, 2] = torch.clamp(per_image_boxes[:, 2],
                                                max=per_image_origin_hw[1])
            per_image_boxes[:, 3] = torch.clamp(per_image_boxes[:, 3],
                                                max=per_image_origin_hw[0])

            # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
            per_image_boxes[:, 2:] -= per_image_boxes[:, :2]

            input_h, input_w = int(
                per_image_masks.shape[-2] / per_image_scale), int(
                    per_image_masks.shape[-1] / per_image_scale)
            per_image_masks = F.interpolate(
                per_image_masks.float().unsqueeze(0),
                size=(input_h, input_w),
                mode='nearest').squeeze(0)
            per_image_origin_hw = per_image_origin_hw.int()
            per_image_masks = per_image_masks[:, 0:per_image_origin_hw[0],
                                              0:per_image_origin_hw[1]]
            per_image_masks = (per_image_masks > mask_threshold).int()

            for object_score, object_class, object_mask, object_box in zip(
                    per_image_scores, per_image_classes, per_image_masks,
                    per_image_boxes):
                object_score = float(object_score)
                object_class = int(object_class)
                object_box = object_box.tolist()
                object_mask = np.asfortranarray(object_mask).astype(np.uint8)

                if object_class == -1:
                    break

                image_result = {
                    'image_id':
                    val_dataset.image_ids[index],
                    'category_id':
                    val_dataset.coco_label_to_cat_id[object_class],
                    'score':
                    object_score,
                    'bbox':
                    object_box,
                    'segmentation':
                    val_dataset.transform_mask_to_rle_mask(object_mask),
                }
                results.append(image_result)

            image_ids.append(val_dataset.image_ids[index])

            print('{}/{}'.format(index, len(val_dataset)), end='\r')

        end = time.time()

    if len(results) == 0:
        return None

    # load results in COCO evaluation tool
    coco_true = val_dataset.coco
    coco_pred = coco_true.loadRes(results)

    variable_definitions = {
        0: 'IoU=0.5:0.95,area=all,maxDets=100,mAP',
        1: 'IoU=0.5,area=all,maxDets=100,mAP',
        2: 'IoU=0.75,area=all,maxDets=100,mAP',
        3: 'IoU=0.5:0.95,area=small,maxDets=100,mAP',
        4: 'IoU=0.5:0.95,area=medium,maxDets=100,mAP',
        5: 'IoU=0.5:0.95,area=large,maxDets=100,mAP',
        6: 'IoU=0.5:0.95,area=all,maxDets=1,mAR',
        7: 'IoU=0.5:0.95,area=all,maxDets=10,mAR',
        8: 'IoU=0.5:0.95,area=all,maxDets=100,mAR',
        9: 'IoU=0.5:0.95,area=small,maxDets=100,mAR',
        10: 'IoU=0.5:0.95,area=medium,maxDets=100,mAR',
        11: 'IoU=0.5:0.95,area=large,maxDets=100,mAR',
    }
    result_dict = collections.OrderedDict()

    coco_eval_segm = COCOeval(coco_true, coco_pred, 'segm')
    coco_eval_segm.params.imgIds = image_ids
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    segm_eval_result = coco_eval_segm.stats

    result_dict['sgem_eval_result'] = {}
    for i, var in enumerate(segm_eval_result):
        result_dict['sgem_eval_result'][variable_definitions[i]] = var

    coco_eval_box = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval_box.params.imgIds = image_ids
    coco_eval_box.evaluate()
    coco_eval_box.accumulate()
    coco_eval_box.summarize()
    box_eval_result = coco_eval_box.stats

    result_dict['box_eval_result'] = {}
    for i, var in enumerate(box_eval_result):
        result_dict['box_eval_result'][variable_definitions[i]] = var * 100

    # per image data load time(ms) and inference time(ms)
    per_image_load_time = data_time.avg / config.batch_size * 1000
    per_image_inference_time = batch_time.avg / config.batch_size * 1000

    result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
    result_dict[
        'per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

    return result_dict


def validate_segmentation(val_dataset, val_loader, model, decoder, config):
    # switch to evaluate mode
    model.eval()

    assert config.dataset_name in ['COCO']

    func_dict = {
        'COCO': evaluate_coco_segmentation,
    }
    with torch.no_grad():
        result_dict = func_dict[config.dataset_name](val_dataset, val_loader,
                                                     model, decoder, config)

    return result_dict


def compute_segmentation_test_loss(val_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for data in tqdm(val_loader):
            images, gt_annots = data['image'], data['annots']
            boxes, masks, classes = gt_annots['box'], gt_annots[
                'mask'], gt_annots['class']
            if model_on_cuda:
                images, boxes, masks, classes = images.cuda(), boxes.cuda(
                ), masks.cuda(), classes.cuda()
            targets = {
                'box': boxes,
                'mask': masks,
                'class': classes,
            }

            outs_tuple = model(images)
            loss_dict = criterion(targets, *outs_tuple)

            loss = sum(loss_dict.values())

            losses.update(loss.item(), images.size(0))

        return losses.avg


def train_segmentation(train_loader, model, criterion, optimizer, scheduler,
                       epoch, logger, config):
    '''
    train classification model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank() if config.distributed else None
    if config.distributed:
        gpus_num = torch.cuda.device_count()
        iters = len(train_loader.dataset) // (
            config.batch_size * gpus_num) if config.distributed else len(
                train_loader.dataset) // config.batch_size
    else:
        iters = len(train_loader.dataset) // config.batch_size

    prefetcher = SegmentationDataPrefetcher(train_loader)
    images, boxes, masks, classes = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, boxes, masks, classes = images.cuda(), boxes.cuda(
        ), masks.cuda(), classes.cuda()
        targets = {
            'box': boxes,
            'mask': masks,
            'class': classes,
        }

        outs_tuple = model(images)
        loss_dict = criterion(targets, *outs_tuple)

        loss = sum(loss_dict.values())

        if loss == 0.:
            optimizer.zero_grad()
            continue

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), images.size(0))

        images, boxes, masks, classes = prefetcher.next()

        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, total_loss: {loss.item():.4f}'
            for key, value in loss_dict.items():
                log_info += f', {key}: {value.item():.4f}'
            logger.info(log_info) if (config.distributed and local_rank
                                      == 0) or not config.distributed else None

        iter_index += 1

    scheduler.step()

    return losses.avg
