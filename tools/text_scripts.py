import cv2

import os

import collections
import numpy as np
import json
import re
import time
from tqdm import tqdm

from nltk.metrics.distance import edit_distance

import torch
import torch.nn.functional as F

from torch.cuda.amp import autocast

from simpleAICV.text_detection.common import AverageMeter, PrecisionRecallMeter


def all_reduce_operation_in_group_for_variables(variables, operator, group):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables


def test_text_recognition_for_all_dataset(val_loader_list, model, criterion,
                                          config):

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader) in enumerate(
            zip(config.val_dataset_name_list, val_loader_list)):
        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = test_text_recognition_for_per_sub_dataset(
            per_sub_dataset_loader, model, criterion, config)

        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def test_text_recognition_for_per_sub_dataset(val_loader, model, criterion,
                                              config):
    str_acc_edit_distance_dict = test_str_acc_edit_distance_for_per_sub_dataset(
        val_loader, model, criterion, config.converter, config)
    order_pr_dict = test_order_PR_for_per_sub_dataset(val_loader, model,
                                                      criterion,
                                                      config.converter, config)
    chars_pr_dict = test_chars_PR_for_per_sub_dataset(val_loader, model,
                                                      criterion,
                                                      config.converter, config)
    lcs_pr_dict = test_lcs_PR_for_per_sub_dataset(val_loader,
                                                  model,
                                                  criterion,
                                                  config.converter,
                                                  config,
                                                  ignore_threhold=1000)
    all_dict = collections.OrderedDict()

    for key, value in str_acc_edit_distance_dict.items():
        all_dict[key] = value
    for key, value in order_pr_dict.items():
        all_dict[key] = value
    for key, value in chars_pr_dict.items():
        all_dict[key] = value
    for key, value in lcs_pr_dict.items():
        all_dict[key] = value

    return all_dict


def test_str_acc_edit_distance_for_per_sub_dataset(val_loader, model,
                                                   criterion, converter,
                                                   config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    keep_character = "".join(config.all_char_table)
    garbage_char = converter.garbage_char

    # switch to evaluate mode
    model.eval()

    correct_str_nums, not_included_str_nums, total_str_nums, ne_distances = 0, 0, 0, 0

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(val_loader)):
            images, targets = data['image'], data['label']
            if model_on_cuda:
                images = images.cuda()

            trans_targets, target_lengths = config.converter.encode(targets)
            if model_on_cuda:
                trans_targets, target_lengths = trans_targets.cuda(
                ), target_lengths.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            input_lengths = torch.IntTensor([outputs.shape[1]] *
                                            outputs.shape[0])
            if model_on_cuda:
                input_lengths = input_lengths.cuda()

            loss = criterion(
                F.log_softmax(outputs, dim=2).permute(1, 0, 2), trans_targets,
                input_lengths, target_lengths)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss = loss / float(config.gpus_num)

            losses.update(loss, images.size(0))

            _, pred_indexes = outputs.max(dim=2)
            pred_strs = converter.decode(pred_indexes.cpu().numpy(),
                                         input_lengths.cpu().numpy())
            pred_probs, _ = (F.softmax(outputs, dim=2)).max(dim=2)
            pred_probs = pred_probs.cpu().numpy()

            correct_str_nums, not_included_str_nums, ne_distances = compute_strs_acc_edit_distance_per_batch(
                correct_str_nums,
                not_included_str_nums,
                ne_distances,
                pred_strs,
                pred_probs,
                targets,
                keep_character,
                garbage_char,
                case_insensitve=True)

            total_str_nums += images.size(0)

            end = time.time()

    torch.distributed.barrier()

    [correct_str_nums, not_included_str_nums, total_str_nums,
     ne_distances] = all_reduce_operation_in_group_for_variables(
         variables=[
             correct_str_nums, not_included_str_nums, total_str_nums,
             ne_distances
         ],
         operator=torch.distributed.ReduceOp.SUM,
         group=config.group)

    if total_str_nums != 0:
        str_acc = correct_str_nums / float(total_str_nums) * 100
        not_included_str_percent = not_included_str_nums / float(
            total_str_nums) * 100
        final_edit_distance = ne_distances / float(total_str_nums) * 100
    else:
        str_acc, not_included_str_percent, final_edit_distance = 0, 0, 0

    # avg_loss
    test_loss = losses.avg

    # per image data load time(ms) and inference time(ms)
    per_image_load_time = data_time.avg / (config.batch_size //
                                           config.gpus_num) * 1000
    per_image_inference_time = batch_time.avg / (config.batch_size //
                                                 config.gpus_num) * 1000

    str_acc_edit_distance_dict = {
        'per_image_load_time': per_image_load_time,
        'per_image_inference_time': per_image_inference_time,
        'test_loss': test_loss,
        'str_acc': str_acc,
        'not_included_str_percent': not_included_str_percent,
        'final_edit_distance': final_edit_distance
    }

    return str_acc_edit_distance_dict


def compute_strs_acc_edit_distance_per_batch(correct_str_nums,
                                             not_included_str_nums,
                                             ne_distances,
                                             pred_strs,
                                             pred_probs,
                                             targets,
                                             keep_character,
                                             garbage_char,
                                             case_insensitve=True):
    keep_character = f'[^{keep_character}]'
    for per_pred_str, per_pred_prob, per_target in zip(pred_strs, pred_probs,
                                                       targets):
        not_include_char = ""
        for per_char in list(per_target):
            if per_char not in keep_character:
                not_include_char += per_char

        if any(per_char not in keep_character
               for per_char in list(per_target)):
            not_included_str_nums += 1

        # convert not include char to garbage char
        per_target_convert = ""
        for per_char in per_target:
            if per_char not in keep_character:
                per_target_convert += garbage_char
            else:
                per_target_convert += per_char

        per_target = per_target_convert

        per_pred_str = per_pred_str.replace(' ', '')
        per_target = per_target.replace(' ', '')

        if per_target == "㍿":
            continue

        if per_target == "":
            continue

        if case_insensitve:
            per_pred_str = per_pred_str.lower()
            per_target = per_target.lower()

        if per_pred_str == per_target:
            correct_str_nums += 1

        # ICDAR2019 Normalized Edit Distance
        # https://arxiv.org/pdf/1909.07741.pdf IV. TASKS B. TASK 2 - END-TO-END TEXT SPOTTING
        # edit_distance():Levenshtein Distance
        # 两个字符串之间，由一个转成另一个所需的最少编辑操作次数
        # ne_distances [0,1]
        if len(per_target) == 0 or len(per_pred_str) == 0:
            ne_distances += 0
        elif len(per_target) > len(per_pred_str):
            ne_distances += 1 - edit_distance(per_pred_str,
                                              per_target) / len(per_target)
        else:
            ne_distances += 1 - edit_distance(per_pred_str,
                                              per_target) / len(per_pred_str)

    return correct_str_nums, not_included_str_nums, ne_distances


def test_order_PR_for_per_sub_dataset(val_loader, model, criterion, converter,
                                      config):
    keep_character = "".join(config.all_char_table)
    garbage_char = converter.garbage_char

    # switch to evaluate mode
    model.eval()

    c_char_nums, p_char_nums, t_char_nums = 0, 0, 0

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(val_loader)):
            images, targets = data['image'], data['label']
            if model_on_cuda:
                images = images.cuda()

            trans_targets, target_lengths = config.converter.encode(targets)
            if model_on_cuda:
                trans_targets, target_lengths = trans_targets.cuda(
                ), target_lengths.cuda()

            outputs = model(images)

            input_lengths = torch.IntTensor([outputs.shape[1]] *
                                            outputs.shape[0])
            if model_on_cuda:
                input_lengths = input_lengths.cuda()

            _, pred_indexes = outputs.max(dim=2)

            pred_strs = converter.decode(pred_indexes.cpu().numpy(),
                                         input_lengths.cpu().numpy())

            c_char_nums, p_char_nums, t_char_nums = compute_order_PR_per_batch(
                c_char_nums,
                p_char_nums,
                t_char_nums,
                pred_strs,
                targets,
                keep_character,
                garbage_char,
                case_insensitve=True)

    torch.distributed.barrier()

    [c_char_nums, p_char_nums,
     t_char_nums] = all_reduce_operation_in_group_for_variables(
         variables=[c_char_nums, p_char_nums, t_char_nums],
         operator=torch.distributed.ReduceOp.SUM,
         group=config.group)

    if p_char_nums != 0:
        order_precision = c_char_nums / float(p_char_nums) * 100
    else:
        order_precision = 0

    if t_char_nums != 0:
        order_recall = c_char_nums / float(t_char_nums) * 100
    else:
        order_recall = 0

    order_pr_dict = {
        'order_precision': order_precision,
        'order_recall': order_recall,
    }

    return order_pr_dict


def compute_order_PR_per_batch(c_char_nums,
                               p_char_nums,
                               t_char_nums,
                               pred_strs,
                               targets,
                               keep_character,
                               garbage_char,
                               case_insensitve=True):
    keep_character = f'[^{keep_character}]'
    for per_pred_str, per_target in zip(pred_strs, targets):
        # convert not include char to garbage char
        per_target_convert = ""
        for per_char in per_target:
            if per_char not in keep_character:
                per_target_convert += garbage_char
            else:
                per_target_convert += per_char

        per_target = per_target_convert

        per_pred_str = per_pred_str.replace(' ', '')
        per_target = per_target.replace(' ', '')

        if per_target == "㍿":
            continue

        if per_target == "":
            continue

        if case_insensitve:
            per_pred_str = per_pred_str.lower()
            per_target = per_target.lower()

        length = min(len(per_pred_str), len(per_target))

        for i in range(length):
            if per_pred_str[i] == per_target[i]:
                c_char_nums += 1

        p_char_nums += len(per_pred_str)
        t_char_nums += len(per_target)

    return c_char_nums, p_char_nums, t_char_nums


def test_chars_PR_for_per_sub_dataset(val_loader, model, criterion, converter,
                                      config):
    support_chars_set = set(config.all_char_table)
    garbage_char = converter.garbage_char

    # switch to evaluate mode
    model.eval()

    correct_char_nums, not_include_target_char_nums, pred_char_nums, target_char_nums = 0, 0, 0, 0

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(val_loader)):
            images, targets = data['image'], data['label']
            if model_on_cuda:
                images = images.cuda()

            trans_targets, target_lengths = config.converter.encode(targets)
            if model_on_cuda:
                trans_targets, target_lengths = trans_targets.cuda(
                ), target_lengths.cuda()

            outputs = model(images)

            input_lengths = torch.IntTensor([outputs.shape[1]] *
                                            outputs.shape[0])
            if model_on_cuda:
                input_lengths = input_lengths.cuda()

            _, pred_indexes = outputs.max(dim=2)

            pred_strs = converter.decode(pred_indexes.cpu().numpy(),
                                         input_lengths.cpu().numpy())

            correct_char_nums, pred_char_nums, target_char_nums, not_include_target_char_nums = compute_chars_PR_per_batch(
                correct_char_nums, pred_char_nums, target_char_nums,
                not_include_target_char_nums, pred_strs, targets,
                support_chars_set, garbage_char)

    torch.distributed.barrier()

    [
        correct_char_nums, pred_char_nums, target_char_nums,
        not_include_target_char_nums
    ] = all_reduce_operation_in_group_for_variables(
        variables=[
            correct_char_nums, pred_char_nums, target_char_nums,
            not_include_target_char_nums
        ],
        operator=torch.distributed.ReduceOp.SUM,
        group=config.group)

    if pred_char_nums != 0:
        chars_precision = correct_char_nums / float(pred_char_nums) * 100
        if chars_precision > 100:
            chars_precision = 1 * 100
    else:
        chars_precision = 0

    if target_char_nums != 0:
        chars_recall = correct_char_nums / float(target_char_nums) * 100
        if chars_recall > 100:
            chars_recall = 1 * 100
    else:
        chars_recall = 0

    if target_char_nums != 0:
        not_include_chars_percent = not_include_target_char_nums / float(
            target_char_nums) * 100
    else:
        not_include_chars_percent = 0

    chars_pr_dict = {
        'chars_precision': chars_precision,
        'chars_recall': chars_recall,
        'not_include_chars_percent': not_include_chars_percent,
    }

    return chars_pr_dict


def compute_chars_PR_per_batch(correct_char_nums, pred_char_nums,
                               target_char_nums, not_include_target_char_nums,
                               pred_strs, targets, support_chars_set,
                               garbage_char):
    # none:"㍿",space:" "
    batch_none_target_nums, batch_space_target_nums = 0, 0
    batch_space_pred_nums, batch_correct_pred_nums = 0, 0
    batch_pred_length, batch_target_length = 0, 0

    for per_pred_str, per_target in zip(pred_strs, targets):
        # convert not include char to garbage char
        per_target_convert = ""
        for per_char in per_target:
            if per_char not in support_chars_set:
                not_include_target_char_nums += 1
                per_target_convert += garbage_char
            else:
                per_target_convert += per_char

        per_target = per_target_convert

        filter_space_per_target = per_target.replace(' ', '')
        if filter_space_per_target == "㍿":
            continue

        per_pred_str, per_target = list(per_pred_str), list(per_target)
        pred_length, target_length = len(per_pred_str), len(per_target)
        batch_pred_length += pred_length
        batch_target_length += target_length
        for i in range(target_length):
            if per_target[i] == ' ':
                batch_space_target_nums += 1
            if per_target[i] == '㍿':
                batch_none_target_nums += 1

        for i in range(pred_length):
            if per_pred_str[i] == ' ':
                batch_space_pred_nums += 1
                continue
            for j in range(len(per_target)):
                if per_target[j] == '㍿' or per_target[j] == ' ':
                    continue
                if per_pred_str[i] == per_target[j]:
                    batch_correct_pred_nums += 1
                    per_target.remove(per_target[j])
                    break

    correct_char_nums += min(batch_correct_pred_nums + batch_none_target_nums,
                             batch_pred_length - batch_space_pred_nums)
    pred_char_nums += (batch_pred_length - batch_space_pred_nums)
    target_char_nums += (batch_target_length - batch_space_target_nums)

    return correct_char_nums, pred_char_nums, target_char_nums, not_include_target_char_nums


def test_lcs_PR_for_per_sub_dataset(val_loader,
                                    model,
                                    criterion,
                                    converter,
                                    config,
                                    ignore_threhold=1000):
    keep_character = "".join(config.all_char_table)
    garbage_char = converter.garbage_char

    num_chars_set = set(config.num_char_table)
    alpha_chars_set = set(config.alpha_char_table)

    common_standard_chinese_char_first_set = set(
        config.common_standard_chinese_char_first_table)
    common_standard_chinese_char_second_set = set(
        config.common_standard_chinese_char_second_table)
    common_standard_chinese_char_third_set = set(
        config.common_standard_chinese_char_third_table)

    # switch to evaluate mode
    model.eval()

    c_num_char_nums, p_num_char_nums, t_num_char_nums = 0, 0, 0
    c_alpha_char_nums, p_alpha_char_nums, t_alpha_char_nums = 0, 0, 0
    c_first_char_nums, p_first_char_nums, t_first_char_nums = 0, 0, 0
    c_second_char_nums, p_second_char_nums, t_second_char_nums = 0, 0, 0
    c_third_char_nums, p_third_char_nums, t_third_char_nums = 0, 0, 0
    c_char_nums, p_char_nums, t_char_nums = 0, 0, 0

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(val_loader)):
            images, targets = data['image'], data['label']
            if model_on_cuda:
                images = images.cuda()

            trans_targets, target_lengths = config.converter.encode(targets)
            if model_on_cuda:
                trans_targets, target_lengths = trans_targets.cuda(
                ), target_lengths.cuda()

            outputs = model(images)

            input_lengths = torch.IntTensor([outputs.shape[1]] *
                                            outputs.shape[0])
            if model_on_cuda:
                input_lengths = input_lengths.cuda()

            _, pred_indexes = outputs.max(dim=2)

            pred_strs = converter.decode(pred_indexes.cpu().numpy(),
                                         input_lengths.cpu().numpy())

            c_num_char_nums, p_num_char_nums, t_num_char_nums, c_alpha_char_nums, p_alpha_char_nums, t_alpha_char_nums, c_first_char_nums, p_first_char_nums, t_first_char_nums, c_second_char_nums, p_second_char_nums, t_second_char_nums, c_third_char_nums, p_third_char_nums, t_third_char_nums, c_char_nums, p_char_nums, t_char_nums = compute_lcs_PR_per_batch(
                c_num_char_nums,
                p_num_char_nums,
                t_num_char_nums,
                c_alpha_char_nums,
                p_alpha_char_nums,
                t_alpha_char_nums,
                c_first_char_nums,
                p_first_char_nums,
                t_first_char_nums,
                c_second_char_nums,
                p_second_char_nums,
                t_second_char_nums,
                c_third_char_nums,
                p_third_char_nums,
                t_third_char_nums,
                c_char_nums,
                p_char_nums,
                t_char_nums,
                pred_strs,
                targets,
                num_chars_set,
                alpha_chars_set,
                common_standard_chinese_char_first_set,
                common_standard_chinese_char_second_set,
                common_standard_chinese_char_third_set,
                keep_character,
                garbage_char,
                ignore_threhold=1000)

    torch.distributed.barrier()

    [
        c_num_char_nums,
        p_num_char_nums,
        t_num_char_nums,
        c_alpha_char_nums,
        p_alpha_char_nums,
        t_alpha_char_nums,
        c_first_char_nums,
        p_first_char_nums,
        t_first_char_nums,
        c_second_char_nums,
        p_second_char_nums,
        t_second_char_nums,
        c_third_char_nums,
        p_third_char_nums,
        t_third_char_nums,
        c_char_nums,
        p_char_nums,
        t_char_nums,
    ] = all_reduce_operation_in_group_for_variables(
        variables=[
            c_num_char_nums, p_num_char_nums, t_num_char_nums,
            c_alpha_char_nums, p_alpha_char_nums, t_alpha_char_nums,
            c_first_char_nums, p_first_char_nums, t_first_char_nums,
            c_second_char_nums, p_second_char_nums, t_second_char_nums,
            c_third_char_nums, p_third_char_nums, t_third_char_nums,
            c_char_nums, p_char_nums, t_char_nums
        ],
        operator=torch.distributed.ReduceOp.SUM,
        group=config.group)

    if p_num_char_nums != 0:
        if t_num_char_nums < ignore_threhold:
            num_lcs_precision = -1
        else:
            num_lcs_precision = c_num_char_nums / float(p_num_char_nums) * 100
    else:
        num_lcs_precision = 0

    if t_num_char_nums != 0:
        if t_num_char_nums < ignore_threhold:
            num_lcs_recall = -1
        else:
            num_lcs_recall = c_num_char_nums / float(t_num_char_nums) * 100
    else:
        num_lcs_recall = 0

    if p_alpha_char_nums != 0:
        if t_alpha_char_nums < ignore_threhold:
            alpha_lcs_precision = -1
        else:
            alpha_lcs_precision = c_alpha_char_nums / float(
                p_alpha_char_nums) * 100
    else:
        alpha_lcs_precision = 0

    if t_alpha_char_nums != 0:
        if t_alpha_char_nums < ignore_threhold:
            alpha_lcs_recall = -1
        else:
            alpha_lcs_recall = c_alpha_char_nums / float(
                t_alpha_char_nums) * 100
    else:
        alpha_lcs_recall = 0

    if p_first_char_nums != 0:
        if t_first_char_nums < ignore_threhold:
            first_lcs_precision = -1
        else:
            first_lcs_precision = c_first_char_nums / float(
                p_first_char_nums) * 100
    else:
        first_lcs_precision = 0

    if t_first_char_nums != 0:
        if t_first_char_nums < ignore_threhold:
            first_lcs_recall = -1
        else:
            first_lcs_recall = c_first_char_nums / float(
                t_first_char_nums) * 100
    else:
        first_lcs_recall = 0

    if p_second_char_nums != 0:
        if t_second_char_nums < ignore_threhold:
            second_lcs_precision = -1
        else:
            second_lcs_precision = c_second_char_nums / float(
                p_second_char_nums) * 100
    else:
        second_lcs_precision = 0

    if t_second_char_nums != 0:
        if t_second_char_nums < ignore_threhold:
            second_lcs_recall = -1
        else:
            second_lcs_recall = c_second_char_nums / float(
                t_second_char_nums) * 100
    else:
        second_lcs_recall = 0

    if p_third_char_nums != 0:
        if t_third_char_nums < ignore_threhold:
            third_lcs_precision = -1
        else:
            third_lcs_precision = c_third_char_nums / float(
                p_third_char_nums) * 100
    else:
        third_lcs_precision = 0

    if t_third_char_nums != 0:
        if t_third_char_nums < ignore_threhold:
            third_lcs_recall = -1
        else:
            third_lcs_recall = c_third_char_nums / float(
                t_third_char_nums) * 100
    else:
        third_lcs_recall = 0

    if p_char_nums != 0:
        if t_char_nums < ignore_threhold:
            lcs_precision = -1
        else:
            lcs_precision = c_char_nums / float(p_char_nums) * 100
    else:
        lcs_precision = 0

    if t_char_nums != 0:
        if t_char_nums < ignore_threhold:
            lcs_recall = -1
        else:
            lcs_recall = c_char_nums / float(t_char_nums) * 100
    else:
        lcs_recall = 0

    lcs_pr_dict = {
        'num_lcs_precision': num_lcs_precision,
        'num_lcs_recall': num_lcs_recall,
        'alpha_lcs_precision': alpha_lcs_precision,
        'alpha_lcs_recall': alpha_lcs_recall,
        'first_lcs_precision': first_lcs_precision,
        'first_lcs_recall': first_lcs_recall,
        'second_lcs_precision': second_lcs_precision,
        'second_lcs_recall': second_lcs_recall,
        'third_lcs_precision': third_lcs_precision,
        'third_lcs_recall': third_lcs_recall,
        'lcs_precision': lcs_precision,
        'lcs_recall': lcs_recall,
    }

    return lcs_pr_dict


def compute_lcs_PR_per_batch(c_num_char_nums,
                             p_num_char_nums,
                             t_num_char_nums,
                             c_alpha_char_nums,
                             p_alpha_char_nums,
                             t_alpha_char_nums,
                             c_first_char_nums,
                             p_first_char_nums,
                             t_first_char_nums,
                             c_second_char_nums,
                             p_second_char_nums,
                             t_second_char_nums,
                             c_third_char_nums,
                             p_third_char_nums,
                             t_third_char_nums,
                             c_char_nums,
                             p_char_nums,
                             t_char_nums,
                             pred_strs,
                             targets,
                             num_chars_set,
                             alpha_chars_set,
                             common_standard_chinese_char_first_set,
                             common_standard_chinese_char_second_set,
                             common_standard_chinese_char_third_set,
                             keep_character,
                             garbage_char,
                             ignore_threhold=1000):

    def get_lcs_dp_matrix(str1, str2):
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp

    keep_character = f'[^{keep_character}]'

    keep_num = "".join(list(num_chars_set))
    keep_num = f'[^{keep_num}]'

    keep_alpha = "".join(list(alpha_chars_set))
    keep_alpha = f'[^{keep_alpha}]'

    keep_first = "".join(list(common_standard_chinese_char_first_set))
    keep_first = f'[^{keep_first}]'

    keep_second = "".join(list(common_standard_chinese_char_second_set))
    keep_second = f'[^{keep_second}]'

    keep_third = "".join(list(common_standard_chinese_char_third_set))
    keep_third = f'[^{keep_third}]'

    for per_pred_str, per_target_str in zip(pred_strs, targets):
        # convert not include char to garbage char
        per_target_convert = ""
        for per_char in per_target_str:
            if per_char not in keep_character:
                per_target_convert += garbage_char
            else:
                per_target_convert += per_char

        per_target_str = per_target_convert

        # remove space
        per_pred_str = per_pred_str.replace(' ', '')
        per_target_str = per_target_str.replace(' ', '')

        if per_target_str == "㍿":
            continue

        if per_target_str == "":
            continue

        per_num_pred_str = re.sub(keep_num, '', per_pred_str)
        per_num_target_str = re.sub(keep_num, '', per_target_str)

        per_alpha_pred_str = re.sub(keep_alpha, '', per_pred_str)
        per_alpha_target_str = re.sub(keep_alpha, '', per_target_str)

        per_first_pred_str = re.sub(keep_first, '', per_pred_str)
        per_first_target_str = re.sub(keep_first, '', per_target_str)

        per_second_pred_str = re.sub(keep_second, '', per_pred_str)
        per_second_target_str = re.sub(keep_second, '', per_target_str)

        per_third_pred_str = re.sub(keep_third, '', per_pred_str)
        per_third_target_str = re.sub(keep_third, '', per_target_str)

        if len(per_pred_str) == 0 or len(per_target_str) == 0:
            c_char_nums += 0
        else:
            dp = get_lcs_dp_matrix(per_pred_str, per_target_str)
            c_char_nums += dp[-1][-1]
        p_char_nums += len(per_pred_str)
        t_char_nums += len(per_target_str)

        if len(per_num_pred_str) == 0 or len(per_num_target_str) == 0:
            c_num_char_nums += 0
        else:
            dp = get_lcs_dp_matrix(per_num_pred_str, per_num_target_str)
            c_num_char_nums += dp[-1][-1]
        p_num_char_nums += len(per_num_pred_str)
        t_num_char_nums += len(per_num_target_str)

        if len(per_alpha_pred_str) == 0 or len(per_alpha_target_str) == 0:
            c_alpha_char_nums += 0
        else:
            dp = get_lcs_dp_matrix(per_alpha_pred_str, per_alpha_target_str)
            c_alpha_char_nums += dp[-1][-1]
        p_alpha_char_nums += len(per_alpha_pred_str)
        t_alpha_char_nums += len(per_alpha_target_str)

        if len(per_first_pred_str) == 0 or len(per_first_target_str) == 0:
            c_first_char_nums += 0
        else:
            dp = get_lcs_dp_matrix(per_first_pred_str, per_first_target_str)
            c_first_char_nums += dp[-1][-1]
        p_first_char_nums += len(per_first_pred_str)
        t_first_char_nums += len(per_first_target_str)

        if len(per_second_pred_str) == 0 or len(per_second_target_str) == 0:
            c_second_char_nums += 0
        else:
            dp = get_lcs_dp_matrix(per_second_pred_str, per_second_target_str)
            c_second_char_nums += dp[-1][-1]
        p_second_char_nums += len(per_second_pred_str)
        t_second_char_nums += len(per_second_target_str)

        if len(per_third_pred_str) == 0 or len(per_third_target_str) == 0:
            c_third_char_nums += 0
        else:
            dp = get_lcs_dp_matrix(per_third_pred_str, per_third_target_str)
            c_third_char_nums += dp[-1][-1]
        p_third_char_nums += len(per_third_pred_str)
        t_third_char_nums += len(per_third_target_str)

    return c_num_char_nums, p_num_char_nums, t_num_char_nums, c_alpha_char_nums, p_alpha_char_nums, t_alpha_char_nums, c_first_char_nums, p_first_char_nums, t_first_char_nums, c_second_char_nums, p_second_char_nums, t_second_char_nums, c_third_char_nums, p_third_char_nums, t_third_char_nums, c_char_nums, p_char_nums, t_char_nums


def train_text_recognition(train_loader, model, criterion, optimizer,
                           scheduler, epoch, logger, config):
    '''
    train semantic segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, targets = data['image'], data['label']
        images = images.cuda()

        trans_targets, target_lengths = config.converter.encode(targets)
        trans_targets, target_lengths = trans_targets.cuda(
        ), target_lengths.cuda()

        if torch.any(torch.isinf(images)):
            continue

        if torch.any(torch.isnan(images)):
            continue

        if torch.sum(images) == 0:
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    # [B,W,num_classes]
                    outputs = model(images)

                    input_lengths = torch.IntTensor([outputs.shape[1]] *
                                                    outputs.shape[0]).cuda()
                    # [B,W,num_classes]->[W,B,num_classes]
                    outputs = outputs.permute(1, 0, 2)

                    loss_value = {}
                    for loss_name in criterion.keys():
                        if loss_name == "CTCLoss":
                            temp_loss = criterion[loss_name](outputs,
                                                             trans_targets,
                                                             input_lengths,
                                                             target_lengths)
                            temp_loss = config.loss_ratio[loss_name] * temp_loss
                            loss_value[loss_name] = temp_loss
                        elif loss_name == 'ACELoss':
                            temp_loss = criterion[loss_name](outputs,
                                                             trans_targets)
                            temp_loss = config.loss_ratio[loss_name] * temp_loss
                            loss_value[loss_name] = temp_loss
                        else:
                            log_info = f'Unsupport loss: {loss_name}'
                            logger.info(log_info) if local_rank == 0 else None
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        # [B,W,num_classes]
                        outputs = model(images)

                        input_lengths = torch.IntTensor(
                            [outputs.shape[1]] * outputs.shape[0]).cuda()
                        # [B,W,num_classes]->[W,B,num_classes]
                        outputs = outputs.permute(1, 0, 2)

                        loss_value = {}
                        for loss_name in criterion.keys():
                            if loss_name == "CTCLoss":
                                temp_loss = criterion[loss_name](
                                    outputs, trans_targets, input_lengths,
                                    target_lengths)
                                temp_loss = config.loss_ratio[
                                    loss_name] * temp_loss
                                loss_value[loss_name] = temp_loss
                            elif loss_name == 'ACELoss':
                                temp_loss = criterion[loss_name](outputs,
                                                                 trans_targets)
                                temp_loss = config.loss_ratio[
                                    loss_name] * temp_loss
                                loss_value[loss_name] = temp_loss
                            else:
                                log_info = f'Unsupport loss: {loss_name}'
                                logger.info(
                                    log_info) if local_rank == 0 else None
        else:
            if iter_index % config.accumulation_steps == 0:
                # [B,W,num_classes]
                outputs = model(images)

                input_lengths = torch.IntTensor([outputs.shape[1]] *
                                                outputs.shape[0]).cuda()
                # [B,W,num_classes]->[W,B,num_classes]
                outputs = outputs.permute(1, 0, 2)

                loss_value = {}
                for loss_name in criterion.keys():
                    if loss_name == "CTCLoss":
                        temp_loss = criterion[loss_name](outputs,
                                                         trans_targets,
                                                         input_lengths,
                                                         target_lengths)
                        temp_loss = config.loss_ratio[loss_name] * temp_loss
                        loss_value[loss_name] = temp_loss
                    elif loss_name == 'ACELoss':
                        temp_loss = criterion[loss_name](outputs,
                                                         trans_targets)
                        temp_loss = config.loss_ratio[loss_name] * temp_loss
                        loss_value[loss_name] = temp_loss
                    else:
                        log_info = f'Unsupport loss: {loss_name}'
                        logger.info(log_info) if local_rank == 0 else None
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    # [B,W,num_classes]
                    outputs = model(images)

                    input_lengths = torch.IntTensor([outputs.shape[1]] *
                                                    outputs.shape[0]).cuda()
                    # [B,W,num_classes]->[W,B,num_classes]
                    outputs = outputs.permute(1, 0, 2)

                    loss_value = {}
                    for loss_name in criterion.keys():
                        if loss_name == "CTCLoss":
                            temp_loss = criterion[loss_name](outputs,
                                                             trans_targets,
                                                             input_lengths,
                                                             target_lengths)
                            temp_loss = config.loss_ratio[loss_name] * temp_loss
                            loss_value[loss_name] = temp_loss
                        elif loss_name == 'ACELoss':
                            temp_loss = criterion[loss_name](outputs,
                                                             trans_targets)
                            temp_loss = config.loss_ratio[loss_name] * temp_loss
                            loss_value[loss_name] = temp_loss
                        else:
                            log_info = f'Unsupport loss: {loss_name}'
                            logger.info(log_info) if local_rank == 0 else None

        loss = 0.
        for key, value in loss_value.items():
            loss += value

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        loss = loss / config.accumulation_steps
        for key, value in loss_value.items():
            loss_value[key] = value / config.accumulation_steps

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                config.scaler.scale(loss).backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    config.scaler.scale(loss).backward()
        else:
            if iter_index % config.accumulation_steps == 0:
                loss.backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    loss.backward()

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
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

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def test_text_detection_for_all_dataset(val_loader_list, model, criterion,
                                        decoder, config):
    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        model = config.ema_model.ema_model

    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader) in enumerate(
            zip(config.val_dataset_name_list, val_loader_list)):
        connect_char = "[+]"
        per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = test_text_detection_for_per_sub_dataset(
            per_sub_dataset_loader, model, criterion, decoder, config)

        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def test_text_detection_for_per_sub_dataset(test_loader, model, criterion,
                                            decoder, config):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precision_recall = PrecisionRecallMeter()

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for i, data in tqdm(enumerate(test_loader)):
            images, annots, scales, sizes = data['image'], data[
                'annots'], data['scale'], data['size']
            if model_on_cuda:
                images = images.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            preds = model(images)

            loss_value = criterion(preds, annots)
            loss = sum(loss_value.values())
            losses.update(loss, images.size(0))

            batch_boxes, batch_scores = decoder(preds, sizes)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            origin_shapes = annots['shape']
            pred_correct_num, gt_correct_num, pred_num, gt_num = compute_text_detection_pr_per_batch(
                batch_boxes, batch_scores, origin_shapes, sizes, config, preds)

            # please keep same variable on different gpus has same data type for all reduce operation
            [pred_correct_num, gt_correct_num, pred_num,
             gt_num] = all_reduce_operation_in_group_for_variables(
                 variables=[
                     pred_correct_num, gt_correct_num, pred_num, gt_num
                 ],
                 operator=torch.distributed.ReduceOp.SUM,
                 group=config.group)

            precision_recall.update(pred_correct_num, gt_correct_num, pred_num,
                                    gt_num)

            end = time.time()

    precision_recall.compute()
    precision = precision_recall.precision * 100
    recall = precision_recall.recall * 100

    f1 = 2. / ((1. / precision) +
               (1. / recall)) if precision != 0 and recall != 0 else 0

    # avg_loss
    test_loss = losses.avg

    # per image data load time(ms) and inference time(ms)
    per_image_load_time = data_time.avg / (config.batch_size //
                                           config.gpus_num) * 1000
    per_image_inference_time = batch_time.avg / (config.batch_size //
                                                 config.gpus_num) * 1000

    result_dict = {
        'per_image_load_time': per_image_load_time,
        'per_image_inference_time': per_image_inference_time,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    return result_dict


def compute_text_detection_pr_per_batch(batch_boxes, batch_scores, shapes,
                                        sizes, config, preds):
    probability_map, threshold_map = preds[:, 0, :, :], preds[:, 1, :, :]
    probability_map, threshold_map = probability_map.cpu().detach().numpy(
    ), threshold_map.cpu().detach().numpy()

    # 计算Precision时若insection_pred_ious大于precision_iou_threshold认为两个框重叠
    # precision_iou_threshold=0.5
    precision_iou_threshold = config.precision_iou_threshold
    # 计算Recall时若insection_target_ious大于recall_iou_threshold认为两个框重叠
    # recall_iou_threshold=0.5
    recall_iou_threshold = config.recall_iou_threshold
    # 一对多/多对一惩罚系数
    # punish_factor=1.0
    punish_factor = config.punish_factor
    # 一对多和多对一时若有match_count_threshold个框与一个框的iou都大于阈值，则执行一对多或多对一逻辑
    # match_count_threshold=2
    match_count_threshold = config.match_count_threshold

    pred_correct_num, gt_correct_num, pred_num, gt_num = 0.0, 0.0, 0.0, 0.0
    for per_image_pred_boxes, per_image_shapes, per_image_size, per_image_probability_map, per_image_threshold_map in zip(
            batch_boxes, shapes, sizes, probability_map, threshold_map):
        per_image_gt_boxes = [
            per_shape['points'] for per_shape in per_image_shapes
        ]
        insection_pred_ious, insection_target_ious = compute_pred_gt_ious(
            per_image_pred_boxes, per_image_gt_boxes, per_image_size)

        per_image_pred_correct_num, per_image_gt_correct_num = 0.0, 0.0
        per_image_pred_num, per_image_gt_num = len(per_image_pred_boxes), len(
            per_image_gt_boxes)

        pred_boxes_flag = np.zeros((len(per_image_pred_boxes), ),
                                   dtype=np.float32)
        gt_boxes_flag = np.zeros((len(per_image_gt_boxes), ), dtype=np.float32)
        '''
        ###########################################################################
        统计参与指标计算的pred_box数量和gt_box数量说明:
        1. gt_num = gt_box总量 - 被标记为ignore的gt_box数量
        2. pred_num = pred_box总量 - 判定为预测标签为ignore的gt_box的pred_box数量,
        判定为ignore的pred_box情况j较为复杂,只有当pred_box与标有ignore的gt_box有对应关系时,
        才会被记作ignore
        3. pred_box与gt_box对应关系有三种: 一对一、一对多、多对一,在这三种情况中,
        统计与ignore标签gt_box对应的pred_box
        ###########################################################################
        '''

        # 一对一匹配统计
        per_image_pred_correct_num, per_image_gt_correct_num, pred_boxes_flag, gt_boxes_flag, one2one_per_image_pred_ignore_nums = one_to_one_match_count(
            insection_pred_ious,
            insection_target_ious,
            per_image_pred_correct_num,
            per_image_gt_correct_num,
            per_image_shapes,
            pred_boxes_flag,
            gt_boxes_flag,
            precision_iou_threshold,
            recall_iou_threshold,
        )

        # 一个GT与多个预测匹配统计
        per_image_pred_correct_num, per_image_gt_correct_num, pred_boxes_flag, gt_boxes_flag, one2many_per_image_pred_ignore_nums = one_to_many_match_count(
            insection_pred_ious,
            insection_target_ious,
            per_image_pred_correct_num,
            per_image_gt_correct_num,
            per_image_shapes,
            pred_boxes_flag,
            gt_boxes_flag,
            precision_iou_threshold,
            recall_iou_threshold,
            punish_factor,
            match_count_threshold,
        )

        # 一个预测与多个GT匹配统计
        per_image_pred_correct_num, per_image_gt_correct_num, pred_boxes_flag, gt_boxes_flag, many2one_per_image_pred_ignore_nums = many_to_one_match_count(
            insection_pred_ious,
            insection_target_ious,
            per_image_pred_correct_num,
            per_image_gt_correct_num,
            per_image_shapes,
            pred_boxes_flag,
            gt_boxes_flag,
            precision_iou_threshold,
            recall_iou_threshold,
            punish_factor,
            match_count_threshold,
        )

        # 获取所有标为'ignore'的gt_boxes的索引
        per_image_gt_ignore_boxes_indexes = []
        for gt_idx in range(len(per_image_shapes)):
            if 'ignore' in per_image_shapes[gt_idx].keys():
                if per_image_shapes[gt_idx]['ignore']:
                    per_image_gt_ignore_boxes_indexes.append(1)
                else:
                    per_image_gt_ignore_boxes_indexes.append(0)
            else:
                if per_image_shapes[gt_idx]['label'] == '###':
                    per_image_gt_ignore_boxes_indexes.append(1)
                else:
                    per_image_gt_ignore_boxes_indexes.append(0)
        per_image_gt_ignore_boxes_indexes = np.array(
            per_image_gt_ignore_boxes_indexes, dtype=np.float32)

        # 若经过以上三次匹配后仍然有标为'ignore'的gt_boxes未与pred_boxes匹配,统计在这种情况下,预测框忽略的数量
        remaining_pred_ignore_box_nums = 0
        # 求未匹配的标为'ignore'的gt_boxes
        gt_ignore_remain_box_flag = np.logical_and(
            (1 - gt_boxes_flag), per_image_gt_ignore_boxes_indexes)

        # 三个条件需同时满足：
        # 1.经过1对1、1对多、多对1后，仍有gt_box未匹配
        # 2.经过1对1、1对多、多对1后，仍有N个gt_box未匹配，且在N个gt中有被标记为ignore的box
        # 3.经过1对1、1对多、多对1后，仍有pred_box未匹配
        if (gt_boxes_flag.shape[0] > gt_boxes_flag.sum()) and (
                pred_boxes_flag.shape[0] > pred_boxes_flag.sum()) and (
                    gt_ignore_remain_box_flag.sum() > 0):
            for pred_idx in range(len(pred_boxes_flag)):
                for gt_idx in range(len(gt_boxes_flag)):
                    if gt_ignore_remain_box_flag[gt_idx]:
                        if (1 - pred_boxes_flag[pred_idx]) and (
                                insection_target_ious[gt_idx, pred_idx] > 0
                                and insection_pred_ious[gt_idx, pred_idx] > 0):
                            remaining_pred_ignore_box_nums += 1
                            break

        per_image_pred_correct_num = float(int(per_image_pred_correct_num))
        per_image_gt_correct_num = float(int(per_image_gt_correct_num))
        per_image_pred_num = float(int(per_image_pred_num))
        per_image_gt_num = float(int(per_image_gt_num))

        per_image_pred_num = per_image_pred_num - one2one_per_image_pred_ignore_nums - one2many_per_image_pred_ignore_nums - many2one_per_image_pred_ignore_nums - remaining_pred_ignore_box_nums
        per_image_gt_num = per_image_gt_num - sum(
            per_image_gt_ignore_boxes_indexes)

        # 修正,避免出现precision/recall大于1的情况
        per_image_pred_num = per_image_pred_correct_num if per_image_pred_correct_num > per_image_pred_num else per_image_pred_num
        per_image_gt_num = per_image_gt_correct_num if per_image_gt_correct_num > per_image_gt_num else per_image_gt_num

        pred_correct_num += per_image_pred_correct_num
        gt_correct_num += per_image_gt_correct_num
        pred_num += per_image_pred_num
        gt_num += per_image_gt_num

        pred_correct_num = float(int(pred_correct_num))
        gt_correct_num = float(int(gt_correct_num))
        pred_num = float(int(pred_num))
        gt_num = float(int(gt_num))

    return pred_correct_num, gt_correct_num, pred_num, gt_num


def one_to_one_match_count(insection_pred_ious, insection_target_ious,
                           per_image_pred_correct_num,
                           per_image_gt_correct_num, per_image_shapes,
                           pred_boxes_flag, gt_boxes_flag,
                           precision_iou_threshold, recall_iou_threshold):
    per_image_pred_ignore_nums, per_image_gt_ignore_nums = 0, 0

    for gt_idx in range(len(per_image_shapes)):
        # 对单个gt_box,获取insection_target_ious对应整行中大于阈值的pred_box索引
        target_iou_match_gt_pred_idxs = np.where(
            insection_target_ious[gt_idx, :] > recall_iou_threshold)[0]
        # 若索引数量不是1,则不属于一对一情况,continue
        if target_iou_match_gt_pred_idxs.shape[0] != 1:
            continue
        # 对刚才找到的唯一pred_box索引,获取insection_target_ious对应整列中大于阈值的gt_box索引
        target_iou_match_pred_gt_idxs = np.where(
            insection_target_ious[:, target_iou_match_gt_pred_idxs[0]] >
            recall_iou_threshold)[0]
        # 若索引数量不是1,则不属于一对一情况,continue
        if target_iou_match_pred_gt_idxs.shape[0] != 1:
            continue
        # 对单个gt_box,获取insection_pred_ious对应整行中大于阈值的pred_box索引
        pred_iou_match_gt_pred_idxs = np.where(
            insection_pred_ious[gt_idx, :] > precision_iou_threshold)[0]
        # 若索引数量不是1,则不属于一对一情况,continue
        if pred_iou_match_gt_pred_idxs.shape[0] != 1:
            continue
        # 对刚才找到的唯一pred_box索引,获取insection_pred_ious对应整列中大于阈值的gt_box索引
        pred_iou_match_pred_gt_idxs = np.where(
            insection_pred_ious[:, pred_iou_match_gt_pred_idxs[0]] >
            precision_iou_threshold)[0]
        # 若索引数量不是1,则不属于一对一情况,continue
        if pred_iou_match_pred_gt_idxs.shape[0] != 1:
            continue

        # 一对一情况要求在insection_target_ious、insection_pred_ious中IOU均大于阈值,且匹配的gt_box和pred_box必须是一对一
        if 'ignore' in per_image_shapes[gt_idx].keys():
            if per_image_shapes[gt_idx]['ignore']:
                per_image_gt_ignore_nums += 1
                per_image_pred_ignore_nums += 1
            else:
                per_image_gt_correct_num += 1
                per_image_pred_correct_num += 1
        else:
            if per_image_shapes[gt_idx]['label'] == '###':
                per_image_gt_ignore_nums += 1
                per_image_pred_ignore_nums += 1
            else:
                per_image_gt_correct_num += 1
                per_image_pred_correct_num += 1

        gt_boxes_flag[gt_idx] = 1
        pred_boxes_flag[target_iou_match_gt_pred_idxs[0]] = 1

    return per_image_pred_correct_num, per_image_gt_correct_num, pred_boxes_flag, gt_boxes_flag, per_image_pred_ignore_nums


def one_to_many_match_count(insection_pred_ious, insection_target_ious,
                            per_image_pred_correct_num,
                            per_image_gt_correct_num, per_image_shapes,
                            pred_boxes_flag, gt_boxes_flag,
                            precision_iou_threshold, recall_iou_threshold,
                            punish_factor, match_count_threshold):
    per_image_gt_ignore_nums, per_image_pred_ignore_nums = 0, 0

    for gt_idx in range(len(per_image_shapes)):
        # 若gt已经在前面匹配过,则跳过
        if gt_boxes_flag[gt_idx] == 1:
            continue
        # 对单个gt_box,获取insection_target_ious对应整行中大于阈值的pred_box索引
        # 当一个gt匹配多个pred时，gt面积往往较大且gt与pred交集面积较小，故insection_target_ious仅要求iou大于0即可
        target_iou_match_gt_pred_idxs = np.where(
            insection_target_ious[gt_idx, :] > 0)[0]
        # 若pred_box索引数量小于match_count_threshold,则认为不是一对多情况
        if target_iou_match_gt_pred_idxs.shape[0] < match_count_threshold:
            continue
        # 对单个gt_box,获取insection_pred_ious对应整行中大于阈值的pred_box索引
        # 当一个gt匹配多个pred时,pred与gt交集面积/pred面积往往较大,故insection_pred_ious要求大于precision_iou_threshold
        pred_iou_match_gt_pred_idxs = np.where(
            (insection_pred_ious[gt_idx, :] > precision_iou_threshold)
            & (pred_boxes_flag == 0))[0]
        pred_iou_match_gt_pred_nums = pred_iou_match_gt_pred_idxs.shape[0]
        # 若pred_box索引数量小于1,则认为不是一对多情况
        if pred_iou_match_gt_pred_nums < 1:
            continue

        if pred_iou_match_gt_pred_nums == 1:
            # 如果insection_pred_ious中一个gt只匹配一个pred,则看看insection_target_ious中同一个gt是否也只匹配一个pred,如果是,转为1对1关系
            if (insection_pred_ious[gt_idx, pred_iou_match_gt_pred_idxs[0]] >
                    precision_iou_threshold) and (
                        insection_target_ious[gt_idx,
                                              pred_iou_match_gt_pred_idxs[0]] >
                        recall_iou_threshold):
                if 'ignore' in per_image_shapes[gt_idx].keys():
                    if per_image_shapes[gt_idx]['ignore']:
                        per_image_gt_ignore_nums += 1
                        per_image_pred_ignore_nums += 1
                    else:
                        per_image_gt_correct_num += 1
                        per_image_pred_correct_num += 1
                else:
                    if per_image_shapes[gt_idx]['label'] == '###':
                        per_image_gt_ignore_nums += 1
                        per_image_pred_ignore_nums += 1
                    else:
                        per_image_gt_correct_num += 1
                        per_image_pred_correct_num += 1

                gt_boxes_flag[gt_idx] = 1
                # 注意此时pred_iou_match_gt_pred_idxs索引只有一个
                pred_boxes_flag[pred_iou_match_gt_pred_idxs[0]] = 1
        # 若 insection_pred_ious中一个gt匹配上多个pred
        # 将N个pred的insection_target_ious值相加看是否大于阈值,若是,则符合一对多关系
        elif (np.sum(insection_target_ious[gt_idx,
                                           pred_iou_match_gt_pred_idxs]) >
              recall_iou_threshold):
            if 'ignore' in per_image_shapes[gt_idx].keys():
                if per_image_shapes[gt_idx]['ignore']:
                    per_image_gt_ignore_nums += punish_factor
                    per_image_pred_ignore_nums += (
                        pred_iou_match_gt_pred_nums * punish_factor)
                else:
                    per_image_gt_correct_num += punish_factor
                    per_image_pred_correct_num += (
                        pred_iou_match_gt_pred_nums * punish_factor)
            else:
                if per_image_shapes[gt_idx]['label'] == '###':
                    per_image_gt_ignore_nums += punish_factor
                    per_image_pred_ignore_nums += (
                        pred_iou_match_gt_pred_nums * punish_factor)
                else:
                    per_image_gt_correct_num += punish_factor
                    per_image_pred_correct_num += (
                        pred_iou_match_gt_pred_nums * punish_factor)

            gt_boxes_flag[gt_idx] = 1
            # 注意此时pred_iou_match_gt_pred_idxs有多个pred_box索引
            pred_boxes_flag[pred_iou_match_gt_pred_idxs] = 1

    return per_image_pred_correct_num, per_image_gt_correct_num, pred_boxes_flag, gt_boxes_flag, per_image_pred_ignore_nums


def many_to_one_match_count(insection_pred_ious, insection_target_ious,
                            per_image_pred_correct_num,
                            per_image_gt_correct_num, per_image_shapes,
                            pred_boxes_flag, gt_boxes_flag,
                            precision_iou_threshold, recall_iou_threshold,
                            punish_factor, match_count_threshold):
    per_image_gt_ignore_nums, per_image_pred_ignore_nums = 0, 0

    for pred_idx in range(pred_boxes_flag.shape[0]):
        # 若pred已经在前面匹配过,则跳过
        if pred_boxes_flag[pred_idx] == 1:
            continue
        # 对单个pred_box,获取insection_target_ious对应整行中大于阈值的gt_box索引
        # 当一个pred匹配多个gt时，pred面积往往较大且gt与pred交集面积较小，故insection_pred_ious仅要求iou大于0即可
        pred_iou_match_pred_gt_idxs = np.where(
            insection_pred_ious[:, pred_idx] > 0)[0]
        # 若gt_box索引数量小于match_count_threshold,则认为不是一对多情况
        if pred_iou_match_pred_gt_idxs.shape[0] < match_count_threshold:
            continue
        # 对单个pred_box,获取insection_target_ious对应整行中大于阈值的gt_box索引
        # 当一个pred匹配多个gt时,gt与pred交集面积/gt面积往往较大,故insection_target_ious要求大于precision_iou_threshold
        target_iou_match_pred_gt_idxs = np.where(
            (insection_target_ious[:, pred_idx] > precision_iou_threshold)
            & (gt_boxes_flag == 0))[0]
        target_iou_match_pred_gt_nums = target_iou_match_pred_gt_idxs.shape[0]
        # 若gt_box索引数量小于1,则认为不是一对多情况
        if target_iou_match_pred_gt_nums < 1:
            continue

        if target_iou_match_pred_gt_nums == 1:
            # 如果insection_pred_ious中一个pred只匹配一个gt,则看看insection_target_ious中同一个pred是否也只匹配一个gt,如果是,转为1对1关系
            if ((insection_pred_ious[target_iou_match_pred_gt_idxs[0],
                                     pred_idx] > precision_iou_threshold) and
                (insection_target_ious[target_iou_match_pred_gt_idxs[0],
                                       pred_idx] > recall_iou_threshold)):
                if 'ignore' in per_image_shapes[
                        target_iou_match_pred_gt_idxs[0]].keys():
                    if per_image_shapes[
                            target_iou_match_pred_gt_idxs[0]]['ignore']:
                        per_image_gt_ignore_nums += 1
                        per_image_pred_ignore_nums += 1
                    else:
                        per_image_gt_correct_num += 1
                        per_image_pred_correct_num += 1
                else:
                    if per_image_shapes[target_iou_match_pred_gt_idxs[0]][
                            'label'] == '###':
                        per_image_gt_ignore_nums += 1
                        per_image_pred_ignore_nums += 1
                    else:
                        per_image_gt_correct_num += 1
                        per_image_pred_correct_num += 1

                # 注意此时target_iou_match_pred_gt_idxs索引只有一个
                gt_boxes_flag[target_iou_match_pred_gt_idxs[0]] = 1
                pred_boxes_flag[pred_idx] = 1
        elif (np.sum(insection_pred_ious[target_iou_match_pred_gt_idxs,
                                         pred_idx]) > precision_iou_threshold):
            pred_boxes_flag[pred_idx] = 1
            # 注意此时target_iou_match_pred_gt_idxs索引不止一个
            gt_boxes_flag[target_iou_match_pred_gt_idxs] = 1

            # 多个gt对应一个pred时，每个gt是否为ignore
            gt_ignore_flags = []
            for gt_idx in target_iou_match_pred_gt_idxs:
                if 'ignore' in per_image_shapes[gt_idx].keys():
                    if per_image_shapes[gt_idx]['ignore']:
                        gt_ignore_flags.append(1)
                    else:
                        gt_ignore_flags.append(0)
                else:
                    if per_image_shapes[gt_idx]['label'] == '###':
                        gt_ignore_flags.append(1)
                    else:
                        gt_ignore_flags.append(0)

            # 判断多个gt是否全为ignore
            if sum(gt_ignore_flags) == len(target_iou_match_pred_gt_idxs):
                per_image_gt_ignore_nums += len(target_iou_match_pred_gt_idxs)
                per_image_pred_ignore_nums += 1
            else:
                per_image_gt_correct_num += (
                    target_iou_match_pred_gt_nums * punish_factor -
                    sum(gt_ignore_flags))
                per_image_pred_correct_num += punish_factor
                per_image_gt_ignore_nums += sum(gt_ignore_flags)

    return per_image_pred_correct_num, per_image_gt_correct_num, pred_boxes_flag, gt_boxes_flag, per_image_pred_ignore_nums


def compute_pred_gt_ious(pred_boxes, gt_boxes, size):
    insection_pred_ious = np.zeros((len(gt_boxes), len(pred_boxes)),
                                   dtype=np.float32)
    insection_target_ious = np.zeros((len(gt_boxes), len(pred_boxes)),
                                     dtype=np.float32)
    h, w = size[0], size[1]

    for gt_idx, per_gt_box in enumerate(gt_boxes):
        gt_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(gt_mask, [per_gt_box.astype(np.int32)], 1.0)
        for pred_idx, per_pred_box in enumerate(pred_boxes):
            pred_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(pred_mask, [per_pred_box.astype(np.int32)], 1.0)
            insection_mask = gt_mask * pred_mask
            insection_pred_ious[gt_idx][pred_idx] = insection_mask.sum() / (
                pred_mask.sum() + 1e-4)
            insection_target_ious[gt_idx][pred_idx] = insection_mask.sum() / (
                gt_mask.sum() + 1e-4)

    return insection_pred_ious, insection_target_ious


def train_text_detection(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train semantic segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, targets = data['image'], data['annots']
        images = images.cuda()

        if torch.any(torch.isinf(images)):
            continue

        if torch.any(torch.isnan(images)):
            continue

        if torch.sum(images) == 0:
            continue

        if config.use_amp:
            with autocast():
                if iter_index % config.accumulation_steps == 0:
                    preds = model(images)
                    loss_value = criterion(preds, targets)
                else:
                    # not reduce gradient while iter_index % config.accumulation_steps != 0
                    with model.no_sync():
                        preds = model(images)
                        loss_value = criterion(preds, targets)
        else:
            if iter_index % config.accumulation_steps == 0:
                preds = model(images)
                loss_value = criterion(preds, targets)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    preds = model(images)
                    loss_value = criterion(preds, targets)

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        loss = loss / config.accumulation_steps
        for key, value in loss_value.items():
            loss_value[key] = value / config.accumulation_steps

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                config.scaler.scale(loss).backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    config.scaler.scale(loss).backward()
        else:
            if iter_index % config.accumulation_steps == 0:
                loss.backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    loss.backward()

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if hasattr(config,
                           'clip_max_norm') and config.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
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

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss
