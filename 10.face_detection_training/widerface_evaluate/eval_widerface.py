import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import cv2
import collections
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

import torch
import torch.nn as nn

from tools.utils import get_logger, set_seed

FACE_CLASSES = [
    'face',
]

FACE_CLASSES_COLOR = [
    (0, 255, 0),
]


def compute_bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = ((boxes[n, 2] - boxes[n, 0] + 1) *
                          (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps


def get_gt_boxes_from_mat(config):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(config.gt_mat_path)
    hard_mat = loadmat(config.hard_mat_path)
    medium_mat = loadmat(config.medium_mat_path)
    easy_mat = loadmat(config.easy_mat_path)

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def norm_pred_box_score(pred):
    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score) / diff

    return pred


def eval_per_image(pred, gt, ignore, iou_threshold=0.5):
    preds = pred.copy()
    gts = gt.copy()
    pred_recall = np.zeros(preds.shape[0])
    recall_list = np.zeros(gts.shape[0])
    proposal_list = np.ones(preds.shape[0])

    preds[:, 2] = preds[:, 2] + preds[:, 0]
    preds[:, 3] = preds[:, 3] + preds[:, 1]
    gts[:, 2] = gts[:, 2] + gts[:, 0]
    gts[:, 3] = gts[:, 3] + gts[:, 1]

    overlaps = compute_bbox_overlaps(preds[:, :4], gts)

    for h in range(preds.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_threshold:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list


def compute_per_image_pr_info(thresh_num, pred_info, proposal_list,
                              pred_recall):
    per_image_pr_info = np.zeros((thresh_num, 2)).astype(np.float32)
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            per_image_pr_info[t, 0] = 0
            per_image_pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            per_image_pr_info[t, 0] = len(p_index)
            per_image_pr_info[t, 1] = pred_recall[r_index]

    return per_image_pr_info


def compute_per_dataset_pr_info(thresh_num, pr_curve, face_count):
    per_dataset_pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        per_dataset_pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        per_dataset_pr_curve[i, 1] = pr_curve[i, 1] / face_count

    return per_dataset_pr_curve


def compute_voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def compute_ap_metrics(pred_result_dict, config):
    pred_result_dict = norm_pred_box_score(pred_result_dict)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes_from_mat(
        config)

    event_num = len(event_list)
    thresh_num = 1000
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

    aps = []
    for setting_id in range(3):
        face_count = 0
        per_setting_pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)
        gt_list = setting_gts[setting_id]

        # ['easy', 'medium', 'hard']
        for i in tqdm(range(event_num)):
            event_name = str(event_list[i][0][0])

            image_list = file_list[i][0]
            pred_list = pred_result_dict[event_name]
            image_gt_list = gt_list[i][0]
            image_gt_bbx_list = facebox_list[i][0]

            for j in range(len(image_list)):
                per_image_name = str(image_list[j][0][0]) + '.jpg'
                per_image_pred_info = pred_list[per_image_name]

                per_image_gt_boxes = image_gt_bbx_list[j][0].astype(np.float32)
                keep_index = image_gt_list[j][0]
                face_count += len(keep_index)

                if len(per_image_gt_boxes) == 0 or len(
                        per_image_pred_info) == 0:
                    continue

                ignore = np.zeros(per_image_gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                per_image_pred_recall, per_image_proposal_list = eval_per_image(
                    per_image_pred_info,
                    per_image_gt_boxes,
                    ignore,
                    iou_threshold=0.5)

                per_image_pr_info = compute_per_image_pr_info(
                    thresh_num, per_image_pred_info, per_image_proposal_list,
                    per_image_pred_recall)
                per_setting_pr_curve += per_image_pr_info

        per_setting_pr_curve = compute_per_dataset_pr_info(
            thresh_num, per_setting_pr_curve, face_count)

        per_setting_propose = per_setting_pr_curve[:, 0]
        per_setting_recall = per_setting_pr_curve[:, 1]

        ap = compute_voc_ap(per_setting_recall, per_setting_propose)
        aps.append(ap)

    return aps


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Face Detection Testing')
    parser.add_argument('--work-dir',
                        type=str,
                        help='path for get testing config')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from eval_config import config
    log_dir = os.path.join(args.work_dir, 'log')

    set_seed(config.seed)

    os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    logger = get_logger('test', log_dir)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in ['model']:
                log_info = f'{key}: {value}'
                logger.info(log_info)

    model = config.model
    decoder = config.decoder

    model = model.cuda()
    model = model.eval()

    eval_image_path_list = []
    for per_event_name in os.listdir(config.eval_image_dir):
        per_event_image_dir = os.path.join(config.eval_image_dir,
                                           per_event_name)
        for per_image_name in os.listdir(per_event_image_dir):
            if '.jpg' in per_image_name:
                per_image_path = os.path.join(per_event_image_dir,
                                              per_image_name)
                eval_image_path_list.append(
                    [per_event_name, per_image_name, per_image_path])

    print(f'eval image num:{len(eval_image_path_list)}')

    with torch.no_grad():
        eval_image_result_dict = collections.OrderedDict()
        for per_event_name, per_image_name, per_image_path in tqdm(
                eval_image_path_list):
            per_image = cv2.imdecode(
                np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_BGR2RGB)
            per_image = per_image.astype(np.float32)

            per_origin_image = per_image.copy()
            per_origin_image = cv2.cvtColor(per_origin_image,
                                            cv2.COLOR_RGB2BGR)
            per_image_origin_h, per_image_origin_w = per_origin_image.shape[
                0], per_origin_image.shape[1]

            resize_factor = config.input_image_size[0] / max(
                per_image_origin_h, per_image_origin_w)
            resize_h, resize_w = int(round(
                per_image_origin_h * resize_factor)), int(
                    round(per_image_origin_w * resize_factor))
            per_image = cv2.resize(per_image, (resize_w, resize_h))

            per_image = per_image / 255.

            per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

            input_image = np.zeros(
                (1, config.input_image_size[0], config.input_image_size[0], 3),
                dtype=np.float32)
            input_image[0, 0:per_image_h, 0:per_image_w, :] = per_image
            input_image = torch.from_numpy(input_image)
            # 1 H W 3 ->1 3 H W
            input_image = input_image.permute(0, 3, 1, 2)
            input_image = input_image.cuda()

            per_image_out = model(input_image)

            per_image_pred_scores, per_image_pred_classes, per_image_pred_boxes = decoder(
                per_image_out)
            per_image_pred_scores, per_image_pred_classes, per_image_pred_boxes = per_image_pred_scores[
                0], per_image_pred_classes[0], per_image_pred_boxes[0]

            per_image_pred_boxes = per_image_pred_boxes / resize_factor

            per_image_pred_scores = per_image_pred_scores[
                per_image_pred_classes > -1]
            per_image_pred_boxes = per_image_pred_boxes[per_image_pred_classes
                                                        > -1]
            per_image_pred_classes = per_image_pred_classes[
                per_image_pred_classes > -1]

            # clip boxes
            per_image_pred_boxes[:, 0] = np.maximum(per_image_pred_boxes[:, 0],
                                                    0)
            per_image_pred_boxes[:, 1] = np.maximum(per_image_pred_boxes[:, 1],
                                                    0)
            per_image_pred_boxes[:, 2] = np.minimum(per_image_pred_boxes[:, 2],
                                                    per_image_origin_w)
            per_image_pred_boxes[:, 3] = np.minimum(per_image_pred_boxes[:, 3],
                                                    per_image_origin_h)

            if per_event_name not in eval_image_result_dict.keys():
                eval_image_result_dict[
                    per_event_name] = collections.OrderedDict()
            eval_image_result_dict[per_event_name][per_image_name] = []
            for per_box_score, per_box in zip(per_image_pred_scores,
                                              per_image_pred_boxes):
                per_box = per_box.astype(np.int32).tolist()
                per_box_score = per_box_score.astype(np.float32)

                x = int(per_box[0])
                y = int(per_box[1])
                w = int(per_box[2]) - int(per_box[0])
                h = int(per_box[3]) - int(per_box[1])
                confidence = round(per_box_score, 3)

                eval_image_result_dict[per_event_name][per_image_name].append(
                    [x, y, w, h, confidence])
            eval_image_result_dict[per_event_name][per_image_name] = np.array(
                eval_image_result_dict[per_event_name][per_image_name],
                dtype=np.float32)

            if config.save_image_result:
                save_event_image_dir = os.path.join(config.save_image_dir,
                                                    per_event_name)
                if not os.path.exists(save_event_image_dir):
                    os.makedirs(save_event_image_dir)

                # draw all label boxes
                for per_box_score, per_box in zip(per_image_pred_scores,
                                                  per_image_pred_boxes):
                    per_box = per_box.astype(np.int32)
                    class_color = FACE_CLASSES_COLOR[0]

                    left_top, right_bottom = (per_box[0],
                                              per_box[1]), (per_box[2],
                                                            per_box[3])

                    cv2.rectangle(per_origin_image,
                                  left_top,
                                  right_bottom,
                                  color=class_color,
                                  thickness=1,
                                  lineType=cv2.LINE_AA)

                    text = f'{per_box_score:.3f}'
                    text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
                    fill_right_bottom = (max(left_top[0] + text_size[0],
                                             right_bottom[0]),
                                         left_top[1] - text_size[1] - 3)
                    cv2.rectangle(per_origin_image,
                                  left_top,
                                  fill_right_bottom,
                                  color=class_color,
                                  thickness=-1,
                                  lineType=cv2.LINE_AA)
                    cv2.putText(per_origin_image,
                                text, (left_top[0], left_top[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color=(0, 0, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA)

                cv2.imencode('.jpg', per_origin_image)[1].tofile(
                    os.path.join(save_event_image_dir, f'{per_image_name}'))

        # compute metric
        aps = compute_ap_metrics(eval_image_result_dict, config)

        log_info = f'Easy   Val AP: {aps[0]}'
        logger.info(log_info)
        log_info = f'Medium Val AP: {aps[1]}'
        logger.info(log_info)
        log_info = f'Hard   Val AP: {aps[2]}'
        logger.info(log_info)

    return


if __name__ == '__main__':
    main()
