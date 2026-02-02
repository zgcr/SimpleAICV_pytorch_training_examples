import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import functools
import numpy as np
import time

from numba import jit

from ocr_const_value import OcrConstValue
from text_detect_post_process_curve_line import find_rs_polygons
from rectification_curve_line import RectificationCurve


class PostProcess:

    def __init__(self,
                 hard_border_threshold=None,
                 box_score_threshold=0.5,
                 min_area_size=9,
                 max_box_num=1000,
                 rectangle_similarity=0.6,
                 min_box_size=3,
                 line_text_expand_ratio=1.2,
                 curve_text_expand_ratio=1.5):
        self.hard_border_threshold = hard_border_threshold
        self.box_score_threshold = box_score_threshold
        self.min_area_size = min_area_size
        self.max_box_num = max_box_num
        self.rectangle_similarity = rectangle_similarity
        self.min_box_size = min_box_size
        self.line_text_expand_ratio = line_text_expand_ratio
        self.curve_text_expand_ratio = curve_text_expand_ratio

    def __call__(self, preds, ori_image, image_name, isNeedDetectModel):
        if isNeedDetectModel:
            probability_map, threshold_map = preds
            probability_map, threshold_map = probability_map.cpu().detach(
            ).numpy(), threshold_map.cpu().detach().numpy()
            probability_map, threshold_map = np.squeeze(probability_map,
                                                        axis=1), np.squeeze(
                                                            threshold_map,
                                                            axis=1)

            binary_map = np.where(
                probability_map > self.hard_border_threshold, 1.0,
                0.0) if self.hard_border_threshold else np.where(
                    probability_map > threshold_map, 1.0, 0.0)
            probability_map = np.squeeze(probability_map, axis=0)

            minVal = probability_map.min()
            maxVal = probability_map.max()

            probability_map = (probability_map -
                               minVal) * OcrConstValue.PIXEL_MAX_VALUE_F / (
                                   maxVal - minVal)

            binary_map = np.squeeze(binary_map, axis=0)

            ori_height, ori_width, _ = ori_image.shape
            outputHeight = 960
            outputWidth = 960
        else:
            probability_map, binary_map = preds
            ori_height, ori_width, _ = ori_image.shape
            outputHeight, outputWidth, _ = ori_image.shape
        ratio = min(
            float(outputWidth) / float(ori_width),
            float(outputHeight) / float(ori_height))
        newWidth = int(ratio * ori_width)
        newHeight = int(ratio * ori_height)
        heightPadding = outputHeight - newHeight
        widthPadding = outputWidth - newWidth
        padding = [
            float(widthPadding),
            float(heightPadding),
            float(newWidth) / float(ori_width),
            float(newHeight) / float(ori_height)
        ]
        scale_param = {
            'srcHeight': ori_height,
            'srcWidth': ori_width,
            'dstHeight': outputHeight,
            'dstWidth': outputWidth,
            'ratioHeight': ori_height * 1.0 / outputHeight,
            'ratioWidth': ori_width * 1.0 / outputWidth,
            'padding': padding
        }

        boxScoreThresh = 0.5
        thresholdPara = {
            "boxScoreThresh": boxScoreThresh,
            "rectangle_similarity": self.rectangle_similarity,
            "min_area_size": self.min_area_size,
            "line_text_expand_ratio": self.line_text_expand_ratio,
            "curve_text_expand_ratio": self.curve_text_expand_ratio
        }
        candidateBboxes = find_rs_polygons(probability_map, binary_map,
                                           scale_param, thresholdPara,
                                           image_name, isNeedDetectModel)
        return candidateBboxes


def GetTwoPointDist(point1, point2):
    return ((1.0 * (point1[0] - point2[0]) * (point1[0] - point2[0]) + 1.0 *
             (point1[1] - point2[1]) * (point1[1] - point2[1]))**0.5)


def cmp1(a, b):
    if a[1] >= b[1]:
        return 1
    else:
        return -1


def cmp2(a, b):
    if a[0] >= b[0]:
        return 1
    else:
        return -1


def FindEndPoint(box):
    cosList = []
    x = np.zeros(2)
    y = np.zeros(2)
    for i in range(0, len(box)):
        if i == 0:
            pPre = box[len(box) - 1]
        else:
            pPre = box[i - 1]
        point = box[i]
        if i == (len(box) - 1):
            pNext = box[0]
        else:
            pNext = box[i + 1]
        x[0] = point[0] - pPre[0]
        x[1] = point[1] - pPre[1]
        y[0] = point[0] - pNext[0]
        y[1] = point[1] - pNext[1]
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        cosList.append([i, x.dot(y) / (Lx * Ly)])
    cosList = sorted(cosList, key=functools.cmp_to_key(cmp1))
    endPoint = cosList[-4:]
    endPoint = sorted(endPoint, key=functools.cmp_to_key(cmp2))

    if (endPoint[2][0] - endPoint[1][0]) == 1:
        if (box[endPoint[2][0]][0] > box[endPoint[0][0]][0]):
            if (box[endPoint[2][0]][1] > box[endPoint[1][0]][1]):
                rightTopPoint = box[endPoint[1][0]]
                rightDownPoint = box[endPoint[2][0]]
            else:
                rightTopPoint = box[endPoint[2][0]]
                rightDownPoint = box[endPoint[1][0]]
            if (box[endPoint[3][0]][1] > box[endPoint[0][0]][1]):
                leftTopPoint = box[endPoint[0][0]]
                leftDownPoint = box[endPoint[3][0]]
            else:
                leftTopPoint = box[endPoint[3][0]]
                leftDownPoint = box[endPoint[0][0]]
        else:
            if (box[endPoint[2][0]][1] > box[endPoint[1][0]][1]):
                leftTopPoint = box[endPoint[1][0]]
                leftDownPoint = box[endPoint[2][0]]
            else:
                leftTopPoint = box[endPoint[2][0]]
                leftDownPoint = box[endPoint[1][0]]
            if (box[endPoint[3][0]][1] > box[endPoint[0][0]][1]):
                rightTopPoint = box[endPoint[0][0]]
                rightDownPoint = box[endPoint[3][0]]
            else:
                rightTopPoint = box[endPoint[3][0]]
                rightDownPoint = box[endPoint[0][0]]
    else:
        if (box[endPoint[2][0]][0] > box[endPoint[0][0]][0]):
            if (box[endPoint[3][0]][1] > box[endPoint[2][0]][1]):
                rightTopPoint = box[endPoint[2][0]]
                rightDownPoint = box[endPoint[3][0]]
            else:
                rightTopPoint = box[endPoint[3][0]]
                rightDownPoint = box[endPoint[2][0]]
            if (box[endPoint[1][0]][1] > box[endPoint[0][0]][1]):
                leftTopPoint = box[endPoint[0][0]]
                leftDownPoint = box[endPoint[1][0]]
            else:
                leftTopPoint = box[endPoint[1][0]]
                leftDownPoint = box[endPoint[0][0]]
        else:
            if (box[endPoint[3][0]][1] > box[endPoint[2][0]][1]):
                leftTopPoint = box[endPoint[2][0]]
                leftDownPoint = box[endPoint[3][0]]
            else:
                leftTopPoint = box[endPoint[3][0]]
                leftDownPoint = box[endPoint[2][0]]
            if (box[endPoint[1][0]][1] > box[endPoint[0][0]][1]):
                rightTopPoint = box[endPoint[0][0]]
                rightDownPoint = box[endPoint[1][0]]
            else:
                rightTopPoint = box[endPoint[1][0]]
                rightDownPoint = box[endPoint[0][0]]
    return leftTopPoint, leftDownPoint, rightTopPoint, rightDownPoint


@jit()
def FillCicle(fillMask, leftCenter0, leftCenter1, rightCenter0, rightCenter1,
              leftRadius, rightRadius, VectorLeft0, VectorLeft1, VectorRight0,
              VectorRight1):
    MaskH, MaskW = fillMask.shape
    for i in range(0, MaskW):
        for j in range(0, MaskH):
            point = [i, j]
            distLeft = (1.0 * (point[0] - leftCenter0) *
                        (point[0] - leftCenter0) + 1.0 *
                        (point[1] - leftCenter1) *
                        (point[1] - leftCenter1))**0.5
            distRight = (1.0 * (point[0] - rightCenter0) *
                         (point[0] - rightCenter0) + 1.0 *
                         (point[1] - rightCenter1) *
                         (point[1] - rightCenter1))**0.5
            if distLeft <= leftRadius:
                # vector = [i - leftCenter0, j - leftCenter1]
                # result = VectorLeft0 * vector[1] - vector[0] * VectorLeft1
                # if result <= 0:
                fillMask[j, i] = 1
            if distRight <= rightRadius:
                # vector = [i - rightCenter0, j - rightCenter1]
                # result = VectorRight0 * vector[1] - vector[0] * VectorRight1
                # if result >= 0:
                fillMask[j, i] = 1


def GtToBezierBoxes(image, box, postProcess, keep_image_name):

    imageH, imageW, _ = image.shape
    fillMask = np.zeros((imageH, imageW), dtype=np.float32)
    cv2.fillPoly(fillMask, [box], 1.0)

    leftTopPoint, leftDownPoint, rightTopPoint, rightDownPoint = FindEndPoint(
        box)

    leftCenter = [(leftTopPoint[0] + leftDownPoint[0]) // 2,
                  (leftTopPoint[1] + leftDownPoint[1]) // 2]
    rightCenter = [(rightTopPoint[0] + rightDownPoint[0]) // 2,
                   (rightTopPoint[1] + rightDownPoint[1]) // 2]

    leftRadius = GetTwoPointDist(leftTopPoint, leftDownPoint) / 2
    rightRadius = GetTwoPointDist(rightTopPoint, rightDownPoint) / 2

    VectorLeft = [
        leftTopPoint[0] - leftCenter[0], leftTopPoint[1] - leftCenter[1]
    ]
    VectorRight = [
        rightTopPoint[0] - rightCenter[0], rightTopPoint[1] - rightCenter[1]
    ]

    startTime = time.time()
    FillCicle(fillMask, leftCenter[0], leftCenter[1], rightCenter[0],
              rightCenter[1], leftRadius, rightRadius, VectorLeft[0],
              VectorLeft[1], VectorRight[0], VectorRight[1])
    endTime = time.time()

    print("fill circle cost:", endTime - startTime)

    candidateBboxes = postProcess([fillMask * 255, fillMask], image,
                                  keep_image_name, False)

    # 图像校正
    RectificationResult = RectificationCurve(image, candidateBboxes,
                                             keep_image_name, False)

    if len(RectificationResult['linesData']) > 0:
        return RectificationResult['linesData'][0]
    else:
        return None
