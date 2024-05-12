import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import copy
import cv2
import numpy as np
import pyclipper

from ocr_const_value import OcrConstValue

TAG = "TextDetectDBProcStraight: "


def distance(x1, y1, x2, y2):
    """
    计算两点之间的欧氏距离
    """
    distance = np.float32(((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))**0.5)
    return distance


def UnClip(minBox, perimeter, unClipRatio):

    distance = (unClipRatio * cv2.contourArea(minBox)) / perimeter
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(minBox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    outBox = offset.Execute(distance)
    outBox = np.squeeze(outBox)
    return outBox


def GetMinBox(contour):
    """
    计算最小面积外接矩形
    Args:
        contour: 轮廓
    Return:
        minBox：外接矩形坐标，顺时针排列
        boxWidth：矩形宽
        boxHeight：矩形高
    """
    textRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(textRect)
    # 对x坐标进行从小到大排序
    x_sorted = box[np.argsort(box[:, 0]), :]
    index1, index2, index3, index4 = 0, 0, 0, 0
    if x_sorted[1][1] > x_sorted[0][1]:
        index1 = 0
        index4 = 1
    else:
        index1 = 1
        index4 = 0
    if x_sorted[3][1] > x_sorted[2][1]:
        index2 = 2
        index3 = 3
    else:
        index2 = 3
        index3 = 2
    minBox = [
        x_sorted[index1], x_sorted[index2], x_sorted[index3], x_sorted[index4]
    ]
    boxWidth, boxHeight = textRect[1]
    return minBox, boxWidth, boxHeight


def BoxScoreFast(probability, contour):
    """
    计算轮廓区域的分数
    Args:
        probability: 概率图
        contour：轮廓
    Return:
        textBox：文本坐标框
    """
    height, width = probability.shape

    box = copy.deepcopy(contour)
    box = np.squeeze(box)

    maskMat = np.zeros((height, width), dtype=np.int32)
    cv2.fillPoly(maskMat, [box.astype(np.int32)], 1.0)
    area = maskMat.sum()
    score = (probability * maskMat).sum() / area
    return score


def get_length(box):
    """
    计算四边形周长
    Args:
        minBox: 四边形四点坐标
    Return:
        perimeter：周长
    """
    sideOne = distance(box[0][0], box[0][1], box[1][0], box[1][1])
    sideTwo = distance(box[1][0], box[1][1], box[2][0], box[2][1])
    sideThree = distance(box[2][0], box[2][1], box[3][0], box[3][1])
    sideFour = distance(box[0][0], box[0][1], box[3][0], box[3][1])
    perimeter = sideOne + sideTwo + sideThree + sideFour
    return perimeter


def GetBoxFromContour(probability, scaleParam, boxScoreThresh, unClipRatio,
                      contour, minBox, boxWidth, boxHeight):
    """
    
    Args:
        probability: 概率图
        scaleParam：resize参数
        boxScoreThresh：box阈值分数
        unClipRatio：扩框倍数
        contour：轮廓
        minBox：轮廓对应的最小面积外界矩形
        boxWidth：矩形框
        boxHeight：矩形高
    Return:
        textBox：文本坐标框
    """
    textBox = {}
    minSize = 3
    # 计算概率图分数
    score = BoxScoreFast(probability,
                         contour) / OcrConstValue.PIXEL_MAX_VALUE_F
    if score < boxScoreThresh:
        print(TAG + "boxScore < boxScoreThresh, boxScore is:", score)
        return {'resultCode': OcrConstValue.FAIL}
    # 计算原图尺寸
    for i in range(0, len(minBox)):
        minBox[i][0] = minBox[i][0] / scaleParam['padding'][2]
        minBox[i][0] = (min)((max)(minBox[i][0], 0),
                             scaleParam['srcWidth'] - 1)

        minBox[i][1] = minBox[i][1] / scaleParam['padding'][3]
        minBox[i][1] = (min)((max)(minBox[i][1], 0),
                             scaleParam['srcHeight'] - 1)

    boxArea = cv2.contourArea(minBox)
    if boxArea < 10:
        print(TAG + "minBox Area < 10, minBox Area is:", boxArea)
        return {'resultCode': OcrConstValue.FAIL}

    # 扩框
    perimeter = get_length(minBox)
    clipBox = UnClip(minBox, perimeter, unClipRatio)
    clipBox = np.array(clipBox, dtype=np.float32)
    if len(clipBox) == 0:
        return {'resultCode': OcrConstValue.FAIL}
    clipMinBox, clipMinBoxWidth, clipMinBoxHeight = GetMinBox(clipBox)
    if min(clipMinBoxWidth, clipMinBoxHeight) < minSize + 2:
        print(TAG + "min(clipMinBoxWidth, clipMinBoxHeight) < minSize + 2")
        return {'resultCode': OcrConstValue.FAIL}

    for i in range(0, len(clipMinBox)):
        clipMinBox[i][0] = (min)((max)(clipMinBox[i][0], 0),
                                 scaleParam['srcWidth'] - 1)
        clipMinBox[i][1] = (min)((max)(clipMinBox[i][1], 0),
                                 scaleParam['srcHeight'] - 1)

    textBox = {
        "resultCode": OcrConstValue.SUCCESSFUL,
        "candidateBbox": clipMinBox,
        "score": score
    }
    return textBox
