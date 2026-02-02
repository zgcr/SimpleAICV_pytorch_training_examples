import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import numpy as np

from ocr_const_value import OcrConstValue

TAG = "RectificationCurve："
MIN_DISTANCE = 1.0


def distance(x1, y1, x2, y2):
    """
    计算两点之间的欧氏距离
    """
    distance = np.float32(((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))**0.5)

    return distance


def LineOrientationEstimate(coords):
    """
    判断是不是竖排文本
    Args:
        coords：文本框坐标点
    Return:
        isVertical: 是否是竖排文本
    """
    isVertical = False
    x1 = (coords[0][0] + coords[len(coords) - 1][0]) / 2.0
    y1 = (coords[0][1] + coords[len(coords) - 1][1]) / 2.0
    x2 = (coords[len(coords) // 2 - 1][0] + coords[len(coords) // 2][0]) / 2.0
    y2 = (coords[len(coords) // 2 - 1][1] + coords[len(coords) // 2][1]) / 2.0
    estHeight = distance(x1, y1, x2, y2)

    x3 = (coords[0][0] + coords[len(coords) // 2 - 1][0]) / 2.0
    y3 = (coords[0][1] + coords[len(coords) // 2 - 1][1]) / 2.0
    x4 = (coords[len(coords) // 2][0] + coords[len(coords) - 1][0]) / 2.0
    y4 = (coords[len(coords) // 2][1] + coords[len(coords) - 1][1]) / 2.0
    estWidth = distance(x3, y3, x4, y4)

    if estHeight >= estWidth:
        isVertical = True if (abs(y2 - y1) > abs(x2 - x1)) else False
    else:
        isVertical = True if (abs(y4 - y3) > abs(x4 - x3)) else False

    if isVertical and len(coords) == 4:
        heightWidthRatio = 0.0
        if estHeight >= estWidth and estWidth > 0:
            heightWidthRatio = estHeight / estWidth
        elif estHeight < estWidth and estHeight > 0:
            heightWidthRatio = estWidth / estHeight
        print(TAG + "heightWidthRatio is: ", heightWidthRatio)
        thrHeightWidthRatio = 1.78
        if heightWidthRatio < thrHeightWidthRatio:
            isVertical = False

    return isVertical


def RectificationCurve(srcImage, candidateBboxes, imageName,
                       isNeedDetectModel):
    """
    文本框校正
    Args:
        srcImage：原始图片
        candidateBboxes: 候选框
    Return:
        candidateBboxes：所有坐标点
        boxWidth：矩形宽
        boxHeight：矩形高
    """
    print(TAG + "start RectificationCurve...")
    linesHorizontal = []
    horizontalBboxes = []
    index = 0
    for candidateBbox in candidateBboxes:
        line = {}
        line['confidence'] = candidateBbox['score']
        line['coords'] = candidateBbox['candidateBbox']
        if not LineOrientationEstimate(line['coords']):
            line['lineId'] = index
            index += 1
            linesHorizontal.append(line)
            horizontalBboxes.append(candidateBbox)

    candidateBboxes = horizontalBboxes
    windowHeight, windowWidth, _ = srcImage.shape
    image = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    targetImageHeight = 32
    scale = 1.0
    rectResult = {}
    rectResult['candidateBboxes'] = []
    rectResult['linesData'] = []
    rectResult['linesH'] = []
    rectResult['linesW'] = []
    rectResult['linesHorizontal'] = []
    for i in range(0, len(candidateBboxes)):
        line_coords = candidateBboxes[i]['candidateBbox']
        dstImage, dstH, dstW = GetLineImg(image, windowWidth, windowHeight,
                                          line_coords, targetImageHeight,
                                          scale, imageName, i,
                                          isNeedDetectModel)

        if len(dstImage) > 0:
            rectResult['candidateBboxes'].append(candidateBboxes[i])
            rectResult['linesData'].append(dstImage)
            rectResult['linesH'].append(dstH)
            rectResult['linesW'].append(dstW)
            rectResult['linesHorizontal'].append(linesHorizontal[i])
    return rectResult


def GetLineImg(sourceImage, sourceHeight, sourceWidth, lineCoords,
               targetImageHeight, scale, imageName, index, isNeedDetectModel):
    print(TAG + "start GetLineImg...")
    nBox = (int)((len(lineCoords) / 2) - 1)
    lineImg = []
    dstH = []
    dstW = []
    hei = 0
    wei = 0
    for j in range(0, nBox):
        x0 = lineCoords[j][0]
        y0 = lineCoords[j][1]
        x1 = lineCoords[j + 1][0]
        y1 = lineCoords[j + 1][1]
        x2 = lineCoords[OcrConstValue.TWO_INT *
                        (nBox + OcrConstValue.ONE_INT) - j -
                        OcrConstValue.TWO_INT][0]
        y2 = lineCoords[OcrConstValue.TWO_INT *
                        (nBox + OcrConstValue.ONE_INT) - j -
                        OcrConstValue.TWO_INT][1]
        x3 = lineCoords[OcrConstValue.TWO_INT *
                        (nBox + OcrConstValue.ONE_INT) - j -
                        OcrConstValue.ONE_INT][0]
        y3 = lineCoords[OcrConstValue.TWO_INT *
                        (nBox + OcrConstValue.ONE_INT) - j -
                        OcrConstValue.ONE_INT][1]
        hei += distance(x0, y0, x3, y3)
        wei += distance(x0, y0, x1, y1) + distance(x2, y2, x3, y3)
    hei /= nBox
    wei /= nBox * 2
    if hei < MIN_DISTANCE:
        return lineImg, dstH, dstW
    if wei < MIN_DISTANCE:
        return lineImg, dstH, dstW
    for j in range(0, nBox):
        coords = []
        coords.append(lineCoords[j])
        coords.append(lineCoords[j + 1])
        coords.append(
            lineCoords[OcrConstValue.TWO_INT * (nBox + OcrConstValue.ONE_INT) -
                       j - OcrConstValue.TWO_INT])
        coords.append(
            lineCoords[OcrConstValue.TWO_INT * (nBox + OcrConstValue.ONE_INT) -
                       j - OcrConstValue.ONE_INT])
        if ((coords[0][0] == coords[1][0]) and
            (coords[1][0] == coords[2][0]) and
            (coords[2][0] == coords[3][0])) or (
                (coords[0][1] == coords[1][1]) and
                (coords[1][1] == coords[2][1]) and
                (coords[2][1] == coords[3][1])):
            continue
        if ((abs(coords[0][0] - coords[1][0]) <= 1) and
            (abs(coords[0][1] - coords[1][1]) <= 1)) or (
                (abs(coords[1][0] - coords[2][0]) <= 1) and
                (abs(coords[1][1] - coords[2][1]) <= 1)) or (
                    (abs(coords[2][0] - coords[3][0]) <= 1) and
                    (abs(coords[2][1] - coords[3][1]) <= 1)) or (
                        (abs(coords[3][0] - coords[0][0]) <= 1) and
                        (abs(coords[3][1] - coords[0][1]) <= 1)):
            continue
        dstImage, dstH_, dstW_ = RectificationGray(sourceImage, coords, hei,
                                                   wei)
        if nBox > 1:
            isVertical = 0
        else:
            isVertical = is_vertical(hei, wei)
        if isNeedDetectModel:
            imageResize, dstH_, dstW_ = NormalizeLineImg(
                dstImage, dstH_, dstW_, targetImageHeight, isVertical, scale)
        else:
            imageResize = dstImage
        lineImg.append(imageResize)
        dstH.append(dstH_)
        dstW.append(dstW_)
    imageConcatH, lineH, lineW = ConcatImgH(lineImg, dstH, dstW)

    return imageConcatH, lineH, lineW


def ConcatImgH(srcImgs, srcHs, srcWs):
    dstImg = []
    dstH = 0
    dstW = 0
    if len(srcImgs) > 1:
        dstH = srcHs[0]
        for i in range(1, len(srcHs)):
            if srcHs[i] != dstH:
                print(TAG +
                      "Concat image failed! srcH[%d] no equal srcHs[0]!" % (i))
                return dstImg, dstH, dstW
        for i in range(0, len(srcWs)):
            dstW += srcWs[i]
        dstImgBufSize = dstH * dstW
        if dstImgBufSize == 0:
            return dstImg, dstH, dstW
        imageConcatH = np.zeros((dstH, dstW))
        colInd = 0
        for i in range(0, len(srcImgs)):
            imageConcatH[:, colInd:colInd + srcWs[i]] = srcImgs[i]
            colInd += srcWs[i]
        return imageConcatH, dstH, dstW
    else:
        if (len(srcImgs) == 0) or (len(srcHs) == 0) or (len(srcWs) == 0):
            return dstImg, dstH, dstW
        return srcImgs[0], srcHs[0], srcWs[0]


def RectificationGray(srcImage, line, hei, wei):
    srcTri = np.float32(line)
    maxWidth = wei
    maxHeight = hei
    desWid = maxWidth
    desHei = maxHeight
    zone = desWid * desHei
    desImage = []
    if zone <= 0:
        return desImage

    dstTri = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight],
                         [0, maxHeight]])
    desImage = PerspectiveTransform(srcImage, srcTri, dstTri)
    desImage = desImage[0:np.int32(maxHeight), 0:np.int32(maxWidth)]
    return desImage, np.int32(desHei), np.int32(desWid)


def PerspectiveTransform(srcImage, srcPoints, dstPoints):
    M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    height, width = srcImage.shape
    out_image = cv2.warpPerspective(srcImage, M, (height, width))
    return out_image


def is_vertical(hei, wid):
    if wid == 0:
        return 1
    isVertical = 0
    if (1.0 * hei / wid) > 1.5:
        isVertical = 1
    return isVertical


def NormalizeLineImg(image, height, width, targetSize, direction, lineScale):
    imageResize = []
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    dstW = 0
    dstH = 0
    if direction == 0:
        aspecOri = lineScale * width / height
        if aspecOri * targetSize <= 0:
            return imageResize
        width = int(aspecOri * targetSize + 0.5)
        if targetSize * width <= 0:
            return imageResize
        imageResize = cv2.resize(image, (width, targetSize), 0, 0,
                                 cv2.INTER_AREA)
        dstW = width
        dstH = targetSize
    else:
        aspecOri = lineScale * height / width
        if aspecOri * targetSize <= 0:
            return imageResize
        height = int(aspecOri * targetSize + 0.5)
        if targetSize * height <= 0:
            return imageResize
        imageResize = cv2.resize(image, (targetSize, height), 0, 0,
                                 cv2.INTER_AREA)
        dstW = targetSize
        dstH = height
    return imageResize, dstH, dstW
