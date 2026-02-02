import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import cv2
import copy
import functools
import numpy as np
import time

from math import fabs, atan, cos
from numba import jit

from ocr_const_value import OcrConstValue
from text_detect_post_process_straight_line import GetBoxFromContour, BoxScoreFast, GetMinBox, UnClip

TAG = "TextDetectDBpostprocCurve："
PIXEL_MAX_VALUE = 255


@jit()
def thin_image(Mat):
    """
    提取骨架点
    Args:
        Mat：二值图 * 255
    Return:
        skeletonMat：细化后的骨架图
        skeletonPoints：骨架点坐标
    """
    row, col = Mat.shape
    deleteList = np.zeros((row, col))
    neighbourhood = np.zeros(9)
    inOddIterations = True
    while True:
        count = 0
        for i in range(1, row - 1):
            dataLast = Mat[i - 1]
            data = Mat[i]
            dataNext = Mat[i + 1]
            for j in range(1, col - 1):
                if data[j] == PIXEL_MAX_VALUE:
                    whitePointCount = 0
                    neighbourhood[0] = 1
                    if dataLast[j] == PIXEL_MAX_VALUE:
                        neighbourhood[1] = 1
                    else:
                        neighbourhood[1] = 0
                    if dataLast[j + 1] == PIXEL_MAX_VALUE:
                        neighbourhood[2] = 1
                    else:
                        neighbourhood[2] = 0
                    if data[j + 1] == PIXEL_MAX_VALUE:
                        neighbourhood[3] = 1
                    else:
                        neighbourhood[3] = 0
                    if dataNext[j + 1] == PIXEL_MAX_VALUE:
                        neighbourhood[4] = 1
                    else:
                        neighbourhood[4] = 0
                    if dataNext[j] == PIXEL_MAX_VALUE:
                        neighbourhood[5] = 1
                    else:
                        neighbourhood[5] = 0
                    if dataNext[j - 1] == PIXEL_MAX_VALUE:
                        neighbourhood[6] = 1
                    else:
                        neighbourhood[6] = 0
                    if data[j - 1] == PIXEL_MAX_VALUE:
                        neighbourhood[7] = 1
                    else:
                        neighbourhood[7] = 0
                    if dataLast[j - 1] == PIXEL_MAX_VALUE:
                        neighbourhood[8] = 1
                    else:
                        neighbourhood[8] = 0
                    whitePointCount = sum(neighbourhood) - neighbourhood[0]
                    if whitePointCount >= 2 and whitePointCount <= 6:
                        ap = 0
                        if neighbourhood[1] == 0 and neighbourhood[2] == 1:
                            ap += 1
                        if neighbourhood[2] == 0 and neighbourhood[3] == 1:
                            ap += 1
                        if neighbourhood[3] == 0 and neighbourhood[4] == 1:
                            ap += 1
                        if neighbourhood[4] == 0 and neighbourhood[5] == 1:
                            ap += 1
                        if neighbourhood[5] == 0 and neighbourhood[6] == 1:
                            ap += 1
                        if neighbourhood[6] == 0 and neighbourhood[7] == 1:
                            ap += 1
                        if neighbourhood[7] == 0 and neighbourhood[8] == 1:
                            ap += 1
                        if neighbourhood[8] == 0 and neighbourhood[1] == 1:
                            ap += 1
                        if ap == 1:
                            if ((inOddIterations == True)
                                    and (neighbourhood[1] * neighbourhood[3] *
                                         neighbourhood[5] == 0)
                                    and (neighbourhood[3] * neighbourhood[5] *
                                         neighbourhood[7] == 0)):
                                # deleteList.append([i, j])
                                deleteList[i, j] = 1
                                count += 1
                            elif ((inOddIterations == False)
                                  and (neighbourhood[1] * neighbourhood[3] *
                                       neighbourhood[7] == 0)
                                  and (neighbourhood[1] * neighbourhood[5] *
                                       neighbourhood[7] == 0)):
                                # deleteList.append([i, j])
                                deleteList[i, j] = 1
                                count += 1

        if count == 0:
            break
        for i in range(0, row):
            for j in range(0, col):
                if deleteList[i][j] == 1:
                    Mat[i][j] = 0
        deleteList.fill(0)
        if inOddIterations is True:
            inOddIterations = False
        else:
            inOddIterations = True


def get_skeleton(skeletonMat):
    """
    提取骨架点
    Args:
        skeletonMat：二值图
    Return:
        skeletonMat：细化后的骨架图
        skeletonPoints：骨架点坐标
    """
    thin_image(skeletonMat)


def find_rs_polygons(probability, binary, scaleParam, thresholdPara, imageName,
                     isNeedDetectModel):
    """
    提取文本框
    Args:
        probability: 概率图
        binary：二值图
        scaleParam：所有缩放参数
        boxScoreThresh：阈值 0.5
    Return:
        candidateBboxes：所有坐标点
        boxWidth：矩形宽
        boxHeight：矩形高
    """

    print(TAG + "start FindRsPolygons")

    # 加上腐蚀膨胀，解决小范围粘连问题
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    binary = cv2.erode(src=binary, kernel=horizontalStructure)
    binary = cv2.dilate(src=binary, kernel=horizontalStructure)
    del horizontalStructure
    binaryH, binaryW = binary.shape

    # 获取轮廓
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(TAG + "contours size is ", len(contours))

    startTime = time.time()
    # 获取全图骨架点

    binary = binary * 255
    skeletonMat = binary.astype(np.uint8)
    get_skeleton(skeletonMat)
    skeletonPoints = np.argwhere(skeletonMat > 0)
    tmp = np.zeros_like(skeletonPoints)
    tmp[:, 0], tmp[:, 1] = skeletonPoints[:, 1], skeletonPoints[:, 0]
    skeletonPoints = tmp
    skeletonPoints = skeletonPoints.tolist()

    endTime = time.time()
    print(TAG + "get_skeleton spend time:", endTime - startTime)

    candidateBboxes = []
    contourIndex = 0
    for contour in contours:
        contourArea = cv2.contourArea(contour)
        if contourArea < thresholdPara['min_area_size']:
            continue
        # 获取轮廓最小面积外接矩形
        minBox, boxWidth, boxHeight = GetMinBox(contour)
        minBox = np.array(minBox)
        boxArea = cv2.contourArea(minBox)
        if boxHeight < 0.0001:
            continue
        startTime = time.time()
        areaBorder = thresholdPara['rectangle_similarity']
        boxScoreThresh = thresholdPara['boxScoreThresh']
        line_text_expand_ratio = thresholdPara['line_text_expand_ratio']
        curve_text_expand_ratio = thresholdPara['curve_text_expand_ratio']
        if not isNeedDetectModel:
            print(TAG + "GetPolygonsFromContour, isNeedDetectModel：",
                  isNeedDetectModel)
            result = GetPolygonsFromContour(probability, scaleParam,
                                            boxScoreThresh,
                                            curve_text_expand_ratio, contour,
                                            skeletonPoints, skeletonMat,
                                            binary, imageName,
                                            isNeedDetectModel)
            if result['resultCode'] == OcrConstValue.SUCCESSFUL:
                candidateBboxes.append(result)
            contourIndex += 1
            continue
        if (contourArea / boxArea >= areaBorder) or (boxArea < 1000):
            print(TAG + "GetBoxFromContour")
            result = GetBoxFromContour(probability, scaleParam, boxScoreThresh,
                                       line_text_expand_ratio, contour, minBox,
                                       boxWidth, boxHeight)
        else:
            print(TAG + "GetPolygonsFromContour, isNeedDetectModel：",
                  isNeedDetectModel)
            pMin = [binaryW, binaryH]
            pMax = [0, 0]
            for j in range(0, len(contour)):
                if contour[j][0][0] < pMin[0]:
                    pMin[0] = contour[j][0][0]
                if contour[j][0][1] < pMin[1]:
                    pMin[1] = contour[j][0][1]
                if contour[j][0][0] > pMax[0]:
                    pMax[0] = contour[j][0][0]
                if contour[j][0][1] > pMax[1]:
                    pMax[1] = contour[j][0][1]
            width = pMax[0] - pMin[0]
            height = pMax[1] - pMin[1]
            # k = cv2.isContourConvex(contour)
            hullsI = cv2.convexHull(contour,
                                    clockwise=False,
                                    returnPoints=False)
            hull = cv2.convexHull(contour, clockwise=False)
            hullsI = np.sort(hullsI, axis=None)
            if len(hullsI) > 3:
                defects = cv2.convexityDefects(contour, hullsI)

            if len(defects) > 0:
                maxDis = 0
                maxInd = 0
                for j in range(0, len(defects)):
                    d = defects[j][0][3]
                    fp = defects[j][0][2]
                    if d > maxDis:
                        maxDis = d
                        maxInd = j
                startPointInd = defects[maxInd][0][0]
                endPointsInd = defects[maxInd][0][1]
                disInd = defects[maxInd][0][2]
                point1 = contour[startPointInd]
                point2 = contour[endPointsInd]
                far = contour[disInd]
                pointHull = []
                tmp = []
                if point1[0][0] > point2[0][0]:
                    tmp = point2
                    point2 = point1
                    point1 = tmp
                a = point2[0][1] - point1[0][1]
                b = point2[0][0] - point1[0][0]
                c = point2[0][0] * point1[0][1] - point1[0][0] * point2[0][1]
                x = (b * b * far[0][0] - a * b * far[0][1] - a * c) / (a * a +
                                                                       b * b)
                y = (-a * b * far[0][0] + a * a * far[0][1] - b * c) / (a * a +
                                                                        b * b)
                MAX_SLOPE = 100
                if (x >= point1[0][0] and x <= point2[0][0]):
                    pointHull.append(x)
                    pointHull.append(y)
                else:
                    pointHull = point1 if x < point1[0][0] else point2
                if far[0][0] != pointHull[0][0]:
                    slope = (far[0][1] - pointHull[1]) / (far[0][0] -
                                                          pointHull[0])
                else:
                    slope = MAX_SLOPE
                maxDis = (pow((far[0][0] - pointHull[0]), 2) + pow(
                    (far[0][1] - pointHull[1]), 2))**0.5
                if ((maxDis /
                     (pMax[0] - pMin[0])) > 0.4) and (fabs(slope) < 0.6) and (
                         (pMax[0] - pMin[0]) > (pMax[1] - pMin[1])):
                    imgDraw = np.zeros((width, height), 0)
                    imgDraw = cv2.drawContours(image=imgDraw,
                                               contours=contour,
                                               contourIdx=-1,
                                               thickness=0,
                                               lineType=0)
                    LINE_THICKNESS = 5
                    imgDraw = cv2.line(imgDraw, (far[0] - pMin[0], 0),
                                       (far[0] - pMin[0], pMax[1] - pMin[1]),
                                       0, LINE_THICKNESS)
                    result = FindRsPolygonsCropped(probability, imgDraw, pMin,
                                                   scaleParam, boxScoreThresh,
                                                   thresholdPara, imageName,
                                                   isNeedDetectModel)
                else:
                    result = GetPolygonsFromContour(
                        probability, scaleParam, boxScoreThresh,
                        curve_text_expand_ratio, contour, skeletonPoints,
                        skeletonMat, binary, imageName, isNeedDetectModel)

            else:
                result = GetPolygonsFromContour(probability, scaleParam,
                                                boxScoreThresh,
                                                curve_text_expand_ratio,
                                                contour, skeletonPoints,
                                                skeletonMat, binary, imageName,
                                                isNeedDetectModel)
        if result['resultCode'] == OcrConstValue.SUCCESSFUL:
            candidateBboxes.append(result)
        contourIndex += 1

    return candidateBboxes


def GetPolygonsFromContour(probability, scaleParam, boxScoreThresh,
                           unClipRatio, contour, skeletonPoints, skeletonMat,
                           binary, imageName, isNeedDetectModel):
    minSize = 3
    probabilityH, probabilityW = probability.shape

    # 轮廓折线化
    epsilon = 0.001 * cv2.arcLength(contour, True)
    polygonPoints = cv2.approxPolyDP(contour, epsilon, True)

    if len(polygonPoints) < OcrConstValue.FOUR_INT:
        return {'resultCode': OcrConstValue.FAIL}
    polygonPoints = np.squeeze(polygonPoints)

    # 计算轮廓置信度
    score = BoxScoreFast(probability,
                         contour) / OcrConstValue.PIXEL_MAX_VALUE_F

    if score < boxScoreThresh:
        return {'resultCode': OcrConstValue.FAIL}

    if isNeedDetectModel:
        # 扩框
        clipBox = UnClip(polygonPoints, cv2.arcLength(polygonPoints, True),
                         unClipRatio)
    else:
        clipBox = polygonPoints[::-1]

    if len(clipBox) < OcrConstValue.FOUR_INT:
        return {'resultCode': OcrConstValue.FAIL}
    # 最小外接矩形
    clipMinBox, boxWidth, _ = GetMinBox(clipBox)
    if boxWidth < (minSize + OcrConstValue.TWO_INT):
        return {'resultCode': OcrConstValue.FAIL}

    # 防止出边界
    for j in range(0, len(clipBox)):
        clipBox[j][0] = (min)((max)(clipBox[j][0], 0),
                              scaleParam['dstWidth'] - 1)
        clipBox[j][1] = (min)((max)(clipBox[j][1], 0),
                              scaleParam['dstHeight'] - 1)
    startTime = time.time()

    # 找到当前轮廓的骨架点且去除了毛刺
    contourSkeletonPoints, endPoints = GetContourSkeleton(
        skeletonPoints, contour, skeletonMat, imageName)

    if len(contourSkeletonPoints) <= 0:
        print(TAG + "contourSkeletonPoints is empty")
        return {'resultCode': OcrConstValue.FAIL}

    endTime = time.time()
    print(TAG + "GetContourSkeleton time is: %f ms." % (endTime - startTime))

    # 还原成原始尺寸
    RestoreSize(clipBox, scaleParam)
    RestoreSize(contourSkeletonPoints, scaleParam)

    boxArea = cv2.contourArea(clipBox)
    if boxArea < OcrConstValue.TEN_INT:
        return {'resultCode': OcrConstValue.FAIL}

    # 计算骨架点的bezier拟合曲线
    bezierPoints = []
    res = GetBezierPointOfOneLine(contourSkeletonPoints, bezierPoints)
    if res != OcrConstValue.SUCCESSFUL:
        print(TAG + "GetBezierPointOfOneLine error")
        return {'resultCode': OcrConstValue.FAIL}

    # 获取上下边界
    topSidePoints = []
    downSidePoints = []
    res += GetPointsOfTopAndDownSide(clipBox, bezierPoints, scaleParam,
                                     topSidePoints, downSidePoints, imageName,
                                     isNeedDetectModel)
    if res != OcrConstValue.SUCCESSFUL:
        print(TAG + "GetPointsOfTopAndDownSide error")
        return {'resultCode': OcrConstValue.FAIL}

    # 过滤所有多边形框，看有无自相交的不合法多边形，如果有说明标注不合法
    res += CheckPolygonValidity(topSidePoints, downSidePoints)
    if res != OcrConstValue.SUCCESSFUL:
        print(TAG + "CheckPolygonValidity error")
        return {'resultCode': OcrConstValue.FAIL}

    # 将上下边界组合在一起
    candidateBbox = []
    ConvertOneSidePointToCandidateBbox(topSidePoints, downSidePoints,
                                       candidateBbox)
    textBox = {
        'resultCode': OcrConstValue.SUCCESSFUL,
        "candidateBbox": candidateBbox,
        "score": score
    }
    return textBox


def ConvertOneSidePointToCandidateBbox(topSidePoints, downSidePoints,
                                       candidateBbox):
    pointNum = len(topSidePoints)
    if pointNum < OcrConstValue.FOUR_INT:
        print(TAG + "No candidateBboxes")
        return
    polygon = np.zeros((pointNum * 2, 2), dtype=np.float32)
    for i in range(0, pointNum):
        polygon[i][0] = (float)(topSidePoints[i][0])
        polygon[i][1] = (float)(topSidePoints[i][1])
        polygon[pointNum * OcrConstValue.TWO_INT -
                (pointNum - i)][0] = (float)(downSidePoints[i][0])
        polygon[pointNum * OcrConstValue.TWO_INT -
                (pointNum - i)][1] = (float)(downSidePoints[i][1])
    for i in range(0, len(polygon)):
        candidateBbox.append(polygon[i])


def CheckPolygonValidity(topSidesPoints, downSidesPoints):
    leftEdgeTopPoint = topSidesPoints[0]
    leftEdgeDownPoint = downSidesPoints[-1]
    rightEdgeTopPoint = topSidesPoints[-1]
    rightEdgeDownPoint = downSidesPoints[0]

    # 判断是否存在与下边相交的上边
    for i in range(0, len(topSidesPoints) - 1):
        topEdgeLeftPoint = topSidesPoints[i]
        topEdgeRightPoint = topSidesPoints[i + 1]
        for j in range(len(downSidesPoints) - 1, 0, -1):
            downEdgeLeftPoint = downSidesPoints[j]
            downEdgeRightPoint = downSidesPoints[j - 1]
            if EdgeIsIntersect(topEdgeLeftPoint, topEdgeRightPoint,
                               downEdgeLeftPoint, downEdgeRightPoint):
                return OcrConstValue.FAIL

    # 判断左右两边是否相交
    if EdgeIsIntersect(leftEdgeTopPoint, leftEdgeDownPoint, rightEdgeTopPoint,
                       rightEdgeDownPoint):
        return OcrConstValue.FAIL

    # 判断左边是否和上边相交
    for i in range(1, len(topSidesPoints) - 1):
        topEdgeLeftPoint = topSidesPoints[i]
        topEdgeRightPoint = topSidesPoints[i + 1]
        if EdgeIsIntersect(leftEdgeTopPoint, leftEdgeDownPoint,
                           topEdgeLeftPoint, topEdgeRightPoint):
            return OcrConstValue.FAIL

    # 判断左边是否和下边相交
    for i in range(len(downSidesPoints) - 2, 1, -1):
        downEdgeLeftPoint = downSidesPoints[i]
        downEdgeRightPoint = downSidesPoints[i - 1]
        if EdgeIsIntersect(leftEdgeTopPoint, leftEdgeDownPoint,
                           downEdgeLeftPoint, downEdgeRightPoint):
            return OcrConstValue.FAIL

    # 判断右边是否和上边相交
    for i in range(1, len(topSidesPoints) - 2):
        topEdgeLeftPoint = topSidesPoints[i]
        topEdgeRightPoint = topSidesPoints[i + 1]
        if EdgeIsIntersect(rightEdgeTopPoint, rightEdgeDownPoint,
                           topEdgeLeftPoint, topEdgeRightPoint):
            return OcrConstValue.FAIL

    #  判断右边是否和下边相交
    for i in range(len(downSidesPoints) - 1, 1, -1):
        downEdgeLeftPoint = downSidesPoints[i]
        downEdgeRightPoint = downSidesPoints[i - 1]
        if EdgeIsIntersect(rightEdgeTopPoint, rightEdgeDownPoint,
                           downEdgeLeftPoint, downEdgeRightPoint):
            return OcrConstValue.FAIL
    return OcrConstValue.SUCCESSFUL


def EdgeIsIntersect(edge1Point1, edge1Point2, edge2Point1, edge2Point2):
    # 快速排斥，以l1、l2为对角线的矩形不相交，否则两线段不相交
    # 矩形1最右端大于矩形2最左端
    # 矩形2最右端大于矩形1最左端
    # 矩形1最高端大于矩形2最低端
    # 矩形2最高端大于矩形1最低端
    return (((max(edge1Point1[0], edge1Point2[0])) >= (min(
        edge2Point1[0], edge2Point2[0]))) and
            ((max(edge2Point1[0], edge2Point2[0])) >= (min(
                edge1Point1[0], edge1Point2[0]))) and
            ((max(edge1Point1[1], edge1Point2[1])) >= (min(
                edge2Point1[1], edge2Point2[1]))) and
            ((max(edge2Point1[1], edge2Point2[1]))
             >= (min(edge1Point1[1], edge1Point2[1])))) and (
                 (CrossProduct(edge1Point1, edge1Point2, edge2Point1) *
                  CrossProduct(edge1Point1, edge1Point2, edge2Point2) <= 0) and
                 (CrossProduct(edge2Point1, edge2Point2, edge1Point1) *
                  CrossProduct(edge2Point1, edge2Point2, edge1Point2) <= 0))


def CrossProduct(p1, p2, p3):
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return (x1 * y2 - x2 * y1)


def GetPointsOfTopAndDownSide(clipBox, bezierPoints, scaleParam, topSidePoints,
                              downSidePoints, imageName, isNeedDetectModel):
    result = OcrConstValue.SUCCESSFUL

    filledPolygon = []
    FillPolygon(clipBox, filledPolygon)

    # 获取上边的两个端点及两个扩展的端点
    # 左端点
    resultLeft, leftTopIndex, leftDownIndex, leftTopExtendPoint, leftDownExtendPoint = GetTwoEndPoints(
        filledPolygon, bezierPoints[0], bezierPoints[1], scaleParam, True,
        imageName)
    # 右端点
    resultRight, rightTopIndex, rightDownIndex, rightTopExtendPoint, rightDownExtendPoint = GetTwoEndPoints(
        filledPolygon, bezierPoints[OcrConstValue.SIX_INT],
        bezierPoints[OcrConstValue.FIVE_INT], scaleParam, False, imageName)
    result = resultLeft + resultRight
    if result != OcrConstValue.SUCCESSFUL or len(
            leftTopExtendPoint) == 0 or len(leftDownExtendPoint) == 0 or len(
                rightTopExtendPoint) == 0 or len(rightDownExtendPoint) == 0:
        print(TAG + "get two end points fail!")
        return OcrConstValue.FAIL

    GetOneSidePoints(filledPolygon, leftTopIndex, rightTopIndex, topSidePoints)
    GetOneSidePoints(filledPolygon, rightDownIndex, leftDownIndex,
                     downSidePoints)
    list(reversed(downSidePoints))
    topSidePoints.insert(0, leftTopExtendPoint)
    topSidePoints.append(rightTopExtendPoint)
    downSidePoints.insert(0, rightDownExtendPoint)
    downSidePoints.append(leftDownExtendPoint)

    print(TAG + "topSidePoints size is %d after reduce" % (len(topSidePoints)))
    print(TAG + "downSidePoints size is %d after reduce" %
          (len(downSidePoints)))

    topBezierPoints = []
    GetBezierPointOfOneLine(topSidePoints, topBezierPoints)
    downBezierPoints = []
    GetBezierPointOfOneLine(downSidePoints, downBezierPoints)

    print(TAG + "topBezierPoints size is %d" % (len(topBezierPoints)))
    print(TAG + "downBezierPoints size is %d" % (len(downBezierPoints)))

    topSidePoints.clear()
    topSidePoints += topBezierPoints
    downSidePoints.clear()
    downSidePoints += downBezierPoints

    PreventExceedingBoundaries(topSidePoints, scaleParam)
    PreventExceedingBoundaries(downSidePoints, scaleParam)

    return OcrConstValue.SUCCESSFUL


def PreventExceedingBoundaries(points, scaleParam):
    for i in range(0, len(points)):
        points[i][0] = (min)((max)(points[i][0], 0),
                             scaleParam['srcWidth'] - 1)
        points[i][1] = (min)((max)(points[i][1], 0),
                             scaleParam['srcHeight'] - 1)


def ExtendTopAndDownPoint(topSidePoints, downSidePoints, scaleParam):
    if len(topSidePoints) != len(downSidePoints):
        return
    width = abs(topSidePoints[0][0] - topSidePoints[-1][0])
    width += abs(downSidePoints[0][0] - downSidePoints[-1][0])

    height = abs(topSidePoints[0][1] - downSidePoints[0][1])
    height += abs(topSidePoints[-1][1] - downSidePoints[-1][1])

    if height != 0:
        ratio = width * 1.0 / height
        for i in range(0, len(downSidePoints)):
            ExtendPoints(topSidePoints[i],
                         downSidePoints[len(downSidePoints) - 1 - i],
                         scaleParam, ratio)


def ExtendPoints(topPoint, downPoint, scaleParam, ratio):
    height = GetTwoPointDist(topPoint, downPoint)
    lowScale = 0.1
    highScale = 0.5

    upScale = lowScale * height
    downScale = lowScale * height

    # 弯曲长文本，底部压字过多，故将downScale放大较多
    longTextRatio = 20
    if ratio > longTextRatio:
        downScale = highScale * height
    if topPoint[0] == downPoint[0]:
        topPoint[1] = max(0, (int)(topPoint[1] - upScale))
        downPoint[1] = min(scaleParam['srcHeight'] - 1,
                           (int)(downPoint[1] + downScale))
        return

    k = GetSlope(topPoint, downPoint)
    b = topPoint[1] - k * topPoint[0]

    topPoint[0] = min(max(0, (int)(topPoint[0] + upScale * cos(atan(k)))),
                      scaleParam['srcWidth'] - 1)
    topPoint[1] = min(max(0, (int)(k * topPoint[0] + b)),
                      scaleParam['srcHeight'] - 1)
    downPoint[0] = min(max(0, (int)(downPoint[0] - downScale * cos(atan(k)))),
                       scaleParam['srcWidth'] - 1)
    downPoint[1] = min(max(0, (int)(k * downPoint[0] + b)),
                       scaleParam['srcHeight'] - 1)


def GetOneSidePoints(contours, startIndex, endIndex, oneSidePoints):
    if startIndex < endIndex:
        for i in range(startIndex, endIndex + 1):
            oneSidePoints.append(contours[i])
    else:
        for i in range(startIndex, len(contours)):
            oneSidePoints.append(contours[i])
        for i in range(0, endIndex + 1):
            oneSidePoints.append(contours[i])


def GetTwoEndPoints(contours, point1, point2, scaleParam, isLeft, imageName):
    result = OcrConstValue.SUCCESSFUL
    if point1[0] == point2[0]:
        result1, topIndex, downIndex = GetPointIfVertical(
            contours, point1, isLeft)
        if result1 != OcrConstValue.SUCCESSFUL:
            return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
        result2, topExtendPoint, downExtendPoint = GetExtendPointIfVertical(
            contours, topIndex, downIndex, point1, scaleParam)
        if result2 != OcrConstValue.SUCCESSFUL:
            return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
        result = result1 + result2
    elif point1[1] == point2[1]:
        result1, topIndex, downIndex = GetPointIfHorizontal(contours, point1)
        if result1 != OcrConstValue.SUCCESSFUL:
            return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
        result2, topExtendPoint, downExtendPoint = GetExtendPointIfHorizontal(
            contours, topIndex, downIndex, point1, isLeft, scaleParam)
        if result2 != OcrConstValue.SUCCESSFUL:
            return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
        result = result1 + result2
    else:
        result1, topIndex, downIndex = GetPointIfNormal(
            contours, point1, point2, isLeft)
        if result1 != OcrConstValue.SUCCESSFUL:
            return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL

        result2, topExtendPoint, downExtendPoint = GetExtendPointIfNormal(
            contours, topIndex, downIndex, point1, point2, isLeft, scaleParam,
            imageName)
        if result2 != OcrConstValue.SUCCESSFUL:
            return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL

        result = result1 + result2
    return result, topIndex, downIndex, topExtendPoint, downExtendPoint


# 垂直情况,x相同
def GetPointIfVertical(contours, point, isLeft):
    temp = []
    for i in range(0, len(contours)):
        if abs(contours[i][1] - point[1] < 1):
            temp.append(i)
    if len(temp) < OcrConstValue.TWO_INT:
        return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
    topIndex = temp[0]
    downIndex = temp[0]
    for i in range(1, len(temp)):
        if contours[i][0] < contours[topIndex][0]:
            downIndex = temp[i]
        if contours[i][0] < contours[downIndex][0]:
            topIndex = temp[i]
    if isLeft:
        tempIndex = downIndex
        downIndex = topIndex
        topIndex = tempIndex
    return OcrConstValue.SUCCESSFUL, topIndex, downIndex


def GetExtendPointIfVertical(contours, topIndex, downIndex, point1,
                             scaleParam):
    temp = []
    index = 0
    while index < len(contours):
        nextIndex = index + 1
        if index == len(contours) - 1:
            nextIndex = 0
        if (abs(contours[index][0] - point1[0])
                < 1) or ((contours[index][0] - point1[0]) *
                         (contours[nextIndex][0] - point1[0]) < 0):
            temp.append(index)
        index += 1
    if len(temp) <= 0:
        return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
    crossPointIndex = temp[0]
    minDist = GetTwoPointDist(contours[crossPointIndex], point1)
    for i in range(0, len(temp)):
        currentDist = GetTwoPointDist(contours[temp[i]], point1)
        if minDist > currentDist:
            crossPointIndex = temp[i]
            minDist = currentDist
    y = contours[crossPointIndex][1]

    y = (int)((y + contours[topIndex][1]) / 2.0)

    topExtendPoint = [contours[topIndex][0], y]
    downExtendPoint = [contours[downIndex][0], y]
    return OcrConstValue.SUCCESSFUL, topExtendPoint, downExtendPoint


# 垂直情况,y相同
def GetPointIfHorizontal(contours, point):
    temp = []
    for i in range(0, len(contours)):
        if abs(contours[i][0] - point[0] < 1):
            temp.append(i)
    if len(temp) < OcrConstValue.TWO_INT:
        return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
    topIndex = temp[0]
    downIndex = temp[0]
    for i in range(1, len(temp)):
        if contours[i][1] < contours[topIndex][1]:
            topIndex = temp[i]
        if contours[i][1] < contours[downIndex][1]:
            downIndex = temp[i]
    return OcrConstValue.SUCCESSFUL, topIndex, downIndex


def GetExtendPointIfHorizontal(contours, topIndex, downIndex, point1, isLeft,
                               scaleParam):
    width = scaleParam['srcWidth']
    temp = []
    index = 0
    while index < len(contours):
        nextIndex = index + 1
        if index == len(contours) - 1:
            nextIndex = 0
        if (abs(contours[index][1] - point1[1])
                < 1) or ((contours[index][1] - point1[1]) *
                         (contours[nextIndex][1] - point1[0]) < 0):
            temp.append(index)
        index += 1
    if len(temp) <= 0:
        return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL
    crossPointIndex = temp[0]
    minDist = GetTwoPointDist(contours[crossPointIndex], point1)
    for i in range(0, len(temp)):
        currentDist = GetTwoPointDist(contours[temp[i]], point1)
        if minDist > currentDist:
            crossPointIndex = temp[i]
            minDist = currentDist
    x = contours[crossPointIndex][0]
    x = (int)((x + contours[topIndex][0]) / 2.0)
    if isLeft:
        x = max(0, x - OcrConstValue.FIVE_INT)
    else:
        x = min(width, x + OcrConstValue.FIVE_INT)

    topExtendPoint = [x, contours[topIndex][1]]
    downExtendPoint = [x, contours[downIndex][1]]
    return OcrConstValue.SUCCESSFUL, topExtendPoint, downExtendPoint


def GetPointIfNormal(contours, point1, point2, isLeft):
    k = -1 / GetSlope(point1, point2)
    b = point1[1] - k * point1[0]
    temp = GetCrossPointsOfLineAndContours(contours, k, b)
    if len(temp) < OcrConstValue.TWO_INT:
        print(TAG + "GetPointIfNormal: temp.size() < 2")
        return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL

    topPoints = []
    downPoints = []
    for i in range(0, len(temp)):
        if JudgingTopDownPoints(point1, point2, contours[temp[i]], isLeft):
            topPoints.append(temp[i])
        else:
            downPoints.append(temp[i])
    if (len(topPoints) <= 0) or (len(downPoints) <= 0):
        print(TAG +
              "GetPointIfNormal: topPoints is empty or  downPoints is empty")
        return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL

    topIndex = topPoints[0]
    minDist = GetTwoPointDist(contours[topIndex], point1)
    for i in range(1, len(topPoints)):
        currentDist = GetTwoPointDist(contours[topPoints[i]], point1)
        if minDist > currentDist:
            topIndex = topPoints[i]
            minDist = currentDist

    downIndex = downPoints[0]
    minDist = GetTwoPointDist(contours[downIndex], point1)
    for i in range(1, len(downPoints)):
        currentDist = GetTwoPointDist(contours[downPoints[i]], point1)
        if minDist > currentDist:
            downIndex = downPoints[i]
            minDist = currentDist
    return OcrConstValue.SUCCESSFUL, topIndex, downIndex


def GetExtendPointIfNormal(contours, topIndex, downIndex, point1, point2,
                           isLeft, scaleParam, imageName):
    width = scaleParam['srcWidth']
    height = scaleParam['srcHeight']
    topPoint = contours[topIndex]
    downPoint = contours[downIndex]

    if topPoint[0] == downPoint[0]:
        return GetExtendPointIfHorizontal(contours, topIndex, downIndex,
                                          point1, isLeft, scaleParam)
    elif topPoint[1] == downPoint[1]:
        return GetExtendPointIfVertical(contours, topIndex, downIndex, point1,
                                        scaleParam)

    centerK = GetSlope(point1, point2)
    centerB = point1[1] - centerK * point1[0]
    temp = GetCrossPointsOfLineAndContours(contours, centerK, centerB)
    if len(temp) <= 0:
        print(
            TAG +
            "GetExtendPointIfNormal, cross points of line and contours is empty"
        )
        return OcrConstValue.FAIL, OcrConstValue.FAIL, OcrConstValue.FAIL

    crossPointIndex = temp[0]
    minDist = GetTwoPointDist(contours[crossPointIndex], point1)
    for i in range(1, len(temp)):
        currentDist = GetTwoPointDist(contours[temp[i]], point1)
        if minDist > currentDist:
            crossPointIndex = temp[i]
            minDist = currentDist

    extendRatio = 10
    k = GetSlope(topPoint, downPoint)
    b = contours[crossPointIndex][1] - k * (contours[crossPointIndex][0] +
                                            extendRatio)
    if isLeft:
        b = contours[crossPointIndex][1] - k * (contours[crossPointIndex][0] -
                                                extendRatio)

    topExtendX = (k * (topPoint[1] - b) + topPoint[0]) / (k * k + 1)
    topExtendK = -1 / k
    topExtendB = topPoint[1] - topExtendK * (topPoint[0] + extendRatio)
    if isLeft:
        topExtendB = topPoint[1] - topExtendK * (topPoint[0] - extendRatio)
    topExtendX = (int)((topExtendX + 3 * topPoint[0]) / 4.0)
    topExtendY = (int)(topExtendK * topExtendX + topExtendB)

    downExtendX = (k * (downPoint[1] - b) + downPoint[0]) / (k * k + 1)
    downExtendK = -1 / k
    downExtendB = downPoint[1] - downExtendK * (downPoint[0] + extendRatio)
    if isLeft:
        downExtendB = downPoint[1] - downExtendK * (downPoint[0] - extendRatio)
    downExtendX = (int)((downExtendX + 3 * downPoint[0]) / 4.0)
    downExtendY = (int)(downExtendK * downExtendX + downExtendB)

    topExtendPoint = GetLegalPoint(width, height, topExtendX, topExtendY,
                                   topPoint)
    downExtendPoint = GetLegalPoint(width, height, downExtendX, downExtendY,
                                    downPoint)

    return OcrConstValue.SUCCESSFUL, topExtendPoint, downExtendPoint


def GetLegalPoint(width, height, extendX, extendY, point):
    p = [(int)(extendX), (int)(extendY)]

    if p[0] == point[0]:
        extendY = point[1]
        return [(int)(extendX), (int)(extendY)]

    k = GetSlope(p, point)

    if extendX < 0:
        extendY = -k * point[0] + point[1]
        extendX = 0
    elif (extendX > (width - 1)):
        extendY = k * (width - 1 - point[0]) + point[1]
        extendX = width - 1

    if extendY < 0:
        if k == 0:
            return []
        extendX = -point[1] / k + point[0]
        extendY = 0
    elif (extendY > (height - 1)):
        if k == 0:
            return []
        extendX = (height - 1 - point[1]) / k + point[0]
        extendY = height - 1
    return [(int)(extendX), (int)(extendY)]


def JudgingTopDownPoints(point1, point2, toJudgingPoint, isLeft):
    vector1 = [point2[0] - point1[0], point2[1] - point1[1]]
    vector2 = [toJudgingPoint[0] - point1[0], toJudgingPoint[1] - point1[1]]

    result = vector1[0] * vector2[1] - vector2[0] * vector1[1]

    if isLeft:
        return (result < 0)
    return (result > 0)


def GetCrossPointsOfLineAndContours(contours, k, b):
    temp = []
    index = 0
    while (index < len(contours)):
        nextIndex = index + 1
        if index == len(contours) - 1:
            nextIndex = 0
        dist1 = contours[index][1] - k * contours[index][0] - b
        dist2 = contours[nextIndex][1] - k * contours[nextIndex][0] - b

        if (abs(dist1) < 1) or (abs(dist2) < 1) or (dist1 * dist2 <= 0):
            temp.append(index)
        index += 1
    return temp


def GetSlope(point1, point2):
    return (point2[1] - point1[1]) * 1.0 / (point2[0] - point1[0])


def GetTwoPointDist(point1, point2):
    return ((1.0 * (point1[0] - point2[0]) * (point1[0] - point2[0]) + 1.0 *
             (point1[1] - point2[1]) * (point1[1] - point2[1]))**0.5)


def FillPolygon(polygon, filledPolygon):
    pointNum = len(polygon)
    for i in range(0, pointNum):
        point1 = polygon[i]
        if i == pointNum - 1:
            point2 = polygon[0]
        else:
            point2 = polygon[i + 1]
        if point1[0] == point2[0]:
            if point1[1] > point2[1]:
                for j in range(point1[1] - 1, point2[1] + 1, -1):
                    filledPolygon.append([point1[0], j])
            else:
                for j in range(point1[1] + 1, point2[1] + 1, 1):
                    filledPolygon.append([point1[0], j])
        else:
            lineSlope = (point2[1] - point1[1]) * 1.0 / (point2[0] - point1[0])
            if point1[0] > point2[0]:
                for j in range(point1[0] - 1, point2[0] + 1, -1):
                    currentY = (int)(point1[1] + lineSlope * (j - point1[0]))
                    filledPolygon.append([j, currentY])
            else:
                for j in range(point1[0] + 1, point2[0] + 1, 1):
                    currentY = (int)(point1[1] + lineSlope * (j - point1[0]))
                    filledPolygon.append([j, currentY])


def GetBezierPointOfOneLine(linePoints, bezierPoints):
    if len(linePoints) < OcrConstValue.FOUR_INT:
        return OcrConstValue.FAIL
    linePoints = ReduceOneLinePoints(linePoints)
    BeziercurveNormalizeOneLine(linePoints, bezierPoints)
    return OcrConstValue.SUCCESSFUL


def BeziercurveNormalizeOneLine(line, bezierPoints):
    '''
    获取标准的7个点
    line:文本框坐标点
    '''
    tmp = BezierFitOfOneLine(line, 0)
    tmp = tmp.tolist()
    bezierPoints.append([(int)(tmp[0][0]), (int)(tmp[0][1])])
    for j in range(1, OcrConstValue.SIX_INT):
        u = (float)(j) / 6.0
        x0 = GetBezierCoord(tmp[0][0], tmp[1][0], tmp[2][0], tmp[3][0], u)
        y0 = GetBezierCoord(tmp[0][1], tmp[1][1], tmp[2][1], tmp[3][1], u)
        bezierPoints.append([(int)(x0), (int)(y0)])
    bezierPoints.append(tmp[OcrConstValue.THREE_INT])


def GetBezierCoord(p0, p1, p2, p3, u):
    return ((1 - u) * (1 - u) * (1 - u) * p0 + 3 * (1 - u) * (1 - u) * u * p1 +
            3 * (1 - u) * u * u * p2 + u * u * u * p3)


def BezierFitOfOneLine(coords, start):
    '''
    获取Bezier控制点
    box:文本框标注坐标点
    start：第一个点坐标
    '''
    length = len(coords)
    dt = []
    j = 0
    for i in range(start, start + length - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        dt.append((dx * dx + dy * dy)**0.5)
        j = j + 1
    t = [0.0]
    sumlength = sum(dt)
    for i in range(0, len(dt)):
        t.append(dt[i] / sumlength)
        t[i + 1] = t[i] + t[i + 1]
    Basematrix = np.zeros((len(t), 4))
    for i in range(0, len(t)):
        for j in range(0, 4):
            Basematrix[i][j] = pow(
                (1 - t[i]), 3 - j) * pow(t[i], j) * Combinator(3, j)

    invertBasematrix = np.linalg.pinv(Basematrix)
    controlPoints = np.dot(invertBasematrix, coords)
    controlPoints[0] = coords[0]
    controlPoints[3] = coords[-1]
    return controlPoints


def Combinator(n, m):
    if n < m:
        tmp = n
        n = m
        m = tmp
    return Factorial(n) / (Factorial(m) * Factorial(n - m))


def Factorial(num):
    if num <= 1:
        return 1
    else:
        return num * Factorial(num - 1)


def ReduceOneLinePoints(linePoints):
    requrePoints = 36
    pointNum = len(linePoints) - 1
    if pointNum > requrePoints:
        temp = []
        temp.append(linePoints[0])
        interval = pointNum / (requrePoints - 1 + 1e-8)
        tempIndex = 1
        for j in range(0, requrePoints - 1):
            remainIndex = 1 + (int)(interval * j)
            if (remainIndex == tempIndex) and (remainIndex != 1):
                continue
            else:
                temp.append(linePoints[remainIndex])
                tempIndex = remainIndex
        if (temp[-1][0] != linePoints[pointNum - 1][0]) and (
                temp[-1][1] != linePoints[pointNum - 1][1]):
            temp.append(linePoints[pointNum - 1])
        temp.append(linePoints[pointNum])
        linePoints.clear()
        linePoints += temp
    return linePoints


def RestoreSize(points, scaleParam):
    if (scaleParam['padding'][OcrConstValue.INDEX_TWO]
        ) != 0 and scaleParam['padding'][OcrConstValue.INDEX_THREE] != 0:
        for j in range(0, len(points)):
            points[j][0] = points[j][0] / scaleParam['padding'][
                OcrConstValue.INDEX_TWO]
            points[j][0] = (min)((max)(points[j][0], 0),
                                 scaleParam['srcWidth'] - 1)
            points[j][1] = points[j][1] / scaleParam['padding'][
                OcrConstValue.INDEX_THREE]
            points[j][1] = (min)((max)(points[j][1], 0),
                                 scaleParam['srcHeight'] - 1)


def FindRsPolygonsCropped(probabilityMat, binaryMat, pMin, scaleParam,
                          boxScoreThresh, thresholdPara, imageName,
                          isNeedDetectModel):
    print(TAG + "Enter FindRsPolygonsCropped")
    # 获取轮廓
    contours, _ = cv2.findContours((binaryMat * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
                                   pMin)
    print(TAG + "contours size is ", len(contours))

    skeletonMat = copy.deepcopy(binaryMat * 255)
    skeletonMat = skeletonMat.astype(np.uint8)
    hPro, wPro = probabilityMat.shape
    hBin, wBin = binaryMat.shape
    skeletonMat = cv2.copyMakeBorder(skeletonMat, pMin[1],
                                     hPro - pMin[1] - hBin, pMin[0],
                                     wPro - pMin[0] - wBin,
                                     cv2.BORDER_CONSTANT, 0)
    startTime = time.time()
    # 获取全图骨架点
    skeletonPoints = get_skeleton(skeletonMat)

    endTime = time.time()
    print(TAG + "get_skeleton spend time:", endTime - startTime)

    for contour in contours:
        contourArea = cv2.contourArea(contour)
        if contourArea < thresholdPara['min_area_size']:
            continue
        # 获取轮廓最小面积外接矩形
        minBox, boxWidth, boxHeight = GetMinBox(contour)
        minBox = np.array(minBox)
        boxArea = cv2.contourArea(minBox)
        if boxHeight < 0.0001:
            continue
        startTime = time.time()
        areaBorder = thresholdPara['rectangle_similarity']
        boxScoreThresh = thresholdPara['boxScoreThresh']
        line_text_expand_ratio = thresholdPara['line_text_expand_ratio']
        curve_text_expand_ratio = thresholdPara['curve_text_expand_ratio']
        if (contourArea / boxArea >= areaBorder) or (boxArea < 1000):
            print(TAG + "GetBoxFromContour")
            result = GetBoxFromContour(probabilityMat, scaleParam,
                                       boxScoreThresh, line_text_expand_ratio,
                                       contour, minBox, boxWidth, boxHeight)
        else:
            print(TAG + "GetPolygonsFromContour")
            result = GetPolygonsFromContour(probabilityMat, scaleParam,
                                            boxScoreThresh,
                                            curve_text_expand_ratio, contour,
                                            skeletonPoints, skeletonMat,
                                            binaryMat, imageName,
                                            isNeedDetectModel)
        endTime = time.time()
        print(TAG + "GetXXFromContour time is：" + (endTime - startTime))
    return result


def IsPointInRect(p, contourRect):
    if (p[0] > contourRect[0]) and (p[0] < (
            contourRect[0] + contourRect[2])) and (p[1] > contourRect[1]) and (
                p[1] < (contourRect[1] + contourRect[3])):
        return True
    else:
        return False


def GetContourSkeleton(skeletonPoints, contour, skeletonMat, imageName):
    # 外接矩形（无角度）
    contourRect = cv2.boundingRect(contour)
    contourSkeleton = []
    for point in skeletonPoints:
        if IsPointInRect(point, contourRect):
            distance = cv2.pointPolygonTest(contour, (point[0], point[1]),
                                            False)
            if distance >= 0:
                contourSkeleton.append(point)
    if len(contourSkeleton) <= 0:
        print(TAG + "contourSkeleton size is zero")
        return [], []

    return Deburring(skeletonMat, contourSkeleton, imageName)


# 除毛刺。方法：先找到骨架的端点，然后从端点出发，找到当前连通域中的最长路径
def Deburring(skeletonMat, contourSkeleton, imageName):
    height, width = skeletonMat.shape
    endPoints = []
    finalSkeletonPoints = []
    GetSkeletonEndpoints(skeletonMat, contourSkeleton, endPoints)

    # 保存骨架点
    skeletonPointsName = imageName + "_endPoints.jpg"
    temp = np.ones((height, width, 3), np.uint8)
    for point in contourSkeleton:
        cv2.circle(temp, point, 1, (255, 255, 255), 3)

    if len(endPoints) > 0:
        cv2.circle(temp, endPoints[0], 4, (255, 0, 0), 3)
        for i in range(1, len(endPoints) - 2):
            cv2.circle(temp, endPoints[i], 4, (0, 0, 255), 3)
        cv2.circle(temp, endPoints[len(endPoints) - 1], 4, (0, 0, 255), 3)
        # 大于两个端点需除去毛刺，1个或2个端点的线段可能存在圆环因此也需要找最长路径
        finalSkeletonPoints = GetLongestPath(skeletonMat, endPoints)
    else:
        # 没有端点，则骨架为圆环, 以X最小为起点，顺时针进行排序
        sorted(contourSkeleton, key=functools.cmp_to_key(cmp))
        startPoint = contourSkeleton[0]
        # 排完序了，为什么再比较一遍？
        for point in contourSkeleton:
            if point[0] < startPoint[0]:
                startPoint = point
        labelMat = np.zeros((height, width))
        SortedSkeletonPoints(skeletonMat, labelMat, startPoint,
                             finalSkeletonPoints)

    a = finalSkeletonPoints[0][0]
    b = finalSkeletonPoints[-1][0]
    if finalSkeletonPoints[0][0] > finalSkeletonPoints[-1][0]:
        finalSkeletonPoints = list(reversed(finalSkeletonPoints))
    return finalSkeletonPoints, endPoints


def cmp(a, b):
    if a[0] == b[0]:
        if a[1] >= b[1]:
            return 1
        else:
            return -1
    else:
        if a[0] > b[0]:
            return 1
        else:
            return -1


def SortedSkeletonPoints(skeletonMat, labelMat, startPoint,
                         finalSkeletonPoints):
    labelMat[startPoint[1], startPoint[0]] = OcrConstValue.PIXEL_MAX_VALUE
    finalSkeletonPoints.append(startPoint)
    nextPoints = GetNextPoint(skeletonMat, labelMat, startPoint)
    if len(nextPoints) > 0:
        SortedSkeletonPoints(skeletonMat, labelMat, nextPoints[0],
                             finalSkeletonPoints)


def GetNextPoint(skeletonMat, labelMat, startPoint):
    nextPoints = []
    height, width = skeletonMat.shape
    windowSize = 1
    xStart = max(0, startPoint[0] - windowSize)
    xEnd = min(startPoint[0] + windowSize, width - 1)
    yStart = max(0, startPoint[1] - windowSize)
    yEnd = min(startPoint[1] + windowSize, height - 1)

    for i in range(xStart, xEnd + 1):
        if (skeletonMat[startPoint[1], i]
                == OcrConstValue.PIXEL_MAX_VALUE) and (labelMat[startPoint[1],
                                                                i] == 0):
            nextPoints.append([i, startPoint[0]])
    for j in range(yStart, yEnd + 1):
        if (skeletonMat[j, startPoint[0]]
                == OcrConstValue.PIXEL_MAX_VALUE) and (labelMat[j,
                                                                startPoint[0]]
                                                       == 0):
            nextPoints.append([startPoint[0], j])
    if len(nextPoints) > 0:
        for i in range(xStart, xEnd + 1):
            for j in range(yStart, yEnd + 1):
                if (skeletonMat[j, i]
                        == OcrConstValue.PIXEL_MAX_VALUE) and (labelMat[j, i]
                                                               == 0):
                    nextPoints.append([startPoint[0], j])

    return nextPoints


def GetLongestPath(skeletonMat, endpoints):
    height, width = skeletonMat.shape
    finalSkeletonPoints = []
    for endPoint in endpoints:
        labelMat = np.zeros((height, width), np.float32)
        currentPath = []
        GetAllPathsFromEndPoint(skeletonMat, labelMat, finalSkeletonPoints,
                                currentPath, endPoint)
    return finalSkeletonPoints


def GetAllPathsFromEndPoint(skeletonMat, labelMat, longestPath, currentPath,
                            currentPoint):
    labelMat[currentPoint[1], currentPoint[0]] = OcrConstValue.PIXEL_MAX_VALUE
    currentPath.append(currentPoint)
    surroundingPoints = GetSurroundingPoints(skeletonMat, labelMat,
                                             currentPoint)
    if len(surroundingPoints) <= 0:
        if len(longestPath) < len(currentPath):
            longestPath.clear()
            longestPath += currentPath
    else:
        for sp in surroundingPoints:
            GetAllPathsFromEndPoint(skeletonMat, labelMat, longestPath,
                                    currentPath, sp)
    currentPath.pop()


def GetSurroundingPoints(skeletonMat, labelMat, point):
    height, width = skeletonMat.shape
    windowSize = 1
    xStart = max(0, point[0] - windowSize)
    xEnd = min(point[0] + windowSize, width - 1)
    yStart = max(0, point[1] - windowSize)
    yEnd = min(point[1] + windowSize, height - 1)
    surroundingPoints = []
    for i in range(xStart, xEnd + 1):
        for j in range(yStart, yEnd + 1):
            if (skeletonMat[j, i]
                    == OcrConstValue.PIXEL_MAX_VALUE) and (labelMat[j, i]
                                                           == 0):
                surroundingPoints.append([i, j])
    return surroundingPoints


def GetSkeletonEndpoints(skeletonMat, contourSkeleton, endpoints):
    kernelSize = 3

    for point in contourSkeleton:
        if SkeletonConvolution(skeletonMat, point,
                               kernelSize) == OcrConstValue.TWO_INT:
            endpoints.append(point)


def SkeletonConvolution(skeletonMat, point, kernelSize):
    height, width = skeletonMat.shape

    windowSize = (kernelSize - 1) / OcrConstValue.TWO_INT
    xStart = (int)(max(0, point[0] - windowSize))
    xEnd = (int)(min(point[0] + windowSize, width - 1))
    yStart = (int)(max(0, point[1] - windowSize))
    yEnd = (int)(min(point[1] + windowSize, height - 1))

    neighborSize = 0
    for i in range(xStart, xEnd + 1):
        for j in range(yStart, yEnd + 1):
            if skeletonMat[j][i] == OcrConstValue.PIXEL_MAX_VALUE:
                neighborSize += 1
    return neighborSize
