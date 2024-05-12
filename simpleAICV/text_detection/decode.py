import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import pyclipper
import numpy as np

from shapely.geometry import Polygon

__all__ = [
    'DBNetDecoder',
]


class DBNetDecoder:

    def __init__(self,
                 use_morph_open=False,
                 hard_border_threshold=None,
                 box_score_threshold=0.5,
                 min_area_size=9,
                 max_box_num=1000,
                 rectangle_similarity=0.6,
                 min_box_size=3,
                 line_text_expand_ratio=1.2,
                 curve_text_expand_ratio=1.5):
        self.use_morph_open = use_morph_open
        self.hard_border_threshold = hard_border_threshold
        self.box_score_threshold = box_score_threshold
        self.min_area_size = min_area_size
        self.max_box_num = max_box_num
        self.rectangle_similarity = rectangle_similarity
        self.min_box_size = min_box_size
        self.line_text_expand_ratio = line_text_expand_ratio
        self.curve_text_expand_ratio = curve_text_expand_ratio

    def __call__(self, preds, sizes):
        probability_map, threshold_map = preds[:, 0, :, :], preds[:, 1, :, :]
        probability_map, threshold_map = probability_map.cpu().detach().numpy(
        ), threshold_map.cpu().detach().numpy()

        binary_map = np.where(probability_map > self.hard_border_threshold,
                              1.0,
                              0.0) if self.hard_border_threshold else np.where(
                                  probability_map > threshold_map, 1.0, 0.0)

        batch_boxes, batch_scores = [], []
        for i, per_image_binary_map in enumerate(binary_map):
            image_h, image_w = sizes[i]
            per_image_probability_map = probability_map[i][0:image_h,
                                                           0:image_w]
            per_image_binary_map = per_image_binary_map[0:image_h, 0:image_w]

            if self.use_morph_open:
                per_image_binary_map = cv2.morphologyEx(per_image_binary_map,
                                                        cv2.MORPH_OPEN,
                                                        kernel=np.ones(
                                                            (3, 3),
                                                            dtype=np.uint8),
                                                        iterations=1)

            # 求每张图像上预测出的文本轮廓
            per_image_contours, _ = cv2.findContours(
                (per_image_binary_map * 255).astype(np.uint8), cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE)

            # 使用最小面积和最小预测分数过滤所有文本轮廓
            filter_image_contours, filter_image_areas,filter_image_scores = [], [],[]
            for per_contour in per_image_contours:
                area, score = self.compute_contour_area_and_score(
                    per_contour, per_image_probability_map)
                if area > self.min_area_size and score > self.box_score_threshold:
                    filter_image_contours.append(per_contour)
                    filter_image_areas.append(area)
                    filter_image_scores.append(score)

            # 若文本轮廓数量大于max_box_num,则保留分数最高的max_box_num个本文轮廓
            if len(filter_image_scores) > self.max_box_num:
                topk_indexes = np.argsort(
                    filter_image_scores)[-self.max_box_num:]
                filter_image_contours = [
                    filter_image_contours[idx] for idx in topk_indexes
                ]
                filter_image_areas = [
                    filter_image_areas[idx] for idx in topk_indexes
                ]
                filter_image_scores = [
                    filter_image_scores[idx] for idx in topk_indexes
                ]

            # 计算各文本轮廓最小外接矩形面积
            enclose_boxes = [
                cv2.minAreaRect(per_contour)
                for per_contour in filter_image_contours
            ]
            enclose_boxes_area = [
                per_contour_enclose_box[1][0] * per_contour_enclose_box[1][1]
                for per_contour_enclose_box in enclose_boxes
            ]

            image_matrix = np.array([[0, 0], [image_w, 0], [image_w, image_h],
                                     [0, image_h]])
            image_boxes, image_scores = [], []
            for i in range(len(filter_image_contours)):
                per_contour = np.squeeze(filter_image_contours[i], axis=1)
                # 通过轮廓面积与最小外接矩形面积的比值判断是直线文本还是弯曲文本
                # 如果小于,则是弯曲文本
                if enclose_boxes_area[i] < 1:
                    continue

                if filter_image_areas[i] / enclose_boxes_area[
                        i] < self.rectangle_similarity:
                    epsilon = 1e-3 * cv2.arcLength(per_contour, True)
                    per_contour = cv2.approxPolyDP(per_contour, epsilon, True)
                    per_contour = np.squeeze(per_contour, axis=1)

                # 若轮廓多边形少于4个点则抛弃
                if per_contour.shape[0] < 4:
                    continue

                # 根据expand比例扩大轮廓
                text_expand_ratio = self.curve_text_expand_ratio if filter_image_areas[
                    i] / enclose_boxes_area[
                        i] < self.rectangle_similarity else self.line_text_expand_ratio
                polygon = Polygon(per_contour)
                distance = polygon.area * text_expand_ratio / polygon.length
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(per_contour, pyclipper.JT_ROUND,
                               pyclipper.ET_CLOSEDPOLYGON)
                per_contour = offset.Execute(distance)
                if len(per_contour) != 1:
                    continue

                # 图像h,w的矩形和contour求交集区域，保证每个expand之后的contour不越界
                per_contour = np.array(per_contour, dtype=np.float32)
                pc = pyclipper.Pyclipper()
                pc.AddPath(image_matrix, pyclipper.PT_CLIP, True)
                pc.AddPaths(per_contour, pyclipper.PT_SUBJECT, True)
                per_contour = pc.Execute(pyclipper.CT_INTERSECTION,
                                         pyclipper.PFT_EVENODD,
                                         pyclipper.PFT_EVENODD)
                if len(per_contour) != 1:
                    continue
                per_box = np.array(per_contour, dtype=np.float32)[0]

                enclose_box = cv2.minAreaRect(per_box)
                box_sizes = enclose_box[1]

                # 如果是直线文本，则以最小外接矩形为最终矩形
                if filter_image_areas[i] / enclose_boxes_area[
                        i] >= self.rectangle_similarity:
                    enclose_box = cv2.boxPoints(enclose_box)
                    per_box = self.order_box_points(enclose_box)

                if min(box_sizes) < self.min_box_size:
                    continue

                per_box = per_box.astype(np.int32)

                image_boxes.append(per_box)
                image_scores.append(filter_image_scores[i])

            batch_boxes.append(image_boxes)
            batch_scores.append(image_scores)

        return batch_boxes, batch_scores

    def compute_contour_area_and_score(self, per_contour,
                                       per_image_probability_map):
        h, w = per_image_probability_map.shape
        per_contour_area_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(per_contour_area_mask, [per_contour.astype(np.int32)],
                     1.0)
        area = per_contour_area_mask.sum()
        score = (per_image_probability_map *
                 per_contour_area_mask).sum() / area

        return area, score

    def order_box_points(self, box):
        # 根据box各点x坐标从小到大对点进行排序
        x_sorted = box[np.argsort(box[:, 0]), :]
        # 获取x坐标最小的两个点和x坐标最大的两个点
        left2x_points, right2x_points = x_sorted[:2, :], x_sorted[2:, :]
        # 对left2x_points按y坐标从小到大排序,得到左上角点和左下角点
        left2x_points = left2x_points[np.argsort(left2x_points[:, 1]), :]
        (left_top, left_bottom) = left2x_points
        # 计算左上角点和x坐标最大的两个点的欧氏距离,距离最大的点就是右下角点
        distance = (left_top[np.newaxis][:, 0] - right2x_points[:, 0])**2 + (
            left_top[np.newaxis][:, 1] - right2x_points[:, 1])**2
        (right_bottom,
         right_top) = right2x_points[np.argsort(distance)[::-1], :]
        box = np.array([left_top, right_top, right_bottom, left_bottom],
                       dtype=np.float32)

        return box


if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from tools.path import text_detection_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.text_detection.datasets.text_detection_dataset import TextDetection
    from simpleAICV.text_detection.common import RandomRotate, MainDirectionRandomRotate, Resize, Normalize, TextDetectionCollater, DBNetTextDetectionCollater

    textdetectiondataset = TextDetection(
        text_detection_dataset_path,
        set_name=[
            # 'ICDAR2017RCTW_text_detection',
            # 'ICDAR2019ART_text_detection',
            # 'ICDAR2019LSVT_text_detection',
            # 'ICDAR2019MLT_text_detection',
            'ICDAR2019ReCTS_text_detection',
        ],
        set_type='train',
        transform=transforms.Compose([
            # RandomRotate(angle=[-30, 30], prob=0.3),
            # MainDirectionRandomRotate(angle=[0, 90, 180, 270],
            #                           prob=[0.7, 0.1, 0.1, 0.1]),
            Resize(resize=960),
            #  Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(textdetectiondataset):
        print(per_sample['image'].shape, per_sample['image'].dtype)

        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DBNetTextDetectionCollater(resize=960,
                                          min_box_size=3,
                                          min_max_threshold=[0.3, 0.7],
                                          shrink_ratio=0.6)
    train_loader = DataLoader(textdetectiondataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    decoder = DBNetDecoder(hard_border_threshold=0.3,
                           box_score_threshold=0.5,
                           rectangle_similarity=0.6,
                           min_box_size=3,
                           min_area_size=9,
                           max_box_num=1000,
                           line_text_expand_ratio=1.2,
                           curve_text_expand_ratio=1.5)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']

        probability_mask = annots['probability_mask']
        threshold_mask = annots['threshold_mask']
        batch_boxes, batch_scores = decoder(
            torch.tensor(
                np.concatenate((probability_mask.unsqueeze(1),
                                threshold_mask.unsqueeze(1)),
                               axis=1)), sizes)
        print("1111", len(batch_boxes), len(batch_scores))
        print("2222", batch_boxes[0], batch_scores[0])

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()

        # for i in range(images.shape[0]):
        #     per_image = images[i]
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_boxes = batch_boxes[i]
        #     per_image_scores = batch_scores[i]

        #     for per_box in per_image_boxes:
        #         per_box = np.array(per_box, np.int32)
        #         per_box = per_box.reshape((-1, 1, 2))

        #         cv2.polylines(per_image,
        #                       pts=[per_box],
        #                       isClosed=True,
        #                       color=(0, 255, 0),
        #                       thickness=3)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))

        if count < 5:
            count += 1
        else:
            break
