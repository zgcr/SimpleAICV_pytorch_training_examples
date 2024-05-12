import os
import copy
import cv2
import numpy as np
import random

from pycocotools.coco import COCO
from torch.utils.data import Dataset

COCO_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

COCO_CLASSES_COLOR = [
    (156, 77, 36),
    (218, 3, 199),
    (252, 197, 160),
    (82, 69, 38),
    (132, 17, 27),
    (71, 19, 213),
    (108, 81, 1),
    (49, 54, 81),
    (8, 249, 143),
    (80, 20, 4),
    (75, 227, 112),
    (82, 41, 57),
    (157, 0, 97),
    (0, 209, 246),
    (116, 242, 109),
    (60, 225, 243),
    (2, 125, 5),
    (118, 94, 170),
    (171, 1, 17),
    (54, 97, 38),
    (16, 132, 55),
    (1, 90, 238),
    (112, 4, 197),
    (147, 219, 248),
    (253, 0, 14),
    (103, 77, 249),
    (149, 1, 222),
    (120, 94, 51),
    (88, 29, 129),
    (204, 29, 128),
    (19, 0, 244),
    (92, 154, 54),
    (34, 89, 7),
    (29, 168, 224),
    (111, 25, 1),
    (137, 70, 83),
    (24, 217, 19),
    (47, 170, 155),
    (34, 234, 107),
    (182, 116, 221),
    (102, 243, 211),
    (53, 247, 123),
    (147, 159, 24),
    (194, 147, 121),
    (76, 101, 233),
    (50, 11, 88),
    (253, 33, 83),
    (84, 1, 57),
    (248, 243, 24),
    (244, 79, 35),
    (162, 240, 132),
    (1, 32, 203),
    (208, 10, 8),
    (30, 64, 206),
    (234, 80, 229),
    (31, 253, 207),
    (110, 34, 78),
    (234, 72, 73),
    (92, 3, 16),
    (113, 0, 65),
    (196, 177, 53),
    (63, 92, 139),
    (76, 143, 1),
    (61, 93, 84),
    (82, 130, 157),
    (28, 2, 84),
    (55, 226, 12),
    (34, 99, 82),
    (47, 5, 239),
    (53, 100, 219),
    (132, 37, 147),
    (244, 156, 224),
    (179, 57, 59),
    (2, 27, 76),
    (0, 100, 83),
    (64, 39, 116),
    (170, 46, 246),
    (27, 51, 87),
    (185, 71, 0),
    (107, 247, 29),
]


class SAMACOCOdataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name='train',
                 positive_points_num=9,
                 negative_points_num=0,
                 area_filter_ratio=0.0025,
                 box_noise_pixel=50,
                 mask_noise_pixel=50,
                 transform=None):
        assert set_name in ['train', 'validation'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'sama_coco_{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.image_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()

        # get all mask for all images
        self.all_image_mask_ids = []
        for image_id in self.image_ids:
            annot_ids = self.coco.getAnnIds(imgIds=image_id)
            annots = self.coco.loadAnns(annot_ids)
            annots.sort(key=lambda x: x['area'], reverse=True)
            if len(annots) == 0:
                continue

            image_info = self.coco.loadImgs(image_id)[0]
            image_h, image_w = image_info['height'], image_info['width']

            for annot_idx, annot in enumerate(annots):
                if 'ignore' in annot.keys():
                    continue

                # bbox format:[x_min, y_min, w, h]
                bbox = annot['bbox']

                inter_w = max(
                    0,
                    min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
                inter_h = max(
                    0,
                    min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
                if inter_w * inter_h == 0:
                    continue

                if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
                    continue
                if annot['category_id'] not in self.cat_ids:
                    continue

                if annot['area'] / float(
                        image_h * image_w) < area_filter_ratio:
                    continue

                self.all_image_mask_ids.append([image_id, annot_idx])

        self.cats = sorted(self.coco.loadCats(self.cat_ids),
                           key=lambda x: x['id'])
        self.num_classes = len(self.cats)

        # cat_id is an original cat id,coco_label is set from 0 to 79
        self.cat_id_to_cat_name = {cat['id']: cat['name'] for cat in self.cats}
        self.cat_id_to_coco_label = {
            cat['id']: i
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_id = {
            i: cat['id']
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_name = {
            coco_label: self.cat_id_to_cat_name[cat_id]
            for coco_label, cat_id in self.coco_label_to_cat_id.items()
        }

        self.positive_points_num = positive_points_num
        self.negative_points_num = negative_points_num
        self.area_filter_ratio = area_filter_ratio
        self.box_noise_pixel = box_noise_pixel
        self.mask_noise_pixel = mask_noise_pixel
        self.transform = transform

        print(f'Image Size:{len(self.image_ids)}')
        print(f'Dataset Size:{len(self.all_image_mask_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.all_image_mask_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        # image_mask:[0,1]二值化mask
        image_box, image_mask = self.load_mask(idx)
        origin_image = copy.deepcopy(image)
        origin_box = copy.deepcopy(image_box)
        origin_mask = copy.deepcopy(image_mask)

        size = np.array([origin_image.shape[0],
                         origin_image.shape[1]]).astype(np.float32)
        origin_size = copy.deepcopy(size)

        image_mask_all_points_coords = np.argwhere(image_mask)
        image_mask_all_points_num = len(image_mask_all_points_coords)

        if image_mask_all_points_num < self.positive_points_num:
            positive_prompt_point = np.zeros((0, 3), dtype=np.float32)
        else:
            if self.positive_points_num > 0:
                positive_prompt_point = []
                positive_prompt_points_index = np.random.choice(
                    image_mask_all_points_num, self.positive_points_num)
                for positive_point_idx in positive_prompt_points_index:
                    positive_prompt_point.append([
                        image_mask_all_points_coords[positive_point_idx][1],
                        image_mask_all_points_coords[positive_point_idx][0], 1
                    ])
                positive_prompt_point = np.array(positive_prompt_point,
                                                 dtype=np.float32)
            else:
                positive_prompt_point = np.zeros((0, 3), dtype=np.float32)

        image_not_mask_all_points_coords = np.argwhere(1. - image_mask)
        image_not_mask_all_points_num = len(image_not_mask_all_points_coords)

        if image_not_mask_all_points_num < self.negative_points_num:
            negative_prompt_point = np.zeros((0, 3), dtype=np.float32)
        else:
            if self.negative_points_num > 0:
                negative_prompt_point = []
                negative_prompt_points_index = np.random.choice(
                    image_not_mask_all_points_num, self.negative_points_num)
                for negative_point_idx in negative_prompt_points_index:
                    negative_prompt_point.append([
                        image_not_mask_all_points_coords[negative_point_idx]
                        [1],
                        image_not_mask_all_points_coords[negative_point_idx]
                        [0], 0
                    ])
                negative_prompt_point = np.array(negative_prompt_point,
                                                 dtype=np.float32)
            else:
                negative_prompt_point = np.zeros((0, 3), dtype=np.float32)

        prompt_box = copy.deepcopy(image_box)
        if self.box_noise_pixel > 0:
            prompt_box = self.noise_bbox(prompt_box, origin_size)

        prompt_mask = copy.deepcopy(image_mask)
        prompt_mask = self.noise_mask(prompt_mask, idx)

        sample = {
            'origin_image': origin_image,
            'origin_bbox': origin_box,
            'origin_mask': origin_mask,
            'origin_size': origin_size,
            'image': image,
            'box': image_box,
            'mask': image_mask,
            'size': size,
            'positive_prompt_point': positive_prompt_point,
            'negative_prompt_point': negative_prompt_point,
            'prompt_box': prompt_box,
            'prompt_mask': prompt_mask,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        file_name = self.coco.loadImgs(
            self.all_image_mask_ids[idx][0])[0]['file_name']
        image = cv2.imdecode(
            np.fromfile(os.path.join(self.image_dir, file_name),
                        dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        image_annot_ids = self.coco.getAnnIds(
            imgIds=self.all_image_mask_ids[idx][0])
        image_annots = self.coco.loadAnns(image_annot_ids)
        image_annots.sort(key=lambda x: x['area'], reverse=True)

        target_box = np.array(
            image_annots[self.all_image_mask_ids[idx][1]]['bbox'])
        target_box_label = np.expand_dims(np.array(self.cat_id_to_coco_label[
            image_annots[self.all_image_mask_ids[idx][1]]['category_id']]),
                                          axis=0)
        target_box = np.concatenate((target_box, target_box_label), axis=0)

        # transform bbox targets from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        target_box[2] = target_box[0] + target_box[2]
        target_box[3] = target_box[1] + target_box[3]

        target_mask = self.coco.annToMask(
            image_annots[self.all_image_mask_ids[idx][1]])

        return target_box.astype(np.float32), target_mask.astype(np.float32)

    def noise_bbox(self, properties_bbox, mask_np_shape):
        w, h = properties_bbox[2] - properties_bbox[0], properties_bbox[
            3] - properties_bbox[1]
        x0 = properties_bbox[0] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                w / 2), -w / 2)
        y0 = properties_bbox[1] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                h / 2), -h / 2)
        x1 = properties_bbox[2] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                w / 2), -w / 2)
        y1 = properties_bbox[3] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                h / 2), -h / 2)
        x1 = x1 if x1 <= mask_np_shape[1] else mask_np_shape[1]  # 避免越界
        y1 = y1 if y1 <= mask_np_shape[0] else mask_np_shape[0]
        post_properties_bbox = np.array([x0, y0, x1, y1])
        post_properties_bbox = np.where(post_properties_bbox > 0,
                                        post_properties_bbox, 0)
        if x0 >= x1 or y0 >= y1:
            return properties_bbox.astype(np.float32)
        else:
            return post_properties_bbox.astype(np.float32)

    def noise_mask(self, properties_mask, idx):
        image_info = self.coco.loadImgs(self.all_image_mask_ids[idx][0])[0]
        image_h, image_w = image_info['height'], image_info['width']

        kernel = np.random.randint(1, self.mask_noise_pixel)
        erode_kernel = np.ones((kernel, kernel), np.uint8)
        post_properties_mask = cv2.erode(properties_mask,
                                         erode_kernel,
                                         iterations=1)

        if np.sum(post_properties_mask) / float(
                image_h * image_w) > self.area_filter_ratio:
            return post_properties_mask.astype(np.float32)
        else:
            return properties_mask.astype(np.float32)


if __name__ == '__main__':
    import os
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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import SAMA_COCO_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMCollater

    cocodataset = SAMACOCOdataset(
        SAMA_COCO_path,
        set_name='train',
        positive_points_num=9,
        negative_points_num=9,
        area_filter_ratio=0.0025,
        box_noise_pixel=50,
        mask_noise_pixel=100,
        transform=transforms.Compose([
            SamResize(resize=1024),
            SamRandomHorizontalFlip(prob=0.5),
            #   SamNormalize(
            #       mean=[123.675, 116.28, 103.53],
            #       std=[58.395, 57.12, 57.375]),
        ]))

    count = 0
    for per_sample in tqdm(cocodataset):
        print('1111', per_sample['origin_image'].shape,
              per_sample['origin_bbox'].shape, per_sample['origin_mask'].shape,
              per_sample['origin_size'], per_sample['image'].shape,
              per_sample['box'].shape, per_sample['mask'].shape,
              per_sample['size'], per_sample['positive_prompt_point'].shape,
              per_sample['negative_prompt_point'].shape,
              per_sample['prompt_box'].shape, per_sample['prompt_mask'].shape)
        print('2222', per_sample['origin_image'].dtype,
              per_sample['origin_bbox'].dtype, per_sample['origin_mask'].dtype,
              per_sample['origin_size'].dtype, per_sample['image'].dtype,
              per_sample['box'].dtype, per_sample['mask'].dtype,
              per_sample['size'].dtype,
              per_sample['positive_prompt_point'].dtype,
              per_sample['negative_prompt_point'].dtype,
              per_sample['prompt_box'].dtype, per_sample['prompt_mask'].dtype)
        print('3333', per_sample['origin_bbox'], per_sample['origin_size'],
              per_sample['box'], per_sample['size'],
              per_sample['positive_prompt_point'][0],
              per_sample['negative_prompt_point'][0], per_sample['prompt_box'])

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # origin_image = np.ascontiguousarray(per_sample['origin_image'],
        #                                     dtype=np.uint8)
        # origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
        # origin_bbox = per_sample['origin_bbox']
        # origin_mask = per_sample['origin_mask']

        # origin_image_for_box = copy.deepcopy(origin_image).astype('uint8')
        # per_image_origin_box = (origin_bbox[0:4]).astype(np.int32)
        # per_image_origin_box_class_index = origin_bbox[4].astype(np.int32)
        # class_name, class_color = COCO_CLASSES[
        #     per_image_origin_box_class_index], COCO_CLASSES_COLOR[
        #         per_image_origin_box_class_index]
        # left_top, right_bottom = (per_image_origin_box[0],
        #                           per_image_origin_box[1]), (
        #                               per_image_origin_box[2],
        #                               per_image_origin_box[3])
        # cv2.rectangle(origin_image_for_box,
        #               left_top,
        #               right_bottom,
        #               color=class_color,
        #               thickness=2,
        #               lineType=cv2.LINE_AA)
        # text = f'{class_name}'
        # text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        # fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
        #                      left_top[1] - text_size[1] - 3)
        # cv2.rectangle(origin_image_for_box,
        #               left_top,
        #               fill_right_bottom,
        #               color=class_color,
        #               thickness=-1,
        #               lineType=cv2.LINE_AA)
        # cv2.putText(origin_image_for_box,
        #             text, (left_top[0], left_top[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             color=(0, 0, 0),
        #             thickness=1,
        #             lineType=cv2.LINE_AA)

        # origin_image_for_mask = copy.deepcopy(origin_image).astype('uint8')
        # origin_mask = origin_mask.astype('uint8')
        # per_origin_image_mask = np.zeros((origin_image_for_mask.shape[0],
        #                                   origin_image_for_mask.shape[1], 3))
        # per_origin_image_contours = []
        # origin_mask = np.nonzero(origin_mask == 1.)
        # if len(origin_mask[0]) > 0:
        #     per_origin_image_mask[origin_mask[0], origin_mask[1]] = class_color
        # new_per_origin_image_mask = np.zeros(
        #     (origin_image_for_mask.shape[0], origin_image_for_mask.shape[1]))
        # if len(origin_mask[0]) > 0:
        #     new_per_origin_image_mask[origin_mask[0], origin_mask[1]] = 255
        # contours, _ = cv2.findContours(
        #     new_per_origin_image_mask.astype('uint8'), cv2.RETR_TREE,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # per_origin_image_contours.append(contours)
        # per_origin_image_mask = per_origin_image_mask.astype('uint8')
        # per_origin_image_mask = cv2.cvtColor(per_origin_image_mask,
        #                                      cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_origin_image_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_origin_image_mask[
        #         all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
        #             origin_image_for_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 0.5,
        #             per_origin_image_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_origin_image_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_origin_image_mask[no_class_mask[0],
        #                           no_class_mask[1]] = origin_image_for_mask[
        #                               no_class_mask[0], no_class_mask[1]]
        # for contours in per_origin_image_contours:
        #     cv2.drawContours(per_origin_image_mask, contours, -1,
        #                      (255, 255, 255), 2)

        # cv2.imencode('.jpg', origin_image_for_box)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_origin_image_with_box.jpg'))
        # cv2.imencode('.jpg', per_origin_image_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_origin_image_with_mask.jpg'))

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # box = per_sample['box']
        # mask = per_sample['mask']

        # image_for_box = copy.deepcopy(image)
        # per_image_box = (box[0:4]).astype(np.int32)
        # per_image_origin_box_class_index = origin_bbox[4].astype(np.int32)
        # class_name, class_color = COCO_CLASSES[
        #     per_image_origin_box_class_index], COCO_CLASSES_COLOR[
        #         per_image_origin_box_class_index]
        # left_top, right_bottom = (per_image_box[0],
        #                           per_image_box[1]), (per_image_box[2],
        #                                               per_image_box[3])
        # cv2.rectangle(image_for_box,
        #               left_top,
        #               right_bottom,
        #               color=class_color,
        #               thickness=2,
        #               lineType=cv2.LINE_AA)
        # text = f'{class_name}'
        # text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        # fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
        #                      left_top[1] - text_size[1] - 3)
        # cv2.rectangle(image_for_box,
        #               left_top,
        #               fill_right_bottom,
        #               color=class_color,
        #               thickness=-1,
        #               lineType=cv2.LINE_AA)
        # cv2.putText(image_for_box,
        #             text, (left_top[0], left_top[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             color=(0, 0, 0),
        #             thickness=1,
        #             lineType=cv2.LINE_AA)

        # image_for_mask = copy.deepcopy(image).astype('uint8')
        # mask = mask.astype('uint8')
        # per_image_mask = np.zeros(
        #     (image_for_mask.shape[0], image_for_mask.shape[1], 3))
        # per_image_contours = []
        # mask = np.nonzero(mask == 1.)
        # per_image_mask[mask[0], mask[1]] = class_color
        # new_per_image_mask = np.zeros(
        #     (image_for_mask.shape[0], image_for_mask.shape[1]))
        # new_per_image_mask[mask[0], mask[1]] = 255
        # contours, _ = cv2.findContours(new_per_image_mask.astype('uint8'),
        #                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # per_image_contours.append(contours)
        # per_image_mask = per_image_mask.astype('uint8')
        # per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_image_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_image_mask[all_classes_mask[0],
        #                    all_classes_mask[1]] = cv2.addWeighted(
        #                        image_for_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 0.5,
        #                        per_image_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_image_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_image_mask[no_class_mask[0],
        #                    no_class_mask[1]] = image_for_mask[no_class_mask[0],
        #                                                       no_class_mask[1]]
        # for contours in per_image_contours:
        #     cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 2)

        # cv2.imencode('.jpg', image_for_box)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_box.jpg'))
        # cv2.imencode('.jpg', per_image_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_mask.jpg'))

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # positive_prompt_point = per_sample['positive_prompt_point']
        # negative_prompt_point = per_sample['negative_prompt_point']
        # prompt_box = per_sample['prompt_box']
        # prompt_mask = per_sample['prompt_mask']
        # positive_prompt_point_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # negative_prompt_point_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # prompt_box_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # prompt_mask_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]

        # image_for_prompt_box = copy.deepcopy(image)

        # for per_point in positive_prompt_point:
        #     cv2.circle(image_for_prompt_box,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                positive_prompt_point_color, -1)

        # for per_point in negative_prompt_point:
        #     cv2.circle(image_for_prompt_box,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                negative_prompt_point_color, -1)

        # per_image_prompt_box = (prompt_box[0:4]).astype(np.int32)
        # left_top, right_bottom = (per_image_prompt_box[0],
        #                           per_image_prompt_box[1]), (
        #                               per_image_prompt_box[2],
        #                               per_image_prompt_box[3])
        # cv2.rectangle(image_for_prompt_box,
        #               left_top,
        #               right_bottom,
        #               color=prompt_box_color,
        #               thickness=2,
        #               lineType=cv2.LINE_AA)
        # text = f'prompt_box'
        # text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        # fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
        #                      left_top[1] - text_size[1] - 3)
        # cv2.rectangle(image_for_prompt_box,
        #               left_top,
        #               fill_right_bottom,
        #               color=prompt_box_color,
        #               thickness=-1,
        #               lineType=cv2.LINE_AA)
        # cv2.putText(image_for_prompt_box,
        #             text, (left_top[0], left_top[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             color=(0, 0, 0),
        #             thickness=1,
        #             lineType=cv2.LINE_AA)

        # image_for_prompt_mask = copy.deepcopy(image).astype('uint8')

        # for per_point in positive_prompt_point:
        #     cv2.circle(image_for_prompt_mask,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                positive_prompt_point_color, -1)

        # for per_point in negative_prompt_point:
        #     cv2.circle(image_for_prompt_mask,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                negative_prompt_point_color, -1)

        # prompt_mask = prompt_mask.astype('uint8')
        # per_image_prompt_mask = np.zeros((image_for_prompt_mask.shape[0],
        #                                   image_for_prompt_mask.shape[1], 3))
        # per_image_prompt_contours = []
        # prompt_mask = np.nonzero(prompt_mask == 1.)
        # if len(prompt_mask[0]) > 0:
        #     per_image_prompt_mask[prompt_mask[0],
        #                           prompt_mask[1]] = prompt_mask_color
        # new_per_image_prompt_mask = np.zeros(
        #     (image_for_prompt_mask.shape[0], image_for_prompt_mask.shape[1]))
        # new_per_image_prompt_mask[prompt_mask[0], prompt_mask[1]] = 255
        # contours, _ = cv2.findContours(
        #     new_per_image_prompt_mask.astype('uint8'), cv2.RETR_TREE,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # per_image_prompt_contours.append(contours)
        # per_image_prompt_mask = per_image_prompt_mask.astype('uint8')
        # per_image_prompt_mask = cv2.cvtColor(per_image_prompt_mask,
        #                                      cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_image_prompt_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_image_prompt_mask[
        #         all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
        #             image_for_prompt_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 0.5,
        #             per_image_prompt_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_image_prompt_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_image_prompt_mask[no_class_mask[0],
        #                           no_class_mask[1]] = image_for_prompt_mask[
        #                               no_class_mask[0], no_class_mask[1]]
        # for contours in per_image_prompt_contours:
        #     cv2.drawContours(per_image_prompt_mask, contours, -1,
        #                      (255, 255, 255), 2)

        # cv2.imencode('.jpg', image_for_prompt_box)[1].tofile(
        #     os.path.join(temp_dir,
        #                  f'idx_{count}_image_with_prompt_point_box.jpg'))
        # cv2.imencode('.jpg', per_image_prompt_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_prompt_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader

    # collater = SAMCollater(resize=1024,
    #                        positive_point_num_range=[1, 9],
    #                        negative_point_num_range=0,
    #                        batch_align_random_point_num=False,
    #                        positive_negative_point_num_ratio=None)
    collater = SAMCollater(resize=1024,
                           positive_point_num_range=[1, 9],
                           negative_point_num_range=[1, 9],
                           batch_align_random_point_num=True,
                           positive_negative_point_num_ratio=1)
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        print('3333', len(origin_images), len(origin_bboxs), len(origin_masks),
              len(origin_sizes))

        print('4444', len(input_images), len(input_boxs), len(input_masks),
              len(sizes))
        print('5555', input_images[0].shape, input_images[1].shape,
              input_boxs[0].shape, input_boxs[1].shape, input_masks[0].shape,
              input_masks[1].shape, sizes[0], sizes[1])
        print('6666', input_images[0].dtype, input_boxs[0].dtype,
              input_masks[0].dtype, sizes[0].dtype)

        print('7777', len(input_positive_prompt_points),
              len(input_negative_prompt_points), len(input_prompt_points))
        if input_positive_prompt_points[
                0] is not None and input_negative_prompt_points[0] is not None:
            print('8888', input_positive_prompt_points[0].shape,
                  input_positive_prompt_points[1].shape,
                  input_negative_prompt_points[0].shape,
                  input_negative_prompt_points[1].shape,
                  input_prompt_points[0].shape, input_prompt_points[1].shape)
            print('9999', input_positive_prompt_points[0].dtype,
                  input_negative_prompt_points[0].dtype,
                  input_prompt_points[0].dtype)

        print('9191', len(input_prompt_boxs), len(input_prompt_masks),
              batch_images.shape, batch_masks.shape, len(batch_prompts),
              torch.unique(batch_masks))
        print('9292', input_prompt_boxs[0].shape, input_prompt_boxs[1].shape,
              input_prompt_masks[0].shape, input_prompt_masks[1].shape)
        print('9393', input_prompt_boxs[0].dtype, input_prompt_masks[0].dtype)

        print('9494', batch_prompts[0]['prompt_point'].shape,
              batch_prompts[0]['prompt_box'].shape,
              batch_prompts[0]['prompt_mask'].shape)
        print('9595', batch_prompts[1]['prompt_point'].shape,
              batch_prompts[1]['prompt_box'].shape,
              batch_prompts[1]['prompt_mask'].shape)
        print('9696', batch_prompts[0]['prompt_point'].dtype,
              batch_prompts[0]['prompt_box'].dtype,
              batch_prompts[0]['prompt_mask'].dtype)

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for i, (per_origin_image, per_origin_image_box,
                per_origin_image_mask) in enumerate(
                    zip(origin_images, origin_bboxs, origin_masks)):
            origin_image = np.ascontiguousarray(per_origin_image,
                                                dtype=np.uint8)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
            origin_bbox = per_origin_image_box
            origin_mask = per_origin_image_mask

            origin_image_for_box = copy.deepcopy(origin_image).astype('uint8')
            per_image_origin_box = (origin_bbox[0:4]).astype(np.int32)
            box_color = [int(np.random.choice(range(256))) for _ in range(3)]
            left_top, right_bottom = (per_image_origin_box[0],
                                      per_image_origin_box[1]), (
                                          per_image_origin_box[2],
                                          per_image_origin_box[3])
            cv2.rectangle(origin_image_for_box,
                          left_top,
                          right_bottom,
                          color=box_color,
                          thickness=2,
                          lineType=cv2.LINE_AA)
            text = 'object_box'
            text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
            fill_right_bottom = (max(left_top[0] + text_size[0],
                                     right_bottom[0]),
                                 left_top[1] - text_size[1] - 3)
            cv2.rectangle(origin_image_for_box,
                          left_top,
                          fill_right_bottom,
                          color=box_color,
                          thickness=-1,
                          lineType=cv2.LINE_AA)
            cv2.putText(origin_image_for_box,
                        text, (left_top[0], left_top[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

            origin_image_for_mask = copy.deepcopy(origin_image).astype('uint8')
            origin_mask = origin_mask.astype('uint8')
            per_origin_image_mask = np.zeros(
                (origin_image_for_mask.shape[0],
                 origin_image_for_mask.shape[1], 3))
            per_origin_image_contours = []
            origin_mask = np.nonzero(origin_mask == 1.)
            mask_color = [int(np.random.choice(range(256))) for _ in range(3)]
            if len(origin_mask[0]) > 0:
                per_origin_image_mask[origin_mask[0],
                                      origin_mask[1]] = mask_color
            new_per_origin_image_mask = np.zeros(
                (origin_image_for_mask.shape[0],
                 origin_image_for_mask.shape[1]))
            if len(origin_mask[0]) > 0:
                new_per_origin_image_mask[origin_mask[0], origin_mask[1]] = 255
            contours, _ = cv2.findContours(
                new_per_origin_image_mask.astype('uint8'), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            per_origin_image_contours.append(contours)
            per_origin_image_mask = per_origin_image_mask.astype('uint8')
            per_origin_image_mask = cv2.cvtColor(per_origin_image_mask,
                                                 cv2.COLOR_RGBA2BGR)
            all_classes_mask = np.nonzero(per_origin_image_mask != 0)
            if len(all_classes_mask[0]) > 0:
                per_origin_image_mask[
                    all_classes_mask[0],
                    all_classes_mask[1]] = cv2.addWeighted(
                        origin_image_for_mask[all_classes_mask[0],
                                              all_classes_mask[1]], 0.5,
                        per_origin_image_mask[all_classes_mask[0],
                                              all_classes_mask[1]], 1, 0)
            no_class_mask = np.nonzero(per_origin_image_mask == 0)
            if len(no_class_mask[0]) > 0:
                per_origin_image_mask[
                    no_class_mask[0],
                    no_class_mask[1]] = origin_image_for_mask[no_class_mask[0],
                                                              no_class_mask[1]]
            for contours in per_origin_image_contours:
                cv2.drawContours(per_origin_image_mask, contours, -1,
                                 (255, 255, 255), 2)

            cv2.imencode('.jpg', origin_image_for_box)[1].tofile(
                os.path.join(temp_dir,
                             f'idx_{count}_{i}_origin_image_with_box.jpg'))
            cv2.imencode('.jpg', per_origin_image_mask)[1].tofile(
                os.path.join(temp_dir,
                             f'idx_{count}_{i}_origin_image_with_mask.jpg'))

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for i, (per_image, per_image_box, per_image_mask) in enumerate(
                zip(input_images, input_boxs, input_masks)):
            per_image = per_image.permute(1, 2, 0).cpu().numpy()
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

            per_image_box = per_image_box.cpu().numpy()
            per_image_mask = per_image_mask.cpu().numpy()

            box_color = [int(np.random.choice(range(256))) for _ in range(3)]
            mask_color = [int(np.random.choice(range(256))) for _ in range(3)]

            image_for_box = copy.deepcopy(per_image).astype('uint8')
            per_image_box = (per_image_box[0:4]).astype(np.int32)
            left_top, right_bottom = (per_image_box[0],
                                      per_image_box[1]), (per_image_box[2],
                                                          per_image_box[3])
            cv2.rectangle(image_for_box,
                          left_top,
                          right_bottom,
                          color=box_color,
                          thickness=2,
                          lineType=cv2.LINE_AA)
            text = f'object_box'
            text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
            fill_right_bottom = (max(left_top[0] + text_size[0],
                                     right_bottom[0]),
                                 left_top[1] - text_size[1] - 3)
            cv2.rectangle(image_for_box,
                          left_top,
                          fill_right_bottom,
                          color=box_color,
                          thickness=-1,
                          lineType=cv2.LINE_AA)
            cv2.putText(image_for_box,
                        text, (left_top[0], left_top[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

            image_for_mask = copy.deepcopy(per_image).astype('uint8')
            per_image_mask = per_image_mask.astype('uint8')

            per_image_draw_mask = np.zeros(
                (image_for_mask.shape[0], image_for_mask.shape[1], 3))
            per_image_contours = []
            per_image_mask = np.nonzero(per_image_mask == 1.)
            if len(per_image_mask[0]) > 0:
                per_image_draw_mask[per_image_mask[0],
                                    per_image_mask[1]] = mask_color
            new_per_image_draw_mask = np.zeros(
                (image_for_mask.shape[0], image_for_mask.shape[1]))
            if len(per_image_mask[0]) > 0:
                new_per_image_draw_mask[per_image_mask[0],
                                        per_image_mask[1]] = 255
            contours, _ = cv2.findContours(
                new_per_image_draw_mask.astype('uint8'), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            per_image_contours.append(contours)
            per_image_draw_mask = per_image_draw_mask.astype('uint8')
            per_image_draw_mask = cv2.cvtColor(per_image_draw_mask,
                                               cv2.COLOR_RGBA2BGR)
            all_classes_mask = np.nonzero(per_image_draw_mask != 0)
            if len(all_classes_mask[0]) > 0:
                per_image_draw_mask[all_classes_mask[0],
                                    all_classes_mask[1]] = cv2.addWeighted(
                                        image_for_mask[all_classes_mask[0],
                                                       all_classes_mask[1]],
                                        0.5, per_image_draw_mask[
                                            all_classes_mask[0],
                                            all_classes_mask[1]], 1, 0)
            no_class_mask = np.nonzero(per_image_draw_mask == 0)
            if len(no_class_mask[0]) > 0:
                per_image_draw_mask[no_class_mask[0],
                                    no_class_mask[1]] = image_for_mask[
                                        no_class_mask[0], no_class_mask[1]]
            for contours in per_image_contours:
                cv2.drawContours(per_image_draw_mask, contours, -1,
                                 (255, 255, 255), 2)

            cv2.imencode('.jpg', image_for_box)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}_image_with_box.jpg'))
            cv2.imencode('.jpg', per_image_draw_mask)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}_image_with_mask.jpg'))

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for i, (per_image, per_image_positive_prompt_points,
                per_image_negative_prompt_points, per_image_prompt_box,
                per_image_prompt_mask) in enumerate(
                    zip(input_images, input_positive_prompt_points,
                        input_negative_prompt_points, input_prompt_boxs,
                        input_prompt_masks)):
            per_image = per_image.permute(1, 2, 0).cpu().numpy()
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

            if per_image_positive_prompt_points is not None:
                per_image_positive_prompt_points = per_image_positive_prompt_points.cpu(
                ).numpy()
            if per_image_negative_prompt_points is not None:
                per_image_negative_prompt_points = per_image_negative_prompt_points.cpu(
                ).numpy()
            per_image_prompt_box = per_image_prompt_box.cpu().numpy()
            per_image_prompt_mask = per_image_prompt_mask.cpu().numpy()

            per_image_prompt_mask = cv2.resize(
                per_image_prompt_mask, (per_image_prompt_mask.shape[1] * 4,
                                        per_image_prompt_mask.shape[0] * 4),
                interpolation=cv2.INTER_NEAREST)

            positive_prompt_point_color = [
                int(np.random.choice(range(256))) for _ in range(3)
            ]
            negative_prompt_point_color = [
                int(np.random.choice(range(256))) for _ in range(3)
            ]
            prompt_box_color = [
                int(np.random.choice(range(256))) for _ in range(3)
            ]
            prompt_mask_color = [
                int(np.random.choice(range(256))) for _ in range(3)
            ]

            image_for_prompt_box = copy.deepcopy(per_image)

            if per_image_positive_prompt_points is not None:
                for per_point in per_image_positive_prompt_points:
                    cv2.circle(image_for_prompt_box,
                               (int(per_point[0]), int(per_point[1])), 10,
                               positive_prompt_point_color, -1)

            if per_image_negative_prompt_points is not None:
                for per_point in per_image_negative_prompt_points:
                    cv2.circle(image_for_prompt_box,
                               (int(per_point[0]), int(per_point[1])), 10,
                               negative_prompt_point_color, -1)

            per_image_prompt_box = (per_image_prompt_box[0:4]).astype(np.int32)
            left_top, right_bottom = (per_image_prompt_box[0],
                                      per_image_prompt_box[1]), (
                                          per_image_prompt_box[2],
                                          per_image_prompt_box[3])
            cv2.rectangle(image_for_prompt_box,
                          left_top,
                          right_bottom,
                          color=prompt_box_color,
                          thickness=2,
                          lineType=cv2.LINE_AA)
            text = f'prompt_box'
            text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
            fill_right_bottom = (max(left_top[0] + text_size[0],
                                     right_bottom[0]),
                                 left_top[1] - text_size[1] - 3)
            cv2.rectangle(image_for_prompt_box,
                          left_top,
                          fill_right_bottom,
                          color=prompt_box_color,
                          thickness=-1,
                          lineType=cv2.LINE_AA)
            cv2.putText(image_for_prompt_box,
                        text, (left_top[0], left_top[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

            image_for_prompt_mask = copy.deepcopy(per_image).astype('uint8')

            if per_image_positive_prompt_points is not None:
                for per_point in per_image_positive_prompt_points:
                    cv2.circle(image_for_prompt_mask,
                               (int(per_point[0]), int(per_point[1])), 10,
                               positive_prompt_point_color, -1)

            if per_image_negative_prompt_points is not None:
                for per_point in per_image_negative_prompt_points:
                    cv2.circle(image_for_prompt_mask,
                               (int(per_point[0]), int(per_point[1])), 10,
                               negative_prompt_point_color, -1)

            per_image_prompt_mask = per_image_prompt_mask.astype('uint8')
            per_image_prompt_draw_mask = np.zeros(
                (image_for_prompt_mask.shape[0],
                 image_for_prompt_mask.shape[1], 3))
            per_image_prompt_contours = []
            per_image_prompt_mask = np.nonzero(per_image_prompt_mask == 1.)
            if len(per_image_prompt_mask[0]) > 0:
                per_image_prompt_draw_mask[
                    per_image_prompt_mask[0],
                    per_image_prompt_mask[1]] = prompt_mask_color
            new_per_image_prompt_draw_mask = np.zeros(
                (image_for_prompt_mask.shape[0],
                 image_for_prompt_mask.shape[1]))
            if len(per_image_prompt_mask[0]) > 0:
                new_per_image_prompt_draw_mask[per_image_prompt_mask[0],
                                               per_image_prompt_mask[1]] = 255
            contours, _ = cv2.findContours(
                new_per_image_prompt_draw_mask.astype('uint8'), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            per_image_prompt_contours.append(contours)
            per_image_prompt_draw_mask = per_image_prompt_draw_mask.astype(
                'uint8')
            per_image_prompt_draw_mask = cv2.cvtColor(
                per_image_prompt_draw_mask, cv2.COLOR_RGBA2BGR)
            all_classes_mask = np.nonzero(per_image_prompt_draw_mask != 0)
            if len(all_classes_mask[0]) > 0:
                per_image_prompt_draw_mask[
                    all_classes_mask[0],
                    all_classes_mask[1]] = cv2.addWeighted(
                        image_for_prompt_mask[all_classes_mask[0],
                                              all_classes_mask[1]], 0.5,
                        per_image_prompt_draw_mask[all_classes_mask[0],
                                                   all_classes_mask[1]], 1, 0)
            no_class_mask = np.nonzero(per_image_prompt_draw_mask == 0)
            if len(no_class_mask[0]) > 0:
                per_image_prompt_draw_mask[
                    no_class_mask[0],
                    no_class_mask[1]] = image_for_prompt_mask[no_class_mask[0],
                                                              no_class_mask[1]]
            for contours in per_image_prompt_contours:
                cv2.drawContours(per_image_prompt_draw_mask, contours, -1,
                                 (255, 255, 255), 2)

            cv2.imencode('.jpg', image_for_prompt_box)[1].tofile(
                os.path.join(
                    temp_dir,
                    f'idx_{count}_{i}_image_with_prompt_point_box.jpg'))
            cv2.imencode('.jpg', per_image_prompt_draw_mask)[1].tofile(
                os.path.join(temp_dir,
                             f'idx_{count}_{i}_image_with_prompt_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
