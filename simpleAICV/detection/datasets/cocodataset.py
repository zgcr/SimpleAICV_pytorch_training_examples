import os
import cv2
import math
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

COCO_CLASSES_COLOR = [
    (156, 77, 36), (218, 3, 199), (252, 197, 160), (82, 69, 38), (132, 17, 27),
    (71, 19, 213), (108, 81, 1), (49, 54, 81), (8, 249, 143), (80, 20, 4),
    (75, 227, 112), (82, 41, 57), (157, 0, 97), (0, 209, 246), (116, 242, 109),
    (60, 225, 243), (2, 125, 5), (118, 94, 170), (171, 1, 17), (54, 97, 38),
    (16, 132, 55), (1, 90, 238), (112, 4, 197), (147, 219, 248), (253, 0, 14),
    (103, 77, 249), (149, 1, 222),
    (120, 94, 51), (88, 29, 129), (204, 29, 128), (19, 0, 244), (92, 154, 54),
    (34, 89, 7), (29, 168, 224), (111, 25, 1), (137, 70, 83), (24, 217, 19),
    (47, 170, 155), (34, 234, 107), (182, 116, 221), (102, 243, 211),
    (53, 247, 123), (147, 159, 24), (194, 147, 121), (76, 101, 233),
    (50, 11, 88), (253, 33, 83), (84, 1, 57), (248, 243, 24), (244, 79, 35),
    (162, 240, 132), (1, 32, 203), (208, 10, 8), (30, 64, 206), (234, 80, 229),
    (31, 253, 207), (110, 34, 78), (234, 72, 73), (92, 3, 16), (113, 0, 65),
    (196, 177, 53), (63, 92, 139), (76, 143, 1), (61, 93, 84), (82, 130, 157),
    (28, 2, 84), (55, 226, 12), (34, 99, 82), (47, 5, 239), (53, 100, 219),
    (132, 37, 147), (244, 156, 224), (179, 57, 59), (2, 27, 76), (0, 100, 83),
    (64, 39, 116), (170, 46, 246), (27, 51, 87), (185, 71, 0), (107, 247, 29)
]


class CocoDetection(Dataset):

    def __init__(self,
                 root_dir,
                 set_name='train2017',
                 filter_no_object_image=False,
                 transform=None):
        assert set_name in ['train2017', 'val2017'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'instances_{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.image_ids = sorted(self.coco.getImgIds())

        if filter_no_object_image:
            # filter image id without annotation,from 118287 ids to 117266 ids
            ids = []
            for image_id in self.image_ids:
                annot_ids = self.coco.getAnnIds(imgIds=image_id)
                annots = self.coco.loadAnns(annot_ids)
                if len(annots) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids
            self.image_ids = sorted(self.image_ids)

        self.cat_ids = self.coco.getCatIds()
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

        self.transform = transform

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        path = os.path.join(self.image_dir, file_name)

        image = self.load_image(idx)
        annots = self.load_annots(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'path': path,
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        image = cv2.imdecode(
            np.fromfile(os.path.join(self.image_dir, file_name),
                        dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annots = self.coco.loadAnns(annot_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        targets = np.zeros((0, 5))
        if len(annots) == 0:
            return targets.astype(np.float32)

        # filter annots
        for annot in annots:
            if 'ignore' in annot.keys():
                continue
            # bbox format:[x_min, y_min, w, h]
            bbox = annot['bbox']

            inter_w = max(0, min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
            inter_h = max(0, min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
            if inter_w * inter_h == 0:
                continue
            if bbox[2] * bbox[3] <= 1 or bbox[2] <= 1 or bbox[3] <= 1:
                continue
            if annot['category_id'] not in self.cat_ids:
                continue

            target = np.zeros((1, 5))
            target[0, :4] = bbox
            target[0, 4] = self.cat_id_to_coco_label[annot['category_id']]
            targets = np.append(targets, target, axis=0)

        # transform bbox targets from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        targets[:, 2] = targets[:, 0] + targets[:, 2]
        targets[:, 3] = targets[:, 1] + targets[:, 3]

        return targets.astype(np.float32)


class MosaicResizeCocoDetection(CocoDetection):
    '''
    When using MosaicResizeCocoDetection class, don't use YoloStyleResize/RetinaStyleResize data augment.
    Only use mixup after use mosaic augmentation.
    Total mixup prob:mosaic_prob * mixup_prob.
    If current_epoch > stop_mosaic_epoch,stop using mosaic augmentation.
    '''

    def __init__(self,
                 root_dir,
                 set_name='train2017',
                 resize=640,
                 stride=32,
                 use_multi_scale=True,
                 multi_scale_range=[0.25, 2.0],
                 mosaic_prob=0.5,
                 mosaic_multi_scale_range=[0.4, 1.0],
                 mixup_prob=0.5,
                 mixup_ratio=[0.5, 0.5],
                 current_epoch=1,
                 stop_mosaic_epoch=100,
                 filter_no_object_image=False,
                 transform=None):
        assert set_name in ['train2017', 'val2017'], 'Wrong set name!'

        self.resize = resize
        self.stride = stride
        self.use_multi_scale = use_multi_scale
        self.multi_scale_range = multi_scale_range
        self.mosaic_prob = mosaic_prob
        self.mosaic_multi_scale_range = mosaic_multi_scale_range
        self.mixup_prob = mixup_prob
        self.mixup_ratio = mixup_ratio
        self.current_epoch = current_epoch
        self.stop_mosaic_epoch = stop_mosaic_epoch
        self.transform = transform

        assert len(self.multi_scale_range) == 2
        assert self.multi_scale_range[0] < self.multi_scale_range[1]
        assert len(self.mosaic_multi_scale_range) == 2
        assert self.mosaic_multi_scale_range[
            0] < self.mosaic_multi_scale_range[1]
        assert len(self.mixup_ratio) == 2, 'wrong mixup ratio num!'
        assert self.mixup_ratio[0] + self.mixup_ratio[
            1] == 1.0, 'wrong mixup ratio total number!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'instances_{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.image_ids = sorted(self.coco.getImgIds())

        if filter_no_object_image:
            # filter image id without annotation,from 118287 ids to 117266 ids
            ids = []
            for image_id in self.image_ids:
                annot_ids = self.coco.getAnnIds(imgIds=image_id)
                annots = self.coco.loadAnns(annot_ids)
                if len(annots) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids
            self.image_ids = sorted(self.image_ids)

        self.cat_ids = self.coco.getCatIds()
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

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if np.random.uniform(
                0, 1
        ) < self.mosaic_prob and self.current_epoch <= self.stop_mosaic_epoch:
            # use mosaic augmentation
            # mosaic center x, y
            x_ctr, y_ctr = [int(self.resize), int(self.resize)]
            # 4 images ids
            image_ids = [idx] + [
                np.random.randint(0, len(self.image_ids)) for _ in range(3)
            ]
            # 4 images annots
            image_annots = []
            # combined image by 4 images
            combined_img = np.zeros(
                (int(self.resize * 2), int(self.resize * 2), 3),
                dtype=np.float32)

            for i, idx in enumerate(image_ids):
                image = self.load_image(idx)
                annots = self.load_annots(idx)

                h, w, _ = image.shape

                if self.use_multi_scale:
                    scale_range = [
                        int(self.mosaic_multi_scale_range[0] * self.resize),
                        int(self.mosaic_multi_scale_range[1] * self.resize)
                    ]
                    resize_list = [
                        i // self.stride * self.stride
                        for i in range(scale_range[0], scale_range[1] +
                                       self.stride)
                    ]
                    resize_list = list(set(resize_list))

                    random_idx = np.random.randint(0, len(resize_list))
                    final_resize = resize_list[random_idx]
                else:
                    final_resize = self.resize

                factor = final_resize / max(h, w)

                resize_h, resize_w = math.ceil(h * factor), math.ceil(w *
                                                                      factor)
                image = cv2.resize(image, (resize_w, resize_h))
                annots[:, :4] *= factor

                # top left img
                if i == 0:
                    # xmin, ymin, xmax, ymax (large image)
                    x1a, y1a, x2a, y2a = max(x_ctr - resize_w,
                                             0), max(y_ctr - resize_h,
                                                     0), x_ctr, y_ctr
                    # xmin, ymin, xmax, ymax (small image)
                    x1b, y1b, x2b, y2b = resize_w - (x2a - x1a), resize_h - (
                        y2a - y1a), resize_w, resize_h
                # top right img
                elif i == 1:
                    x1a, y1a, x2a, y2a = x_ctr, max(y_ctr - resize_h, 0), min(
                        x_ctr + resize_w, int(self.resize * 2)), y_ctr
                    x1b, y1b, x2b, y2b = 0, resize_h - (y2a - y1a), min(
                        resize_w, x2a - x1a), resize_h
                # bottom left img
                elif i == 2:
                    x1a, y1a, x2a, y2a = max(x_ctr - resize_w,
                                             0), y_ctr, x_ctr, min(
                                                 int(self.resize * 2),
                                                 y_ctr + resize_h)
                    x1b, y1b, x2b, y2b = resize_w - (x2a - x1a), 0, max(
                        x_ctr, resize_w), min(y2a - y1a, resize_h)
                # bottom right img
                elif i == 3:
                    x1a, y1a, x2a, y2a = x_ctr, y_ctr, min(
                        x_ctr + resize_w,
                        int(self.resize * 2)), min(int(self.resize * 2),
                                                   y_ctr + resize_h)
                    x1b, y1b, x2b, y2b = 0, 0, min(resize_w, x2a - x1a), min(
                        y2a - y1a, resize_h)

                # combined_img[ymin:ymax, xmin:xmax]
                combined_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
                padw, padh = x1a - x1b, y1a - y1b

                # annot coordinates transform
                if annots.shape[0] > 0:
                    annots[:, 0] = annots[:, 0] + padw
                    annots[:, 1] = annots[:, 1] + padh
                    annots[:, 2] = annots[:, 2] + padw
                    annots[:, 3] = annots[:, 3] + padh

                image_annots.append(annots)

            image_annots = np.concatenate(image_annots, axis=0)
            image_annots[:, 0:4] = np.clip(image_annots[:, 0:4], 0,
                                           int(self.resize * 2))

            image_annots = image_annots[image_annots[:, 2] -
                                        image_annots[:, 0] > 1]
            image_annots = image_annots[image_annots[:, 3] -
                                        image_annots[:, 1] > 1]

            if image_annots.shape[0] > 0 and np.random.uniform(
                    0, 1) < self.mixup_prob:
                # mixup combine images annots
                mixup_image_annots = []
                # mixup combined image by 2 images
                mixup_combined_img = np.zeros(
                    (int(self.resize * 2), int(self.resize * 2), 3),
                    dtype=np.float32)

                second_image_id = np.random.randint(0, len(self.image_ids))
                second_image = self.load_image(second_image_id)
                second_image_annots = self.load_annots(second_image_id)

                h, w, _ = second_image.shape
                second_image_multi_scale_range = [
                    min(2.0, self.multi_scale_range[0]),
                    min(2.0, self.multi_scale_range[1]),
                ]

                if self.use_multi_scale:
                    scale_range = [
                        int(second_image_multi_scale_range[0] * self.resize),
                        int(second_image_multi_scale_range[1] * self.resize)
                    ]
                    resize_list = [
                        i // self.stride * self.stride
                        for i in range(scale_range[0], scale_range[1] +
                                       self.stride)
                    ]
                    resize_list = list(set(resize_list))

                    random_idx = np.random.randint(0, len(resize_list))
                    final_resize = resize_list[random_idx]
                else:
                    final_resize = self.resize

                factor = final_resize / max(h, w)

                resize_h, resize_w = math.ceil(h * factor), math.ceil(w *
                                                                      factor)
                second_image = cv2.resize(second_image, (resize_w, resize_h))
                second_image_annots[:, :4] *= factor

                mixup_combined_img[
                    0:combined_img.shape[0],
                    0:combined_img.shape[1], :] = 0.5 * combined_img

                mixup_combined_img[
                    0:second_image.shape[0],
                    0:second_image.shape[1], :] = mixup_combined_img[
                        0:second_image.shape[0],
                        0:second_image.shape[1], :] + 0.5 * second_image

                mixup_image_annots = np.concatenate(
                    [image_annots, second_image_annots], axis=0)
                mixup_image_annots[:,
                                   0:4] = np.clip(mixup_image_annots[:, 0:4],
                                                  0, int(self.resize * 2))

                mixup_image_annots = mixup_image_annots[
                    mixup_image_annots[:, 2] - mixup_image_annots[:, 0] > 1]
                mixup_image_annots = mixup_image_annots[
                    mixup_image_annots[:, 3] - mixup_image_annots[:, 1] > 1]

                combined_img = mixup_combined_img
                image_annots = mixup_image_annots

            scale = np.array(1.).astype(np.float32)
            size = np.array([int(self.resize * 2),
                             int(self.resize * 2)]).astype(np.float32)

            combine_h, combine_w, _ = combined_img.shape

            pad_w = 0 if combine_w % 32 == 0 else 32 - combine_w % 32
            pad_h = 0 if combine_h % 32 == 0 else 32 - combine_h % 32

            padded_image = np.zeros((combine_h + pad_h, combine_w + pad_w, 3),
                                    dtype=np.uint8)

            padded_image[:combine_h, :combine_w, :] = combined_img
            padded_image = padded_image.astype(np.float32)
            image_annots = image_annots.astype(np.float32)

            sample = {
                'image': padded_image,
                'annots': image_annots,
                'scale': scale,
                'size': size,
            }

            if self.transform:
                sample = self.transform(sample)

            return sample

        else:
            image = self.load_image(idx)
            annots = self.load_annots(idx)

            scale = np.array(1.).astype(np.float32)
            size = np.array([image.shape[0],
                             image.shape[1]]).astype(np.float32)

            h, w, _ = image.shape

            if self.use_multi_scale:
                scale_range = [
                    int(self.multi_scale_range[0] * self.resize),
                    int(self.multi_scale_range[1] * self.resize)
                ]
                resize_list = [
                    i // self.stride * self.stride
                    for i in range(scale_range[0], scale_range[1] +
                                   self.stride)
                ]
                resize_list = list(set(resize_list))

                random_idx = np.random.randint(0, len(resize_list))
                final_resize = resize_list[random_idx]
            else:
                final_resize = self.resize

            factor = final_resize / max(h, w)

            resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
            image = cv2.resize(image, (resize_w, resize_h))

            pad_w = 0 if resize_w % 32 == 0 else 32 - resize_w % 32
            pad_h = 0 if resize_h % 32 == 0 else 32 - resize_h % 32

            padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                    dtype=np.float32)
            padded_image[:resize_h, :resize_w, :] = image

            factor = np.float32(factor)
            annots[:, :4] *= factor
            scale *= factor

            sample = {
                'image': image,
                'annots': annots,
                'scale': scale,
                'size': size,
            }

            if self.transform:
                sample = self.transform(sample)

            return sample


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

    from tools.path import COCO2017_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionResize, DetectionCollater, DETRDetectionCollater

    cocodataset = CocoDetection(
        COCO2017_path,
        set_name='train2017',
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            DetectionResize(resize=640,
                            stride=32,
                            resize_type='yolo_style',
                            multi_scale=False,
                            multi_scale_range=[0.8, 1.0]),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(cocodataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['annots'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # annots = per_sample['annots']

        # # draw all label boxes
        # for per_annot in annots:
        #     per_box = (per_annot[0:4]).astype(np.int32)
        #     per_box_class_index = per_annot[4].astype(np.int32)

        #     class_name, class_color = COCO_CLASSES[
        #         per_box_class_index], COCO_CLASSES_COLOR[per_box_class_index]
        #     left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
        #                                                         per_box[3])
        #     cv2.rectangle(image,
        #                   left_top,
        #                   right_bottom,
        #                   color=class_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)

        #     text = f'{class_name}'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(image,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=class_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(image,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DetectionCollater(resize=640,
                                 resize_type='yolo_style',
                                 max_annots_num=100)
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('2222', images.shape, annots.shape, scales.shape, sizes.shape)
        print('2222', images.dtype, annots.dtype, scales.dtype, sizes.dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # annots = annots.cpu().numpy()

        # for i, (per_image, per_image_annot) in enumerate(zip(images, annots)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     # draw all label boxes
        #     for per_annot in per_image_annot:
        #         per_box = (per_annot[0:4]).astype(np.int32)
        #         per_box_class_index = per_annot[4].astype(np.int32)

        #         if per_box_class_index == -1:
        #             continue

        #         class_name, class_color = COCO_CLASSES[
        #             per_box_class_index], COCO_CLASSES_COLOR[
        #                 per_box_class_index]
        #         left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
        #                                                             per_box[3])
        #         cv2.rectangle(per_image,
        #                       left_top,
        #                       right_bottom,
        #                       color=class_color,
        #                       thickness=2,
        #                       lineType=cv2.LINE_AA)

        #         text = f'{class_name}'
        #         text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #         fill_right_bottom = (max(left_top[0] + text_size[0],
        #                                  right_bottom[0]),
        #                              left_top[1] - text_size[1] - 3)
        #         cv2.rectangle(per_image,
        #                       left_top,
        #                       fill_right_bottom,
        #                       color=class_color,
        #                       thickness=-1,
        #                       lineType=cv2.LINE_AA)
        #         cv2.putText(per_image,
        #                     text, (left_top[0], left_top[1] - 2),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     color=(0, 0, 0),
        #                     thickness=1,
        #                     lineType=cv2.LINE_AA)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))

        if count < 5:
            count += 1
        else:
            break

    cocodataset = CocoDetection(
        COCO2017_path,
        set_name='train2017',
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            DetectionResize(resize=800,
                            stride=32,
                            resize_type='retina_style',
                            multi_scale=False,
                            multi_scale_range=[0.8, 1.0]),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(cocodataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['annots'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        # temp_dir = './temp5'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # annots = per_sample['annots']

        # # draw all label boxes
        # for per_annot in annots:
        #     per_box = (per_annot[0:4]).astype(np.int32)
        #     per_box_class_index = per_annot[4].astype(np.int32)

        #     class_name, class_color = COCO_CLASSES[
        #         per_box_class_index], COCO_CLASSES_COLOR[per_box_class_index]
        #     left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
        #                                                         per_box[3])
        #     cv2.rectangle(image,
        #                   left_top,
        #                   right_bottom,
        #                   color=class_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)

        #     text = f'{class_name}'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(image,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=class_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(image,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DETRDetectionCollater(resize=800,
                                     resize_type='retina_style',
                                     max_annots_num=100)
    train_loader = DataLoader(cocodataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('2222', images.shape, annots.shape, scales.shape, sizes.shape)
        print('2222', images.dtype, annots.dtype, scales.dtype, sizes.dtype)

        # temp_dir = './temp6'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # annots = annots.cpu().numpy()

        # for i, (per_image, per_image_annot) in enumerate(zip(images, annots)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     # draw all label boxes
        #     for per_annot in per_image_annot:
        #         per_box = (per_annot[0:4]).astype(np.int32)
        #         per_box_class_index = per_annot[4].astype(np.int32)

        #         if per_box_class_index == -1:
        #             continue

        #         class_name, class_color = COCO_CLASSES[
        #             per_box_class_index], COCO_CLASSES_COLOR[
        #                 per_box_class_index]
        #         left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
        #                                                             per_box[3])
        #         cv2.rectangle(per_image,
        #                       left_top,
        #                       right_bottom,
        #                       color=class_color,
        #                       thickness=2,
        #                       lineType=cv2.LINE_AA)

        #         text = f'{class_name}'
        #         text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #         fill_right_bottom = (max(left_top[0] + text_size[0],
        #                                  right_bottom[0]),
        #                              left_top[1] - text_size[1] - 3)
        #         cv2.rectangle(per_image,
        #                       left_top,
        #                       fill_right_bottom,
        #                       color=class_color,
        #                       thickness=-1,
        #                       lineType=cv2.LINE_AA)
        #         cv2.putText(per_image,
        #                     text, (left_top[0], left_top[1] - 2),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     color=(0, 0, 0),
        #                     thickness=1,
        #                     lineType=cv2.LINE_AA)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))

        if count < 5:
            count += 1
        else:
            break

    mosaiccocodataset = MosaicResizeCocoDetection(
        COCO2017_path,
        set_name='train2017',
        resize=640,
        stride=32,
        use_multi_scale=True,
        multi_scale_range=[0.5, 2.0],
        mosaic_prob=0.5,
        mosaic_multi_scale_range=[0.4, 1.0],
        mixup_prob=0.5,
        mixup_ratio=[0.5, 0.5],
        current_epoch=1,
        stop_mosaic_epoch=100,
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(mosaiccocodataset):
        print('3333', per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'])
        print('3333', per_sample['image'].dtype, per_sample['annots'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        # temp_dir = './temp3'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # annots = per_sample['annots']

        # # draw all label boxes
        # for per_annot in annots:
        #     per_box = (per_annot[0:4]).astype(np.int32)
        #     per_box_class_index = per_annot[4].astype(np.int32)
        #     class_name, class_color = COCO_CLASSES[
        #         per_box_class_index], COCO_CLASSES_COLOR[per_box_class_index]
        #     left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
        #                                                         per_box[3])
        #     cv2.rectangle(image,
        #                   left_top,
        #                   right_bottom,
        #                   color=class_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)

        #     text = f'{class_name}'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(image,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=class_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(image,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DetectionCollater(resize=1280,
                                 resize_type='yolo_style',
                                 max_annots_num=100)
    mosaic_train_loader = DataLoader(mosaiccocodataset,
                                     batch_size=4,
                                     shuffle=True,
                                     num_workers=2,
                                     collate_fn=collater)

    count = 0
    for data in tqdm(mosaic_train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('4444', images.shape, annots.shape, scales.shape, sizes.shape)
        print('4444', images.dtype, annots.dtype, scales.dtype, sizes.dtype)

        # temp_dir = './temp4'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # annots = annots.cpu().numpy()

        # for i, (per_image, per_image_annot) in enumerate(zip(images, annots)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     # draw all label boxes
        #     for per_annot in per_image_annot:
        #         per_box = (per_annot[0:4]).astype(np.int32)
        #         per_box_class_index = per_annot[4].astype(np.int32)

        #         if per_box_class_index == -1:
        #             continue

        #         class_name, class_color = COCO_CLASSES[
        #             per_box_class_index], COCO_CLASSES_COLOR[
        #                 per_box_class_index]
        #         left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
        #                                                             per_box[3])
        #         cv2.rectangle(per_image,
        #                       left_top,
        #                       right_bottom,
        #                       color=class_color,
        #                       thickness=2,
        #                       lineType=cv2.LINE_AA)

        #         text = f'{class_name}'
        #         text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #         fill_right_bottom = (max(left_top[0] + text_size[0],
        #                                  right_bottom[0]),
        #                              left_top[1] - text_size[1] - 3)
        #         cv2.rectangle(per_image,
        #                       left_top,
        #                       fill_right_bottom,
        #                       color=class_color,
        #                       thickness=-1,
        #                       lineType=cv2.LINE_AA)
        #         cv2.putText(per_image,
        #                     text, (left_top[0], left_top[1] - 2),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     color=(0, 0, 0),
        #                     thickness=1,
        #                     lineType=cv2.LINE_AA)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))

        if count < 5:
            count += 1
        else:
            break
