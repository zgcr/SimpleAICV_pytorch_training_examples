import os
import cv2
import numpy as np

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


class SamaCocoDetection(Dataset):

    def __init__(self, root_dir, set_name='train', transform=None):
        assert set_name in ['train', 'validation'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'sama_coco_{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.image_ids = self.coco.getImgIds()

        if 'train' in set_name:
            # filter image id without annotation,from 118287 ids to 117266 ids
            ids = []
            for image_id in self.image_ids:
                annot_ids = self.coco.getAnnIds(imgIds=image_id)
                annots = self.coco.loadAnns(annot_ids)
                if len(annots) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids

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
        image = self.load_image(idx)
        annots = self.load_annots(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
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
            if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
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

    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionResize, DetectionCollater, BatchAlignDETRDetectionCollater

    cocodataset = SamaCocoDetection(
        SAMA_COCO_path,
        set_name='train',
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
    # collater = DetectionCollater(resize=800,
    #                              resize_type='retina_style',
    #                              max_annots_num=100)
    collater = DetectionCollater(resize=640,
                                 resize_type='yolo_style',
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

    cocodataset = SamaCocoDetection(
        SAMA_COCO_path,
        set_name='train',
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
    collater = BatchAlignDETRDetectionCollater(max_annots_num=100)
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
