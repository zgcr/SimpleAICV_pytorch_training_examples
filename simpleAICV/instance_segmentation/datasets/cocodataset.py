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
    (241, 23, 78),
    (63, 71, 49),
    (67, 79, 143),
    (32, 250, 205),
    (136, 228, 157),
    (135, 125, 104),
    (151, 46, 171),
    (129, 37, 28),
    (3, 248, 159),
    (154, 129, 58),
    (93, 155, 200),
    (201, 98, 152),
    (187, 194, 70),
    (122, 144, 121),
    (168, 31, 32),
    (168, 68, 189),
    (173, 68, 45),
    (200, 81, 154),
    (171, 114, 139),
    (216, 211, 39),
    (187, 119, 238),
    (201, 120, 112),
    (129, 16, 164),
    (211, 3, 208),
    (169, 41, 248),
    (100, 77, 159),
    (140, 104, 243),
    (26, 165, 41),
    (225, 176, 197),
    (35, 212, 67),
    (160, 245, 68),
    (7, 87, 70),
    (52, 107, 85),
    (103, 64, 188),
    (245, 76, 17),
    (248, 154, 59),
    (77, 45, 123),
    (210, 95, 230),
    (172, 188, 171),
    (250, 44, 233),
    (161, 71, 46),
    (144, 14, 134),
    (231, 142, 186),
    (34, 1, 200),
    (144, 42, 108),
    (222, 70, 139),
    (138, 62, 77),
    (178, 99, 61),
    (17, 94, 132),
    (93, 248, 254),
    (244, 116, 204),
    (138, 165, 238),
    (44, 216, 225),
    (224, 164, 12),
    (91, 126, 184),
    (116, 254, 49),
    (70, 250, 105),
    (252, 237, 54),
    (196, 136, 21),
    (234, 13, 149),
    (66, 43, 47),
    (2, 73, 234),
    (118, 181, 5),
    (105, 99, 225),
    (150, 253, 92),
    (59, 2, 121),
    (176, 190, 223),
    (91, 62, 47),
    (198, 124, 140),
    (100, 135, 185),
    (20, 207, 98),
    (216, 38, 133),
    (17, 202, 208),
    (216, 135, 81),
    (212, 203, 33),
    (108, 135, 76),
    (28, 47, 170),
    (142, 128, 121),
    (23, 161, 179),
    (33, 183, 224),
]


class CocoInstanceSegmentation(Dataset):

    def __init__(self, root_dir, set_name='train2017', transform=None):
        assert set_name in ['train2017', 'val2017'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'instances_{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.image_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()

        # filter image id without annotation,from 118287 ids to 117266 ids
        ids = []
        for image_id in self.image_ids:
            annot_ids = self.coco.getAnnIds(imgIds=image_id)
            annots = self.coco.loadAnns(annot_ids)
            if len(annots) == 0:
                continue

            image_info = self.coco.loadImgs(image_id)[0]
            image_h, image_w = image_info['height'], image_info['width']

            illegal_flag = False
            for annot in annots:
                if 'ignore' in annot.keys():
                    illegal_flag = True
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
                    illegal_flag = True
                    continue

                if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
                    illegal_flag = True
                    continue
                if annot['category_id'] not in self.cat_ids:
                    illegal_flag = True
                    continue

            if illegal_flag:
                continue

            ids.append(image_id)

        self.image_ids = ids

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
        image_boxes, image_masks = self.load_mask(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)
        origin_size = size.copy()

        sample = {
            'image': image,
            'box': image_boxes,
            'mask': image_masks,
            'scale': scale,
            'size': size,
            'origin_size': origin_size,
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

    def load_mask(self, idx):
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annots = self.coco.loadAnns(annot_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        target_boxes = np.zeros((0, 5))
        target_masks = np.zeros((image_h, image_w, 0))

        if len(annots) == 0:
            return target_boxes.astype(np.float32), target_masks.astype(
                np.float32)

        # filter annots
        for annot in annots:
            if 'ignore' in annot.keys():
                continue

            # bbox format:[x_min, y_min, w, h]
            bbox = annot['bbox'].copy()

            inter_w = max(0, min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
            inter_h = max(0, min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
            if inter_w * inter_h == 0:
                continue
            if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
                continue
            if annot['category_id'] not in self.cat_ids:
                continue

            target_box = np.zeros((1, 5))
            target_box[0, :4] = bbox
            target_box[0, 4] = self.cat_id_to_coco_label[annot['category_id']]
            target_boxes = np.append(target_boxes, target_box, axis=0)

            target_mask = np.zeros((image_h, image_w, 1))
            annot_mask = self.coco.annToMask(annot).copy()
            target_mask[:, :, 0] = annot_mask
            target_masks = np.append(target_masks, target_mask, axis=2)

        # transform bbox targets from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        target_boxes[:, 2] = target_boxes[:, 0] + target_boxes[:, 2]
        target_boxes[:, 3] = target_boxes[:, 1] + target_boxes[:, 3]

        assert target_boxes.shape[0] == target_masks.shape[-1]

        return target_boxes.astype(np.float32), target_masks.astype(np.float32)


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

    from simpleAICV.instance_segmentation.common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, InstanceSegmentationCollater

    cocodataset = CocoInstanceSegmentation(
        COCO2017_path,
        set_name='val2017',
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=800,
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=False,
                                       multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(cocodataset):
        print('1111', per_sample['image'].shape, per_sample['box'].shape,
              per_sample['mask'].shape, per_sample['scale'],
              per_sample['size'], per_sample['origin_size'])
        print('1111', per_sample['image'].dtype, per_sample['box'].dtype,
              per_sample['mask'].dtype, per_sample['scale'].dtype,
              per_sample['size'].dtype, per_sample['origin_size'].dtype)

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # masks = per_sample['mask']
        # mask_jpg = np.zeros((image.shape[0], image.shape[1], 3))

        # masks_num = masks.shape[2]

        # masks_class_color = []
        # for _ in range(masks_num):
        #     masks_class_color.append(list(np.random.choice(range(256),
        #                                                    size=3)))

        # print("1212", masks_num, len(masks_class_color), masks_class_color[0])
        # for i in range(masks_num):
        #     per_mask = masks[:, :, i]
        #     per_mask_color = np.array(
        #         (masks_class_color[i][0], masks_class_color[i][1],
        #          masks_class_color[i][2]))

        #     per_mask = np.expand_dims(per_mask, axis=-1)
        #     per_mask = np.tile(per_mask, (1, 1, 3))
        #     mask_color = np.expand_dims(np.expand_dims(per_mask_color, axis=0),
        #                                 axis=0)

        #     per_mask = per_mask * mask_color
        #     image = 0.5 * per_mask + image
        #     mask_jpg += per_mask

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))
        # cv2.imencode('.jpg', mask_jpg)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = InstanceSegmentationCollater(resize=800,
                                            resize_type='yolo_style')
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, boxes, masks, scales, sizes, origin_sizes = data[
            'image'], data['box'], data['mask'], data['scale'], data[
                'size'], data['origin_size']
        print('2222', images.shape, len(boxes), len(masks), scales.shape,
              sizes.shape, origin_sizes.shape)
        print('2222', images.dtype, boxes[0].dtype, masks[0].dtype,
              scales.dtype, sizes.dtype, origin_sizes.dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()

        # for i, (per_image, per_image_masks) in enumerate(zip(images, masks)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_mask_jpg = np.zeros(
        #         (per_image.shape[0], per_image.shape[1], 3))

        #     per_image_masks_num = per_image_masks.shape[0]

        #     per_image_masks_class_color = []
        #     for _ in range(per_image_masks_num):
        #         per_image_masks_class_color.append(
        #             list(np.random.choice(range(256), size=3)))

        #     for i in range(per_image_masks_num):
        #         per_mask = per_image_masks[i, :, :]
        #         per_mask_color = np.array((per_image_masks_class_color[i][0],
        #                                    per_image_masks_class_color[i][1],
        #                                    per_image_masks_class_color[i][2]))

        #         per_mask = np.expand_dims(per_mask, axis=-1)
        #         per_mask = np.tile(per_mask, (1, 1, 3))
        #         mask_color = np.expand_dims(np.expand_dims(per_mask_color,
        #                                                    axis=0),
        #                                     axis=0)

        #         per_mask = per_mask * mask_color
        #         per_image = 0.5 * per_mask + per_image
        #         per_image_mask_jpg += per_mask

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
        #     cv2.imencode('.jpg', per_image_mask_jpg)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

        if count < 5:
            count += 1
        else:
            break