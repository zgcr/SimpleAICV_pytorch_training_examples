import os
import copy
import cv2
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


class SamaCocoInstanceSegmentation(Dataset):

    def __init__(self,
                 root_dir,
                 set_name='train',
                 filter_no_object_image=False,
                 transform=None):
        assert set_name in ['train', 'validation'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'sama_coco_{set_name}.json')
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
        image_boxes, image_masks = self.load_mask(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)
        origin_size = size.copy()

        sample = {
            'path': path,
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
            bbox = annot['bbox']

            inter_w = max(0, min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
            inter_h = max(0, min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
            if inter_w * inter_h == 0:
                continue
            if bbox[2] * bbox[3] <= 1 or bbox[2] <= 1 or bbox[3] <= 1:
                continue
            if annot['category_id'] not in self.cat_ids:
                continue

            target_box = np.zeros((1, 5))
            target_box[0, :4] = bbox
            target_box[0, 4] = self.cat_id_to_coco_label[annot['category_id']]
            target_boxes = np.append(target_boxes, target_box, axis=0)

            target_mask = np.zeros((image_h, image_w, 1))
            annot_mask = self.coco.annToMask(annot)
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

    from tools.path import SAMA_COCO_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.instance_segmentation_common import InstanceSegmentationResize, RandomHorizontalFlip, Normalize, InstanceSegmentationTrainCollater

    cocodataset = SamaCocoInstanceSegmentation(
        SAMA_COCO_path,
        set_name='train',
        filter_no_object_image=True,
        transform=transforms.Compose([
            InstanceSegmentationResize(resize=1024,
                                       stride=32,
                                       resize_type='yolo_style',
                                       multi_scale=True,
                                       multi_scale_range=[0.8, 1.0]),
            RandomHorizontalFlip(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(cocodataset):
        print('1111', per_sample['path'])
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
        # image_not_draw = copy.deepcopy(image)
        # mask = per_sample['mask']
        # masks_num = mask.shape[2]

        # masks_class_color = []
        # for _ in range(masks_num):
        #     masks_class_color.append(list(np.random.choice(range(256),
        #                                                    size=3)))
        # print("1212", masks_num, len(masks_class_color), masks_class_color[0])

        # per_image_mask = np.zeros((image.shape[0], image.shape[1], 3))
        # per_image_contours = []
        # for i in range(masks_num):
        #     per_mask = mask[:, :, i]
        #     per_mask_color = np.array(
        #         (masks_class_color[i][0], masks_class_color[i][1],
        #          masks_class_color[i][2]))

        #     per_object_mask = np.nonzero(per_mask == 1.)
        #     per_image_mask[per_object_mask[0],
        #                    per_object_mask[1]] = per_mask_color

        #     # get contours
        #     new_per_image_mask = np.zeros((image.shape[0], image.shape[1]))
        #     new_per_image_mask[per_object_mask[0], per_object_mask[1]] = 255
        #     contours, _ = cv2.findContours(new_per_image_mask.astype(np.uint8),
        #                                    cv2.RETR_TREE,
        #                                    cv2.CHAIN_APPROX_SIMPLE)
        #     per_image_contours.append(contours)

        # per_image_mask = per_image_mask.astype(np.uint8)
        # per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

        # all_object_mask = np.nonzero(per_image_mask != 0)
        # per_image_mask[all_object_mask[0],
        #                all_object_mask[1]] = cv2.addWeighted(
        #                    image[all_object_mask[0], all_object_mask[1]], 0.5,
        #                    per_image_mask[all_object_mask[0],
        #                                   all_object_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_image_mask == 0)
        # per_image_mask[no_class_mask[0],
        #                no_class_mask[1]] = image[no_class_mask[0],
        #                                          no_class_mask[1]]
        # for contours in per_image_contours:
        #     cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 2)

        # cv2.imencode('.jpg', image_not_draw)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))
        # cv2.imencode('.jpg', per_image_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = InstanceSegmentationTrainCollater(resize=1024,
                                                 resize_type='yolo_style')
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, labels, sizes = data['image'], data['mask'], data[
            'label'], data['size']
        print('1111', images.shape, len(masks), len(labels), sizes.shape)

        for per_image_masks, per_image_labels in zip(masks, labels):
            print('2222', per_image_masks.shape, per_image_labels.shape)
            print('3333', per_image_labels)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()

        # for image_idx, (per_image, per_image_masks,
        #                 per_image_labels) in enumerate(
        #                     zip(images, masks, labels)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)
        #     per_image_not_draw = copy.deepcopy(per_image)
        #     per_image_masks = per_image_masks.permute(
        #         1, 2, 0).cpu().numpy().astype(np.uint8)
        #     per_image_masks_num = per_image_masks.shape[2]

        #     per_image_masks_class_color = []
        #     for _ in range(per_image_masks_num):
        #         per_image_masks_class_color.append(
        #             list(np.random.choice(range(256), size=3)))
        #     print("1212", per_image_masks_num,
        #           len(per_image_masks_class_color),
        #           per_image_masks_class_color[0])

        #     per_image_new_mask = np.zeros(
        #         (per_image.shape[0], per_image.shape[1], 3))
        #     per_image_contours = []
        #     for i in range(per_image_masks_num):
        #         per_mask = per_image_masks[:, :, i]
        #         per_mask_color = np.array((per_image_masks_class_color[i][0],
        #                                    per_image_masks_class_color[i][1],
        #                                    per_image_masks_class_color[i][2]))

        #         per_object_mask = np.nonzero(per_mask == 1.)
        #         per_image_new_mask[per_object_mask[0],
        #                            per_object_mask[1]] = per_mask_color

        #         # get contours
        #         new_per_image_mask = np.zeros(
        #             (per_image.shape[0], per_image.shape[1]))
        #         new_per_image_mask[per_object_mask[0],
        #                            per_object_mask[1]] = 255
        #         contours, _ = cv2.findContours(
        #             new_per_image_mask.astype(np.uint8), cv2.RETR_TREE,
        #             cv2.CHAIN_APPROX_SIMPLE)
        #         per_image_contours.append(contours)

        #     per_image_new_mask = per_image_new_mask.astype(np.uint8)
        #     per_image_new_mask = cv2.cvtColor(per_image_new_mask,
        #                                       cv2.COLOR_RGBA2BGR)

        #     all_object_mask = np.nonzero(per_image_new_mask != 0)
        #     per_image_new_mask[all_object_mask[0],
        #                        all_object_mask[1]] = cv2.addWeighted(
        #                            per_image[all_object_mask[0],
        #                                      all_object_mask[1]], 0.5,
        #                            per_image_new_mask[all_object_mask[0],
        #                                               all_object_mask[1]], 1,
        #                            0)
        #     no_class_mask = np.nonzero(per_image_new_mask == 0)
        #     per_image_new_mask[no_class_mask[0],
        #                        no_class_mask[1]] = per_image[no_class_mask[0],
        #                                                      no_class_mask[1]]
        #     for contours in per_image_contours:
        #         cv2.drawContours(per_image_new_mask, contours, -1,
        #                          (255, 255, 255), 2)

        #     cv2.imencode('.jpg', per_image_not_draw)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{image_idx}.jpg'))
        #     cv2.imencode('.jpg', per_image_new_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{image_idx}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
