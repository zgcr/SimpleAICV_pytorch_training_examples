import os
import cv2
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset

# background class and 80 classes
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
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

# background class and 80 classes
# background is black color
COCO_CLASSES_COLOR = [
    (0, 0, 0), (156, 77, 36), (218, 3, 199), (252, 197, 160), (82, 69, 38),
    (132, 17, 27), (71, 19, 213), (108, 81, 1), (49, 54, 81), (8, 249, 143),
    (80, 20, 4), (75, 227, 112), (82, 41, 57), (157, 0, 97), (0, 209, 246),
    (116, 242, 109), (60, 225, 243), (2, 125, 5), (118, 94, 170), (171, 1, 17),
    (54, 97, 38), (16, 132, 55), (1, 90, 238), (112, 4, 197), (147, 219, 248),
    (253, 0, 14), (103, 77, 249), (149, 1, 222), (120, 94, 51), (88, 29, 129),
    (204, 29, 128), (19, 0, 244), (92, 154, 54), (34, 89, 7), (29, 168, 224),
    (111, 25, 1), (137, 70, 83), (24, 217, 19), (47, 170, 155), (34, 234, 107),
    (182, 116, 221), (102, 243, 211), (53, 247, 123), (147, 159, 24),
    (194, 147, 121), (76, 101, 233), (50, 11, 88), (253, 33, 83), (84, 1, 57),
    (248, 243, 24), (244, 79, 35), (162, 240, 132), (1, 32, 203), (208, 10, 8),
    (30, 64, 206), (234, 80, 229), (31, 253, 207),
    (110, 34, 78), (234, 72, 73), (92, 3, 16), (113, 0, 65), (196, 177, 53),
    (63, 92, 139), (76, 143, 1), (61, 93, 84), (82, 130, 157), (28, 2, 84),
    (55, 226, 12), (34, 99, 82), (47, 5, 239), (53, 100, 219), (132, 37, 147),
    (244, 156, 224), (179, 57, 59), (2, 27, 76), (0, 100, 83), (64, 39, 116),
    (170, 46, 246), (27, 51, 87), (185, 71, 0), (107, 247, 29)
]


class CocoSemanticSegmentation(Dataset):

    def __init__(self, root_dir, set_name='train2017', transform=None):
        assert set_name in ['train2017', 'val2017'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'instances_{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.image_ids = sorted(self.coco.getImgIds())

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
        self.num_classes = len(self.cats) + 1

        # cat_id is an original cat id,coco_label is set from 1 to 80
        self.cat_id_to_cat_name = {cat['id']: cat['name'] for cat in self.cats}

        self.cat_id_to_coco_label = {}
        self.coco_label_to_cat_id = {}
        # coco label 0 is background class
        # cat id don't have background class ,set background class cat id is 0
        self.cat_id_to_coco_label[0] = 0
        self.coco_label_to_cat_id[0] = 0

        for i, cat in enumerate(self.cats):
            self.cat_id_to_coco_label[cat['id']] = i + 1

        for i, cat in enumerate(self.cats):
            self.coco_label_to_cat_id[i + 1] = cat['id']

        self.coco_label_to_cat_name = {}
        for coco_label, cat_id in self.coco_label_to_cat_id.items():
            if cat_id in self.cat_id_to_cat_name.keys():
                self.coco_label_to_cat_name[
                    coco_label] = self.cat_id_to_cat_name[cat_id]

        self.transform = transform

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        path = os.path.join(self.image_dir, file_name)

        image = self.load_image(idx)
        mask = self.load_mask(idx)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'path': path,
            'image': image,
            'mask': mask,
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

    def load_mask(self, idx):
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annots = self.coco.loadAnns(annot_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        mask = np.zeros((image_h, image_w))
        # filter annots
        for annot in annots:
            if 'ignore' in annot.keys():
                continue

            per_binary_mask = self.coco.annToMask(annot)
            per_mask_coco_label = self.cat_id_to_coco_label[
                annot['category_id']]

            # 先保留新mask之外区域原来的mask类别赋值
            keep_mask = mask * (1 - per_binary_mask)
            # 新mask区域用新的mask类别代替原来的mask类别
            per_new_mask = per_binary_mask * per_mask_coco_label
            mask = keep_mask + per_new_mask

        # class 0 is the background class

        return mask.astype(np.float32)


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

    import copy

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.semantic_segmentation.common import YoloStyleResize, RandomHorizontalFlip, Normalize, SemanticSegmentationCollater

    cocodataset = CocoSemanticSegmentation(
        COCO2017_path,
        set_name='train2017',
        transform=transforms.Compose([
            YoloStyleResize(resize=512),
            RandomHorizontalFlip(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(cocodataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['size'].dtype)

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image_not_draw = copy.deepcopy(image)
        # mask = per_sample['mask']

        # all_classes = np.unique(mask)
        # print('1212', all_classes)
        # all_colors = []
        # for per_class in all_classes:
        #     per_class = int(per_class)
        #     if per_class < 0 or per_class > 255:
        #         continue
        #     class_name, class_color = COCO_CLASSES[
        #         per_class], COCO_CLASSES_COLOR[per_class]
        #     all_colors.append(class_color)
        # all_classes = list(all_classes)

        # if len(all_classes) == 0:
        #     continue

        # per_image_mask = np.zeros((image.shape[0], image.shape[1], 3))
        # per_image_contours = []
        # for idx, per_class in enumerate(all_classes):
        #     if per_class < 0 or per_class > 255:
        #         continue

        #     per_class_mask = np.nonzero(mask == per_class)
        #     per_image_mask[per_class_mask[0],
        #                    per_class_mask[1]] = all_colors[idx]
        #     # get contours
        #     new_per_image_mask = np.zeros((image.shape[0], image.shape[1]))
        #     new_per_image_mask[per_class_mask[0], per_class_mask[1]] = 255
        #     contours, _ = cv2.findContours(new_per_image_mask.astype(np.uint8),
        #                                    cv2.RETR_TREE,
        #                                    cv2.CHAIN_APPROX_SIMPLE)
        #     per_image_contours.append(contours)

        # per_image_mask = per_image_mask.astype(np.uint8)
        # per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

        # all_classes_mask = np.nonzero(per_image_mask != 0)
        # per_image_mask[all_classes_mask[0],
        #                all_classes_mask[1]] = cv2.addWeighted(
        #                    image[all_classes_mask[0], all_classes_mask[1]],
        #                    0.5, per_image_mask[all_classes_mask[0],
        #                                        all_classes_mask[1]], 1, 0)
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
    collater = SemanticSegmentationCollater(resize=512)
    train_loader = DataLoader(cocodataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('2222', images.shape, masks.shape, sizes.shape)
        print('2222', images.dtype, masks.dtype, sizes.dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # masks = masks.cpu().numpy()

        # for i, (per_image,
        #         per_image_mask_targets) in enumerate(zip(images, masks)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)
        #     per_image_not_draw = copy.deepcopy(per_image)

        #     all_classes = np.unique(per_image_mask_targets)
        #     print('1212', all_classes)
        #     all_colors = []
        #     for per_class in all_classes:
        #         per_class = int(per_class)
        #         if per_class < 0 or per_class > 255:
        #             continue

        #         class_name, class_color = COCO_CLASSES[
        #             per_class], COCO_CLASSES_COLOR[per_class]
        #         all_colors.append(class_color)
        #     all_classes = list(all_classes)

        #     if len(all_classes) == 0:
        #         continue

        #     per_image_mask = np.zeros(
        #         (per_image.shape[0], per_image.shape[1], 3))
        #     per_image_contours = []
        #     for idx, per_class in enumerate(all_classes):
        #         if per_class < 0 or per_class > 255:
        #             continue

        #         per_class_mask = np.nonzero(
        #             per_image_mask_targets == per_class)
        #         per_image_mask[per_class_mask[0],
        #                        per_class_mask[1]] = all_colors[idx]
        #         # get contours
        #         new_per_image_mask = np.zeros(
        #             (per_image.shape[0], per_image.shape[1]))
        #         new_per_image_mask[per_class_mask[0], per_class_mask[1]] = 255
        #         contours, _ = cv2.findContours(
        #             new_per_image_mask.astype(np.uint8), cv2.RETR_TREE,
        #             cv2.CHAIN_APPROX_SIMPLE)
        #         per_image_contours.append(contours)

        #     per_image_mask = per_image_mask.astype(np.uint8)
        #     per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

        #     all_classes_mask = np.nonzero(per_image_mask != 0)
        #     per_image_mask[all_classes_mask[0],
        #                    all_classes_mask[1]] = cv2.addWeighted(
        #                        per_image[all_classes_mask[0],
        #                                  all_classes_mask[1]], 0.5,
        #                        per_image_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 1, 0)
        #     no_class_mask = np.nonzero(per_image_mask == 0)
        #     per_image_mask[no_class_mask[0],
        #                    no_class_mask[1]] = per_image[no_class_mask[0],
        #                                                  no_class_mask[1]]
        #     for contours in per_image_contours:
        #         cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255),
        #                          2)

        #     cv2.imencode('.jpg', per_image_not_draw)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
        #     cv2.imencode('.jpg', per_image_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
