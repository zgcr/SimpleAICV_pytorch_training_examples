import os
import cv2
import torch
import numpy as np
import random
import math
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch.nn.functional as F

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

colors = [
    (39, 129, 113),
    (164, 80, 133),
    (83, 122, 114),
    (99, 81, 172),
    (95, 56, 104),
    (37, 84, 86),
    (14, 89, 122),
    (80, 7, 65),
    (10, 102, 25),
    (90, 185, 109),
    (106, 110, 132),
    (169, 158, 85),
    (188, 185, 26),
    (103, 1, 17),
    (82, 144, 81),
    (92, 7, 184),
    (49, 81, 155),
    (179, 177, 69),
    (93, 187, 158),
    (13, 39, 73),
    (12, 50, 60),
    (16, 179, 33),
    (112, 69, 165),
    (15, 139, 63),
    (33, 191, 159),
    (182, 173, 32),
    (34, 113, 133),
    (90, 135, 34),
    (53, 34, 86),
    (141, 35, 190),
    (6, 171, 8),
    (118, 76, 112),
    (89, 60, 55),
    (15, 54, 88),
    (112, 75, 181),
    (42, 147, 38),
    (138, 52, 63),
    (128, 65, 149),
    (106, 103, 24),
    (168, 33, 45),
    (28, 136, 135),
    (86, 91, 108),
    (52, 11, 76),
    (142, 6, 189),
    (57, 81, 168),
    (55, 19, 148),
    (182, 101, 89),
    (44, 65, 179),
    (1, 33, 26),
    (122, 164, 26),
    (70, 63, 134),
    (137, 106, 82),
    (120, 118, 52),
    (129, 74, 42),
    (182, 147, 112),
    (22, 157, 50),
    (56, 50, 20),
    (2, 22, 177),
    (156, 100, 106),
    (21, 35, 42),
    (13, 8, 121),
    (142, 92, 28),
    (45, 118, 33),
    (105, 118, 30),
    (7, 185, 124),
    (46, 34, 146),
    (105, 184, 169),
    (22, 18, 5),
    (147, 71, 73),
    (181, 64, 91),
    (31, 39, 184),
    (164, 179, 33),
    (96, 50, 18),
    (95, 15, 106),
    (113, 68, 54),
    (136, 116, 112),
    (119, 139, 130),
    (31, 139, 34),
    (66, 6, 127),
    (62, 39, 2),
    (49, 99, 180),
    (49, 119, 155),
    (153, 50, 183),
    (125, 38, 3),
    (129, 87, 143),
    (49, 87, 40),
    (128, 62, 120),
    (73, 85, 148),
    (28, 144, 118),
    (29, 9, 24),
    (175, 45, 108),
    (81, 175, 64),
    (178, 19, 157),
    (74, 188, 190),
    (18, 114, 2),
    (62, 128, 96),
    (21, 3, 150),
    (0, 6, 95),
    (2, 20, 184),
    (122, 37, 185),
]


class CocoDetection(Dataset):
    def __init__(self,
                 image_root_dir,
                 annotation_root_dir,
                 set='train2017',
                 transform=None):
        self.image_root_dir = image_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(
            os.path.join(self.annotation_root_dir,
                         'instances_' + self.set_name + '.json'))

        self.load_classes()

    def load_classes(self):
        self.image_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()
        self.categories = self.coco.loadCats(self.cat_ids)
        self.categories.sort(key=lambda x: x['id'])

        # category_id is an original id,coco_id is set from 0 to 79
        self.category_id_to_coco_label = {
            category['id']: i
            for i, category in enumerate(self.categories)
        }
        self.coco_label_to_category_id = {
            v: k
            for k, v in self.category_id_to_coco_label.items()
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)

        sample = {'img': img, 'annot': annot, 'scale': 1.}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_root_dir, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for _, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.find_coco_label_from_category_id(
                a['category_id'])

            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def find_coco_label_from_category_id(self, category_id):
        return self.category_id_to_coco_label[category_id]

    def find_category_id_from_coco_label(self, coco_label):
        return self.coco_label_to_category_id[coco_label]

    def num_classes(self):
        return 80

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])


class COCODataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_annot = sample['img'], sample['annot']
        except StopIteration:
            self.next_input = None
            self.next_annot = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_annot = self.next_annot.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        annot = self.next_annot
        self.preload()
        return input, annot


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * (-1)

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * (-1)

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class RandomFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) < self.flip_prob:
            image, annots, scale = sample['img'], sample['annot'], sample[
                'scale']
            image = image[:, ::-1, :]

            _, width, _ = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = width - x2
            annots[:, 2] = width - x1

            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class RandomCrop(object):
    def __init__(self, crop_prob=0.5):
        self.crop_prob = crop_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.crop_prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_left_trans)))
            crop_ymin = max(0,
                            int(max_bbox[1] - random.uniform(0, max_up_trans)))
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_right_trans)))
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_down_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_xmin
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_ymin

            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class RandomTranslate(object):
    def __init__(self, translate_prob=0.5):
        self.translate_prob = translate_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.translate_prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            tx = random.uniform(-(max_left_trans - 1), (max_right_trans - 1))
            ty = random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty

            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class Resize(object):
    def __init__(self, resize=600):
        self.resize = resize

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        height, width, _ = image.shape

        max_image_size = max(height, width)
        resize_factor = self.resize / max_image_size
        resize_height, resize_width = int(height * resize_factor), int(
            width * resize_factor)

        image = cv2.resize(image, (resize_width, resize_height))

        new_image = np.zeros((self.resize, self.resize, 3))
        new_image[0:resize_height, 0:resize_width] = image

        annots[:, :4] *= resize_factor
        scale = scale * resize_factor

        return {
            'img': torch.from_numpy(new_image),
            'annot': torch.from_numpy(annots),
            'scale': scale
        }


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm
    coco = CocoDetection(
        image_root_dir=
        '/home/zgcr/Downloads/datasets/COCO2017/images/train2017/',
        annotation_root_dir=
        "/home/zgcr/Downloads/datasets/COCO2017/annotations/",
        set='train2017',
        transform=transforms.Compose([
            RandomFlip(),
            Resize(resize=600),
        ]))

    print(len(coco))
    print(coco.category_id_to_coco_label)

    print(coco[0]['img'].shape, coco[0]['annot'], coco[0]['scale'])

    # retinanet resize method
    # resize=400,per_image_average_area=223743,input shape=[667,667]
    # resize=500,per_image_average_area=347964,input shape=[833,833]
    # resize=600,per_image_average_area=502820,input shape=[1000,1000]
    # resize=700,per_image_average_area=682333,input shape=[1166,1166]
    # resize=800,per_image_average_area=891169,input shape=[1333,1333]

    # my resize method
    # resize=600,per_image_average_area=258182,input shape=[600,600]
    # resize=667,per_image_average_area=318986,input shape=[667,667]
    # resize=700,per_image_average_area=351427,input shape=[700,700]
    # resize=800,per_image_average_area=459021,input shape=[800,800]
    # resize=833,per_image_average_area=497426,input shape=[833,833]
    # resize=900,per_image_average_area=580988,input shape=[900,900]
    # resize=1000,per_image_average_area=717349,input shape=[1000,1000]
    # resize=1166,per_image_average_area=974939,input shape=[1166,1166]
    # resize=1333,per_image_average_area=1274284,input shape=[1333,1333]
