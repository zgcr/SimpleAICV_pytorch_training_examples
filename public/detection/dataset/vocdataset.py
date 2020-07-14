import os
import cv2
import numpy as np
import random
import math
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
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
]


class VocDetection(Dataset):
    def __init__(self,
                 root_dir,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None,
                 keep_difficult=False):
        self.root_dir = root_dir
        self.image_set = image_sets
        self.transform = transform
        self.categories = VOC_CLASSES

        self.category_id_to_voc_label = dict(
            zip(self.categories, range(len(self.categories))))
        self.voc_label_to_category_id = {
            v: k
            for k, v in self.category_id_to_voc_label.items()
        }

        self.keep_difficult = keep_difficult

        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root_dir, 'VOC' + year)
            for line in open(
                    os.path.join(rootpath, 'ImageSets', 'Main',
                                 name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self.load_image(img_id)

        target = ET.parse(self._annopath % img_id).getroot()
        annot = self.load_annotations(target)

        sample = {'img': img, 'annot': annot, 'scale': 1.}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, img_id):
        img = cv2.imread(self._imgpath % img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, target):
        annotations = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            bndbox = []
            for pt in pts:
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)
            label_idx = self.category_id_to_voc_label[name]
            bndbox.append(label_idx)
            annotations += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        annotations = np.array(annotations)
        # format:[[x1, y1, x2, y2, label_ind], ... ]
        return annotations

    def find_category_id_from_voc_label(self, voc_label):
        return self.voc_label_to_category_id[voc_label]

    def image_aspect_ratio(self, idx):
        img_id = self.ids[idx]
        image = self.load_image(img_id)
        #w/h
        return float(image.shape[1]) / float(image.shape[0])

    def __len__(self):
        return len(self.ids)


class VOCDataPrefetcher():
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
    voc = VocDetection(root_dir='/home/zgcr/Downloads/datasets/VOCdataset',
                       image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                       transform=transforms.Compose([
                           RandomFlip(),
                           Resize(resize=600),
                       ]),
                       keep_difficult=False)

    print(voc.image_set)
    print(voc.categories)
    print(voc.category_id_to_voc_label)
    print(voc.voc_label_to_category_id)
    print(len(voc.ids))
    print(voc[0]['img'].shape, voc[0]['annot'], voc[0]['scale'])
    print(voc[0])

    voc2 = VocDetection(root_dir='/home/zgcr/Downloads/datasets/VOCdataset',
                        image_sets=[('2007', 'test')],
                        transform=transforms.Compose([
                            Resize(resize=600),
                        ]),
                        keep_difficult=False)

    print(voc2.image_set)
    print(voc2.categories)
    print(voc2.category_id_to_voc_label)
    print(voc2.voc_label_to_category_id)
    print(len(voc2.ids))
    print(voc2[0]['img'].shape, voc2[0]['annot'], voc2[0]['scale'])
    print(voc2[0])
