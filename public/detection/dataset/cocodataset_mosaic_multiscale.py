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

coco_class_colors = [(241, 23, 78), (63, 71, 49),
                     (67, 79, 143), (32, 250, 205), (136, 228, 157),
                     (135, 125, 104), (151, 46, 171), (129, 37, 28),
                     (3, 248, 159), (154, 129, 58), (93, 155, 200),
                     (201, 98, 152), (187, 194, 70), (122, 144, 121),
                     (168, 31, 32), (168, 68, 189), (173, 68, 45),
                     (200, 81, 154), (171, 114, 139), (216, 211, 39),
                     (187, 119, 238), (201, 120, 112), (129, 16, 164),
                     (211, 3, 208), (169, 41, 248), (100, 77, 159),
                     (140, 104, 243), (26, 165, 41), (225, 176, 197),
                     (35, 212, 67), (160, 245, 68), (7, 87, 70), (52, 107, 85),
                     (103, 64, 188), (245, 76, 17), (248, 154, 59),
                     (77, 45, 123), (210, 95, 230), (172, 188, 171),
                     (250, 44, 233), (161, 71, 46), (144, 14, 134),
                     (231, 142, 186), (34, 1, 200), (144, 42, 108),
                     (222, 70, 139), (138, 62, 77),
                     (178, 99, 61), (17, 94, 132), (93, 248, 254),
                     (244, 116, 204), (138, 165, 238), (44, 216, 225),
                     (224, 164, 12), (91, 126, 184), (116, 254, 49),
                     (70, 250, 105), (252, 237, 54), (196, 136, 21),
                     (234, 13, 149), (66, 43, 47), (2, 73, 234), (118, 181, 5),
                     (105, 99, 225), (150, 253, 92), (59, 2, 121),
                     (176, 190, 223), (91, 62, 47), (198, 124, 140),
                     (100, 135, 185), (20, 207, 98), (216, 38, 133),
                     (17, 202, 208), (216, 135, 81), (212, 203, 33),
                     (108, 135, 76), (28, 47, 170), (142, 128, 121),
                     (23, 161, 179), (33, 183, 224)]


class CocoDetection(Dataset):
    def __init__(self,
                 image_root_dir,
                 annotation_root_dir,
                 set='train2017',
                 resize=416,
                 use_mosaic=False,
                 mosaic_center_range=[0.5, 1.5],
                 transform=None):
        self.image_root_dir = image_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.set_name = set
        self.resize = resize
        self.use_mosaic = use_mosaic
        self.mosaic_center_range = mosaic_center_range
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
        if self.use_mosaic:
            # mosaic center x, y
            x_ctr, y_ctr = [
                int(
                    random.uniform(self.resize * self.mosaic_center_range[0],
                                   self.resize * self.mosaic_center_range[1]))
                for _ in range(2)
            ]
            # all 4 image indices
            imgs_indices = [idx] + [
                random.randint(0,
                               len(self.image_ids) - 1) for _ in range(3)
            ]

            final_annots = []
            # combined image by 4 images
            combined_img = np.full((self.resize * 2, self.resize * 2, 3),
                                   114,
                                   dtype=np.uint8)

            for i, index in enumerate(imgs_indices):
                img = self.load_image(index)
                annot = self.load_annotations(index)

                origin_height, origin_width, _ = img.shape
                resize_factor = self.resize / max(origin_height, origin_width)
                resize_height, resize_width = int(
                    origin_height * resize_factor), int(origin_width *
                                                        resize_factor)

                img = cv2.resize(img, (resize_width, resize_height))
                annot[:, :4] *= resize_factor

                # top left img
                if i == 0:
                    # combined image coordinates
                    x1a, y1a, x2a, y2a = max(x_ctr - resize_width,
                                             0), max(y_ctr - resize_height,
                                                     0), x_ctr, y_ctr
                    # single img choosen area
                    x1b, y1b, x2b, y2b = resize_width - (
                        x2a - x1a), resize_height - (
                            y2a - y1a), resize_width, resize_height
                # top right img
                elif i == 1:
                    x1a, y1a, x2a, y2a = x_ctr, max(y_ctr - resize_height,
                                                    0), min(
                                                        x_ctr + resize_width,
                                                        self.resize * 2), y_ctr
                    x1b, y1b, x2b, y2b = 0, resize_height - (y2a - y1a), min(
                        resize_width, x2a - x1a), resize_height
                # bottom left img
                elif i == 2:
                    x1a, y1a, x2a, y2a = max(x_ctr - resize_width,
                                             0), y_ctr, x_ctr, min(
                                                 self.resize * 2,
                                                 y_ctr + resize_height)
                    x1b, y1b, x2b, y2b = resize_width - (x2a - x1a), 0, max(
                        x_ctr, resize_width), min(y2a - y1a, resize_height)
                # bottom right img
                elif i == 3:
                    x1a, y1a, x2a, y2a = x_ctr, y_ctr, min(
                        x_ctr + resize_width,
                        self.resize * 2), min(self.resize * 2,
                                              y_ctr + resize_height)
                    x1b, y1b, x2b, y2b = 0, 0, min(
                        resize_width, x2a - x1a), min(y2a - y1a, resize_height)

                # combined_img[ymin:ymax, xmin:xmax]
                combined_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
                padw, padh = x1a - x1b, y1a - y1b

                # annot coordinates transform
                if annot.shape[0] > 0:
                    annot[:, 0] = annot[:, 0] + padw
                    annot[:, 1] = annot[:, 1] + padh
                    annot[:, 2] = annot[:, 2] + padw
                    annot[:, 3] = annot[:, 3] + padh

                final_annots.append(annot)

            final_annots = np.concatenate(final_annots, axis=0)
            final_annots[:, 0:4] = np.clip(final_annots[:, 0:4], 0,
                                           self.resize * 2)

            final_annots = final_annots[final_annots[:, 2] -
                                        final_annots[:, 0] > 1]
            final_annots = final_annots[final_annots[:, 3] -
                                        final_annots[:, 1] > 1]

            sample = {'img': combined_img, 'annot': final_annots, 'scale': 1.}

        else:
            img = self.load_image(idx)
            annot = self.load_annotations(idx)
            scale = 1.

            origin_height, origin_width, _ = img.shape
            resize_factor = self.resize / max(origin_height, origin_width)
            resize_height, resize_width = int(
                origin_height * resize_factor), int(origin_width *
                                                    resize_factor)

            img = cv2.resize(img, (resize_width, resize_height))
            annot[:, :4] *= resize_factor
            scale *= resize_factor

            sample = {'img': img, 'annot': annot, 'scale': scale}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_root_dir, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)

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
            if a['bbox'][2] > 0 and a['bbox'][3] > 0:
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


class Collater():
    def __init__(self):
        pass

    def next(self, data):
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


class MultiScaleCollater():
    def __init__(self,
                 resize=416,
                 multi_scale_range=[0.5, 1.5],
                 stride=32,
                 use_mosaic=False):
        if use_mosaic:
            self.resize = resize * 2
        else:
            self.resize = resize

        self.multi_scale_range = multi_scale_range
        self.stride = stride

    def next(self, data):
        min_resize = int(
            ((self.resize + self.stride) * self.multi_scale_range[0]) //
            self.stride * self.stride)
        max_resize = int(
            ((self.resize + self.stride) * self.multi_scale_range[1]) //
            self.stride * self.stride)

        final_resize = random.choice(range(min_resize, max_resize,
                                           self.stride))

        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s['scale'] for s in data]

        imgs = torch.from_numpy(np.stack(imgs, axis=0))
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = F.interpolate(imgs,
                             size=(final_resize, final_resize),
                             mode='bilinear',
                             align_corners=False)

        resize_factor = final_resize / self.resize

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * (-1)

            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot[:, 0:4] = annot[:, 0:4] * resize_factor
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * (-1)

        return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class RandomFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if np.random.uniform(0, 1) < self.flip_prob:
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
            crop_xmax = min(
                w, int(max_bbox[2] + random.uniform(0, max_right_trans)))
            crop_ymax = min(
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


class PadToSquare(object):
    def __init__(self, resize=416, use_mosaic=False):
        if use_mosaic:
            self.resize = resize * 2
        else:
            self.resize = resize

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        height, width, _ = image.shape

        new_image = np.zeros((self.resize, self.resize, 3))
        new_image[0:height, 0:width] = image

        return {
            'img': torch.from_numpy(new_image),
            'annot': torch.from_numpy(annots),
            'scale': scale
        }


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        image = image / 255.
        sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class RandomNoise(object):
    def __init__(self, gaussian_noise=50, noise_prob=0.5):
        self.noise_prob = noise_prob
        self.gaussian_noise = gaussian_noise

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if np.random.uniform(0, 1) < self.noise_prob:
            gaussian_noise = min(self.gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            gaussian_noise = random.randint(0, gaussian_noise)
            noise = np.random.normal(0, gaussian_noise, image.shape)
            image = image + noise
            image[image < 0] = 0
            image[image > 255] = 255

        return {'img': image, 'annot': annots, 'scale': scale}


class RandomBlur(object):
    def __init__(self, blur_kernel=3, blur_prob=0.5):
        self.blur_prob = blur_prob
        self.blur_kernel = blur_kernel

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if np.random.uniform(0, 1) < self.blur_prob:
            image = cv2.GaussianBlur(image,
                                     (self.blur_kernel, self.blur_kernel), 0)

        return {'img': image, 'annot': annots, 'scale': scale}


class RandomHSV(object):
    def __init__(self, hue=0.1, saturation=1.5, exposure=1.5, hsv_prob=0.5):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.hsv_prob = hsv_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if np.random.uniform(0, 1) < self.hsv_prob:

            final_hue = self.generate_random_number(-self.hue, self.hue)

            final_sat = self.generate_random_number(1, self.saturation)
            if random.randint(0, 1) % 2:
                final_sat = 1. / final_sat

            final_exp = self.generate_random_number(1, self.exposure)
            if random.randint(0, 1) % 2:
                final_exp = 1. / final_exp

            if final_hue != 0 or final_sat != 1 or final_exp != 1:
                # RGB to HSV
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hsv_split = cv2.split(hsv_image)
                hsv_split[1] *= final_sat
                hsv_split[2] *= final_exp
                hsv_split[0] += 179 * final_hue
                hsv_image = cv2.merge(hsv_split)
                # HSV to RGB
                image = np.clip(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB), 0,
                                255)

        return {'img': image, 'annot': annots, 'scale': scale}

    def generate_random_number(self, min_value, max_value):
        if min_value > max_value:
            temp = min_value
            min_value = max_value
            max_value = temp
        return np.random.uniform(0, 1) * (max_value - min_value) + min_value


class Resize(object):
    def __init__(self, resize=600):
        self.resize = resize

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        height, width, _ = image.shape

        if annots.shape[0] == 0:
            return sample

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
        resize=416,
        use_mosaic=True,
        mosaic_center_range=[0.5, 1.5],
        transform=transforms.Compose([
            RandomFlip(flip_prob=0.5),
            RandomTranslate(translate_prob=0.5),
            PadToSquare(resize=416, use_mosaic=True),
        ]))

    print(len(coco.image_ids))
    print(coco.category_id_to_coco_label)

    print(coco[0]['img'].shape, coco[0]['annot'].shape, coco[0]['scale'])

    # retinanet resize method
    # resize=400,per_image_average_area=223743,input shape=[667,667]
    # resize=500,per_image_average_area=347964,input shape=[833,833]
    # resize=600,per_image_average_area=502820,input shape=[1000,1000]
    # resize=700,per_image_average_area=682333,input shape=[1166,1166]
    # resize=800,per_image_average_area=891169,input shape=[1333,1333]

    # yolov3 resize method(my resize method)
    # resize=600,per_image_average_area=258182,input shape=[600,600]
    # resize=667,per_image_average_area=318986,input shape=[667,667]
    # resize=700,per_image_average_area=351427,input shape=[700,700]
    # resize=800,per_image_average_area=459021,input shape=[800,800]
    # resize=833,per_image_average_area=497426,input shape=[833,833]
    # resize=900,per_image_average_area=580988,input shape=[900,900]
    # resize=1000,per_image_average_area=717349,input shape=[1000,1000]
    # resize=1166,per_image_average_area=974939,input shape=[1166,1166]
    # resize=1333,per_image_average_area=1274284,input shape=[1333,1333]