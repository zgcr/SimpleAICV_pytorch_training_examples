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

        sample = {'img': img, 'boxes': annot, 'scale': 1.}

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
        # for segmentation annotation,if iscrowd=0(False),segmentation annotation format is polygon;
        # attention:some iscrowd=0 segmentation annotation may use RLE format too.
        # if iscrowd=1(True),segmentation annotation format is RLE.
        # polygon:multi points set(x,y)
        # uncompressed RLE:have 'counts' and 'size' keys:value
        # size:image w/h
        # counts array:all pixel on this image,reshape for One-dimension array(cow first),then count
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
            self.next_input, self.next_boxes = sample['img'], sample['boxes']
        except StopIteration:
            self.next_input = None
            self.next_boxes = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_boxes = self.next_boxes.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        boxes = self.next_boxes
        self.preload()
        return input, boxes


class Collater():
    def __init__(self):
        pass

    def next(self, data):
        imgs = [s['img'] for s in data]
        annots = [s['boxes'] for s in data]
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


if __name__ == '__main__':
    # import torchvision.transforms as transforms
    # from tqdm import tqdm
    # coco = CocoDetection(
    #     image_root_dir=
    #     '/home/zgcr/Downloads/datasets/COCO2017/images/train2017/',
    #     annotation_root_dir=
    #     "/home/zgcr/Downloads/datasets/COCO2017/annotations/",
    #     set='train2017',
    #     transform=transforms.Compose([
    #         RandomFlip(),
    #         Resize(resize=600),
    #     ]))

    # print(len(coco))
    # print(coco.category_id_to_coco_label)

    # print(coco[0]['img'].shape, coco[0]['annot'], coco[0]['scale'])

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

    # CLASS_COLOR = [(np.random.randint(255), np.random.randint(255),
    #                 np.random.randint(255)) for _ in range(20)]
    # print(CLASS_COLOR)
    # print(len(CLASS_COLOR))

    def mask2polygon(mask):
        contours, hierarchy = cv2.findContours(
            (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            contour_list = contour.flatten().tolist()
            if len(contour_list) > 4:
                segmentation.append(contour_list)

        return segmentation

    def polygons_to_mask(img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        polygons = np.asarray(polygons,
                              np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
        shape = polygons.shape
        polygons = polygons.reshape(shape[0], -1, 2)
        cv2.fillPoly(mask, polygons, color=1)  # 非int32 会报错
        return mask

    def mask2rle(img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle2mask(rle, input_shape):
        width, height = input_shape[:2]

        mask = np.zeros(width * height).astype(np.uint8)

        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start + lengths[index])] = 1
            current_position += lengths[index]
        return mask.reshape(height, width).T

    def transform_mask_to_bounding_box(img):
        # return max and min of a mask to draw bounding box
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    import numpy as np
    mask = np.ones((100, 100))
    for i in range(10):
        for j in range(10):
            mask[i][j] = 0
    mask2polygon(mask)
