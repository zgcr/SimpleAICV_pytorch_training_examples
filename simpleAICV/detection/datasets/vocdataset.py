import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset

VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

VOC_CLASSES_COLOR = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                     (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
                     (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                     (192, 0, 128), (64, 128, 128), (192, 128, 128),
                     (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                     (0, 64, 128)]


class VocDetection(Dataset):

    def __init__(self,
                 root_dir,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None,
                 keep_difficult=False):

        self.annotpath = os.path.join('%s', 'Annotations', '%s.xml')
        self.imagepath = os.path.join('%s', 'JPEGImages', '%s.jpg')

        self.cats = VOC_CLASSES
        self.num_classes = len(self.cats)

        self.cat_to_voc_label = {cat: i for i, cat in enumerate(self.cats)}
        self.voc_label_to_cat = {i: cat for i, cat in enumerate(self.cats)}

        self.keep_difficult = keep_difficult

        self.ids = []
        for (year, name) in image_sets:
            rootpath = os.path.join(root_dir, 'VOC' + year)
            for line in open(
                    os.path.join(rootpath, 'ImageSets', 'Main',
                                 name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        self.transform = transform

        print(f'Dataset Size:{len(self.ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.ids)

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
        image = cv2.imdecode(
            np.fromfile(self.imagepath % self.ids[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        annots = ET.parse(self.annotpath % self.ids[idx]).getroot()

        targets = []
        size = annots.find('size')
        h, w = int(size.find('height').text), int(size.find('width').text)
        for obj in annots.iter('object'):
            difficult = (int(obj.find('difficult').text) == 1)
            if not self.keep_difficult and difficult:
                continue

            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            target = []
            for pt in pts:
                cur_pt = float(bbox.find(pt).text)
                target.append(cur_pt)

            if target[2] - target[0] < 1 or target[3] - target[1] < 1:
                continue

            if target[0] < 0 or target[1] < 0 or target[2] > w or target[3] > h:
                continue

            if class_name not in self.cats:
                continue

            target.append(self.cat_to_voc_label[class_name])
            # [xmin, ymin, xmax, ymax, voc_label]
            targets += [target]

        if len(targets) == 0:
            targets = np.zeros((0, 5))
        else:
            targets = np.array(targets)

        # format:[[x1, y1, x2, y2, voc_label], ... ]
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

    from tools.path import VOCdataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm
    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize, DetectionCollater

    vocdataset = VocDetection(
        root_dir=VOCdataset_path,
        image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            # RandomCrop(prob=0.5),
            # RandomTranslate(prob=0.5),
            YoloStyleResize(resize=640,
                            divisor=32,
                            stride=32,
                            multi_scale=False,
                            multi_scale_range=[0.5, 1.0]),
            # RetinaStyleResize(resize=800, multi_scale=True),
            # Normalize(),
        ]),
        keep_difficult=True)

    count = 0
    for per_sample in tqdm(vocdataset):
        print(per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'])

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # annots = per_sample['annots']

        # # draw all label boxes
        # for per_annot in annots:
        #     per_box = (per_annot[0:4]).astype(np.int32)
        #     per_box_class_index = per_annot[4].astype(np.int32)
        #     class_name, class_color = VOC_CLASSES[
        #         per_box_class_index], VOC_CLASSES_COLOR[per_box_class_index]
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
    collater = DetectionCollater()
    train_loader = DataLoader(vocdataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print(images.shape, annots.shape, sizes.shape)
        print(images.dtype, annots.dtype, scales.dtype, sizes.dtype)
        if count < 10:
            count += 1
        else:
            break
