import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

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

VOC_CLASSES_COLOR = [(174, 175, 133), (17, 84, 219), (46, 138, 54),
                     (253, 151, 96), (75, 242, 162), (173, 150, 67),
                     (232, 46, 160), (83, 226, 155), (80, 150, 1),
                     (127, 246, 43), (167, 126, 221), (132, 20, 125),
                     (192, 240, 135), (111, 67, 22), (56, 53, 178),
                     (74, 215, 29), (14, 69, 126), (191, 60, 67),
                     (56, 119, 196), (84, 48, 194)]


class VocDetection(Dataset):
    def __init__(self,
                 root_dir,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None,
                 keep_difficult=False):
        self.root_dir = root_dir
        self.image_set = image_sets
        self.transform = transform
        self.cats = VOC_CLASSES
        self.num_classes = len(self.cats)

        self.cat_to_voc_label = {cat: i for i, cat in enumerate(self.cats)}
        self.voc_label_to_cat = {i: cat for i, cat in enumerate(self.cats)}

        self.keep_difficult = keep_difficult

        self.annotpath = os.path.join('%s', 'Annotations', '%s.xml')
        self.imagepath = os.path.join('%s', 'JPEGImages', '%s.jpg')

        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root_dir, 'VOC' + year)
            for line in open(
                    os.path.join(rootpath, 'ImageSets', 'Main',
                                 name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annots, origin_hw = self.load_annots(idx)

        sample = {
            'image': image,
            'annots': annots,
            'scale': np.float32(1.),
            'origin_hw': origin_hw,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        img = cv2.imread(self.imagepath % self.ids[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)

    def load_annots(self, idx):
        target = ET.parse(self.annotpath % self.ids[idx]).getroot()
        annots = []

        size = target.find('size')
        h, w = int(size.find('height').text), int(size.find('width').text)
        origin_hw = np.array([h, w])

        for obj in target.iter('object'):
            difficult = (int(obj.find('difficult').text) == 1)
            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for pt in pts:
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)

            if bndbox[2] - bndbox[0] < 1 or bndbox[3] - bndbox[1] < 1:
                continue

            if bndbox[0] < 0 or bndbox[1] < 0 or bndbox[2] > w or bndbox[3] > h:
                continue

            if name not in self.cats:
                continue

            bndbox.append(self.cat_to_voc_label[name])
            # [xmin, ymin, xmax, ymax, voc_label]
            annots += [bndbox]

        # format:[[x1, y1, x2, y2, voc_label], ... ]
        annots = np.array(annots)

        return annots.astype(np.float32), origin_hw.astype(np.float32)


if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from tools.path import VOCdataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm
    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize, DetectionCollater

    vocdataset = VocDetection(
        root_dir=VOCdataset_path,
        image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        transform=transforms.Compose([
            RandomHorizontalFlip(),
            RandomCrop(),
            RandomTranslate(),
            Normalize(),
            #YoloStyleResize(resize=640),
            RetinaStyleResize(resize=800, multi_scale=True),
        ]),
        keep_difficult=False)

    count = 0
    for per_sample in tqdm(vocdataset):
        print(per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['origin_hw'])
        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DetectionCollater()
    train_loader = DataLoader(vocdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater.next)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, origin_hws = data['image'], data[
            'annots'], data['scale'], data['origin_hw']
        print(images.shape, annots.shape, origin_hws.shape)
        print(images.dtype, annots.dtype, scales.dtype, origin_hws.dtype)
        if count < 10:
            count += 1
        else:
            break
