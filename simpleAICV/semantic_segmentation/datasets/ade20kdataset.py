import os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

ADE20K_CLASSES = [
    'wall',
    'building, edifice',
    'sky',
    'floor, flooring',
    'tree',
    'ceiling',
    'road, routev',
    'bed',
    'windowpane, window',
    'grass',
    'cabinet',
    'sidewalk, pavement',
    'person, individual, someone, somebody, mortal, soul',
    'earth, ground',
    'door, double door',
    'table',
    'mountain, mount',
    'plant, flora, plant life',
    'curtain, drape, drapery, mantle, pall',
    'chair',
    'car, auto, automobile, machine, motorcar',
    'water',
    'painting, picture',
    'sofa, couch, lounge',
    'shelf',
    'house',
    'sea',
    'mirror',
    'rug, carpet, carpeting',
    'field',
    'armchair',
    'seat',
    'fence, fencing',
    'desk',
    'rock, stone',
    'wardrobe, closet, press',
    'lamp',
    'bathtub, bathing tub, bath, tub',
    'railing, rail',
    'cushion',
    'base, pedestal, stand',
    'box',
    'column, pillar',
    'signboard, sign',
    'chest of drawers, chest, bureau, dresser',
    'counter',
    'sand',
    'sink',
    'skyscraper',
    'fireplace, hearth, open fireplace',
    'refrigerator, icebox',
    'grandstand, covered stand',
    'path',
    'stairs, steps',
    'runway',
    'case, display case, showcase, vitrine',
    'pool table, billiard table, snooker table',
    'pillow',
    'screen door, screen',
    'stairway, staircase',
    'river',
    'bridge, span',
    'bookcase',
    'blind, screen',
    'coffee table, cocktail table',
    'toilet, can, commode, crapper, pot, potty, stool, throne',
    'flower',
    'book',
    'hill',
    'bench',
    'countertop',
    'stove, kitchen stove, range, kitchen range, cooking stove',
    'palm, palm tree',
    'kitchen island',
    'computer, computing machine, computing device, data processor, electronic computer, information processing system',
    'swivel chair',
    'boat',
    'bar',
    'arcade machine',
    'hovel, hut, hutch, shack, shanty',
    'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle',
    'towel',
    'light, light source',
    'truck, motortruck',
    'tower',
    'chandelier, pendant, pendent',
    'awning, sunshade, sunblind',
    'streetlight, street lamp',
    'booth, cubicle, stall, kiosk',
    'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box',
    'airplane, aeroplane, plane',
    'dirt track',
    'apparel, wearing apparel, dress, clothes',
    'pole',
    'land, ground, soil',
    'bannister, banister, balustrade, balusters, handrail',
    'escalator, moving staircase, moving stairway',
    'ottoman, pouf, pouffe, puff, hassock',
    'bottle',
    'buffet, counter, sideboard',
    'poster, posting, placard, notice, bill, card',
    'stage',
    'van',
    'ship',
    'fountain',
    'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
    'canopy',
    'washer, automatic washer, washing machine',
    'plaything, toy',
    'swimming pool, swimming bath, natatorium',
    'stool',
    'barrel, cask',
    'basket, handbasket',
    'waterfall, falls',
    'tent, collapsible shelter',
    'bag',
    'minibike, motorbike',
    'cradle',
    'oven',
    'ball',
    'food, solid food',
    'step, stair',
    'tank, storage tank',
    'trade name, brand name, brand, marque',
    'microwave, microwave oven',
    'pot, flowerpot',
    'animal, animate being, beast, brute, creature, fauna',
    'bicycle, bike, wheel, cycle',
    'lake',
    'dishwasher, dish washer, dishwashing machine',
    'screen, silver screen, projection screen',
    'blanket, cover',
    'sculpture',
    'hood, exhaust hood',
    'sconce',
    'vase',
    'traffic light, traffic signal, stoplight',
    'tray',
    'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
    'fan',
    'pier, wharf, wharfage, dock',
    'crt screen',
    'plate',
    'monitor, monitoring device',
    'bulletin board, notice board',
    'shower',
    'radiator',
    'glass, drinking glass',
    'clock',
    'flag',
]

# RGB color
ADK20K_CLASSES_COLOR = [
    (120, 120, 120),
    (180, 120, 120),
    (6, 230, 230),
    (80, 50, 50),
    (4, 200, 3),
    (120, 120, 80),
    (140, 140, 140),
    (204, 5, 255),
    (230, 230, 230),
    (4, 250, 7),
    (224, 5, 255),
    (235, 255, 7),
    (150, 5, 61),
    (120, 120, 70),
    (8, 255, 51),
    (255, 6, 82),
    (143, 255, 140),
    (204, 255, 4),
    (255, 51, 7),
    (204, 70, 3),
    (0, 102, 200),
    (61, 230, 250),
    (255, 6, 51),
    (11, 102, 255),
    (255, 7, 71),
    (255, 9, 224),
    (9, 7, 230),
    (220, 220, 220),
    (255, 9, 92),
    (112, 9, 255),
    (8, 255, 214),
    (7, 255, 224),
    (255, 184, 6),
    (10, 255, 71),
    (255, 41, 10),
    (7, 255, 255),
    (224, 255, 8),
    (102, 8, 255),
    (255, 61, 6),
    (255, 194, 7),
    (255, 122, 8),
    (0, 255, 20),
    (255, 8, 41),
    (255, 5, 153),
    (6, 51, 255),
    (235, 12, 255),
    (160, 150, 20),
    (0, 163, 255),
    (140, 140, 140),
    (250, 10, 15),
    (20, 255, 0),
    (31, 255, 0),
    (255, 31, 0),
    (255, 224, 0),
    (153, 255, 0),
    (0, 0, 255),
    (255, 71, 0),
    (0, 235, 255),
    (0, 173, 255),
    (31, 0, 255),
    (11, 200, 200),
    (255, 82, 0),
    (0, 255, 245),
    (0, 61, 255),
    (0, 255, 112),
    (0, 255, 133),
    (255, 0, 0),
    (255, 163, 0),
    (255, 102, 0),
    (194, 255, 0),
    (0, 143, 255),
    (51, 255, 0),
    (0, 82, 255),
    (0, 255, 41),
    (0, 255, 173),
    (10, 0, 255),
    (173, 255, 0),
    (0, 255, 153),
    (255, 92, 0),
    (255, 0, 255),
    (255, 0, 245),
    (255, 0, 102),
    (255, 173, 0),
    (255, 0, 20),
    (255, 184, 184),
    (0, 31, 255),
    (0, 255, 61),
    (0, 71, 255),
    (255, 0, 204),
    (0, 255, 194),
    (0, 255, 82),
    (0, 10, 255),
    (0, 112, 255),
    (51, 0, 255),
    (0, 194, 255),
    (0, 122, 255),
    (0, 255, 163),
    (255, 153, 0),
    (0, 255, 10),
    (255, 112, 0),
    (143, 255, 0),
    (82, 0, 255),
    (163, 255, 0),
    (255, 235, 0),
    (8, 184, 170),
    (133, 0, 255),
    (0, 255, 92),
    (184, 0, 255),
    (255, 0, 31),
    (0, 184, 255),
    (0, 214, 255),
    (255, 0, 112),
    (92, 255, 0),
    (0, 224, 255),
    (112, 224, 255),
    (70, 184, 160),
    (163, 0, 255),
    (153, 0, 255),
    (71, 255, 0),
    (255, 0, 163),
    (255, 204, 0),
    (255, 0, 143),
    (0, 255, 235),
    (133, 255, 0),
    (255, 0, 235),
    (245, 0, 255),
    (255, 0, 122),
    (255, 245, 0),
    (10, 190, 212),
    (214, 255, 0),
    (0, 204, 255),
    (20, 0, 255),
    (255, 255, 0),
    (0, 153, 255),
    (0, 41, 255),
    (0, 255, 204),
    (41, 0, 255),
    (41, 255, 0),
    (173, 0, 255),
    (0, 245, 255),
    (71, 0, 255),
    (122, 0, 255),
    (0, 255, 184),
    (0, 92, 255),
    (184, 255, 0),
    (0, 133, 255),
    (255, 214, 0),
    (25, 194, 194),
    (102, 255, 0),
    (92, 0, 255),
]


class ADE20KSemanticSegmentation(Dataset):

    def __init__(self,
                 root_dir,
                 image_sets='training',
                 reduce_zero_label=False,
                 transform=None):
        assert image_sets in ['training', 'validation']

        self.imagepath = os.path.join(root_dir, 'images', image_sets, '%s.jpg')
        self.maskpath = os.path.join(root_dir, 'annotations', image_sets,
                                     '%s.png')

        self.cats = ADE20K_CLASSES
        self.num_classes = len(self.cats)

        self.cat_to_ade20k_label = {
            cat: i + 1
            for i, cat in enumerate(self.cats)
        }
        self.ade20k_label_to_cat = {
            i + 1: cat
            for i, cat in enumerate(self.cats)
        }

        self.ids = []
        for per_image_name in os.listdir(
                os.path.join(root_dir, 'images', image_sets)):
            image_name = per_image_name.split('.')[0]
            per_image_path = self.imagepath % image_name
            per_mask_path = self.maskpath % image_name

            if not os.path.exists(per_image_path) or not os.path.exists(
                    per_mask_path):
                continue
            self.ids.append(image_name)

        self.reduce_zero_label = reduce_zero_label
        self.transform = transform

        print(f'Dataset Size:{len(self.ids)}')
        if self.reduce_zero_label:
            print(f'Dataset Class Num:{self.num_classes}')
        else:
            print(f'Dataset Class Num:{self.num_classes+1}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'image': image,
            'mask': mask,
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

    def load_mask(self, idx):
        # h,w
        mask = np.array(Image.open(self.maskpath % self.ids[idx]),
                        dtype=np.uint8)
        # If class 0 is the background class and you want to ignore it when calculating the evaluation index,
        # you need to set reduce_zero_label=True.
        if self.reduce_zero_label:
            # avoid using underflow conversion
            mask[mask == 0] = 255
            mask = mask - 1
            # background class 0 transform to class 255,class 1~150 transform to 0~149
            mask[mask == 254] = 255

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

    from tools.path import ADE20Kdataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.semantic_segmentation.common import Resize, RandomCrop, RandomHorizontalFlip, PhotoMetricDistortion, Normalize, SemanticSegmentationCollater

    ade20kdataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        reduce_zero_label=True,
        transform=transforms.Compose([
            Resize(image_scale=(2048, 512),
                   multi_scale=True,
                   multi_scale_range=(0.5, 2.0)),
            RandomCrop(crop_size=(512, 512),
                       cat_max_ratio=0.75,
                       ignore_index=255),
            RandomHorizontalFlip(prob=0.5),
            PhotoMetricDistortion(),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(ade20kdataset):
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['scale'], per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mask = per_sample['mask']
        # mask_jpg = np.zeros((image.shape[0], image.shape[1], 3))

        # all_classes = np.unique(mask)
        # print("1212", all_classes)
        # for per_class in all_classes:
        #     per_class = int(per_class)
        #     if per_class < 0 or per_class >= 255:
        #         continue
        #     class_name, class_color = ADE20K_CLASSES[
        #         per_class], ADK20K_CLASSES_COLOR[per_class]
        #     class_color = np.array(
        #         (class_color[2], class_color[1], class_color[0]))
        #     per_mask = (mask == per_class).astype(np.float32)
        #     per_mask = np.expand_dims(per_mask, axis=-1)
        #     per_mask = np.tile(per_mask, (1, 1, 3))
        #     mask_color = np.expand_dims(np.expand_dims(class_color, axis=0),
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
    collater = SemanticSegmentationCollater(divisor=32, ignore_index=255)
    train_loader = DataLoader(ade20kdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('2222', images.shape, masks.shape, scales.shape, sizes.shape)
        print('2222', images.dtype, masks.dtype, scales.dtype, sizes.dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # masks = masks.cpu().numpy()

        # for i, (per_image,
        #         per_image_mask_targets) in enumerate(zip(images, masks)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_mask_jpg = np.zeros(
        #         (per_image.shape[0], per_image.shape[1], 3))

        #     all_classes = np.unique(per_image_mask_targets)
        #     print("2323", all_classes)
        #     for per_class in all_classes:
        #         per_class = int(per_class)
        #         if per_class < 0 or per_class >= 255:
        #             continue
        #         class_name, class_color = ADE20K_CLASSES[
        #             per_class], ADK20K_CLASSES_COLOR[per_class]
        #         class_color = np.array(
        #             (class_color[2], class_color[1], class_color[0]))
        #         per_image_mask = (per_image_mask_targets == per_class).astype(
        #             np.float32)
        #         per_image_mask = np.expand_dims(per_image_mask, axis=-1)
        #         per_image_mask = np.tile(per_image_mask, (1, 1, 3))
        #         mask_color = np.expand_dims(np.expand_dims(class_color,
        #                                                    axis=0),
        #                                     axis=0)

        #         per_image_mask = per_image_mask * mask_color
        #         per_image = 0.5 * per_image_mask + per_image
        #         per_image_mask_jpg += per_image_mask

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
        #     cv2.imencode('.jpg', per_image_mask_jpg)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

        if count < 10:
            count += 1
        else:
            break

    ade20kdataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        reduce_zero_label=False,
        transform=transforms.Compose([
            Resize(image_scale=(2048, 512),
                   multi_scale=True,
                   multi_scale_range=(0.5, 2.0)),
            RandomCrop(crop_size=(512, 512),
                       cat_max_ratio=0.75,
                       ignore_index=None),
            RandomHorizontalFlip(prob=0.5),
            PhotoMetricDistortion(),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(ade20kdataset):
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['scale'], per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        # temp_dir = './temp3'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mask = per_sample['mask']
        # mask_jpg = np.zeros((image.shape[0], image.shape[1], 3))

        # all_classes = np.unique(mask)
        # print("1212", all_classes)
        # for per_class in all_classes:
        #     if per_class == 0:
        #         continue
        #     per_class = int(per_class)
        #     if per_class < 0 or per_class >= 255:
        #         continue
        #     class_name, class_color = ADE20K_CLASSES[
        #         per_class - 1], ADK20K_CLASSES_COLOR[per_class - 1]
        #     class_color = np.array(
        #         (class_color[2], class_color[1], class_color[0]))
        #     per_mask = (mask == per_class).astype(np.float32)
        #     per_mask = np.expand_dims(per_mask, axis=-1)
        #     per_mask = np.tile(per_mask, (1, 1, 3))
        #     mask_color = np.expand_dims(np.expand_dims(class_color, axis=0),
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
    collater = SemanticSegmentationCollater(divisor=32, ignore_index=None)
    train_loader = DataLoader(ade20kdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('2222', images.shape, masks.shape, scales.shape, sizes.shape)
        print('2222', images.dtype, masks.dtype, scales.dtype, sizes.dtype)

        # temp_dir = './temp4'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # masks = masks.cpu().numpy()

        # for i, (per_image,
        #         per_image_mask_targets) in enumerate(zip(images, masks)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_mask_jpg = np.zeros(
        #         (per_image.shape[0], per_image.shape[1], 3))

        #     all_classes = np.unique(per_image_mask_targets)
        #     print("2323", all_classes)
        #     for per_class in all_classes:
        #         if per_class == 0:
        #             continue
        #         per_class = int(per_class)
        #         if per_class < 0 or per_class >= 255:
        #             continue
        #         class_name, class_color = ADE20K_CLASSES[
        #             per_class - 1], ADK20K_CLASSES_COLOR[per_class - 1]
        #         class_color = np.array(
        #             (class_color[2], class_color[1], class_color[0]))
        #         per_image_mask = (per_image_mask_targets == per_class).astype(
        #             np.float32)
        #         per_image_mask = np.expand_dims(per_image_mask, axis=-1)
        #         per_image_mask = np.tile(per_image_mask, (1, 1, 3))
        #         mask_color = np.expand_dims(np.expand_dims(class_color,
        #                                                    axis=0),
        #                                     axis=0)

        #         per_image_mask = per_image_mask * mask_color
        #         per_image = 0.5 * per_image_mask + per_image
        #         per_image_mask_jpg += per_image_mask

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
        #     cv2.imencode('.jpg', per_image_mask_jpg)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

        if count < 10:
            count += 1
        else:
            break