#-*- encoding: utf-8 -*-
import os
import ast
import cv2
import json
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from pycocotools.coco import COCO
from torch.utils.data import Dataset

COCO_CLASSES = [
    'Person',
    'Sneakers',
    'Chair',
    'Other Shoes',
    'Hat',
    'Car',
    'Lamp',
    'Glasses',
    'Bottle',
    'Desk',
    'Cup',
    'Street Lights',
    'Cabinet/shelf',
    'Handbag/Satchel',
    'Bracelet',
    'Plate',
    'Picture/Frame',
    'Helmet',
    'Book',
    'Gloves',
    'Storage box',
    'Boat',
    'Leather Shoes',
    'Flower',
    'Bench',
    'Potted Plant',
    'Bowl/Basin',
    'Flag',
    'Pillow',
    'Boots',
    'Vase',
    'Microphone',
    'Necklace',
    'Ring',
    'SUV',
    'Wine Glass',
    'Belt',
    'Moniter/TV',
    'Backpack',
    'Umbrella',
    'Traffic Light',
    'Speaker',
    'Watch',
    'Tie',
    'Trash bin Can',
    'Slippers',
    'Bicycle',
    'Stool',
    'Barrel/bucket',
    'Van',
    'Couch',
    'Sandals',
    'Bakset',
    'Drum',
    'Pen/Pencil',
    'Bus',
    'Wild Bird',
    'High Heels',
    'Motorcycle',
    'Guitar',
    'Carpet',
    'Cell Phone',
    'Bread',
    'Camera',
    'Canned',
    'Truck',
    'Traffic cone',
    'Cymbal',
    'Lifesaver',
    'Towel',
    'Stuffed Toy',
    'Candle',
    'Sailboat',
    'Laptop',
    'Awning',
    'Bed',
    'Faucet',
    'Tent',
    'Horse',
    'Mirror',
    'Power outlet',
    'Sink',
    'Apple',
    'Air Conditioner',
    'Knife',
    'Hockey Stick',
    'Paddle',
    'Pickup Truck',
    'Fork',
    'Traffic Sign',
    'Ballon',
    'Tripod',
    'Dog',
    'Spoon',
    'Clock',
    'Pot',
    'Cow',
    'Cake',
    'Dinning Table',
    'Sheep',
    'Hanger',
    'Blackboard/Whiteboard',
    'Napkin',
    'Other Fish',
    'Orange/Tangerine',
    'Toiletry',
    'Keyboard',
    'Tomato',
    'Lantern',
    'Machinery Vehicle',
    'Fan',
    'Green Vegetables',
    'Banana',
    'Baseball Glove',
    'Airplane',
    'Mouse',
    'Train',
    'Pumpkin',
    'Soccer',
    'Skiboard',
    'Luggage',
    'Nightstand',
    'Tea pot',
    'Telephone',
    'Trolley',
    'Head Phone',
    'Sports Car',
    'Stop Sign',
    'Dessert',
    'Scooter',
    'Stroller',
    'Crane',
    'Remote',
    'Refrigerator',
    'Oven',
    'Lemon',
    'Duck',
    'Baseball Bat',
    'Surveillance Camera',
    'Cat',
    'Jug',
    'Broccoli',
    'Piano',
    'Pizza',
    'Elephant',
    'Skateboard',
    'Surfboard',
    'Gun',
    'Skating and Skiing shoes',
    'Gas stove',
    'Donut',
    'Bow Tie',
    'Carrot',
    'Toilet',
    'Kite',
    'Strawberry',
    'Other Balls',
    'Shovel',
    'Pepper',
    'Computer Box',
    'Toilet Paper',
    'Cleaning Products',
    'Chopsticks',
    'Microwave',
    'Pigeon',
    'Baseball',
    'Cutting/chopping Board',
    'Coffee Table',
    'Side Table',
    'Scissors',
    'Marker',
    'Pie',
    'Ladder',
    'Snowboard',
    'Cookies',
    'Radiator',
    'Fire Hydrant',
    'Basketball',
    'Zebra',
    'Grape',
    'Giraffe',
    'Potato',
    'Sausage',
    'Tricycle',
    'Violin',
    'Egg',
    'Fire Extinguisher',
    'Candy',
    'Fire Truck',
    'Billards',
    'Converter',
    'Bathtub',
    'Wheelchair',
    'Golf Club',
    'Briefcase',
    'Cucumber',
    'Cigar/Cigarette ',
    'Paint Brush',
    'Pear',
    'Heavy Truck',
    'Hamburger',
    'Extractor',
    'Extention Cord',
    'Tong',
    'Tennis Racket',
    'Folder',
    'American Football',
    'earphone',
    'Mask',
    'Kettle',
    'Tennis',
    'Ship',
    'Swing',
    'Coffee Machine',
    'Slide',
    'Carriage',
    'Onion',
    'Green beans',
    'Projector',
    'Frisbee',
    'Washing Machine/Drying Machine',
    'Chicken',
    'Printer',
    'Watermelon',
    'Saxophone',
    'Tissue',
    'Toothbrush',
    'Ice cream',
    'Hotair ballon',
    'Cello',
    'French Fries',
    'Scale',
    'Trophy',
    'Cabbage',
    'Hot dog',
    'Blender',
    'Peach',
    'Rice',
    'Wallet/Purse',
    'Volleyball',
    'Deer',
    'Goose',
    'Tape',
    'Tablet',
    'Cosmetics',
    'Trumpet',
    'Pineapple',
    'Golf Ball',
    'Ambulance',
    'Parking meter',
    'Mango',
    'Key',
    'Hurdle',
    'Fishing Rod',
    'Medal',
    'Flute',
    'Brush',
    'Penguin',
    'Megaphone',
    'Corn',
    'Lettuce',
    'Garlic',
    'Swan',
    'Helicopter',
    'Green Onion',
    'Sandwich',
    'Nuts',
    'Speed Limit Sign',
    'Induction Cooker',
    'Broom',
    'Trombone',
    'Plum',
    'Rickshaw',
    'Goldfish',
    'Kiwi fruit',
    'Router/modem',
    'Poker Card',
    'Toaster',
    'Shrimp',
    'Sushi',
    'Cheese',
    'Notepaper',
    'Cherry',
    'Pliers',
    'CD',
    'Pasta',
    'Hammer',
    'Cue',
    'Avocado',
    'Hamimelon',
    'Flask',
    'Mushroon',
    'Screwdriver',
    'Soap',
    'Recorder',
    'Bear',
    'Eggplant',
    'Board Eraser',
    'Coconut',
    'Tape Measur/ Ruler',
    'Pig',
    'Showerhead',
    'Globe',
    'Chips',
    'Steak',
    'Crosswalk Sign',
    'Stapler',
    'Campel',
    'Formula 1 ',
    'Pomegranate',
    'Dishwasher',
    'Crab',
    'Hoverboard',
    'Meat ball',
    'Rice Cooker',
    'Tuba',
    'Calculator',
    'Papaya',
    'Antelope',
    'Parrot',
    'Seal',
    'Buttefly',
    'Dumbbell',
    'Donkey',
    'Lion',
    'Urinal',
    'Dolphin',
    'Electric Drill',
    'Hair Dryer',
    'Egg tart',
    'Jellyfish',
    'Treadmill',
    'Lighter',
    'Grapefruit',
    'Game board',
    'Mop',
    'Radish',
    'Baozi',
    'Target',
    'French',
    'Spring Rolls',
    'Monkey',
    'Rabbit',
    'Pencil Case',
    'Yak',
    'Red Cabbage',
    'Binoculars',
    'Asparagus',
    'Barbell',
    'Scallop',
    'Noddles',
    'Comb',
    'Dumpling',
    'Oyster',
    'Table Teniis paddle',
    'Cosmetics Brush/Eyeliner Pencil',
    'Chainsaw',
    'Eraser',
    'Lobster',
    'Durian',
    'Okra',
    'Lipstick',
    'Cosmetics Mirror',
    'Curling',
    'Table Tennis',
]

COCO_CLASSES_COLOR = [
    (172, 47, 117),
    (192, 67, 251),
    (195, 103, 9),
    (211, 21, 242),
    (36, 87, 70),
    (216, 88, 140),
    (58, 193, 230),
    (39, 87, 174),
    (88, 81, 165),
    (25, 77, 72),
    (9, 148, 115),
    (208, 243, 197),
    (254, 79, 175),
    (192, 82, 99),
    (216, 177, 243),
    (29, 147, 147),
    (142, 167, 32),
    (193, 9, 185),
    (127, 32, 31),
    (202, 244, 151),
    (163, 254, 203),
    (114, 183, 28),
    (34, 128, 128),
    (164, 53, 133),
    (38, 232, 244),
    (17, 79, 132),
    (105, 42, 186),
    (31, 120, 1),
    (65, 231, 169),
    (57, 35, 102),
    (119, 11, 174),
    (82, 91, 128),
    (142, 99, 53),
    (140, 121, 170),
    (84, 203, 68),
    (6, 196, 47),
    (127, 244, 131),
    (204, 100, 180),
    (232, 78, 143),
    (148, 227, 186),
    (23, 207, 141),
    (117, 85, 48),
    (49, 69, 169),
    (163, 192, 95),
    (197, 94, 0),
    (113, 178, 36),
    (162, 48, 93),
    (131, 98, 42),
    (205, 112, 231),
    (149, 201, 127),
    (0, 138, 114),
    (43, 186, 127),
    (23, 187, 130),
    (121, 98, 62),
    (163, 222, 123),
    (195, 82, 174),
    (227, 148, 209),
    (50, 155, 14),
    (41, 58, 193),
    (36, 10, 86),
    (43, 104, 11),
    (2, 51, 80),
    (32, 182, 128),
    (38, 19, 174),
    (42, 115, 184),
    (188, 232, 77),
    (30, 24, 125),
    (2, 3, 94),
    (226, 107, 13),
    (112, 40, 72),
    (19, 95, 72),
    (154, 194, 248),
    (180, 67, 236),
    (61, 14, 96),
    (4, 195, 237),
    (139, 252, 86),
    (205, 121, 109),
    (75, 184, 16),
    (152, 157, 149),
    (110, 25, 208),
    (188, 121, 118),
    (117, 189, 83),
    (161, 104, 160),
    (228, 251, 251),
    (121, 70, 213),
    (31, 13, 71),
    (184, 152, 79),
    (41, 18, 40),
    (182, 207, 11),
    (166, 111, 93),
    (249, 129, 223),
    (118, 44, 216),
    (125, 24, 67),
    (210, 239, 3),
    (234, 204, 230),
    (35, 214, 254),
    (189, 197, 215),
    (43, 32, 11),
    (104, 212, 138),
    (182, 235, 165),
    (125, 156, 111),
    (232, 2, 27),
    (211, 217, 151),
    (53, 51, 174),
    (148, 181, 29),
    (67, 35, 39),
    (137, 73, 41),
    (151, 131, 46),
    (218, 178, 108),
    (3, 31, 9),
    (138, 27, 173),
    (199, 167, 61),
    (85, 97, 44),
    (34, 162, 88),
    (33, 133, 232),
    (255, 36, 0),
    (203, 34, 197),
    (126, 181, 254),
    (80, 190, 136),
    (189, 129, 209),
    (112, 35, 120),
    (91, 168, 116),
    (36, 176, 25),
    (67, 103, 252),
    (35, 114, 30),
    (29, 241, 33),
    (146, 17, 221),
    (84, 253, 2),
    (69, 101, 140),
    (44, 117, 253),
    (66, 111, 91),
    (85, 167, 39),
    (203, 150, 158),
    (145, 198, 199),
    (18, 92, 43),
    (83, 177, 41),
    (93, 174, 149),
    (201, 89, 242),
    (224, 219, 73),
    (28, 235, 209),
    (105, 186, 128),
    (214, 63, 16),
    (106, 164, 94),
    (24, 116, 191),
    (195, 51, 136),
    (184, 91, 93),
    (123, 238, 87),
    (160, 147, 72),
    (199, 87, 13),
    (58, 81, 120),
    (116, 183, 64),
    (203, 220, 164),
    (25, 32, 170),
    (14, 214, 28),
    (20, 210, 68),
    (22, 227, 122),
    (83, 135, 200),
    (61, 141, 5),
    (0, 136, 207),
    (207, 181, 139),
    (4, 167, 92),
    (173, 26, 74),
    (52, 238, 177),
    (219, 51, 227),
    (105, 18, 117),
    (34, 51, 158),
    (181, 58, 171),
    (55, 252, 252),
    (18, 173, 87),
    (193, 70, 234),
    (53, 48, 94),
    (59, 80, 154),
    (124, 163, 58),
    (177, 106, 201),
    (44, 13, 121),
    (70, 38, 167),
    (136, 13, 248),
    (135, 208, 248),
    (22, 248, 79),
    (217, 8, 227),
    (6, 209, 199),
    (212, 217, 194),
    (60, 144, 56),
    (114, 237, 151),
    (24, 4, 100),
    (236, 49, 87),
    (30, 54, 153),
    (20, 97, 101),
    (185, 151, 155),
    (29, 161, 115),
    (53, 119, 179),
    (86, 246, 7),
    (105, 241, 137),
    (182, 128, 83),
    (120, 164, 209),
    (148, 117, 240),
    (3, 126, 42),
    (65, 20, 36),
    (68, 208, 112),
    (175, 138, 237),
    (104, 222, 91),
    (43, 63, 159),
    (148, 198, 9),
    (188, 91, 111),
    (163, 83, 76),
    (18, 113, 74),
    (226, 225, 171),
    (131, 140, 228),
    (58, 129, 113),
    (128, 39, 24),
    (186, 36, 99),
    (69, 134, 3),
    (226, 121, 168),
    (188, 161, 28),
    (68, 26, 224),
    (248, 109, 179),
    (201, 181, 197),
    (161, 135, 125),
    (94, 72, 246),
    (84, 135, 195),
    (213, 219, 108),
    (67, 102, 84),
    (71, 83, 223),
    (0, 133, 91),
    (107, 158, 201),
    (211, 7, 149),
    (229, 220, 136),
    (171, 46, 0),
    (104, 179, 38),
    (89, 74, 243),
    (226, 123, 87),
    (96, 83, 26),
    (206, 32, 115),
    (198, 97, 172),
    (59, 57, 178),
    (173, 233, 132),
    (185, 93, 91),
    (145, 163, 194),
    (148, 173, 185),
    (207, 119, 164),
    (105, 190, 4),
    (241, 242, 205),
    (158, 109, 87),
    (226, 163, 73),
    (218, 183, 26),
    (118, 22, 204),
    (207, 98, 90),
    (51, 230, 46),
    (208, 61, 188),
    (47, 250, 104),
    (128, 138, 203),
    (141, 71, 94),
    (6, 173, 245),
    (158, 15, 169),
    (166, 53, 171),
    (82, 135, 220),
    (65, 169, 66),
    (114, 92, 78),
    (229, 219, 246),
    (100, 159, 221),
    (178, 252, 174),
    (93, 114, 161),
    (12, 224, 233),
    (80, 66, 200),
    (243, 125, 138),
    (112, 218, 155),
    (184, 120, 65),
    (192, 197, 88),
    (34, 207, 3),
    (188, 238, 165),
    (171, 211, 88),
    (70, 148, 134),
    (28, 115, 134),
    (66, 92, 220),
    (102, 101, 123),
    (197, 109, 73),
    (100, 182, 77),
    (149, 251, 159),
    (81, 35, 237),
    (243, 250, 136),
    (254, 25, 21),
    (173, 229, 214),
    (144, 153, 238),
    (119, 165, 127),
    (129, 133, 198),
    (140, 90, 74),
    (251, 182, 78),
    (62, 72, 199),
    (45, 133, 47),
    (187, 170, 195),
    (138, 242, 57),
    (219, 89, 131),
    (125, 206, 82),
    (197, 186, 132),
    (17, 197, 191),
    (94, 152, 131),
    (69, 168, 164),
    (58, 177, 183),
    (152, 161, 146),
    (97, 206, 241),
    (135, 181, 235),
    (46, 240, 244),
    (127, 161, 81),
    (157, 12, 118),
    (46, 118, 32),
    (34, 115, 87),
    (124, 153, 174),
    (242, 107, 52),
    (233, 110, 64),
    (76, 118, 73),
    (146, 4, 7),
    (131, 48, 76),
    (37, 116, 237),
    (48, 109, 210),
    (179, 49, 71),
    (90, 177, 238),
    (29, 3, 74),
    (84, 234, 65),
    (119, 20, 208),
    (42, 242, 28),
    (154, 87, 180),
    (48, 194, 102),
    (9, 2, 116),
    (108, 233, 89),
    (124, 117, 100),
    (90, 68, 105),
    (10, 12, 104),
    (225, 165, 167),
    (160, 122, 33),
    (154, 99, 217),
    (50, 88, 210),
    (20, 222, 156),
    (72, 18, 153),
    (152, 39, 234),
    (123, 28, 152),
    (146, 195, 172),
    (211, 45, 54),
    (0, 138, 195),
    (134, 219, 99),
    (185, 198, 147),
    (162, 50, 242),
    (102, 20, 171),
    (118, 128, 228),
    (109, 105, 194),
    (225, 140, 15),
    (118, 33, 151),
    (140, 133, 38),
    (1, 10, 6),
    (125, 6, 102),
    (166, 75, 114),
    (85, 126, 114),
    (178, 51, 2),
    (76, 157, 9),
    (115, 200, 133),
    (116, 181, 235),
    (193, 43, 79),
    (62, 76, 47),
    (76, 149, 120),
    (18, 89, 13),
    (222, 92, 236),
    (105, 213, 6),
    (100, 48, 40),
    (154, 158, 133),
    (143, 191, 229),
    (37, 59, 113),
]


class Objects365Detection(Dataset):

    def __init__(self, root_dir, set_name='train', transform=None):
        assert set_name in ['train', 'val'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'zhiyuan_objv2_{set_name}.json')

        self.coco = COCO(self.annot_dir)

        self.image_ids = self.coco.getImgIds()

        filter_image_ids = []
        for per_id in tqdm(self.image_ids):
            file_name = self.coco.loadImgs(per_id)[0]['file_name']
            file_name = file_name[10:]
            per_image_path = os.path.join(self.image_dir, file_name)
            if os.path.exists(per_image_path):
                filter_image_ids.append(per_id)
        self.image_ids = filter_image_ids

        self.cat_ids = self.coco.getCatIds()
        self.cats = sorted(self.coco.loadCats(self.cat_ids),
                           key=lambda x: x['id'])
        self.num_classes = len(self.cats)

        # cat_id is an original cat id,coco_label is set from 0 to 364
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
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        file_name = file_name[10:]

        image = cv2.imdecode(
            np.fromfile(os.path.join(self.image_dir, file_name),
                        dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annots = self.coco.loadAnns(annot_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        targets = np.zeros((0, 5))
        if len(annots) == 0:
            return targets.astype(np.float32)

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
            if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
                continue
            if annot['category_id'] not in self.cat_ids:
                continue

            target = np.zeros((1, 5))
            target[0, :4] = bbox
            target[0, 4] = self.cat_id_to_coco_label[annot['category_id']]
            targets = np.append(targets, target, axis=0)

        # transform bbox targets from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        targets[:, 2] = targets[:, 0] + targets[:, 2]
        targets[:, 3] = targets[:, 1] + targets[:, 3]

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

    from tools.path import Objects365_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, DetectionResize, DetectionCollater

    objects365dataset = Objects365Detection(
        Objects365_path,
        set_name='val',
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            DetectionResize(resize=640,
                            stride=32,
                            resize_type='yolo_style',
                            multi_scale=False,
                            multi_scale_range=[0.8, 1.0]),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(objects365dataset):
        print('1111', per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['annots'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annots = per_sample['annots']

        # draw all label boxes
        for per_annot in annots:
            per_box = (per_annot[0:4]).astype(np.int32)
            per_box_class_index = per_annot[4].astype(np.int32)
            class_name, class_color = COCO_CLASSES[
                per_box_class_index], COCO_CLASSES_COLOR[per_box_class_index]
            left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                                per_box[3])
            cv2.rectangle(image,
                          left_top,
                          right_bottom,
                          color=class_color,
                          thickness=2,
                          lineType=cv2.LINE_AA)

            text = f'{class_name}'
            text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
            fill_right_bottom = (max(left_top[0] + text_size[0],
                                     right_bottom[0]),
                                 left_top[1] - text_size[1] - 3)
            cv2.rectangle(image,
                          left_top,
                          fill_right_bottom,
                          color=class_color,
                          thickness=-1,
                          lineType=cv2.LINE_AA)
            cv2.putText(image,
                        text, (left_top[0], left_top[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    # collater = DetectionCollater(resize=800,
    #                              resize_type='retina_style',
    #                              max_annots_num=150)
    collater = DetectionCollater(resize=640,
                                 resize_type='yolo_style',
                                 max_annots_num=150)
    train_loader = DataLoader(objects365dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('2222', images.shape, annots.shape, scales.shape, sizes.shape)
        print('2222', images.dtype, annots.dtype, scales.dtype, sizes.dtype)

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        images = images.permute(0, 2, 3, 1).cpu().numpy()
        annots = annots.cpu().numpy()

        for i, (per_image, per_image_annot) in enumerate(zip(images, annots)):
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

            # draw all label boxes
            for per_annot in per_image_annot:
                per_box = (per_annot[0:4]).astype(np.int32)
                per_box_class_index = per_annot[4].astype(np.int32)

                if per_box_class_index == -1:
                    continue

                class_name, class_color = COCO_CLASSES[
                    per_box_class_index], COCO_CLASSES_COLOR[
                        per_box_class_index]
                left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                                    per_box[3])
                cv2.rectangle(per_image,
                              left_top,
                              right_bottom,
                              color=class_color,
                              thickness=2,
                              lineType=cv2.LINE_AA)

                text = f'{class_name}'
                text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
                fill_right_bottom = (max(left_top[0] + text_size[0],
                                         right_bottom[0]),
                                     left_top[1] - text_size[1] - 3)
                cv2.rectangle(per_image,
                              left_top,
                              fill_right_bottom,
                              color=class_color,
                              thickness=-1,
                              lineType=cv2.LINE_AA)
                cv2.putText(per_image,
                            text, (left_top[0], left_top[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color=(0, 0, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA)

            cv2.imencode('.jpg', per_image)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))

        if count < 5:
            count += 1
        else:
            break