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
    (237, 120, 2),
    (145, 92, 129),
    (144, 60, 159),
    (252, 97, 223),
    (166, 144, 77),
    (236, 102, 36),
    (215, 80, 20),
    (61, 10, 159),
    (33, 26, 163),
    (220, 78, 67),
    (177, 45, 8),
    (125, 59, 8),
    (94, 223, 205),
    (132, 3, 28),
    (69, 121, 111),
    (155, 99, 103),
    (29, 76, 112),
    (56, 4, 106),
    (20, 25, 110),
    (4, 113, 110),
    (44, 46, 99),
    (79, 27, 51),
    (240, 35, 57),
    (110, 221, 98),
    (114, 173, 192),
    (131, 72, 231),
    (58, 79, 54),
    (109, 54, 77),
    (196, 107, 92),
    (177, 101, 2),
    (53, 24, 79),
    (5, 127, 82),
    (8, 76, 30),
    (77, 48, 148),
    (45, 58, 254),
    (94, 145, 65),
    (20, 183, 197),
    (36, 118, 251),
    (186, 187, 128),
    (11, 73, 181),
    (143, 3, 123),
    (122, 33, 11),
    (136, 36, 28),
    (23, 103, 1),
    (83, 58, 208),
    (153, 169, 53),
    (66, 15, 229),
    (139, 129, 223),
    (82, 81, 15),
    (109, 39, 128),
    (2, 141, 23),
    (1, 146, 167),
    (4, 249, 207),
    (13, 68, 84),
    (38, 50, 79),
    (5, 214, 130),
    (104, 114, 76),
    (71, 240, 189),
    (87, 118, 126),
    (14, 23, 77),
    (168, 232, 102),
    (246, 29, 200),
    (78, 143, 116),
    (195, 9, 144),
    (20, 6, 236),
    (139, 108, 94),
    (129, 13, 52),
    (79, 42, 42),
    (232, 252, 111),
    (197, 136, 89),
    (131, 36, 66),
    (89, 87, 44),
    (251, 67, 127),
    (35, 87, 29),
    (99, 33, 7),
    (174, 83, 65),
    (126, 222, 136),
    (168, 103, 35),
    (91, 60, 68),
    (46, 100, 194),
    (172, 113, 224),
    (63, 39, 92),
    (180, 124, 124),
    (76, 144, 76),
    (171, 245, 61),
    (59, 77, 28),
    (48, 94, 105),
    (51, 220, 95),
    (91, 154, 196),
    (104, 70, 34),
    (81, 71, 50),
    (117, 131, 5),
    (108, 10, 25),
    (114, 166, 104),
    (1, 20, 101),
    (111, 59, 118),
    (251, 44, 128),
    (173, 249, 182),
    (52, 90, 13),
    (93, 240, 43),
    (27, 49, 162),
    (20, 46, 78),
    (205, 27, 119),
    (62, 250, 141),
    (1, 78, 228),
    (56, 212, 242),
    (15, 17, 147),
    (177, 2, 173),
    (138, 34, 44),
    (38, 250, 141),
    (94, 76, 19),
    (243, 64, 38),
    (122, 87, 27),
    (39, 37, 128),
    (65, 185, 212),
    (76, 18, 29),
    (41, 169, 104),
    (35, 156, 123),
    (229, 48, 10),
    (45, 136, 26),
    (51, 124, 111),
    (107, 118, 234),
    (111, 249, 159),
    (219, 40, 175),
    (92, 83, 121),
    (155, 71, 142),
    (100, 166, 127),
    (21, 56, 251),
    (38, 86, 46),
    (139, 106, 42),
    (114, 239, 1),
    (132, 195, 159),
    (142, 187, 234),
    (95, 47, 240),
    (4, 33, 206),
    (33, 138, 180),
    (92, 15, 26),
    (200, 25, 228),
    (50, 167, 213),
    (4, 216, 254),
    (74, 124, 167),
    (220, 2, 95),
    (128, 21, 40),
    (229, 177, 109),
    (124, 78, 147),
    (59, 35, 241),
    (42, 147, 236),
    (107, 92, 136),
    (200, 59, 71),
    (32, 213, 15),
    (3, 148, 214),
    (77, 17, 4),
    (69, 166, 14),
    (28, 32, 80),
    (61, 233, 224),
    (220, 76, 140),
    (36, 88, 166),
    (92, 157, 26),
    (46, 78, 124),
    (220, 71, 229),
    (24, 144, 253),
    (83, 10, 48),
    (75, 27, 174),
    (45, 7, 82),
    (83, 198, 177),
    (190, 197, 93),
    (78, 50, 17),
    (6, 233, 56),
    (24, 0, 123),
    (157, 94, 60),
    (165, 27, 82),
    (91, 0, 107),
    (145, 171, 94),
    (135, 11, 202),
    (94, 93, 216),
    (84, 1, 14),
    (78, 48, 67),
    (192, 11, 22),
    (18, 77, 52),
    (118, 130, 50),
    (65, 47, 174),
    (151, 7, 57),
    (136, 32, 143),
    (223, 137, 253),
    (42, 159, 204),
    (57, 146, 30),
    (164, 244, 253),
    (117, 47, 43),
    (185, 68, 173),
    (113, 217, 178),
    (42, 96, 72),
    (158, 138, 207),
    (50, 12, 246),
    (37, 70, 215),
    (199, 98, 133),
    (131, 9, 7),
    (94, 72, 56),
    (241, 119, 143),
    (244, 18, 54),
    (206, 145, 135),
    (89, 150, 105),
    (162, 40, 141),
    (149, 36, 227),
    (123, 73, 44),
    (135, 230, 231),
    (117, 0, 2),
    (90, 18, 90),
    (6, 2, 94),
    (70, 88, 224),
    (126, 217, 50),
    (67, 243, 103),
    (59, 108, 96),
    (20, 6, 81),
    (73, 179, 254),
    (21, 186, 128),
    (0, 84, 111),
    (0, 87, 187),
    (70, 74, 143),
    (239, 74, 90),
    (146, 15, 26),
    (86, 118, 3),
    (76, 245, 247),
    (133, 170, 70),
    (87, 189, 2),
    (252, 152, 70),
    (106, 69, 6),
    (56, 47, 163),
    (62, 127, 205),
    (122, 0, 49),
    (79, 2, 63),
    (34, 138, 25),
    (156, 46, 60),
    (48, 80, 212),
    (60, 231, 14),
    (114, 17, 74),
    (145, 222, 19),
    (18, 159, 8),
    (73, 83, 106),
    (21, 227, 70),
    (45, 7, 156),
    (136, 79, 88),
    (253, 26, 5),
    (104, 15, 100),
    (16, 81, 18),
    (77, 217, 137),
    (62, 99, 118),
    (175, 20, 141),
    (29, 107, 77),
    (168, 117, 147),
    (183, 12, 8),
    (129, 109, 233),
    (254, 140, 117),
    (34, 78, 76),
    (142, 40, 92),
    (161, 135, 19),
    (51, 104, 53),
    (90, 217, 133),
    (88, 22, 14),
    (147, 251, 202),
    (173, 51, 45),
    (200, 138, 35),
    (51, 120, 82),
    (65, 111, 56),
    (93, 213, 72),
    (141, 3, 6),
    (5, 247, 108),
    (168, 31, 37),
    (21, 65, 136),
    (225, 9, 193),
    (4, 80, 137),
    (80, 117, 254),
    (144, 233, 90),
    (37, 235, 200),
    (43, 146, 59),
    (213, 89, 210),
    (130, 5, 77),
    (243, 169, 173),
    (177, 56, 90),
    (114, 81, 78),
    (222, 46, 135),
    (98, 21, 154),
    (100, 124, 194),
    (177, 170, 54),
    (133, 98, 10),
    (198, 251, 71),
    (124, 3, 121),
    (64, 17, 80),
    (74, 87, 32),
    (57, 59, 140),
    (25, 66, 76),
    (7, 245, 4),
    (115, 164, 244),
    (154, 2, 218),
    (48, 42, 131),
    (176, 51, 242),
    (40, 95, 114),
    (34, 12, 85),
    (252, 241, 166),
    (140, 23, 86),
    (29, 114, 108),
    (54, 222, 54),
    (94, 5, 0),
    (160, 4, 17),
    (40, 160, 166),
    (132, 143, 98),
    (59, 158, 90),
    (47, 244, 170),
    (128, 118, 172),
    (31, 80, 6),
    (133, 62, 96),
    (153, 108, 253),
    (110, 25, 39),
    (122, 26, 26),
    (199, 46, 115),
    (30, 164, 163),
    (75, 39, 82),
    (82, 9, 251),
    (211, 6, 37),
    (160, 57, 21),
    (55, 51, 82),
    (154, 89, 4),
    (254, 141, 250),
    (169, 126, 102),
    (64, 164, 61),
    (193, 140, 240),
    (46, 227, 0),
    (82, 107, 171),
    (48, 226, 204),
    (78, 68, 113),
    (92, 32, 64),
    (0, 36, 79),
    (106, 248, 243),
    (12, 116, 136),
    (29, 206, 103),
    (39, 241, 99),
    (52, 181, 122),
    (95, 63, 129),
    (81, 225, 86),
    (117, 10, 252),
    (201, 89, 43),
    (119, 43, 222),
    (207, 243, 159),
    (133, 151, 2),
    (230, 215, 42),
    (107, 73, 231),
    (158, 243, 142),
    (49, 63, 136),
    (193, 176, 8),
    (161, 2, 97),
    (70, 84, 57),
    (29, 118, 45),
    (106, 139, 27),
    (193, 27, 38),
    (4, 7, 195),
    (94, 35, 37),
    (221, 152, 207),
    (75, 160, 169),
    (5, 146, 58),
    (141, 32, 1),
    (123, 57, 149),
    (188, 5, 254),
    (176, 93, 156),
    (21, 115, 157),
    (53, 23, 145),
    (73, 221, 46),
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