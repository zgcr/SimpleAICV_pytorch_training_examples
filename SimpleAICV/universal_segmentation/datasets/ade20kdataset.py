import os
import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset

# background class and 150 class
ADE20K_CLASSES = [
    'background', 'wall', 'building, edifice', 'sky', 'floor, flooring',
    'tree', 'ceiling', 'road, routev', 'bed', 'windowpane, window', 'grass',
    'cabinet', 'sidewalk, pavement',
    'person, individual, someone, somebody, mortal, soul', 'earth, ground',
    'door, double door', 'table', 'mountain, mount',
    'plant, flora, plant life', 'curtain, drape, drapery, mantle, pall',
    'chair', 'car, auto, automobile, machine, motorcar', 'water',
    'painting, picture', 'sofa, couch, lounge', 'shelf', 'house', 'sea',
    'mirror', 'rug, carpet, carpeting', 'field', 'armchair', 'seat',
    'fence, fencing', 'desk', 'rock, stone', 'wardrobe, closet, press', 'lamp',
    'bathtub, bathing tub, bath, tub', 'railing, rail', 'cushion',
    'base, pedestal, stand', 'box', 'column, pillar', 'signboard, sign',
    'chest of drawers, chest, bureau, dresser', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace, hearth, open fireplace', 'refrigerator, icebox',
    'grandstand, covered stand', 'path', 'stairs, steps', 'runway',
    'case, display case, showcase, vitrine',
    'pool table, billiard table, snooker table', 'pillow',
    'screen door, screen', 'stairway, staircase', 'river', 'bridge, span',
    'bookcase', 'blind, screen', 'coffee table, cocktail table',
    'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower',
    'book', 'hill', 'bench', 'countertop',
    'stove, kitchen stove, range, kitchen range, cooking stove',
    'palm, palm tree', 'kitchen island',
    'computer, computing machine, computing device, data processor, electronic computer, information processing system',
    'swivel chair', 'boat', 'bar', 'arcade machine',
    'hovel, hut, hutch, shack, shanty',
    'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle',
    'towel', 'light, light source', 'truck, motortruck', 'tower',
    'chandelier, pendant, pendent', 'awning, sunshade, sunblind',
    'streetlight, street lamp', 'booth, cubicle, stall, kiosk',
    'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box',
    'airplane, aeroplane, plane', 'dirt track',
    'apparel, wearing apparel, dress, clothes', 'pole', 'land, ground, soil',
    'bannister, banister, balustrade, balusters, handrail',
    'escalator, moving staircase, moving stairway',
    'ottoman, pouf, pouffe, puff, hassock', 'bottle',
    'buffet, counter, sideboard',
    'poster, posting, placard, notice, bill, card', 'stage', 'van', 'ship',
    'fountain',
    'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy',
    'washer, automatic washer, washing machine', 'plaything, toy',
    'swimming pool, swimming bath, natatorium', 'stool', 'barrel, cask',
    'basket, handbasket', 'waterfall, falls', 'tent, collapsible shelter',
    'bag', 'minibike, motorbike', 'cradle', 'oven', 'ball', 'food, solid food',
    'step, stair', 'tank, storage tank',
    'trade name, brand name, brand, marque', 'microwave, microwave oven',
    'pot, flowerpot', 'animal, animate being, beast, brute, creature, fauna',
    'bicycle, bike, wheel, cycle', 'lake',
    'dishwasher, dish washer, dishwashing machine',
    'screen, silver screen, projection screen', 'blanket, cover', 'sculpture',
    'hood, exhaust hood', 'sconce', 'vase',
    'traffic light, traffic signal, stoplight', 'tray',
    'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
    'fan', 'pier, wharf, wharfage, dock', 'crt screen', 'plate',
    'monitor, monitoring device', 'bulletin board, notice board', 'shower',
    'radiator', 'glass, drinking glass', 'clock', 'flag'
]

# background class and 150 class RGB color
# background is black color
ADK20K_CLASSES_COLOR = [(0, 0, 0), (251, 39, 22), (114, 58, 11),
                        (79, 126, 122), (48, 198, 87), (87, 30, 109),
                        (206, 155, 9), (52, 252, 41), (33, 129, 3),
                        (44, 125, 81), (79, 88, 208), (19, 182, 21),
                        (166, 206, 10), (30, 1, 230), (138, 48, 26),
                        (129, 43, 54), (58, 227, 156), (2, 139, 223),
                        (94, 121, 55), (213, 244, 168), (19, 179, 181),
                        (111, 234, 23), (159, 132, 47), (181, 251, 145),
                        (59, 228, 83), (64, 11, 246), (120, 135, 29),
                        (91, 6, 70), (77, 118, 159), (147, 98, 12),
                        (198, 24, 48), (97, 104, 6), (9, 45, 87), (52, 97, 13),
                        (127, 108, 56), (45, 171, 151), (0, 54, 239),
                        (67, 210, 206), (134, 10, 179), (80, 13, 1),
                        (122, 28, 2), (9, 96, 30), (177, 120, 117),
                        (90, 191, 157), (97, 25, 1), (69, 54, 191),
                        (143, 214, 245), (96, 168, 220), (129, 119, 214),
                        (114, 89, 131), (155, 27, 121), (152, 248, 96),
                        (232, 146, 170), (182, 32, 103), (205, 56, 244),
                        (21, 33, 181), (58, 99, 78), (180, 98, 235), (2, 4,
                                                                      175),
                        (115, 0, 11), (186, 1, 56), (3, 163, 3), (99, 47, 85),
                        (169, 6, 145), (35, 54, 236), (58, 58, 93),
                        (36, 244, 31), (22, 127, 180), (246, 238, 118),
                        (97, 253, 112), (4, 99, 115), (32, 73, 94),
                        (55, 49, 180), (150, 72, 112), (101, 29, 33),
                        (202, 10, 20), (164, 112, 71), (39, 35, 86),
                        (78, 53, 89), (64, 0, 86), (81, 39, 6), (109, 44, 157),
                        (64, 19, 140), (65, 86, 28), (63, 136, 11),
                        (27, 246, 112), (79, 31, 236), (52, 3, 78),
                        (244, 108, 80), (72, 223, 99), (180, 137, 216),
                        (8, 23, 106), (249, 128, 243), (78, 3, 8),
                        (133, 36, 100), (102, 105, 228), (121, 17, 15),
                        (149, 220, 165), (120, 7, 102), (46, 77, 35),
                        (46, 98, 116), (238, 45, 47), (205, 2, 251),
                        (35, 173, 232), (77, 151, 3), (8, 204, 127),
                        (80, 58, 53), (88, 208, 52), (200, 65, 84),
                        (91, 92, 157), (105, 67, 60), (46, 20, 152),
                        (98, 7, 250), (55, 146, 196), (103, 131, 90),
                        (239, 186, 43), (116, 213, 198), (52, 83, 170),
                        (27, 102, 49), (3, 77, 54), (146, 1, 69), (4, 188, 64),
                        (203, 82, 14), (197, 160, 68), (23, 6, 78),
                        (209, 70, 148), (168, 31, 11), (251, 30, 111),
                        (86, 7, 140), (34, 31, 131), (70, 22, 83), (253, 111,
                                                                    4),
                        (82, 16, 39), (151, 71, 230), (114, 15, 55),
                        (19, 78, 168), (93, 13, 13), (239, 74, 209),
                        (41, 102, 222), (145, 90, 88), (97, 51, 27),
                        (201, 50, 42), (30, 120, 99), (80, 72, 36),
                        (162, 14, 41), (54, 28, 88), (248, 200, 170),
                        (115, 246, 140), (75, 126, 84), (169, 37, 225),
                        (24, 86, 15)]


class ADE20KSemanticSegmentation(Dataset):

    def __init__(self, root_dir, image_sets='training', transform=None):
        assert image_sets in ['training', 'validation']

        self.imagepath = os.path.join(root_dir, 'images', image_sets, '%s.jpg')
        self.maskpath = os.path.join(root_dir, 'annotations', image_sets,
                                     '%s.png')

        self.cats = ADE20K_CLASSES
        self.num_classes = len(self.cats)

        self.cat_to_ade20k_label = {}
        self.ade20k_label_to_cat = {}
        self.cat_to_ade20k_label['background'] = 0
        self.ade20k_label_to_cat[0] = 'background'

        for i, cat in enumerate(self.cats):
            self.cat_to_ade20k_label[cat] = i + 1

        for i, cat in enumerate(self.cats):
            self.ade20k_label_to_cat[i + 1] = cat

        # self.ids = []
        # for per_image_name in tqdm(
        #         os.listdir(os.path.join(root_dir, 'images', image_sets))):
        #     image_name = per_image_name.split('.')[0]
        #     per_image_path = self.imagepath % image_name
        #     per_mask_path = self.maskpath % image_name

        #     if not os.path.exists(per_image_path) or not os.path.exists(
        #             per_mask_path):
        #         continue

        #     per_mask = np.array(Image.open(self.maskpath % image_name),
        #                         dtype=np.uint8)
        #     per_mask_class_ids = list(np.unique(per_mask))
        #     if 0 in per_mask_class_ids:
        #         per_mask_class_ids.remove(0)

        #     if len(per_mask_class_ids) == 0:
        #         continue

        #     self.ids.append(image_name)
        # self.ids = sorted(self.ids)

        self.ids = []
        image_names = sorted(
            os.listdir(os.path.join(root_dir, 'images', image_sets)))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.process_single_image, per_image_name,
                                root_dir, image_sets)
                for per_image_name in image_names
            ]

            results = []
            for future in tqdm(as_completed(futures), total=len(image_names)):
                results.append(future.result())
        self.ids = [result for result in results if result is not None]
        self.ids = sorted(self.ids)

        self.transform = transform

        print(f'Dataset Size:{len(self.ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def process_single_image(self, per_image_name, root_dir, image_sets):
        image_name = per_image_name.split('.')[0]
        per_image_path = os.path.join(root_dir, 'images', image_sets,
                                      f'{image_name}.jpg')
        per_mask_path = os.path.join(root_dir, 'annotations', image_sets,
                                     f'{image_name}.png')

        if not os.path.exists(per_image_path) or not os.path.exists(
                per_mask_path):
            return None

        per_mask = np.array(Image.open(per_mask_path), dtype=np.uint8)
        per_mask_class_ids = list(np.unique(per_mask))
        if 0 in per_mask_class_ids:
            per_mask_class_ids.remove(0)

        if len(per_mask_class_ids) == 0:
            return None

        return image_name

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path = self.imagepath % self.ids[idx]
        image = self.load_image(idx)
        mask = self.load_mask(idx)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)
        origin_size = size.copy()

        sample = {
            'path': path,
            'image': image,
            'mask': mask,
            'size': size,
            'origin_size': origin_size,
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
        # class 0 is the background class
        mask = np.array(Image.open(self.maskpath % self.ids[idx]),
                        dtype=np.uint8)

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

    import copy

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.semantic_segmentation_common import YoloStyleResize, RandomHorizontalFlip, Normalize, SemanticSegmentationTrainCollater

    ade20kdataset = ADE20KSemanticSegmentation(
        ADE20Kdataset_path,
        image_sets='training',
        transform=transforms.Compose([
            YoloStyleResize(resize=512),
            RandomHorizontalFlip(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(ade20kdataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['size'], per_sample['origin_size'])
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

        #     class_name, class_color = ADE20K_CLASSES[
        #         per_class], ADK20K_CLASSES_COLOR[per_class]
        #     all_colors.append(class_color)
        # all_classes = list(all_classes)

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
    collater = SemanticSegmentationTrainCollater(resize=512)
    train_loader = DataLoader(ade20kdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, labels, sizes = data['image'], data['mask'], data[
            'label'], data['size']
        print('1111', images.shape, len(masks), len(labels), sizes.shape)

        for per_image_masks, per_image_labels in zip(masks, labels):
            print('2222', per_image_masks.shape, per_image_labels.shape)
            print('3333', per_image_labels)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()

        # for i, (per_image, per_image_masks,
        #         per_image_labels) in enumerate(zip(images, masks, labels)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)
        #     per_image_not_draw = copy.deepcopy(per_image)

        #     per_image_masks = per_image_masks.cpu().numpy()
        #     per_image_labels = per_image_labels.cpu().numpy()

        #     all_classes = list(np.unique(per_image_labels))
        #     all_classes = [int(x + 1) for x in all_classes]
        #     print('1212', all_classes)

        #     all_colors = []
        #     for per_class in all_classes:
        #         if per_class < 0 or per_class > 255:
        #             continue

        #         class_name, class_color = ADE20K_CLASSES[
        #             per_class], ADK20K_CLASSES_COLOR[per_class]
        #         all_colors.append(class_color)

        #     per_image_mask = np.zeros(
        #         (per_image.shape[0], per_image.shape[1], 3))
        #     per_image_contours = []
        #     for idx, (per_mask,
        #               per_class) in enumerate(zip(per_image_masks,
        #                                           all_classes)):
        #         if per_class < 0 or per_class > 255:
        #             continue

        #         per_class_mask = np.nonzero(per_mask == 1)
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
