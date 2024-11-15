import os
import cv2
import numpy as np
from PIL import Image

import collections

from torch.utils.data import Dataset

FaceSynthetics_19_CLASSES = [
    'background',
    'skin',
    'nose',
    'right_eye',
    'left_eye',
    'right_brow',
    'left_brow',
    'right_ear',
    'left_ear',
    'mouth_interior',
    'top_lip',
    'bottom_lip',
    'neck',
    'hair',
    'beard',
    'clothing',
    'glasses',
    'headwear',
    'facewear',
]

CelebAMask_HQ_19_CLASSES = [
    'background',
    'skin',
    'nose',
    'eye_g',
    'l_eye',
    'r_eye',
    'l_brow',
    'r_brow',
    'l_ear',
    'r_ear',
    'mouth',
    'u_lip',
    'l_lip',
    'hair',
    'hat',
    'ear_r',
    'neck_l',
    'neck',
    'cloth',
]

CLASSES_19_COLOR = [
    (0, 0, 0),
    (172, 194, 217),
    (76, 153, 0),
    (204, 204, 0),
    (51, 51, 255),
    (204, 0, 204),
    (0, 255, 255),
    (255, 204, 204),
    (67, 5, 65),
    (255, 0, 0),
    (102, 204, 0),
    (255, 255, 0),
    (239, 180, 53),
    (0, 0, 204),
    (255, 51, 153),
    (0, 204, 204),
    (0, 51, 0),
    (12, 181, 119),
    (0, 204, 0),
]


class FaceParsingDataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name_list=[
                     'FaceSynthetics',
                 ],
                 set_type='train',
                 cats=FaceSynthetics_19_CLASSES,
                 transform=None):
        assert set_type in ['train', 'val', 'test']

        self.all_image_name_list = set()
        self.all_image_path_dict = collections.OrderedDict()
        self.all_mask_path_dict = collections.OrderedDict()
        for i, per_set_name in enumerate(set_name_list):
            per_set_dir = os.path.join(root_dir, per_set_name, set_type)
            for per_image_name in os.listdir(per_set_dir):
                if '.jpg' in per_image_name:
                    per_image_name = per_image_name
                    per_mask_name = per_image_name.split(".")[0] + '.png'
                    per_image_path = os.path.join(per_set_dir, per_image_name)
                    per_mask_path = os.path.join(per_set_dir, per_mask_name)

                    if os.path.exists(per_image_path) and os.path.exists(
                            per_mask_path):
                        self.all_image_name_list.add(per_image_name)
                        self.all_image_path_dict[
                            per_image_name] = per_image_path
                        self.all_mask_path_dict[per_image_name] = per_mask_path
        self.all_image_name_list = sorted(list(self.all_image_name_list))

        assert len(self.all_image_name_list) == len(
            self.all_image_path_dict) == len(self.all_mask_path_dict)

        self.cats = cats
        self.num_classes = len(self.cats)

        self.cat_to_label = {cat: i for i, cat in enumerate(self.cats)}
        self.label_to_cat = {i: cat for i, cat in enumerate(self.cats)}

        self.transform = transform

        # num_classes数量必须包含背景类
        print(f'Dataset Size:{len(self.all_image_name_list)}')
        print(f'Dataset Class Num:{len(self.cats)}')

    def __len__(self):
        return len(self.all_image_name_list)

    def __getitem__(self, idx):
        image_path = self.all_image_path_dict[self.all_image_name_list[idx]]
        mask_path = self.all_mask_path_dict[self.all_image_name_list[idx]]

        image = self.load_image(idx)
        mask = self.load_mask(idx)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'image_path': image_path,
            'mask_path': mask_path,
            'image': image,
            'mask': mask,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(
                self.all_image_path_dict[self.all_image_name_list[idx]],
                dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        loadmask_path = self.all_mask_path_dict[self.all_image_name_list[idx]]
        mask = np.array(Image.open(loadmask_path).convert('L'), dtype=np.uint8)
        # 255代表忽略区域，视为背景类
        mask[mask >= 255] = 0
        mask[mask <= 0] = 0

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

    from tools.path import face_parsing_dataset_path

    import copy

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.face_parsing.common import RandomCrop, RandomShrink, RandomRGBToGRAY, RandomHorizontalFlip, RandomVerticalFlip, YoloStyleResize, Resize, Normalize, FaceParsingCollater

    face_parsing_trainset = FaceParsingDataset(
        face_parsing_dataset_path,
        set_name_list=[
            'CelebAMask-HQ',
        ],
        set_type='train',
        cats=CelebAMask_HQ_19_CLASSES,
        transform=transforms.Compose([
            RandomCrop(prob=0.5, filter_percent=0.1, random_range=[-0.2, 0.5]),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.5),
            RandomRGBToGRAY(prob=0.1),
            RandomShrink(prob=0.5, random_resize_range=[128, 512]),
            YoloStyleResize(resize=512),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(face_parsing_trainset):
        print('1111', per_sample['image_path'])
        print('1111', per_sample['mask_path'])
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['size'])
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
        #     if per_class <= 0:
        #         continue
        #     class_name, class_color = FaceSynthetics_19_CLASSES[
        #         per_class], CLASSES_19_COLOR[per_class]
        #     all_colors.append(class_color)
        # all_classes = list(all_classes)
        # if 0 in all_classes:
        #     all_classes.remove(0)
        # print('1313', len(all_classes), len(all_colors))

        # per_image_mask = np.zeros((image.shape[0], image.shape[1], 3))
        # for idx, per_class in enumerate(all_classes):
        #     if per_class <= 0:
        #         continue

        #     per_class_mask = np.nonzero(mask == per_class)
        #     per_image_mask[per_class_mask[0],
        #                    per_class_mask[1]] = all_colors[idx]

        # per_image_mask = per_image_mask.astype('uint8')
        # per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

        # all_classes_mask = np.nonzero(per_image_mask != 0)
        # per_image_mask[all_classes_mask[0],
        #                all_classes_mask[1]] = cv2.addWeighted(
        #                    image[all_classes_mask[0], all_classes_mask[1]],
        #                    0.5, per_image_mask[all_classes_mask[0],
        #                                        all_classes_mask[1]], 0.5, 0)
        # no_class_mask = np.nonzero(per_image_mask == 0)
        # per_image_mask[no_class_mask[0],
        #                no_class_mask[1]] = image[no_class_mask[0],
        #                                          no_class_mask[1]]

        # cv2.imencode('.jpg', image_not_draw)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))
        # cv2.imencode('.jpg', per_image_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = FaceParsingCollater(resize=512)
    train_loader = DataLoader(face_parsing_trainset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, sizes = data['image'], data['mask'], data['size']
        print('2222', images.shape, masks.shape, sizes.shape, sizes)
        print('2222', images.dtype, masks.dtype, sizes.dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # masks = masks.cpu().numpy()

        # for i, (per_image,
        #         per_image_mask_targets) in enumerate(zip(images, masks)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)
        #     per_image_not_draw = copy.deepcopy(per_image)

        #     all_classes = np.unique(per_image_mask_targets)
        #     print('1212', all_classes)
        #     all_colors = []
        #     for per_class in all_classes:
        #         per_class = int(per_class)
        #         if per_class <= 0:
        #             continue
        #         class_name, class_color = FaceSynthetics_19_CLASSES[
        #             per_class], CLASSES_19_COLOR[per_class]
        #         all_colors.append(class_color)
        #     all_classes = list(all_classes)
        #     if 0 in all_classes:
        #         all_classes.remove(0)
        #     print('1313', len(all_classes), len(all_colors))

        #     per_image_mask = np.zeros(
        #         (per_image.shape[0], per_image.shape[1], 3))
        #     for idx, per_class in enumerate(all_classes):
        #         if per_class <= 0:
        #             continue

        #         per_class_mask = np.nonzero(
        #             per_image_mask_targets == per_class)
        #         per_image_mask[per_class_mask[0],
        #                        per_class_mask[1]] = all_colors[idx]

        #     per_image_mask = per_image_mask.astype('uint8')
        #     per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)

        #     all_classes_mask = np.nonzero(per_image_mask != 0)
        #     per_image_mask[all_classes_mask[0],
        #                    all_classes_mask[1]] = cv2.addWeighted(
        #                        per_image[all_classes_mask[0],
        #                                  all_classes_mask[1]], 0.5,
        #                        per_image_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 0.5, 0)
        #     no_class_mask = np.nonzero(per_image_mask == 0)
        #     per_image_mask[no_class_mask[0],
        #                    no_class_mask[1]] = per_image[no_class_mask[0],
        #                                                  no_class_mask[1]]

        #     cv2.imencode('.jpg', per_image_not_draw)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
        #     cv2.imencode('.jpg', per_image_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
