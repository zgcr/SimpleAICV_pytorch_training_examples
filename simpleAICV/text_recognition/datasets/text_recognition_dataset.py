import os
import cv2
import collections
import json
import imagesize
import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset


class CNENTextRecognition(Dataset):

    def __init__(self,
                 root_dir,
                 set_name=[
                     'aistudio_baidu_street',
                     'chinese_dataset',
                     'synthetic_chinese_string_dataset_trainsubset0',
                     'synthetic_chinese_string_dataset_trainsubset1',
                     'synthetic_chinese_string_dataset_trainsubset2',
                     'synthetic_chinese_string_dataset_trainsubset3',
                     'meta_self_learning_car',
                     'meta_self_learning_document_trainsubset0',
                     'meta_self_learning_document_trainsubset1',
                     'meta_self_learning_hand',
                     'meta_self_learning_street',
                     'meta_self_learning_syn',
                 ],
                 set_type='train',
                 str_max_length=80,
                 transform=None):
        assert set_type in ['train', 'test'], 'Wrong set name!'

        self.half_full_dict = {
            "，": ",",
            "；": ";",
            "：": ":",
            "？": "?",
            "（": "(",
            "）": ")",
            "！": "!",
        }

        all_image_dirs_list = []
        for per_set_name in set_name:
            per_set_image_dir = os.path.join(
                os.path.join(root_dir, per_set_name), set_type)
            all_image_dirs_list.append(per_set_image_dir)

        all_labels_path_list = []
        for per_set_name in set_name:
            per_set_path = os.path.join(root_dir, per_set_name)
            per_set_label_path = os.path.join(
                per_set_path, f"{per_set_name}_{set_type}.json")
            all_labels_path_list.append(per_set_label_path)

        self.chars_set = set()
        self.image_path_list = []
        self.image_label_dict = collections.OrderedDict()
        for per_set_image_dir_path, per_set_label_path in tqdm(
                zip(all_image_dirs_list, all_labels_path_list)):
            with open(per_set_label_path, 'r', encoding='UTF-8') as json_f:
                per_set_label = json.load(json_f)
                for per_image_name, per_image_label in tqdm(
                        per_set_label.items()):
                    per_image_path = os.path.join(per_set_image_dir_path,
                                                  per_image_name)

                    if not os.path.exists(per_image_path):
                        continue

                    text_image_w, text_image_h = imagesize.get(per_image_path)

                    if text_image_h < 8 or text_image_w < 8:
                        continue

                    if text_image_h / float(text_image_w) > 1.0:
                        continue

                    per_image_convert_label = ""
                    for per_char in per_image_label:
                        if per_char in self.half_full_dict.keys():
                            per_char = self.half_full_dict[per_char]
                        per_image_convert_label += per_char
                    per_image_label = per_image_convert_label

                    if 1 <= len(per_image_label) <= str_max_length:
                        self.image_path_list.append(per_image_path)
                        self.image_label_dict[per_image_name] = per_image_label

                        list_label = list(per_image_label)
                        for per_char in list_label:
                            self.chars_set.add(per_char)

        self.chars_set = list(sorted(self.chars_set, reverse=False))

        self.transform = transform

        print(f"Dataset Num:{len(self.image_path_list)}")
        print(f"Chars Num:{len(self.chars_set)}")

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_label(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'image': image,
            'label': label,
            'scale': scale,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        """
        convert RGB image to gray image
        """
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image.astype(np.float32)

    def load_label(self, idx):
        image_name = self.image_path_list[idx].split("/")[-1]
        label = self.image_label_dict[image_name]

        return label


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

    from tools.path import text_recognition_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm
    from simpleAICV.text_recognition.common import RandomScale, RandomGaussianBlur, RandomRotate, Normalize, Distort, Perspective, Stretch, RandomBrightness, KeepRatioResizeTextRecognitionCollater

    textrecognitiondataset = CNENTextRecognition(
        text_recognition_dataset_path,
        set_name=[
            'aistudio_baidu_street',
            # 'chinese_dataset',
            # 'synthetic_chinese_string_dataset_trainsubset0',
            # 'synthetic_chinese_string_dataset_trainsubset1',
            # 'synthetic_chinese_string_dataset_trainsubset2',
            # 'synthetic_chinese_string_dataset_trainsubset3',
            # 'meta_self_learning_car',
            # 'meta_self_learning_document_trainsubset0',
            # 'meta_self_learning_document_trainsubset1',
            # 'meta_self_learning_hand',
            # 'meta_self_learning_street',
            # 'meta_self_learning_syn',
        ],
        set_type='train',
        str_max_length=80,
        transform=transforms.Compose([
            RandomScale(scale=[0.8, 1.0], prob=0.5),
            RandomGaussianBlur(sigma=[0.5, 1.5], prob=0.5),
            RandomBrightness(brightness=[0.5, 1.5], prob=0.3),
            RandomRotate(angle=[-5, 5], prob=0.5),
            Distort(prob=0.2),
            Stretch(prob=0.2),
            Perspective(prob=0.2),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(textrecognitiondataset):
        print(per_sample['image'].shape, per_sample['image'].dtype,
              per_sample['label'])

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = KeepRatioResizeTextRecognitionCollater(resize_h=32)
    train_loader = DataLoader(textrecognitiondataset,
                              batch_size=256,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, labels = data['image'], data['label']
        print(images.shape, images.dtype, len(labels))

        if count < 5:
            count += 1
        else:
            break

    textrecognitiondataset = CNENTextRecognition(
        text_recognition_dataset_path,
        set_name=[
            'aistudio_baidu_street',
            'chinese_dataset',
            'synthetic_chinese_string_dataset_testsubset',
            'meta_self_learning_car',
            'meta_self_learning_document_testsubset',
            'meta_self_learning_hand',
            'meta_self_learning_street',
            'meta_self_learning_syn',
        ],
        set_type='test',
        str_max_length=80,
        transform=transforms.Compose([
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(textrecognitiondataset):
        print(per_sample['image'].shape, per_sample['image'].dtype,
              per_sample['label'])

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 10:
            count += 1
        else:
            break
