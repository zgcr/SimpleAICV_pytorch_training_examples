import collections
import os
import cv2
import copy
import collections
import json
import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset


class TextDetection(Dataset):

    def __init__(self,
                 root_dir,
                 set_name=[
                     'ICDAR2017RCTW_text_detection',
                     'ICDAR2019ART_text_detection',
                     'ICDAR2019LSVT_text_detection',
                     'ICDAR2019MLT_text_detection',
                     'ICDAR2019ReCTS_text_detection',
                 ],
                 set_type='train',
                 transform=None):
        assert set_type in ['train', 'test'], 'Wrong set name!'

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

        self.image_path_list = []
        self.image_label_dict = collections.OrderedDict()
        for per_set_image_dir_path, per_set_label_path in tqdm(
                zip(all_image_dirs_list, all_labels_path_list)):
            with open(per_set_label_path, 'r', encoding='UTF-8') as json_f:
                per_set_label = json.load(json_f)
                for key, value in tqdm(per_set_label.items()):
                    per_image_path = os.path.join(per_set_image_dir_path, key)
                    if not os.path.exists(per_image_path):
                        continue

                    per_image_label = copy.deepcopy(value)

                    for i in range(len(per_image_label)):
                        per_image_label[i]['points'] = np.array(
                            per_image_label[i]['points']).astype(np.float32)

                    self.image_path_list.append(per_image_path)
                    self.image_label_dict[key] = per_image_label
        self.image_path_list = sorted(self.image_path_list)

        self.transform = transform

        print(f"Dataset Num:{len(self.image_path_list)}")

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        path = self.image_path_list[idx]

        image = self.load_image(idx)
        annots = self.load_annots(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'path': path,
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
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        image_name = self.image_path_list[idx].split('/')[-1]
        shape = self.image_label_dict[image_name]

        return shape


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

    from tools.path import text_detection_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm
    from SimpleAICV.text_detection.common import RandomRotate, MainDirectionRandomRotate, Resize, Normalize, TextDetectionCollater, DBNetTextDetectionCollater

    textdetectiondataset = TextDetection(
        text_detection_dataset_path,
        set_name=[
            'ICDAR2017RCTW_text_detection',
            'ICDAR2019ART_text_detection',
            'ICDAR2019LSVT_text_detection',
            'ICDAR2019MLT_text_detection',
            'ICDAR2019ReCTS_text_detection',
        ],
        set_type='train',
        transform=transforms.Compose([
            RandomRotate(angle=[-30, 30], prob=0.3),
            MainDirectionRandomRotate(angle=[0, 90, 180, 270],
                                      prob=[0.7, 0.1, 0.1, 0.1]),
            Resize(resize=1024),
            #  Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(textdetectiondataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, len(per_sample['annots']),
              per_sample['annots'][0]['points'].shape, per_sample['scale'],
              per_sample['size'])

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # annots = per_sample['annots']

        # for per_annot in annots:
        #     per_box = per_annot['points']
        #     per_box = np.array(per_box, np.int32)
        #     per_box = per_box.reshape((-1, 1, 2))

        #     per_box_type = per_annot['ignore']

        #     if not per_box_type:
        #         color = (0, 255, 0)
        #     else:
        #         color = (255, 0, 0)

        #     cv2.polylines(image,
        #                   pts=[per_box],
        #                   isClosed=True,
        #                   color=color,
        #                   thickness=3)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = TextDetectionCollater(resize=1024)
    train_loader = DataLoader(textdetectiondataset,
                              batch_size=4,
                              shuffle=False,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots = data['image'], data['annots']
        print('2222', images.shape, annots[0][0]['points'].shape)

        if count < 2:
            count += 1
        else:
            break

    textdetectiondataset = TextDetection(
        text_detection_dataset_path,
        set_name=[
            'ICDAR2017RCTW_text_detection',
            'ICDAR2019ART_text_detection',
            'ICDAR2019LSVT_text_detection',
            'ICDAR2019MLT_text_detection',
            'ICDAR2019ReCTS_text_detection',
        ],
        set_type='test',
        transform=transforms.Compose([
            RandomRotate(angle=[-30, 30], prob=0.3),
            MainDirectionRandomRotate(angle=[0, 90, 180, 270],
                                      prob=[0.7, 0.1, 0.1, 0.1]),
            Resize(resize=1024),
            #  Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(textdetectiondataset):
        print('1111', per_sample['path'])
        print('1111', per_sample['image'].shape, len(per_sample['annots']),
              per_sample['annots'][0]['points'].shape, per_sample['scale'],
              per_sample['size'])

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # annots = per_sample['annots']

        # for per_annot in annots:
        #     per_box = per_annot['points']
        #     per_box = np.array(per_box, np.int32)
        #     per_box = per_box.reshape((-1, 1, 2))

        #     per_box_type = per_annot['ignore']

        #     if not per_box_type:
        #         color = (0, 255, 0)
        #     else:
        #         color = (255, 0, 0)

        #     cv2.polylines(image,
        #                   pts=[per_box],
        #                   isClosed=True,
        #                   color=color,
        #                   thickness=3)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DBNetTextDetectionCollater(resize=1024,
                                          min_box_size=3,
                                          min_max_threshold=[0.3, 0.7],
                                          shrink_ratio=0.6)
    train_loader = DataLoader(textdetectiondataset,
                              batch_size=4,
                              shuffle=False,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('2222', images.shape)

        # temp_dir = './temp3'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()

        # for i in range(images.shape[0]):
        #     per_image = images[i]
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     shape = annots['shape'][i]
        #     for per_shape in shape:
        #         per_box = per_shape['points']
        #         per_box = np.array(per_box, np.int32)
        #         per_box = per_box.reshape((-1, 1, 2))

        #         per_box_type = per_shape['ignore']

        #         if not per_box_type:
        #             color = (0, 255, 0)
        #         else:
        #             color = (255, 0, 0)

        #         cv2.polylines(per_image,
        #                       pts=[per_box],
        #                       isClosed=True,
        #                       color=color,
        #                       thickness=3)

        #     probability_mask = annots['probability_mask'][i].numpy() * 255
        #     probability_ignore_mask = annots['probability_ignore_mask'][
        #         i].numpy() * 255
        #     threshold_mask = annots['threshold_mask'][i].numpy() * 255
        #     threshold_ignore_mask = annots['threshold_ignore_mask'][i].numpy(
        #     ) * 255

        #     print("3333", per_image.shape, probability_mask.shape,
        #           probability_ignore_mask.shape, threshold_mask.shape,
        #           threshold_ignore_mask.shape)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
        #     cv2.imencode('.jpg', probability_mask)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count}_{i}_probability_mask.jpg'))
        #     cv2.imencode('.jpg', probability_ignore_mask)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count}_{i}_probability_ignore_mask.jpg'))
        #     cv2.imencode('.jpg', threshold_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_threshold_mask.jpg'))
        #     cv2.imencode('.jpg', threshold_ignore_mask)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count}_{i}_threshold_ignore_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
