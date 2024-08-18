import os
import cv2
import collections
import json

import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset

FACE_CLASSES = [
    'face',
]

FACE_CLASSES_COLOR = [
    (0, 255, 0),
]


class FaceDetectionDataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name_list=[
                     'wider_face',
                 ],
                 set_type='train',
                 transform=None):
        assert set_type in ['train', 'val']

        all_image_names = set()
        all_image_path_dict = collections.OrderedDict()
        all_image_label_dict = collections.OrderedDict()
        for per_set_name in tqdm(set_name_list):
            per_set_image_dir = os.path.join(root_dir, per_set_name, 'images',
                                             set_type)
            per_set_json_path = os.path.join(
                root_dir, per_set_name, 'annotations',
                f'{per_set_name}_{set_type}.json')
            per_set_json_dict = json.load(
                open(per_set_json_path, 'r', encoding='UTF-8'))
            for per_image_name in os.listdir(per_set_image_dir):
                per_image_path = os.path.join(per_set_image_dir,
                                              per_image_name)
                if not per_image_name in per_set_json_dict.keys(
                ) or not os.path.exists(per_image_path):
                    continue
                if len(per_set_json_dict[per_image_name]['face_box']) > 0:
                    all_image_names.add(per_image_name)
                    all_image_path_dict[per_image_name] = per_image_path
                    all_image_label_dict[per_image_name] = per_set_json_dict[
                        per_image_name]['face_box']

        self.all_image_names = list(all_image_names)
        self.all_image_path_dict = all_image_path_dict
        self.all_image_label_dict = all_image_label_dict

        assert len(self.all_image_path_dict) == len(
            self.all_image_label_dict) == len(self.all_image_names)

        self.num_classes = 1
        self.transform = transform

        print(f'Dataset Size:{len(self.all_image_names)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.all_image_names)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annots = self.load_annots(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        path = self.all_image_path_dict[self.all_image_names[idx]]

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
            np.fromfile(self.all_image_path_dict[self.all_image_names[idx]],
                        dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        annots = self.all_image_label_dict[self.all_image_names[idx]]

        targets = np.zeros((0, 5))
        if len(annots) == 0:
            return targets.astype(np.float32)

        for per_annot in annots:
            if per_annot[2] - per_annot[0] <= 1 or per_annot[3] - per_annot[
                    1] <= 1:
                continue

            target = np.zeros((1, 5))
            target[0, :4] = per_annot[0:4]
            target[0, 4] = 0
            targets = np.append(targets, target, axis=0)

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

    from tools.path import face_detection_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm
    from simpleAICV.face_detection.common import YoloStyleResize, RandomCrop, RandomTranslate, RandomHorizontalFlip, RandomVerticalFlip, RandomGaussianBlur, MainDirectionRandomRotate, Normalize, FaceDetectionCollater

    face_detection_dataset = FaceDetectionDataset(
        face_detection_dataset_path,
        set_name_list=[
            'wider_face',
            'UFDD',
        ],
        set_type='train',
        transform=transforms.Compose([
            RandomGaussianBlur(sigma=[0.5, 1.5], prob=0.3),
            MainDirectionRandomRotate(angle=[0, 90, 180, 270],
                                      prob=[0.55, 0.15, 0.15, 0.15]),
            RandomHorizontalFlip(prob=0.5),
            RandomVerticalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            YoloStyleResize(resize=960,
                            divisor=32,
                            stride=32,
                            multi_scale=False,
                            multi_scale_range=[0.8, 1.0]),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(face_detection_dataset):
        print(per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'], per_sample['path'])

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annots = per_sample['annots']

        per_image_name = per_sample['path'].split('/')[-1]

        # draw all label boxes
        for per_annot in annots:
            per_box = (per_annot[0:4]).astype(np.int32)
            class_color = FACE_CLASSES_COLOR[0]
            class_name = FACE_CLASSES

            left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                                per_box[3])

            cv2.rectangle(image,
                          left_top,
                          right_bottom,
                          color=class_color,
                          thickness=1,
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
            os.path.join(temp_dir, f'{per_image_name}'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = FaceDetectionCollater(resize=960)
    train_loader = DataLoader(face_detection_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes, paths = data['image'], data[
            'annots'], data['scale'], data['size'], data['path']
        print(images.shape, annots.shape, sizes.shape)
        print(images.dtype, annots.dtype, scales.dtype, sizes.dtype)

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        images = images.permute(0, 2, 3, 1).cpu().numpy()
        annots = annots.cpu().numpy()

        for i, (per_image, per_image_annot,
                per_image_path) in enumerate(zip(images, annots, paths)):
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

            per_image_name = per_image_path.split('/')[-1]

            # draw all label boxes
            for per_annot in per_image_annot:
                per_box = (per_annot[0:4]).astype(np.int32)
                class_color = FACE_CLASSES_COLOR[0]
                class_name = FACE_CLASSES

                left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                                    per_box[3])

                cv2.rectangle(per_image,
                              left_top,
                              right_bottom,
                              color=class_color,
                              thickness=1,
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
                os.path.join(temp_dir, f'{per_image_name}'))

        if count < 2:
            count += 1
        else:
            break
