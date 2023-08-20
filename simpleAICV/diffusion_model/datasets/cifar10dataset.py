import os
import cv2
import pickle
import numpy as np

from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    '''
    CIFAR10 Dataset:https://www.cs.toronto.edu/~kriz/cifar.html
    '''

    def __init__(self, root_dir, set_name='train', transform=None):
        assert set_name in ['train', 'test'], 'Wrong set name!'

        if set_name == 'train':
            data_name_list = [
                'data_batch_1',
                'data_batch_2',
                'data_batch_3',
                'data_batch_4',
                'data_batch_5',
            ]
        elif set_name == 'test':
            data_name_list = [
                'test_batch',
            ]

        self.images, self.labels = [], []
        # now load the picked numpy arrays
        for per_file_name in data_name_list:
            per_file_path = os.path.join(root_dir, per_file_name)
            with open(per_file_path, 'rb') as f1:
                entry = pickle.load(f1, encoding='latin1')
                self.images.append(entry['data'])
                if 'labels' in entry:
                    self.labels.extend(entry['labels'])
                else:
                    self.labels.extend(entry['fine_labels'])

        # RGB image
        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        # convert to BHWC
        self.images = self.images.transpose((0, 2, 3, 1))

        meta_file_path = os.path.join(root_dir, 'batches.meta')
        with open(meta_file_path, 'rb') as f2:
            meta_data = pickle.load(f2, encoding='latin1')
            sub_class_name_list = meta_data['label_names']

        self.class_name_to_label = {
            sub_class_name: i
            for i, sub_class_name in enumerate(sub_class_name_list)
        }
        self.label_to_class_name = {
            i: sub_class_name
            for i, sub_class_name in enumerate(sub_class_name_list)
        }

        self.transform = transform

        print(f'Dataset Size:{self.images.shape[0]}')
        print(f'Dataset Class Num:{len(self.class_name_to_label)}')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image, label = np.array(image).astype(
            np.float32), np.array(label).astype(np.float32)

        sample = {
            'image': image,
            'label': label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


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

    from tools.path import CIFAR10_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.diffusion_model.common import Opencv2PIL, TorchResize, TorchRandomHorizontalFlip, TorchMeanStdNormalize, ClassificationCollater

    cifar100traindataset = CIFAR10Dataset(
        root_dir=CIFAR10_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

    count = 0
    for per_sample in tqdm(cifar100traindataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # per_sample['image'] = (per_sample['image'] * 0.5 + 0.5) * 255.
        # color = [random.randint(0, 255) for _ in range(3)]
        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # label = per_sample['label']
        # text = f'label:{int(label)}'
        # cv2.putText(image,
        #             text, (30, 30),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1.5,
        #             color=color,
        #             thickness=1)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    train_loader = DataLoader(cifar100traindataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, labels = data['image'], data['label']
        print(images.shape, labels.shape)
        print(images.dtype, labels.dtype)
        if count < 10:
            count += 1
        else:
            break

    cifar100testdataset = CIFAR10Dataset(root_dir=CIFAR10_path,
                                         set_name='test',
                                         transform=transforms.Compose([
                                             Opencv2PIL(),
                                             TorchMeanStdNormalize(
                                                 mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5]),
                                         ]))

    count = 0
    for per_sample in tqdm(cifar100testdataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))
        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    val_loader = DataLoader(cifar100testdataset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collater)

    count = 0
    for data in tqdm(val_loader):
        images, labels = data['image'], data['label']
        print(images.shape, labels.shape)
        print(images.dtype, labels.dtype)
        if count < 10:
            count += 1
        else:
            break