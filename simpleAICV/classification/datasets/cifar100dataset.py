import os
import pickle
import numpy as np

from torch.utils.data import Dataset


class CIFAR100Dataset(Dataset):
    '''
    CIFAR100 Dataset:https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    def __init__(self, root_dir, set_name='train', transform=None):
        assert set_name in ['train', 'test'], 'Wrong set name!'
        set_data_file_path = os.path.join(root_dir, set_name)
        set_meta_file_path = os.path.join(root_dir, 'meta')

        self.images, self.labels = [], []
        with open(set_data_file_path, 'rb') as f1:
            set_data = pickle.load(f1, encoding='latin1')
            self.images = np.array(set_data['data'])
            self.labels = np.array(set_data['fine_labels'])
        # [50000,3072]->[50000,3,32,32]->[50000,32,32,3] B H W 3
        self.images = self.images.reshape(-1, 3, 32, 32).transpose(
            (0, 2, 3, 1))

        with open(set_meta_file_path, 'rb') as f2:
            meta_data = pickle.load(f2, encoding='latin1')
            sub_class_name_list = meta_data['fine_label_names']

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

    from tools.path import CIFAR100_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.classification.common import Opencv2PIL, PIL2Opencv, TorchPad, TorchRandomHorizontalFlip, TorchRandomCrop, Normalize, ClassificationCollater

    cifar100traindataset = CIFAR100Dataset(
        root_dir=CIFAR100_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchPad(padding=4, fill=0, padding_mode='reflect'),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchRandomCrop(32),
            PIL2Opencv(),
            Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(cifar100traindataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))

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

    cifar100testdataset = CIFAR100Dataset(root_dir=CIFAR100_path,
                                          set_name='test',
                                          transform=transforms.Compose([
                                              Normalize(),
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