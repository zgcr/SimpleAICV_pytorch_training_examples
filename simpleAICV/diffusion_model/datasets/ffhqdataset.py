import os
import cv2
import json
import numpy as np

from torch.utils.data import Dataset


class FFHQDataset(Dataset):
    '''
    https://github.com/NVlabs/ffhq-dataset
    '''

    def __init__(self, root_dir, set_name='training', transform=None):
        assert set_name in ['training', 'validation'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, f'ffhq-dataset-v2.json')

        with open(self.label_dir, 'r') as f:
            self.label_info = json.load(f)

        images_name_list = set()
        for key, value in self.label_info.items():
            per_image_set_type = value['category']
            if per_image_set_type == set_name:
                per_image_name = value['image']['file_path'].split('/')[-1]
                images_name_list.add(per_image_name)
        images_name_list = list(sorted(images_name_list))

        self.image_path_list = []
        for per_image_name in images_name_list:
            per_image_path = os.path.join(self.image_dir, per_image_name)
            if os.path.exists(per_image_path):
                self.image_path_list.append(per_image_path)
        self.image_path_list = sorted(self.image_path_list)

        self.transform = transform

        print(f'Dataset Size:{len(self.image_path_list)}')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        path = self.image_path_list[idx]

        image = self.load_image(idx)

        sample = {
            'path': path,
            'image': image,
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

    from tools.path import FFHQ_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.diffusion_model.common import Resize, RandomHorizontalFlip, Normalize, DiffusionCollater

    ffhqtraindataset = FFHQDataset(
        root_dir=FFHQ_path,
        set_name='training',
        transform=transforms.Compose([
            Resize(resize=256),
            RandomHorizontalFlip(prob=0.5),
            #    Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(ffhqtraindataset):
        print(per_sample['path'], per_sample['image'].shape,
              type(per_sample['image']))

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # color = [random.randint(0, 255) for _ in range(3)]
        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DiffusionCollater()
    train_loader = DataLoader(ffhqtraindataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images = data['image']
        print(images.shape)
        print(images.dtype)
        if count < 2:
            count += 1
        else:
            break
