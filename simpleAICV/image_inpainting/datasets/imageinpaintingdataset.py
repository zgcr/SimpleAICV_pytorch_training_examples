import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class ImageInpaintingDataset(Dataset):

    def __init__(self,
                 image_root_dir,
                 mask_root_dir,
                 image_set_name=None,
                 mask_set_name=None,
                 image_transform=None,
                 mask_transform=None,
                 mask_choice='random'):
        assert mask_choice in ['random', 'inorder']
        # if image_set_name is None,read all images in image_root_dir
        # if mask_set_name is None,read all masks in mask_root_dir

        all_image_dir = os.path.join(
            image_root_dir,
            image_set_name) if image_set_name is not None else image_root_dir
        self.image_path_list = []
        for root, folders, files in os.walk(all_image_dir):
            for file_name in files:
                if '.jpg' in file_name:
                    per_image_path = os.path.join(root, file_name)
                    self.image_path_list.append(per_image_path)
        self.image_path_list = sorted(self.image_path_list)

        mask_dir = os.path.join(
            mask_root_dir,
            mask_set_name) if mask_set_name is not None else mask_root_dir
        self.mask_path_list = []
        for root, folders, files in os.walk(mask_dir):
            for file_name in files:
                if '.png' in file_name:
                    per_mask_path = os.path.join(root, file_name)
                    self.mask_path_list.append(per_mask_path)
        self.mask_path_list = sorted(self.mask_path_list)

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_choice = mask_choice

        print(f'Image Dataset Size:{len(self.image_path_list)}')
        print(f'Mask Dataset Size:{len(self.mask_path_list)}')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        sample = {
            'image': image,
            'mask': mask,
        }

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        if self.mask_choice == 'random':
            idx = np.random.randint(0, len(self.mask_path_list))
        elif self.mask_choice == 'inorder':
            idx = idx % len(self.mask_path_list)

        mask = cv2.imdecode(
            np.fromfile(self.mask_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE)

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

    from tools.path import CelebAHQ_path, NVIDIA_Irregular_Mask_Dataset_test_mask_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.image_inpainting.common import Opencv2PIL, TorchResize, TorchRandomResizedCrop, TorchRandomHorizontalFlip, TorchColorJitter, TorchRandomRotation, TorchToTensor, ScaleToRange, ImageInpaintingCollater

    imageinpaintingtraindataset = ImageInpaintingDataset(
        image_root_dir=CelebAHQ_path,
        mask_root_dir=NVIDIA_Irregular_Mask_Dataset_test_mask_path,
        image_set_name='train',
        mask_set_name=None,
        image_transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=512),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchColorJitter(0.05, 0.05, 0.05, 0.05),
            TorchToTensor(),
            ScaleToRange(),
        ]),
        mask_transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=512,
                        interpolation=transforms.InterpolationMode.NEAREST),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchRandomRotation(
                degrees=(0, 45),
                interpolation=transforms.InterpolationMode.NEAREST),
            TorchToTensor(),
        ]),
        mask_choice='random')

    count = 0
    for per_sample in tqdm(imageinpaintingtraindataset):
        print(per_sample['image'].shape, per_sample['mask'].shape,
              type(per_sample['image']), type(per_sample['mask']))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = (per_sample['image'] + 1.) / 2. * 255
        # image = image.transpose(1, 2, 0).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        # mask = per_sample['mask'] * 255
        # mask = mask.squeeze(axis=0).astype(np.uint8)
        # cv2.imencode('.png', mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask.png'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ImageInpaintingCollater()
    train_loader = DataLoader(imageinpaintingtraindataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks = data['image'], data['mask']
        print(images.shape, masks.shape)
        print(images.dtype, masks.dtype)
        if count < 10:
            count += 1
        else:
            break

    imageinpaintingtestdataset = ImageInpaintingDataset(
        image_root_dir=CelebAHQ_path,
        mask_root_dir=NVIDIA_Irregular_Mask_Dataset_test_mask_path,
        image_set_name='val',
        mask_set_name='0.01-0.1',
        image_transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=512),
            TorchToTensor(),
            ScaleToRange(),
        ]),
        mask_transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=512,
                        interpolation=transforms.InterpolationMode.NEAREST),
            TorchToTensor(),
        ]),
        mask_choice='inorder')

    count = 0
    for per_sample in tqdm(imageinpaintingtestdataset):
        print(per_sample['image'].shape, per_sample['mask'].shape,
              type(per_sample['image']), type(per_sample['mask']))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = (per_sample['image'] + 1.) / 2. * 255
        # image = image.transpose(1, 2, 0).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        # mask = per_sample['mask'] * 255
        # mask = mask.squeeze(axis=0).astype(np.uint8)
        # cv2.imencode('.png', mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask.png'))

        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ImageInpaintingCollater()
    train_loader = DataLoader(imageinpaintingtestdataset,
                              batch_size=8,
                              shuffle=False,
                              num_workers=1,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks = data['image'], data['mask']
        print(images.shape, masks.shape)
        print(images.dtype, masks.dtype)
        if count < 5:
            count += 1
        else:
            break