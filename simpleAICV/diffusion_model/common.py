import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from simpleAICV.classification.common import AverageMeter, load_state_dict


class Opencv2PIL:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = Image.fromarray(np.uint8(image))

        sample['image'] = image

        return sample


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = np.asarray(image).astype(np.float32)

        sample['image'] = image

        return sample


class TorchRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.RandomHorizontalFlip = transforms.RandomHorizontalFlip(prob)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = self.RandomHorizontalFlip(image)

        sample['image'] = image

        return sample


class TorchResize:

    def __init__(self, resize=224):
        self.Resize = transforms.Resize([int(resize), int(resize)])

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = self.Resize(image)

        sample['image'] = image

        return sample


class TorchRandomResizedCrop:

    def __init__(self, resize=224, scale=(0.08, 1.0)):
        self.RandomResizedCrop = transforms.RandomResizedCrop(int(resize),
                                                              scale=scale)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = self.RandomResizedCrop(image)

        sample['image'] = image

        return sample


class TorchMeanStdNormalize:

    def __init__(self, mean, std):
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = self.to_tensor(image)
        image = self.Normalize(image)
        # 3 H W ->H W 3
        image = image.permute(1, 2, 0)
        image = image.numpy()

        sample['image'] = image

        return sample


class Resize:

    def __init__(self, resize=224):
        self.resize = int(resize)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = cv2.resize(image, (self.resize, self.resize))

        sample['image'] = image

        return sample


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]

        sample['image'] = image

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image = sample['image']

        image = image / 255.
        image = image.astype(np.float32)

        sample['image'] = image

        return sample


class DiffusionCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]

        images = np.array(images).astype(np.float32)

        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)

        return {
            'image': images,
        }


class DiffusionWithLabelCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        labels = [s['label'] for s in data]

        images = np.array(images).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)
        labels = torch.from_numpy(labels).long()

        return {
            'image': images,
            'label': labels,
        }
