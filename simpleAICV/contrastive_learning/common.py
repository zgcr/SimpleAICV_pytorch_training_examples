import numpy as np

from PIL import ImageFilter, ImageOps

import torch
import torchvision.transforms as transforms


class GaussianBlur:

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image):
        if np.random.uniform(0, 1) < self.prob:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=np.random.uniform(
                    self.radius_min, self.radius_max)))

        return image


class Solarization:

    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, image):
        if np.random.uniform(0, 1) < self.prob:
            image = ImageOps.solarize(image)

        return image


class DINOAugmentation:

    def __init__(self,
                 global_resize=224,
                 local_resize=96,
                 global_crops_scale=(0.14, 1.0),
                 local_crops_scale=(0.05, 0.14),
                 local_crops_number=8,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        # first global crop
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(size=global_resize,
                                         scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        # second global crop
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(size=global_resize,
                                         scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        # transformation for the local small crops
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=local_resize,
                                         scale=local_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.local_crops_number = local_crops_number

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        image, label = sample['image'], sample['label']

        global_and_local_images = []
        global_and_local_images.append(self.global_transform1(image))
        global_and_local_images.append(self.global_transform2(image))

        for _ in range(self.local_crops_number):
            global_and_local_images.append(self.local_transform(image))

        return {
            'image': global_and_local_images,
            'label': label,
        }


class DINOPretrainCollater:

    def __init__(self, global_and_local_crop_nums=10):
        self.global_and_local_crop_nums = global_and_local_crop_nums

    def __call__(self, data):
        batch_images = [s['image'] for s in data]
        labels = [s['label'] for s in data]

        images = []
        for _ in range(self.global_and_local_crop_nums):
            images.append([])

        for i in range(len(batch_images)):
            for j in range(self.global_and_local_crop_nums):
                images[j].append(np.expand_dims(batch_images[i][j], axis=0))

        for i in range(self.global_and_local_crop_nums):
            images[i] = np.concatenate(images[i], axis=0).astype(np.float32)
            images[i] = torch.from_numpy(images[i]).float()
            # len(10) B 3 224 224, B 3 96 96

        labels = np.array(labels).astype(np.float32)
        labels = torch.from_numpy(labels).long()

        return {
            'image': images,
            'label': labels,
        }