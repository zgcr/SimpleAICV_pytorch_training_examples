import copy
import numpy as np

import torch
import torchvision.transforms as transforms


class MAESelfSupervisedPretrainCollater:

    def __init__(self, image_size=224, patch_size=16, norm_label=True):
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.patch_nums = image_size // patch_size
        self.norm_label = norm_label

    def __call__(self, data):
        images = [s['image'] for s in data]

        images = np.array(images, dtype=np.float32)

        images = torch.from_numpy(images)
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)

        unmasked_labels = copy.deepcopy(images)
        unmasked_labels = unmasked_labels.reshape(unmasked_labels.shape[0], 3,
                                                  self.patch_nums,
                                                  self.patch_size,
                                                  self.patch_nums,
                                                  self.patch_size)
        unmasked_labels = torch.einsum('nchpwq->nhwpqc', unmasked_labels)
        unmasked_labels = unmasked_labels.reshape(
            shape=(unmasked_labels.shape[0], self.patch_nums * self.patch_nums,
                   self.patch_size * self.patch_size * 3))

        if self.norm_label:
            mean = unmasked_labels.mean(dim=-1, keepdim=True)
            var = unmasked_labels.var(dim=-1, keepdim=True)
            unmasked_labels = (unmasked_labels - mean) / (var + 1e-4)**0.5

        return {
            'image': images,
            'label': unmasked_labels,
        }