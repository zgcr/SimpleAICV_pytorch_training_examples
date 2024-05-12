import os
import copy
import cv2
import numpy as np
from PIL import Image

import collections

from torch.utils.data import Dataset

FOREGROUND_CLASSES = [
    'global_foreground_area',
    'local_foreground_area',
]

# RGB color
FOREGROUND_CLASSES_COLOR = [
    (255, 255, 255),
    (128, 128, 128),
]


class HumanMattingDataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name_list=[],
                 set_type='train',
                 kernel_size_range=[15, 25],
                 transform=None):
        assert set_type in ['train', 'val'], 'Wrong set name!'
        assert isinstance(kernel_size_range, (list, int))

        self.kernel_size_range = kernel_size_range
        self.transform = transform

        self.all_image_name_list = set()
        self.all_image_path_dict = collections.OrderedDict()
        self.all_mask_path_dict = collections.OrderedDict()
        for _, per_set_name in enumerate(set_name_list):
            per_set_image_dir = os.path.join(root_dir, per_set_name, set_type)
            per_set_mask_dir = os.path.join(root_dir, per_set_name, set_type)
            for per_image_name in os.listdir(per_set_image_dir):
                if '.jpg' in per_image_name:
                    per_image_name = per_image_name
                    per_mask_name = per_image_name.split(".")[0] + '.png'
                    per_image_path = os.path.join(per_set_image_dir,
                                                  per_image_name)
                    per_mask_path = os.path.join(per_set_mask_dir,
                                                 per_mask_name)
                    if os.path.exists(per_image_path) and os.path.exists(
                            per_mask_path):
                        self.all_image_name_list.add(per_image_name)
                        self.all_image_path_dict[
                            per_image_name] = per_image_path
                        self.all_mask_path_dict[per_image_name] = per_mask_path
        self.all_image_name_list = list(self.all_image_name_list)

        assert len(self.all_image_name_list) == len(
            self.all_image_path_dict) == len(self.all_mask_path_dict)

        print(f'Dataset Size:{len(self.all_image_name_list)}')

    def __len__(self):
        return len(self.all_image_name_list)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)

        origin_image = copy.deepcopy(image)
        origin_mask = copy.deepcopy(mask)

        size = np.array([origin_image.shape[0],
                         origin_image.shape[1]]).astype(np.float32)
        origin_size = copy.deepcopy(size)

        image_path = self.all_image_path_dict[self.all_image_name_list[idx]]
        mask_path = self.all_mask_path_dict[self.all_image_name_list[idx]]

        trimap = self.generate_trimap_from_mask(mask)
        fg_map, bg_map = self.generate_fg_bg_map_from_mask(image, mask)

        sample = {
            'image_path': image_path,
            'mask_path': mask_path,
            'origin_image': origin_image,
            'origin_mask': origin_mask,
            'origin_size': origin_size,
            'image': image,
            'mask': mask,
            'trimap': trimap,
            'fg_map': fg_map,
            'bg_map': bg_map,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(
                self.all_image_path_dict[self.all_image_name_list[idx]],
                dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        loadmask_path = self.all_mask_path_dict[self.all_image_name_list[idx]]
        mask = np.array(Image.open(loadmask_path).convert('L'), dtype=np.uint8)
        mask[mask >= 255] = 255
        mask[mask <= 0] = 0
        mask = mask / 255.

        return mask.astype(np.float32)

    def generate_trimap_from_mask(self, alpha):
        if isinstance(self.kernel_size_range, int):
            kernel_size = self.kernel_size_range
        else:
            kernel_size = np.random.randint(self.kernel_size_range[0],
                                            self.kernel_size_range[1])
        alpha_clone = alpha.copy() * 255.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        fg_and_unknown = np.array(
            np.not_equal(alpha_clone, 0).astype(np.float32))
        fg = np.array(np.equal(alpha_clone, 255).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
        erode = cv2.erode(fg, kernel, iterations=1)
        trimap = erode * 255 + (dilate - erode) * 128

        return trimap.astype(np.uint8)

    def generate_fg_bg_map_from_mask(self, image, alpha):
        expand_dim_mask = np.expand_dims(alpha.copy(),
                                         axis=2).astype(np.float32)
        fg_map = image * expand_dim_mask
        bg_map = image * (1. - expand_dim_mask)

        return fg_map.astype(np.float32), bg_map.astype(np.float32)


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

    from tools.path import human_matting_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.human_matting.common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, ResizeHumanMattingCollater

    human_matting_dataset = HumanMattingDataset(
        human_matting_dataset_path,
        set_name_list=[
            'Deep_Automatic_Portrait_Matting',
            'RealWorldPortrait636',
            'P3M10K',
        ],
        set_type='train',
        kernel_size_range=15,
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=1.0),
            # YoloStyleResize(resize=832, divisor=64, stride=64),
            Resize(resize=832),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(human_matting_dataset):
        print('1111', per_sample['origin_image'].shape,
              per_sample['origin_mask'].shape, per_sample['origin_size'],
              per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['trimap'].shape, per_sample['fg_map'].shape,
              per_sample['bg_map'].shape, per_sample['size'],
              per_sample['image_path'], per_sample['mask_path'])
        print('1111', per_sample['origin_image'].dtype,
              per_sample['origin_mask'].dtype, per_sample['origin_size'].dtype,
              per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['trimap'].dtype, per_sample['fg_map'].dtype,
              per_sample['bg_map'].dtype, per_sample['size'].dtype)

        print('1111', np.max(per_sample['mask']), np.min(per_sample['mask']),
              np.unique(per_sample['trimap']))

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # origin_image = np.ascontiguousarray(per_sample['origin_image'],
        #                                     dtype=np.uint8)
        # origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
        # origin_mask = per_sample['origin_mask'] * 255.

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # mask = per_sample['mask'] * 255.

        # trimap = per_sample['trimap']
        # fg_map = per_sample['fg_map']
        # fg_map = cv2.cvtColor(fg_map, cv2.COLOR_RGB2BGR)
        # bg_map = per_sample['bg_map']
        # bg_map = cv2.cvtColor(bg_map, cv2.COLOR_RGB2BGR)

        # cv2.imencode('.jpg', origin_image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_origin.jpg'))
        # cv2.imencode('.jpg', origin_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask_origin.jpg'))
        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image.jpg'))
        # cv2.imencode('.jpg', mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_mask.jpg'))
        # cv2.imencode('.jpg', trimap)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_trimap.jpg'))
        # cv2.imencode('.jpg', fg_map)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_fg_map.jpg'))
        # cv2.imencode('.jpg', bg_map)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_bg_map.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ResizeHumanMattingCollater(resize=832, stride=64)
    train_loader = DataLoader(human_matting_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, sizes, origin_images, origin_masks, origin_sizes = data[
            'image'], data['mask'], data['trimap'], data['fg_map'], data[
                'bg_map'], data['size'], data['origin_image'], data[
                    'origin_mask'], data['origin_size']
        print('2222', images.shape, masks.shape, trimaps.shape, fg_maps.shape,
              bg_maps.shape, sizes.shape, origin_sizes.shape, sizes,
              origin_sizes)
        print('2222', images.dtype, masks.dtype, trimaps.dtype, fg_maps.dtype,
              bg_maps.dtype, sizes.dtype, origin_sizes.dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()
        # fg_maps = fg_maps.permute(0, 2, 3, 1).cpu().numpy()
        # bg_maps = bg_maps.permute(0, 2, 3, 1).cpu().numpy()
        # masks = masks.cpu().numpy()

        # for i, (per_image, per_mask, per_trimap, per_fg_map, per_bg_map,
        #         per_origin_image, per_origin_mask) in enumerate(
        #             zip(images, masks, trimaps, fg_maps, bg_maps,
        #                 origin_images, origin_masks)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_mask = per_mask * 255.

        #     per_origin_image = np.ascontiguousarray(per_origin_image,
        #                                             dtype=np.uint8)
        #     per_origin_image = cv2.cvtColor(per_origin_image,
        #                                     cv2.COLOR_RGB2BGR)
        #     per_origin_mask = per_origin_mask * 255.

        #     per_trimap = np.ascontiguousarray(per_trimap, dtype=np.uint8)

        #     per_fg_map = np.ascontiguousarray(per_fg_map, dtype=np.uint8)
        #     per_fg_map = cv2.cvtColor(per_fg_map, cv2.COLOR_RGB2BGR)
        #     per_bg_map = np.ascontiguousarray(per_bg_map, dtype=np.uint8)
        #     per_bg_map = cv2.cvtColor(per_bg_map, cv2.COLOR_RGB2BGR)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_image.jpg'))
        #     cv2.imencode('.jpg', per_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))
        #     cv2.imencode('.jpg', per_origin_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_image_origin.jpg'))
        #     cv2.imencode('.jpg', per_origin_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_mask_origin.jpg'))

        #     cv2.imencode('.jpg', per_trimap)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_trimap.jpg'))
        #     cv2.imencode('.jpg', per_fg_map)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_fg_map.jpg'))
        #     cv2.imencode('.jpg', per_bg_map)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_bg_map.jpg'))

        if count < 2:
            count += 1
        else:
            break
