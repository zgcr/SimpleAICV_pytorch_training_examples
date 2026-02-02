import os
import cv2
import collections
import numpy as np
from PIL import Image

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class HumanInstanceMattingDataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name_list=[],
                 set_type='train',
                 max_side=2048,
                 kernel_size_range=[15, 15],
                 transform=None):
        assert set_type in ['train', 'val'], 'Wrong set name!'

        self.max_side = max_side
        self.kernel_size_range = kernel_size_range
        self.transform = transform

        self.all_image_name_list = set()
        self.all_image_mask_path_dict = collections.OrderedDict()

        # for _, per_set_name in enumerate(set_name_list):
        #     per_set_image_dir = os.path.join(root_dir, per_set_name, set_type)
        #     per_set_mask_dir = os.path.join(root_dir, per_set_name, set_type)
        #     for per_image_dir_name in tqdm(os.listdir(per_set_image_dir)):
        #         per_image_dir_path = os.path.join(per_set_image_dir,
        #                                           per_image_dir_name)
        #         if os.path.isdir(per_image_dir_path):
        #             per_image_dir_image_num = 0
        #             per_image_dir_mask_num = 0
        #             per_image_path = None
        #             per_image_masks_path_list = []
        #             for per_file_name in sorted(
        #                     os.listdir(per_image_dir_path)):
        #                 if '.jpg' in per_file_name:
        #                     per_image_path = os.path.join(
        #                         per_image_dir_path, per_file_name)

        #                     per_image_dir_image_num += 1
        #                 elif '.png' in per_file_name:
        #                     per_mask_path = os.path.join(
        #                         per_image_dir_path, per_file_name)

        #                     per_mask = np.array(
        #                         Image.open(per_mask_path).convert('L'),
        #                         dtype=np.uint8)
        #                     if np.count_nonzero(per_mask) > 0:
        #                         per_image_masks_path_list.append(per_mask_path)
        #                         per_image_dir_mask_num += 1

        #             if per_image_dir_image_num == 1 and per_image_dir_mask_num >= 1:
        #                 self.all_image_name_list.add(per_image_dir_name)
        #                 self.all_image_mask_path_dict[per_image_dir_name] = [
        #                     per_image_path,
        #                     per_image_masks_path_list,
        #                 ]
        # self.all_image_name_list = sorted(list(self.all_image_name_list))

        for _, per_set_name in enumerate(set_name_list):
            per_set_image_dir = os.path.join(root_dir, per_set_name, set_type)
            per_set_mask_dir = os.path.join(root_dir, per_set_name, set_type)

            image_dir_names = sorted(os.listdir(per_set_image_dir))

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.process_single_image_dir, per_image_dir_name, per_set_image_dir):
                    per_image_dir_name
                    for per_image_dir_name in image_dir_names
                }

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        per_image_dir_name, per_image_path, per_image_masks_path_list = result
                        self.all_image_name_list.add(per_image_dir_name)
                        self.all_image_mask_path_dict[per_image_dir_name] = [
                            per_image_path,
                            per_image_masks_path_list,
                        ]
        self.all_image_name_list = sorted(list(self.all_image_name_list))

        assert len(self.all_image_name_list) == len(
            self.all_image_mask_path_dict)

        print(f'Dataset Size:{len(self.all_image_name_list)}')

    def process_single_image_dir(self, per_image_dir_name, per_set_image_dir):
        per_image_dir_path = os.path.join(per_set_image_dir,
                                          per_image_dir_name)
        if os.path.isdir(per_image_dir_path):
            per_image_dir_image_num = 0
            per_image_dir_mask_num = 0
            per_image_path = None
            per_image_masks_path_list = []
            for per_file_name in sorted(os.listdir(per_image_dir_path)):
                if '.jpg' in per_file_name:
                    per_image_path = os.path.join(per_image_dir_path,
                                                  per_file_name)
                    per_image_dir_image_num += 1
                elif '.png' in per_file_name:
                    per_mask_path = os.path.join(per_image_dir_path,
                                                 per_file_name)
                    per_mask = np.array(Image.open(per_mask_path).convert('L'),
                                        dtype=np.uint8)
                    if np.count_nonzero(per_mask) > 0:
                        per_image_masks_path_list.append(per_mask_path)
                        per_image_dir_mask_num += 1

            if per_image_dir_image_num == 1 and per_image_dir_mask_num >= 1:
                return (per_image_dir_name, per_image_path,
                        per_image_masks_path_list)
        return None

    def __len__(self):
        return len(self.all_image_name_list)

    def __getitem__(self, idx):
        image_path = self.all_image_mask_path_dict[
            self.all_image_name_list[idx]][0]
        masks_path = self.all_image_mask_path_dict[
            self.all_image_name_list[idx]][1]

        image = self.load_image(idx)
        masks = self.load_mask(idx)

        image_h, image_w = image.shape[0], image.shape[1]

        if max(image_h, image_w) > self.max_side:
            factor = self.max_side / max(image_h, image_w)
            resize_w, resize_h = int(image_w * float(factor) +
                                     0.5), int(image_h * float(factor) + 0.5)
            image = cv2.resize(image, (resize_w, resize_h))

            resize_masks = []
            for idx in range(masks.shape[-1]):
                per_mask = masks[:, :, idx]
                per_mask = cv2.resize(per_mask, (resize_w, resize_h))
                resize_masks.append(per_mask)
            resize_masks = np.stack(resize_masks, axis=2)
            masks = resize_masks

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)
        origin_size = size.copy()

        trimaps = self.generate_trimap_from_mask(masks)
        fg_maps, bg_maps = self.generate_fg_bg_map_from_mask(image, masks)

        sample = {
            'image_path': image_path,
            'mask_path': masks_path,
            'image': image,
            'mask': masks,
            'trimap': trimaps,
            'fg_map': fg_maps,
            'bg_map': bg_maps,
            'size': size,
            'origin_size': origin_size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.all_image_mask_path_dict[
                self.all_image_name_list[idx]][0],
                        dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        masks_path = self.all_image_mask_path_dict[
            self.all_image_name_list[idx]][1]

        masks = []
        for per_mask_path in masks_path:
            per_mask = np.array(Image.open(per_mask_path).convert('L'),
                                dtype=np.uint8)
            # 0.9*255
            per_mask[per_mask >= 230] = 255
            # 0.1*255
            per_mask[per_mask <= 25] = 0
            per_mask = per_mask / 255.

            masks.append(per_mask)
        masks = np.stack(masks, axis=2)

        return masks.astype(np.float32)

    def generate_trimap_from_mask(self, alpha):
        alpha_h, alpha_w, object_num = alpha.shape[0], alpha.shape[
            1], alpha.shape[2]
        long_size_length = max(alpha_h, alpha_w)
        side_scale = long_size_length / self.max_side

        if self.kernel_size_range[0] == self.kernel_size_range[1]:
            kernel_size = int(self.kernel_size_range[0] * side_scale)
        else:
            kernel_size = int(
                np.random.randint(self.kernel_size_range[0],
                                  self.kernel_size_range[1]) * side_scale)
        kernel_size = max(3, kernel_size)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))

        trimaps = []
        for idx in range(object_num):
            per_alpha = alpha[:, :, idx].copy() * 255.
            per_fg_and_unknown = np.array(
                np.not_equal(per_alpha, 0).astype(np.float32))
            per_fg = np.array(np.equal(per_alpha, 255).astype(np.float32))
            per_dilate = cv2.dilate(per_fg_and_unknown, kernel, iterations=1)
            per_erode = cv2.erode(per_fg, kernel, iterations=1)
            per_trimap = per_erode * 255 + (per_dilate - per_erode) * 128
            trimaps.append(per_trimap)

        trimaps = np.stack(trimaps, axis=2)

        return trimaps.astype(np.uint8)

    def generate_fg_bg_map_from_mask(self, image, alpha):
        # (h,w,1,n)
        expand_dim_mask = np.expand_dims(alpha.copy(),
                                         axis=2).astype(np.float32)
        # (h,w,3,1)
        expand_dim_image = np.expand_dims(image, axis=3)
        # (h,w,3,n)
        fg_maps = expand_dim_image * expand_dim_mask
        bg_maps = expand_dim_image * (1. - expand_dim_mask)

        return fg_maps.astype(np.float32), bg_maps.astype(np.float32)


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

    from tools.path import human_instance_matting_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.human_instance_matting_common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, HumanInstanceMattingTrainCollater

    human_instance_matting_dataset = HumanInstanceMattingDataset(
        human_instance_matting_path,
        set_name_list=[
            'HIM2K',
            'I-HIM50K',
        ],
        set_type='train',
        max_side=2048,
        kernel_size_range=[15, 15],
        transform=transforms.Compose([
            # YoloStyleResize(resize=1024),
            Resize(resize=1024),
            RandomHorizontalFlip(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(human_instance_matting_dataset):
        print('1111', per_sample['image_path'])
        print('1111', per_sample['mask_path'])
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['trimap'].shape, per_sample['fg_map'].shape,
              per_sample['bg_map'].shape, per_sample['size'],
              per_sample['origin_size'])
        print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['trimap'].dtype, per_sample['fg_map'].dtype,
              per_sample['bg_map'].dtype, per_sample['size'].dtype)

        print('1111', np.max(per_sample['mask']), np.min(per_sample['mask']),
              np.unique(per_sample['trimap']))

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count:05d}_image.jpg'))

        # masks = per_sample['mask'] * 255.
        # trimaps = per_sample['trimap']
        # fg_maps = per_sample['fg_map']
        # bg_maps = per_sample['bg_map']

        # object_nums = masks.shape[-1]
        # for per_object_idx in range(object_nums):
        #     per_mask = masks[:, :, per_object_idx]
        #     per_trimap = trimaps[:, :, per_object_idx]
        #     per_fg_map = fg_maps[:, :, :, per_object_idx]
        #     per_bg_map = bg_maps[:, :, :, per_object_idx]

        #     per_fg_map = cv2.cvtColor(per_fg_map, cv2.COLOR_RGB2BGR)
        #     per_bg_map = cv2.cvtColor(per_bg_map, cv2.COLOR_RGB2BGR)

        #     cv2.imencode('.jpg', per_mask)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count:05d}_mask_{per_object_idx:03d}.jpg'))
        #     cv2.imencode('.jpg', per_trimap)[1].tofile(
        #         os.path.join(
        #             temp_dir,
        #             f'idx_{count:05d}_trimap_{per_object_idx:03d}.jpg'))
        #     cv2.imencode('.jpg', per_fg_map)[1].tofile(
        #         os.path.join(
        #             temp_dir,
        #             f'idx_{count:05d}_fg_map_{per_object_idx:03d}.jpg'))
        #     cv2.imencode('.jpg', per_bg_map)[1].tofile(
        #         os.path.join(
        #             temp_dir,
        #             f'idx_{count:05d}_bg_map_{per_object_idx:03d}.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = HumanInstanceMattingTrainCollater(resize=1024)
    train_loader = DataLoader(human_instance_matting_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, trimaps, fg_maps, bg_maps, labels, sizes = data[
            'image'], data['mask'], data['trimap'], data['fg_map'], data[
                'bg_map'], data['label'], data['size']
        print('1111', images.shape, len(masks), len(trimaps), len(fg_maps),
              len(bg_maps), len(labels), sizes.shape)

        for per_image_masks, per_image_trimaps, per_image_fg_maps, per_image_bg_maps, per_image_labels in zip(
                masks, trimaps, fg_maps, bg_maps, labels):
            print('2222', per_image_masks.shape, per_image_trimaps.shape,
                  per_image_fg_maps.shape, per_image_bg_maps.shape,
                  per_image_labels.shape)
            print('3333', per_image_labels)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # images = images.permute(0, 2, 3, 1).cpu().numpy()

        # for i, (per_image, per_image_masks, per_image_trimaps,
        #         per_image_fg_maps, per_image_bg_maps,
        #         per_image_labels) in enumerate(
        #             zip(images, masks, trimaps, fg_maps, bg_maps, labels)):
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     cv2.imencode('.jpg', per_image)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count:05d}_{i:03d}_image.jpg'))

        #     per_image_masks = per_image_masks.cpu().numpy() * 255.
        #     per_image_trimaps = per_image_trimaps.cpu().numpy()
        #     per_image_fg_maps = per_image_fg_maps.cpu().numpy()
        #     per_image_bg_maps = per_image_bg_maps.cpu().numpy()
        #     per_image_labels = per_image_labels.cpu().numpy()

        #     per_image_object_nums = per_image_masks.shape[0]

        #     for per_object_idx in range(per_image_object_nums):
        #         per_object_mask = per_image_masks[per_object_idx]
        #         per_object_trimap = per_image_trimaps[per_object_idx]
        #         per_object_fg_map = per_image_fg_maps[per_object_idx]
        #         per_object_bg_map = per_image_bg_maps[per_object_idx]

        #         per_object_trimap = np.ascontiguousarray(per_object_trimap,
        #                                                  dtype=np.uint8)

        #         per_object_fg_map = per_object_fg_map.transpose(1, 2, 0)
        #         per_object_fg_map = np.ascontiguousarray(per_object_fg_map,
        #                                                  dtype=np.uint8)
        #         per_object_fg_map = cv2.cvtColor(per_object_fg_map,
        #                                          cv2.COLOR_RGB2BGR)

        #         per_object_bg_map = per_object_bg_map.transpose(1, 2, 0)
        #         per_object_bg_map = np.ascontiguousarray(per_object_bg_map,
        #                                                  dtype=np.uint8)
        #         per_object_bg_map = cv2.cvtColor(per_object_bg_map,
        #                                          cv2.COLOR_RGB2BGR)

        #         cv2.imencode('.jpg', per_object_mask)[1].tofile(
        #             os.path.join(
        #                 temp_dir,
        #                 f'idx_{count:05d}_{i:03d}_mask_{per_object_idx:03d}.jpg'
        #             ))

        #         cv2.imencode('.jpg', per_object_trimap)[1].tofile(
        #             os.path.join(
        #                 temp_dir,
        #                 f'idx_{count:05d}_{i:03d}_trimap_{per_object_idx:03d}.jpg'
        #             ))
        #         cv2.imencode('.jpg', per_object_fg_map)[1].tofile(
        #             os.path.join(
        #                 temp_dir,
        #                 f'idx_{count:05d}_{i:03d}_fg_map_{per_object_idx:03d}.jpg'
        #             ))
        #         cv2.imencode('.jpg', per_object_bg_map)[1].tofile(
        #             os.path.join(
        #                 temp_dir,
        #                 f'idx_{count:05d}_{i:03d}_bg_map_{per_object_idx:03d}.jpg'
        #             ))

        if count < 2:
            count += 1
        else:
            break
