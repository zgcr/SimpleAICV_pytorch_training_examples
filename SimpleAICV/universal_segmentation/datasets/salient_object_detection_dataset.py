import os
import cv2
import collections
import numpy as np
from PIL import Image

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset

FOREGROUND_CLASSES = [
    'foreground',
]

# RGB color
FOREGROUND_CLASSES_COLOR = [
    (0, 255, 0),
]


class SalientObjectDetectionDataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_name_list=[],
                 set_type='train',
                 transform=None):
        assert set_type in [
            'train', 'val', 'test1', 'test2', 'test3', 'test4'
        ], 'Wrong set name!'

        self.transform = transform

        self.all_image_name_list = set()
        self.all_image_path_dict = collections.OrderedDict()
        self.all_mask_path_dict = collections.OrderedDict()

        # for i, per_set_name in enumerate(set_name_list):
        #     per_set_image_dir = os.path.join(root_dir, per_set_name, set_type)
        #     per_set_mask_dir = os.path.join(root_dir, per_set_name, set_type)
        #     for per_image_name in tqdm(os.listdir(per_set_image_dir)):
        #         if '.jpg' in per_image_name:
        #             per_image_name = per_image_name
        #             per_mask_name = per_image_name.split(".")[0] + '.png'
        #             per_image_path = os.path.join(per_set_image_dir,
        #                                           per_image_name)
        #             per_mask_path = os.path.join(per_set_mask_dir,
        #                                          per_mask_name)
        #             if os.path.exists(per_image_path) and os.path.exists(
        #                     per_mask_path):
        #                 per_mask = np.array(
        #                     Image.open(per_mask_path).convert('L'),
        #                     dtype=np.uint8)
        #                 per_mask[per_mask >= 255] = 255
        #                 per_mask[per_mask <= 0] = 0
        #                 per_mask[per_mask > 0] = 1

        #                 per_mask_class_ids = list(np.unique(per_mask))
        #                 if 0 in per_mask_class_ids:
        #                     per_mask_class_ids.remove(0)

        #                 if len(per_mask_class_ids) == 0:
        #                     continue

        #                 self.all_image_name_list.add(per_image_name)
        #                 self.all_image_path_dict[
        #                     per_image_name] = per_image_path
        #                 self.all_mask_path_dict[per_image_name] = per_mask_path
        # self.all_image_name_list = sorted(list(self.all_image_name_list))

        for i, per_set_name in enumerate(set_name_list):
            per_set_image_dir = os.path.join(root_dir, per_set_name, set_type)
            per_set_mask_dir = os.path.join(root_dir, per_set_name, set_type)

            all_image_names = sorted(os.listdir(per_set_image_dir))

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.process_single_image,
                                    per_set_image_dir, per_set_mask_dir,
                                    per_image_name)
                    for per_image_name in all_image_names
                ]

                results = []
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        results.append(result)

                for result in results:
                    per_image_name, per_image_path, per_mask_path = result
                    self.all_image_name_list.add(per_image_name)
                    self.all_image_path_dict[per_image_name] = per_image_path
                    self.all_mask_path_dict[per_image_name] = per_mask_path
        self.all_image_name_list = sorted(list(self.all_image_name_list))

        assert len(self.all_image_name_list) == len(
            self.all_image_path_dict) == len(self.all_mask_path_dict)

        print(f'Dataset Size:{len(self.all_image_name_list)}')

    def process_single_image(self, image_dir, mask_dir, per_image_name):
        if '.jpg' not in per_image_name:
            return None

        per_mask_name = per_image_name.split(".")[0] + '.png'
        per_image_path = os.path.join(image_dir, per_image_name)
        per_mask_path = os.path.join(mask_dir, per_mask_name)

        if not (os.path.exists(per_image_path)
                and os.path.exists(per_mask_path)):
            return None

        per_mask = np.array(Image.open(per_mask_path).convert('L'),
                            dtype=np.uint8)
        per_mask[per_mask >= 255] = 255
        per_mask[per_mask <= 0] = 0
        per_mask[per_mask > 0] = 1

        per_mask_class_ids = list(np.unique(per_mask))
        if 0 in per_mask_class_ids:
            per_mask_class_ids.remove(0)

        if len(per_mask_class_ids) == 0:
            return None

        return per_image_name, per_image_path, per_mask_path

    def __len__(self):
        return len(self.all_image_name_list)

    def __getitem__(self, idx):
        image_path = self.all_image_path_dict[self.all_image_name_list[idx]]
        mask_path = self.all_mask_path_dict[self.all_image_name_list[idx]]

        image = self.load_image(idx)
        mask = self.load_mask(idx)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)
        origin_size = size.copy()

        sample = {
            'image_path': image_path,
            'mask_path': mask_path,
            'image': image,
            'mask': mask,
            'size': size,
            'origin_size': origin_size,
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
        # 0.9*255
        mask[mask >= 230] = 255
        # 0.1*255
        mask[mask <= 25] = 0
        mask = mask / 255.

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

    from tools.path import salient_object_detection_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from SimpleAICV.universal_segmentation.salient_object_detection_common import RandomHorizontalFlip, YoloStyleResize, Resize, Normalize, SalientObjectDetectionSegmentationTrainCollater

    salient_object_detection_dataset = SalientObjectDetectionDataset(
        salient_object_detection_dataset_path,
        set_name_list=[
            'MAGICK',
            'AM2K',
            'DIS5K',
            'HRS10K',
            'HRSOD',
            'UHRSD',
        ],
        set_type='train',
        transform=transforms.Compose([
            YoloStyleResize(resize=1024),
            # Resize(resize=1024),
            RandomHorizontalFlip(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(salient_object_detection_dataset):
        print('1111', per_sample['image_path'])
        print('1111', per_sample['mask_path'])
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['size'], per_sample['origin_size'])
        print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['size'].dtype)

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mask = per_sample['mask'] * 255.

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}.jpg'))
        cv2.imencode('.jpg', mask)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = SalientObjectDetectionSegmentationTrainCollater(resize=1024)
    train_loader = DataLoader(salient_object_detection_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, labels, sizes = data['image'], data['mask'], data[
            'label'], data['size']
        print('1111', images.shape, len(masks), len(labels), sizes.shape)

        for per_image_masks, per_image_labels in zip(masks, labels):
            print('2222', per_image_masks.shape, per_image_labels.shape)
            print('3333', per_image_labels)

        temp_dir = './temp2'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        images = images.permute(0, 2, 3, 1).cpu().numpy()

        for i, (per_image, per_image_masks,
                per_image_labels) in enumerate(zip(images, masks, labels)):
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

            per_image_masks = per_image_masks.cpu().numpy()
            per_image_labels = per_image_labels.cpu().numpy()

            per_mask = per_image_masks[0].copy()
            per_mask = per_mask * 255.

            cv2.imencode('.jpg', per_image)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
            cv2.imencode('.jpg', per_mask)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
