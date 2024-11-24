import os
import copy
import collections
import cv2
import json
import math
import numpy as np
import random

from tqdm import tqdm
from pycocotools import mask as mask_utils

from torch.utils.data import Dataset


class SAMSegmentationDataset(Dataset):

    def __init__(
            self,
            root_dir,
            set_name=[
                'sa_000020',
                'sa_000021',
                'sa_000022',
                'sa_000023',
                'sa_000024',
                'sa_000025',
                'sa_000026',
                'sa_000027',
                'sa_000028',
                'sa_000029',
            ],
            set_type='train',
            per_set_image_choose_max_num={
                'sa_000020': 1000000,
                'sa_000021': 1000000,
                'sa_000022': 1000000,
                'sa_000023': 1000000,
                'sa_000024': 1000000,
                'sa_000025': 1000000,
                'sa_000026': 1000000,
                'sa_000027': 1000000,
                'sa_000028': 1000000,
                'sa_000029': 1000000,
            },
            per_image_mask_chosse_max_num=16,
            positive_points_num=9,
            negative_points_num=9,
            area_filter_ratio=0.0001,
            box_noise_wh_ratio=0.1,
            mask_noise_area_ratio=0.04,
            transform=None):

        self.all_set_image_path_list = collections.OrderedDict()
        self.all_set_image_nums = collections.OrderedDict()
        for per_set_name in tqdm(set_name):
            per_set_dir = os.path.join(root_dir, per_set_name, set_type)
            self.all_set_image_nums[per_set_name] = 0
            self.all_set_image_path_list[per_set_name] = []
            for root, folders, files in os.walk(per_set_dir):
                for file_name in files:
                    if '.jpg' in file_name:
                        per_image_path = os.path.join(root, file_name)
                        json_name = file_name.split('.')[0] + '.json'
                        per_json_path = os.path.join(root, json_name)

                        if os.path.exists(per_image_path) and os.path.exists(
                                per_json_path):
                            self.all_set_image_nums[per_set_name] += 1
                            self.all_set_image_path_list[per_set_name].append(
                                [file_name, per_image_path, per_json_path])

        for key, value in self.all_set_image_path_list.items():
            print(f'set_name:{key},origin_image_num:{len(value)}')

        self.image_path_list = []
        for per_set_name, per_set_image_path_list in self.all_set_image_path_list.items(
        ):
            per_set_image_path_list = sorted(per_set_image_path_list)
            per_set_image_max_num = per_set_image_choose_max_num[per_set_name]
            if len(per_set_image_path_list) > per_set_image_max_num:
                per_set_image_path_list = per_set_image_path_list[
                    0:per_set_image_max_num]

            print(
                f'set_name:{per_set_name},choose_image_num:{len(per_set_image_path_list)}'
            )

            for per_image_info in per_set_image_path_list:
                self.image_path_list.append(per_image_info)
        self.image_path_list = sorted(self.image_path_list)

        # get all mask for all images
        self.all_image_mask_path_list = []
        for per_image_name, per_image_path, per_json_path in tqdm(
                self.image_path_list):
            with open(per_json_path, encoding='utf-8') as f:
                per_image_json_data = json.load(f)
                per_image_annotation = per_image_json_data['annotations']

                per_image_h, per_image_w = per_image_json_data['image'][
                    'height'], per_image_json_data['image']['width']

                per_image_annotation_num = len(per_image_annotation)
                if per_image_annotation_num > per_image_mask_chosse_max_num:
                    per_image_annotation = per_image_annotation[
                        0:per_image_mask_chosse_max_num]

                for mask_list_idx, per_annot in enumerate(
                        per_image_annotation):

                    # bbox format:[x_min, y_min, w, h]
                    per_box = per_annot['bbox']

                    if math.ceil(per_box[2] * per_box[3]) < 25 or math.ceil(
                            per_box[2]) < 5 or math.ceil(per_box[3]) < 5:
                        continue

                    x_min = math.ceil(max(per_box[0], 0))
                    y_min = math.ceil(max(per_box[1], 0))
                    x_max = math.ceil(min(per_box[0] + per_box[2],
                                          per_image_w))
                    y_max = math.ceil(min(per_box[1] + per_box[3],
                                          per_image_h))
                    box_w = math.ceil(x_max - x_min)
                    box_h = math.ceil(y_max - y_min)

                    if box_w * box_h < 25 or box_w < 5 or box_h < 5:
                        continue

                    if per_annot['area'] / float(
                            per_image_h * per_image_w
                    ) < area_filter_ratio or per_annot['area'] / float(
                            per_image_h * per_image_w) > 0.9:
                        continue

                    self.all_image_mask_path_list.append([
                        per_image_name,
                        mask_list_idx,
                        per_image_path,
                        per_json_path,
                        per_image_h,
                        per_image_w,
                    ])

        self.positive_points_num = positive_points_num
        self.negative_points_num = negative_points_num
        self.area_filter_ratio = area_filter_ratio
        self.box_noise_wh_ratio = box_noise_wh_ratio
        self.mask_noise_area_ratio = mask_noise_area_ratio
        self.transform = transform

        print(f'Image Size:{len(self.image_path_list)}')
        print(f'Dataset Size:{len(self.all_image_mask_path_list)}')

    def __len__(self):
        return len(self.all_image_mask_path_list)

    def __getitem__(self, idx):
        _, _, image_path, json_path, _, _ = self.all_image_mask_path_list[idx]

        image = self.load_image(idx)
        # image_mask:[0,1]二值化mask
        image_box, image_mask = self.load_mask(idx)

        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        image_mask_all_points_coords = np.argwhere(image_mask)
        image_mask_all_points_num = len(image_mask_all_points_coords)

        if image_mask_all_points_num < self.positive_points_num:
            positive_prompt_point = np.zeros((0, 3), dtype=np.float32)
        else:
            if self.positive_points_num > 0:
                positive_prompt_point = []
                positive_prompt_points_index = np.random.choice(
                    image_mask_all_points_num, self.positive_points_num)
                for positive_point_idx in positive_prompt_points_index:
                    positive_prompt_point.append([
                        image_mask_all_points_coords[positive_point_idx][1],
                        image_mask_all_points_coords[positive_point_idx][0],
                        1,
                    ])
                positive_prompt_point = np.array(positive_prompt_point,
                                                 dtype=np.float32)
            else:
                positive_prompt_point = np.zeros((0, 3), dtype=np.float32)

        image_not_mask_all_points_coords = np.argwhere(1. - image_mask)
        image_not_mask_all_points_num = len(image_not_mask_all_points_coords)

        if image_not_mask_all_points_num < self.negative_points_num:
            negative_prompt_point = np.zeros((0, 3), dtype=np.float32)
        else:
            if self.negative_points_num > 0:
                negative_prompt_point = []
                negative_prompt_points_index = np.random.choice(
                    image_not_mask_all_points_num, self.negative_points_num)
                for negative_point_idx in negative_prompt_points_index:
                    negative_prompt_point.append([
                        image_not_mask_all_points_coords[negative_point_idx]
                        [1],
                        image_not_mask_all_points_coords[negative_point_idx]
                        [0],
                        0,
                    ])
                negative_prompt_point = np.array(negative_prompt_point,
                                                 dtype=np.float32)
            else:
                negative_prompt_point = np.zeros((0, 3), dtype=np.float32)

        prompt_box = copy.deepcopy(image_box)
        if self.box_noise_wh_ratio > 0:
            prompt_box = self.noise_bbox(prompt_box, size)

        prompt_mask = copy.deepcopy(image_mask)
        prompt_mask = self.noise_mask(prompt_mask, idx)

        sample = {
            'image_path': image_path,
            'json_path': json_path,
            'image': image,
            'box': image_box,
            'mask': image_mask,
            'size': size,
            'positive_prompt_point': positive_prompt_point,
            'negative_prompt_point': negative_prompt_point,
            'prompt_box': prompt_box,
            'prompt_mask': prompt_mask,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        _, _, per_image_path, _, _, _ = self.all_image_mask_path_list[idx]
        image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        _, mask_list_idx, _, per_json_path, _, _ = self.all_image_mask_path_list[
            idx]
        with open(per_json_path, encoding='utf-8') as f:
            per_image_json_data = json.load(f)
            per_image_annotation = per_image_json_data['annotations']
            per_annot = per_image_annotation[mask_list_idx]

            per_image_h, per_image_w = per_image_json_data['image'][
                'height'], per_image_json_data['image']['width']

            target_box = np.array(per_annot['bbox'])

            # transform bbox targets from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
            x_min = math.ceil(max(target_box[0], 0))
            y_min = math.ceil(max(target_box[1], 0))
            x_max = math.ceil(min(target_box[0] + target_box[2], per_image_w))
            y_max = math.ceil(min(target_box[1] + target_box[3], per_image_h))

            target_box = np.array([x_min, y_min, x_max, y_max])

            target_mask = mask_utils.decode(per_annot['segmentation'])

        target_mask[target_mask > 0] = 1

        return target_box.astype(np.float32), target_mask.astype(np.float32)

    def noise_bbox(self, properties_bbox, mask_np_shape):
        w, h = properties_bbox[2] - properties_bbox[0], properties_bbox[
            3] - properties_bbox[1]

        if h / mask_np_shape[0] <= 0.01 or w / mask_np_shape[1] <= 0.01:
            return properties_bbox.astype(np.float32)

        if h <= 10 or w <= 10:
            return properties_bbox.astype(np.float32)

        noise_x, noise_y = w * self.box_noise_wh_ratio, h * self.box_noise_wh_ratio

        if noise_x <= 1 or noise_y <= 1:
            return properties_bbox.astype(np.float32)

        x0 = properties_bbox[0] + max(
            min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
        y0 = properties_bbox[1] + max(
            min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)
        x1 = properties_bbox[2] + max(
            min(np.random.randint(-noise_x, noise_x), w / 2), -w / 2)
        y1 = properties_bbox[3] + max(
            min(np.random.randint(-noise_y, noise_y), h / 2), -h / 2)
        x0 = x0 if x0 >= 0 else 0
        y0 = y0 if y0 >= 0 else 0
        x1 = x1 if x1 <= mask_np_shape[1] else mask_np_shape[1]  # 避免越界
        y1 = y1 if y1 <= mask_np_shape[0] else mask_np_shape[0]
        post_properties_bbox = np.array([x0, y0, x1, y1])
        post_properties_bbox = np.where(post_properties_bbox > 0,
                                        post_properties_bbox, 0)
        if x0 >= x1 or y0 >= y1:
            return properties_bbox.astype(np.float32)
        else:
            return post_properties_bbox.astype(np.float32)

    def noise_mask(self, properties_mask, idx):
        _, _, _, _, image_h, image_w = self.all_image_mask_path_list[idx]

        origin_mask_area = np.sum(properties_mask)
        reduce_mask_area = origin_mask_area * self.mask_noise_area_ratio

        if reduce_mask_area <= 100:
            return properties_mask.astype(np.float32)

        max_kernel = np.sqrt(reduce_mask_area) / 2.
        if int(max_kernel) > 1:
            kernel = np.random.randint(1, max_kernel)
            erode_kernel = np.ones((kernel, kernel), np.uint8)
            post_properties_mask = cv2.erode(properties_mask,
                                             erode_kernel,
                                             iterations=1)
        else:
            post_properties_mask = properties_mask

        if np.sum(post_properties_mask) / float(
                image_h * image_w) > self.area_filter_ratio:
            return post_properties_mask.astype(np.float32)
        else:
            return properties_mask.astype(np.float32)


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

    from tools.path import interactive_segmentation_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater

    samdataset = SAMSegmentationDataset(
        interactive_segmentation_dataset_path,
        set_name=[
            'DIS5K_seg',
            'HRS10K_seg',
            'HRSOD_seg',
            'UHRSD_seg',
            'Deep_Automatic_Portrait_Matting_seg',
            'RealWorldPortrait636_seg',
            'P3M10K_seg',
            # 'sa_000020',
            # 'sa_000021',
            # 'sa_000022',
            # 'sa_000023',
            # 'sa_000024',
            # 'sa_000025',
            # 'sa_000026',
            # 'sa_000027',
            # 'sa_000028',
            # 'sa_000029',
        ],
        set_type='train',
        per_set_image_choose_max_num={
            'DIS5K_seg': 10000000,
            'HRS10K_seg': 10000000,
            'HRSOD_seg': 10000000,
            'UHRSD_seg': 10000000,
            'Deep_Automatic_Portrait_Matting_seg': 10000000,
            'RealWorldPortrait636_seg': 10000000,
            'P3M10K_seg': 10000000,
            # 'sa_000020': 1000000,
            # 'sa_000021': 1000000,
            # 'sa_000022': 1000000,
            # 'sa_000023': 1000000,
            # 'sa_000024': 1000000,
            # 'sa_000025': 1000000,
            # 'sa_000026': 1000000,
            # 'sa_000027': 1000000,
            # 'sa_000028': 1000000,
            # 'sa_000029': 1000000,
        },
        per_image_mask_chosse_max_num=16,
        positive_points_num=9,
        negative_points_num=9,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            SamResize(resize=1024),
            SamRandomHorizontalFlip(prob=0.5),
            #   SamNormalize(
            #       mean=[123.675, 116.28, 103.53],
            #       std=[58.395, 57.12, 57.375]),
        ]))

    count = 0
    for per_sample in tqdm(samdataset):
        print('1111', per_sample['image_path'])
        print('1111', per_sample['json_path'])
        print('1111', per_sample['image'].shape, per_sample['box'].shape,
              per_sample['mask'].shape, per_sample['size'],
              per_sample['positive_prompt_point'].shape,
              per_sample['negative_prompt_point'].shape,
              per_sample['prompt_box'].shape, per_sample['prompt_mask'].shape)
        print('2222', per_sample['image'].dtype, per_sample['box'].dtype,
              per_sample['mask'].dtype, per_sample['size'].dtype,
              per_sample['positive_prompt_point'].dtype,
              per_sample['negative_prompt_point'].dtype,
              per_sample['prompt_box'].dtype, per_sample['prompt_mask'].dtype)
        print('3333', per_sample['box'], per_sample['size'],
              per_sample['positive_prompt_point'][0],
              per_sample['negative_prompt_point'][0], per_sample['prompt_box'])

        # temp_dir = f'./temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # box = per_sample['box']
        # mask = per_sample['mask']

        # image_for_box = copy.deepcopy(image)
        # per_image_box = (box[0:4]).astype(np.int32)
        # box_color = [int(np.random.choice(range(256))) for _ in range(3)]
        # left_top, right_bottom = (per_image_box[0],
        #                           per_image_box[1]), (per_image_box[2],
        #                                               per_image_box[3])
        # cv2.rectangle(image_for_box,
        #               left_top,
        #               right_bottom,
        #               color=box_color,
        #               thickness=2,
        #               lineType=cv2.LINE_AA)
        # text = 'object_box'
        # text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        # fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
        #                      left_top[1] - text_size[1] - 3)
        # cv2.rectangle(image_for_box,
        #               left_top,
        #               fill_right_bottom,
        #               color=box_color,
        #               thickness=-1,
        #               lineType=cv2.LINE_AA)
        # cv2.putText(image_for_box,
        #             text, (left_top[0], left_top[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             color=(0, 0, 0),
        #             thickness=1,
        #             lineType=cv2.LINE_AA)

        # image_for_mask = copy.deepcopy(image).astype('uint8')
        # mask = mask.astype('uint8')
        # per_image_mask = np.zeros(
        #     (image_for_mask.shape[0], image_for_mask.shape[1], 3))
        # per_image_contours = []
        # mask = np.nonzero(mask == 1.)
        # mask_color = [int(np.random.choice(range(256))) for _ in range(3)]
        # per_image_mask[mask[0], mask[1]] = mask_color
        # new_per_image_mask = np.zeros(
        #     (image_for_mask.shape[0], image_for_mask.shape[1]))
        # new_per_image_mask[mask[0], mask[1]] = 255
        # contours, _ = cv2.findContours(new_per_image_mask.astype('uint8'),
        #                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # per_image_contours.append(contours)
        # per_image_mask = per_image_mask.astype('uint8')
        # per_image_mask = cv2.cvtColor(per_image_mask, cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_image_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_image_mask[all_classes_mask[0],
        #                    all_classes_mask[1]] = cv2.addWeighted(
        #                        image_for_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 0.5,
        #                        per_image_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_image_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_image_mask[no_class_mask[0],
        #                    no_class_mask[1]] = image_for_mask[no_class_mask[0],
        #                                                       no_class_mask[1]]
        # for contours in per_image_contours:
        #     cv2.drawContours(per_image_mask, contours, -1, (255, 255, 255), 2)

        # cv2.imencode('.jpg', image_for_box)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_box.jpg'))
        # cv2.imencode('.jpg', per_image_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_mask.jpg'))

        # temp_dir = f'./temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # positive_prompt_point = per_sample['positive_prompt_point']
        # negative_prompt_point = per_sample['negative_prompt_point']
        # prompt_box = per_sample['prompt_box']
        # prompt_mask = per_sample['prompt_mask']
        # positive_prompt_point_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # negative_prompt_point_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # prompt_box_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]
        # prompt_mask_color = [
        #     int(np.random.choice(range(256))) for _ in range(3)
        # ]

        # image_for_prompt_box = copy.deepcopy(image)

        # for per_point in positive_prompt_point:
        #     cv2.circle(image_for_prompt_box,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                positive_prompt_point_color, -1)

        # for per_point in negative_prompt_point:
        #     cv2.circle(image_for_prompt_box,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                negative_prompt_point_color, -1)

        # per_image_prompt_box = (prompt_box[0:4]).astype(np.int32)
        # left_top, right_bottom = (per_image_prompt_box[0],
        #                           per_image_prompt_box[1]), (
        #                               per_image_prompt_box[2],
        #                               per_image_prompt_box[3])
        # cv2.rectangle(image_for_prompt_box,
        #               left_top,
        #               right_bottom,
        #               color=prompt_box_color,
        #               thickness=2,
        #               lineType=cv2.LINE_AA)
        # text = f'prompt_box'
        # text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        # fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
        #                      left_top[1] - text_size[1] - 3)
        # cv2.rectangle(image_for_prompt_box,
        #               left_top,
        #               fill_right_bottom,
        #               color=prompt_box_color,
        #               thickness=-1,
        #               lineType=cv2.LINE_AA)
        # cv2.putText(image_for_prompt_box,
        #             text, (left_top[0], left_top[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             color=(0, 0, 0),
        #             thickness=1,
        #             lineType=cv2.LINE_AA)

        # image_for_prompt_mask = copy.deepcopy(image).astype('uint8')

        # for per_point in positive_prompt_point:
        #     cv2.circle(image_for_prompt_mask,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                positive_prompt_point_color, -1)

        # for per_point in negative_prompt_point:
        #     cv2.circle(image_for_prompt_mask,
        #                (int(per_point[0]), int(per_point[1])), 10,
        #                negative_prompt_point_color, -1)

        # prompt_mask = prompt_mask.astype('uint8')
        # per_image_prompt_mask = np.zeros((image_for_prompt_mask.shape[0],
        #                                   image_for_prompt_mask.shape[1], 3))
        # per_image_prompt_contours = []
        # prompt_mask = np.nonzero(prompt_mask == 1.)
        # if len(prompt_mask[0]) > 0:
        #     per_image_prompt_mask[prompt_mask[0],
        #                           prompt_mask[1]] = prompt_mask_color
        # new_per_image_prompt_mask = np.zeros(
        #     (image_for_prompt_mask.shape[0], image_for_prompt_mask.shape[1]))
        # new_per_image_prompt_mask[prompt_mask[0], prompt_mask[1]] = 255
        # contours, _ = cv2.findContours(
        #     new_per_image_prompt_mask.astype('uint8'), cv2.RETR_TREE,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # per_image_prompt_contours.append(contours)
        # per_image_prompt_mask = per_image_prompt_mask.astype('uint8')
        # per_image_prompt_mask = cv2.cvtColor(per_image_prompt_mask,
        #                                      cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_image_prompt_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_image_prompt_mask[
        #         all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
        #             image_for_prompt_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 0.5,
        #             per_image_prompt_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_image_prompt_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_image_prompt_mask[no_class_mask[0],
        #                           no_class_mask[1]] = image_for_prompt_mask[
        #                               no_class_mask[0], no_class_mask[1]]
        # for contours in per_image_prompt_contours:
        #     cv2.drawContours(per_image_prompt_mask, contours, -1,
        #                      (255, 255, 255), 2)

        # cv2.imencode('.jpg', image_for_prompt_box)[1].tofile(
        #     os.path.join(temp_dir,
        #                  f'idx_{count}_image_with_prompt_point_box.jpg'))
        # cv2.imencode('.jpg', per_image_prompt_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_image_with_prompt_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader

    collater = SAMBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        print('4444', input_images.shape, input_boxs.shape, input_masks.shape,
              sizes)
        print('5555', input_images.dtype, input_boxs.dtype, input_masks.dtype,
              sizes.dtype)
        print('6666', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)
        print('7777', input_prompt_points.dtype, input_prompt_boxs.dtype,
              input_prompt_masks.dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # for i, (per_image, per_image_box, per_image_mask) in enumerate(
        #         zip(input_images, input_boxs, input_masks)):
        #     per_image = per_image.permute(1, 2, 0).cpu().numpy()
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_box = per_image_box.cpu().numpy()
        #     per_image_mask = per_image_mask.squeeze(0)
        #     per_image_mask = per_image_mask.cpu().numpy()

        #     box_color = [int(np.random.choice(range(256))) for _ in range(3)]
        #     mask_color = [int(np.random.choice(range(256))) for _ in range(3)]

        #     image_for_box = copy.deepcopy(per_image).astype('uint8')
        #     per_image_box = (per_image_box[0:4]).astype(np.int32)
        #     left_top, right_bottom = (per_image_box[0],
        #                               per_image_box[1]), (per_image_box[2],
        #                                                   per_image_box[3])
        #     cv2.rectangle(image_for_box,
        #                   left_top,
        #                   right_bottom,
        #                   color=box_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)
        #     text = f'object_box'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(image_for_box,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=box_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(image_for_box,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        #     image_for_mask = copy.deepcopy(per_image).astype('uint8')
        #     per_image_mask = per_image_mask.astype('uint8')

        #     per_image_draw_mask = np.zeros(
        #         (image_for_mask.shape[0], image_for_mask.shape[1], 3))
        #     per_image_contours = []
        #     per_image_mask = np.nonzero(per_image_mask == 1.)
        #     if len(per_image_mask[0]) > 0:
        #         per_image_draw_mask[per_image_mask[0],
        #                             per_image_mask[1]] = mask_color
        #     new_per_image_draw_mask = np.zeros(
        #         (image_for_mask.shape[0], image_for_mask.shape[1]))
        #     if len(per_image_mask[0]) > 0:
        #         new_per_image_draw_mask[per_image_mask[0],
        #                                 per_image_mask[1]] = 255
        #     contours, _ = cv2.findContours(
        #         new_per_image_draw_mask.astype('uint8'), cv2.RETR_TREE,
        #         cv2.CHAIN_APPROX_SIMPLE)
        #     per_image_contours.append(contours)
        #     per_image_draw_mask = per_image_draw_mask.astype('uint8')
        #     per_image_draw_mask = cv2.cvtColor(per_image_draw_mask,
        #                                        cv2.COLOR_RGBA2BGR)
        #     all_classes_mask = np.nonzero(per_image_draw_mask != 0)
        #     if len(all_classes_mask[0]) > 0:
        #         per_image_draw_mask[all_classes_mask[0],
        #                             all_classes_mask[1]] = cv2.addWeighted(
        #                                 image_for_mask[all_classes_mask[0],
        #                                                all_classes_mask[1]],
        #                                 0.5, per_image_draw_mask[
        #                                     all_classes_mask[0],
        #                                     all_classes_mask[1]], 1, 0)
        #     no_class_mask = np.nonzero(per_image_draw_mask == 0)
        #     if len(no_class_mask[0]) > 0:
        #         per_image_draw_mask[no_class_mask[0],
        #                             no_class_mask[1]] = image_for_mask[
        #                                 no_class_mask[0], no_class_mask[1]]
        #     for contours in per_image_contours:
        #         cv2.drawContours(per_image_draw_mask, contours, -1,
        #                          (255, 255, 255), 2)

        #     cv2.imencode('.jpg', image_for_box)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_image_with_box.jpg'))
        #     cv2.imencode('.jpg', per_image_draw_mask)[1].tofile(
        #         os.path.join(temp_dir, f'idx_{count}_{i}_image_with_mask.jpg'))

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # for i, (per_image, per_image_prompt_points, per_image_prompt_box,
        #         per_image_prompt_mask) in enumerate(
        #             zip(input_images, input_prompt_points, input_prompt_boxs,
        #                 input_prompt_masks)):
        #     per_image = per_image.permute(1, 2, 0).cpu().numpy()
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     if per_image_prompt_points is not None:
        #         per_image_prompt_points = per_image_prompt_points.cpu().numpy()
        #     per_image_prompt_box = per_image_prompt_box.cpu().numpy()
        #     per_image_prompt_mask = per_image_prompt_mask.squeeze(0)
        #     per_image_prompt_mask = per_image_prompt_mask.cpu().numpy()

        #     per_image_prompt_mask = cv2.resize(
        #         per_image_prompt_mask, (per_image_prompt_mask.shape[1] * 4,
        #                                 per_image_prompt_mask.shape[0] * 4),
        #         interpolation=cv2.INTER_NEAREST)

        #     prompt_point_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     prompt_box_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     prompt_mask_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]

        #     image_for_prompt_box = copy.deepcopy(per_image)

        #     if per_image_prompt_points is not None:
        #         for per_point in per_image_prompt_points:
        #             cv2.circle(image_for_prompt_box,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        prompt_point_color, -1)

        #     per_image_prompt_box = (per_image_prompt_box[0:4]).astype(np.int32)
        #     left_top, right_bottom = (per_image_prompt_box[0],
        #                               per_image_prompt_box[1]), (
        #                                   per_image_prompt_box[2],
        #                                   per_image_prompt_box[3])
        #     cv2.rectangle(image_for_prompt_box,
        #                   left_top,
        #                   right_bottom,
        #                   color=prompt_box_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)
        #     text = f'prompt_box'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(image_for_prompt_box,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=prompt_box_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(image_for_prompt_box,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        #     image_for_prompt_mask = copy.deepcopy(per_image).astype('uint8')

        #     if per_image_prompt_points is not None:
        #         for per_point in per_image_prompt_points:
        #             cv2.circle(image_for_prompt_mask,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        prompt_point_color, -1)

        #     per_image_prompt_mask = per_image_prompt_mask.astype('uint8')
        #     per_image_prompt_draw_mask = np.zeros(
        #         (image_for_prompt_mask.shape[0],
        #          image_for_prompt_mask.shape[1], 3))
        #     per_image_prompt_contours = []
        #     per_image_prompt_mask = np.nonzero(per_image_prompt_mask == 1.)
        #     if len(per_image_prompt_mask[0]) > 0:
        #         per_image_prompt_draw_mask[
        #             per_image_prompt_mask[0],
        #             per_image_prompt_mask[1]] = prompt_mask_color
        #     new_per_image_prompt_draw_mask = np.zeros(
        #         (image_for_prompt_mask.shape[0],
        #          image_for_prompt_mask.shape[1]))
        #     if len(per_image_prompt_mask[0]) > 0:
        #         new_per_image_prompt_draw_mask[per_image_prompt_mask[0],
        #                                        per_image_prompt_mask[1]] = 255
        #     contours, _ = cv2.findContours(
        #         new_per_image_prompt_draw_mask.astype('uint8'), cv2.RETR_TREE,
        #         cv2.CHAIN_APPROX_SIMPLE)
        #     per_image_prompt_contours.append(contours)
        #     per_image_prompt_draw_mask = per_image_prompt_draw_mask.astype(
        #         'uint8')
        #     per_image_prompt_draw_mask = cv2.cvtColor(
        #         per_image_prompt_draw_mask, cv2.COLOR_RGBA2BGR)
        #     all_classes_mask = np.nonzero(per_image_prompt_draw_mask != 0)
        #     if len(all_classes_mask[0]) > 0:
        #         per_image_prompt_draw_mask[
        #             all_classes_mask[0],
        #             all_classes_mask[1]] = cv2.addWeighted(
        #                 image_for_prompt_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 0.5,
        #                 per_image_prompt_draw_mask[all_classes_mask[0],
        #                                            all_classes_mask[1]], 1, 0)
        #     no_class_mask = np.nonzero(per_image_prompt_draw_mask == 0)
        #     if len(no_class_mask[0]) > 0:
        #         per_image_prompt_draw_mask[
        #             no_class_mask[0],
        #             no_class_mask[1]] = image_for_prompt_mask[no_class_mask[0],
        #                                                       no_class_mask[1]]
        #     for contours in per_image_prompt_contours:
        #         cv2.drawContours(per_image_prompt_draw_mask, contours, -1,
        #                          (255, 255, 255), 2)

        #     cv2.imencode('.jpg', image_for_prompt_box)[1].tofile(
        #         os.path.join(
        #             temp_dir,
        #             f'idx_{count}_{i}_image_with_prompt_point_box.jpg'))
        #     cv2.imencode('.jpg', per_image_prompt_draw_mask)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count}_{i}_image_with_prompt_mask.jpg'))

        if count < 2:
            count += 1
        else:
            break
