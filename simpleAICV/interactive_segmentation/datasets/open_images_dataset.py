import os
import copy
import csv
import cv2
import numpy as np
from PIL import Image

import collections

from torch.utils.data import Dataset


class OpenImagesV7Dataset(Dataset):

    def __init__(self,
                 root_dir,
                 set_type='train',
                 positive_points_num=9,
                 negative_points_num=0,
                 area_filter_ratio=0.0025,
                 box_noise_pixel=50,
                 mask_noise_pixel=50,
                 transform=None):
        assert set_type in ['train', 'validation', 'test'], 'Wrong set name!'

        self.transform = transform

        self.images_dir = os.path.join(root_dir, set_type, 'data')
        self.masks_dir = os.path.join(root_dir, set_type, 'labels', 'masks')

        self.segmentation_info_json_path = os.path.join(
            root_dir, set_type, 'labels', 'segmentations.csv')

        self.all_image_path_dict = collections.OrderedDict()
        for root, folders, files in os.walk(self.images_dir):
            for per_image_name in files:
                if '.jpg' in per_image_name:
                    per_image_path = os.path.join(root, per_image_name)
                    self.all_image_path_dict[per_image_name] = per_image_path

        self.all_mask_path_dict = collections.OrderedDict()
        for root, folders, files in os.walk(self.masks_dir):
            for per_mask_name in files:
                if '.png' in per_mask_name:
                    per_mask_path = os.path.join(root, per_mask_name)
                    self.all_mask_path_dict[per_mask_name] = per_mask_path

        self.all_mask_name_list = set()
        self.all_mask_image_pair_dict = collections.OrderedDict()
        self.all_mask_box_pair_dict = collections.OrderedDict()
        with open(self.segmentation_info_json_path, 'r',
                  encoding='utf-8') as f:
            reader = csv.reader(f)
            for row_idx, per_row in enumerate(reader):
                if row_idx > 0:
                    per_row = per_row[:-1]

                    mask_name = per_row[0]
                    image_name = per_row[1] + '.jpg'

                    per_mask_path = self.all_mask_path_dict[mask_name]
                    per_image_path = self.all_image_path_dict[image_name]

                    if not os.path.isfile(per_mask_path) or not os.path.isfile(
                            per_image_path):
                        continue

                    self.all_mask_name_list.add(mask_name)
                    self.all_mask_image_pair_dict[mask_name] = image_name
                    # x_min,y_min,x_max,y_max:0~1
                    self.all_mask_box_pair_dict[mask_name] = [
                        float(per_row[4]),
                        float(per_row[6]),
                        float(per_row[5]),
                        float(per_row[7]),
                    ]

        self.all_mask_name_list = list(self.all_mask_name_list)

        self.positive_points_num = positive_points_num
        self.negative_points_num = negative_points_num
        self.area_filter_ratio = area_filter_ratio
        self.box_noise_pixel = box_noise_pixel
        self.mask_noise_pixel = mask_noise_pixel
        self.transform = transform

        assert len(self.all_mask_path_dict) == len(
            self.all_mask_name_list) == len(
                self.all_mask_image_pair_dict) == len(
                    self.all_mask_box_pair_dict)

        print(f'Dataset Size:{len(self.all_mask_name_list)}')

    def __len__(self):
        return len(self.all_mask_name_list)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        origin_image = copy.deepcopy(image)
        origin_mask = copy.deepcopy(mask)

        size = np.array([origin_image.shape[0],
                         origin_image.shape[1]]).astype(np.float32)
        origin_size = copy.deepcopy(size)

        image_box = np.array([
            self.all_mask_box_pair_dict[self.all_mask_name_list[idx]][0] *
            image.shape[1],
            self.all_mask_box_pair_dict[self.all_mask_name_list[idx]][1] *
            image.shape[0],
            self.all_mask_box_pair_dict[self.all_mask_name_list[idx]][2] *
            image.shape[1],
            self.all_mask_box_pair_dict[self.all_mask_name_list[idx]][3] *
            image.shape[0],
        ],
                             dtype=np.float32)
        origin_box = copy.deepcopy(image_box)

        image_mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)

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
                        image_mask_all_points_coords[positive_point_idx][0], 1
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
                        [0], 0
                    ])
                negative_prompt_point = np.array(negative_prompt_point,
                                                 dtype=np.float32)
            else:
                negative_prompt_point = np.zeros((0, 3), dtype=np.float32)

        prompt_box = copy.deepcopy(image_box)
        if self.box_noise_pixel > 0:
            prompt_box = self.noise_bbox(prompt_box, origin_size)

        prompt_mask = copy.deepcopy(image_mask)
        prompt_mask = self.noise_mask(prompt_mask, origin_size)

        sample = {
            'origin_image': origin_image,
            'origin_bbox': origin_box,
            'origin_mask': origin_mask,
            'origin_size': origin_size,
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
        image = cv2.imdecode(
            np.fromfile(self.all_image_path_dict[self.all_mask_image_pair_dict[
                self.all_mask_name_list[idx]]],
                        dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        loadmask_path = self.all_mask_path_dict[self.all_mask_name_list[idx]]
        mask = np.array(Image.open(loadmask_path).convert('L'), dtype=np.uint8)
        mask[mask >= 255] = 255
        mask[mask <= 0] = 0
        mask = mask / 255.

        return mask.astype(np.float32)

    def noise_bbox(self, properties_bbox, mask_np_shape):
        w, h = properties_bbox[2] - properties_bbox[0], properties_bbox[
            3] - properties_bbox[1]
        x0 = properties_bbox[0] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                w / 2), -w / 2)
        y0 = properties_bbox[1] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                h / 2), -h / 2)
        x1 = properties_bbox[2] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                w / 2), -w / 2)
        y1 = properties_bbox[3] + max(
            min(np.random.randint(-self.box_noise_pixel, self.box_noise_pixel),
                h / 2), -h / 2)
        x1 = x1 if x1 <= mask_np_shape[1] else mask_np_shape[1]  # 避免越界
        y1 = y1 if y1 <= mask_np_shape[0] else mask_np_shape[0]
        post_properties_bbox = np.array([x0, y0, x1, y1])
        post_properties_bbox = np.where(post_properties_bbox > 0,
                                        post_properties_bbox, 0)
        if x0 >= x1 or y0 >= y1:
            return properties_bbox.astype(np.float32)
        else:
            return post_properties_bbox.astype(np.float32)

    def noise_mask(self, properties_mask, mask_np_shape):
        image_h, image_w = mask_np_shape[0], mask_np_shape[1]

        kernel = np.random.randint(1, self.mask_noise_pixel)
        erode_kernel = np.ones((kernel, kernel), np.uint8)
        post_properties_mask = cv2.erode(properties_mask,
                                         erode_kernel,
                                         iterations=1)

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

    from tools.path import open_images_v7_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMCollater

    open_images_v7_dataset = OpenImagesV7Dataset(
        open_images_v7_dataset_path,
        set_type='test',
        positive_points_num=9,
        negative_points_num=9,
        area_filter_ratio=0.0025,
        box_noise_pixel=50,
        mask_noise_pixel=100,
        transform=transforms.Compose([
            SamResize(resize=1024),
            SamRandomHorizontalFlip(prob=0.5),
            #   SamNormalize(
            #       mean=[123.675, 116.28, 103.53],
            #       std=[58.395, 57.12, 57.375]),
        ]))

    count = 0
    for per_sample in tqdm(open_images_v7_dataset):
        print('1111', per_sample['origin_image'].shape,
              per_sample['origin_bbox'].shape, per_sample['origin_mask'].shape,
              per_sample['origin_size'], per_sample['image'].shape,
              per_sample['box'].shape, per_sample['mask'].shape,
              per_sample['size'], per_sample['positive_prompt_point'].shape,
              per_sample['negative_prompt_point'].shape,
              per_sample['prompt_box'].shape, per_sample['prompt_mask'].shape)
        print('2222', per_sample['origin_image'].dtype,
              per_sample['origin_bbox'].dtype, per_sample['origin_mask'].dtype,
              per_sample['origin_size'].dtype, per_sample['image'].dtype,
              per_sample['box'].dtype, per_sample['mask'].dtype,
              per_sample['size'].dtype,
              per_sample['positive_prompt_point'].dtype,
              per_sample['negative_prompt_point'].dtype,
              per_sample['prompt_box'].dtype, per_sample['prompt_mask'].dtype)
        print('3333', per_sample['origin_bbox'], per_sample['origin_size'],
              per_sample['box'], per_sample['size'],
              per_sample['positive_prompt_point'][0],
              per_sample['negative_prompt_point'][0], per_sample['prompt_box'])

        # temp_dir = './temp1'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # origin_image = np.ascontiguousarray(per_sample['origin_image'],
        #                                     dtype=np.uint8)
        # origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
        # origin_bbox = per_sample['origin_bbox']
        # origin_mask = per_sample['origin_mask']

        # origin_image_for_box = copy.deepcopy(origin_image).astype('uint8')
        # per_image_origin_box = (origin_bbox[0:4]).astype(np.int32)
        # box_color = [int(np.random.choice(range(256))) for _ in range(3)]
        # left_top, right_bottom = (per_image_origin_box[0],
        #                           per_image_origin_box[1]), (
        #                               per_image_origin_box[2],
        #                               per_image_origin_box[3])
        # cv2.rectangle(origin_image_for_box,
        #               left_top,
        #               right_bottom,
        #               color=box_color,
        #               thickness=2,
        #               lineType=cv2.LINE_AA)
        # text = 'object_box'
        # text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        # fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
        #                      left_top[1] - text_size[1] - 3)
        # cv2.rectangle(origin_image_for_box,
        #               left_top,
        #               fill_right_bottom,
        #               color=box_color,
        #               thickness=-1,
        #               lineType=cv2.LINE_AA)
        # cv2.putText(origin_image_for_box,
        #             text, (left_top[0], left_top[1] - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             color=(0, 0, 0),
        #             thickness=1,
        #             lineType=cv2.LINE_AA)

        # origin_image_for_mask = copy.deepcopy(origin_image).astype('uint8')
        # origin_mask = origin_mask.astype('uint8')
        # per_origin_image_mask = np.zeros((origin_image_for_mask.shape[0],
        #                                   origin_image_for_mask.shape[1], 3))
        # per_origin_image_contours = []
        # origin_mask = np.nonzero(origin_mask == 1.)
        # mask_color = [int(np.random.choice(range(256))) for _ in range(3)]
        # if len(origin_mask[0]) > 0:
        #     per_origin_image_mask[origin_mask[0], origin_mask[1]] = mask_color
        # new_per_origin_image_mask = np.zeros(
        #     (origin_image_for_mask.shape[0], origin_image_for_mask.shape[1]))
        # if len(origin_mask[0]) > 0:
        #     new_per_origin_image_mask[origin_mask[0], origin_mask[1]] = 255
        # contours, _ = cv2.findContours(
        #     new_per_origin_image_mask.astype('uint8'), cv2.RETR_TREE,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # per_origin_image_contours.append(contours)
        # per_origin_image_mask = per_origin_image_mask.astype('uint8')
        # per_origin_image_mask = cv2.cvtColor(per_origin_image_mask,
        #                                      cv2.COLOR_RGBA2BGR)
        # all_classes_mask = np.nonzero(per_origin_image_mask != 0)
        # if len(all_classes_mask[0]) > 0:
        #     per_origin_image_mask[
        #         all_classes_mask[0], all_classes_mask[1]] = cv2.addWeighted(
        #             origin_image_for_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 0.5,
        #             per_origin_image_mask[all_classes_mask[0],
        #                                   all_classes_mask[1]], 1, 0)
        # no_class_mask = np.nonzero(per_origin_image_mask == 0)
        # if len(no_class_mask[0]) > 0:
        #     per_origin_image_mask[no_class_mask[0],
        #                           no_class_mask[1]] = origin_image_for_mask[
        #                               no_class_mask[0], no_class_mask[1]]
        # for contours in per_origin_image_contours:
        #     cv2.drawContours(per_origin_image_mask, contours, -1,
        #                      (255, 255, 255), 2)

        # cv2.imencode('.jpg', origin_image_for_box)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_origin_image_with_box.jpg'))
        # cv2.imencode('.jpg', per_origin_image_mask)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}_origin_image_with_mask.jpg'))

        # temp_dir = './temp1'
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

        # temp_dir = './temp1'
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

    # collater = SAMCollater(resize=1024,
    #                        positive_point_num_range=[1, 9],
    #                        negative_point_num_range=0,
    #                        batch_align_random_point_num=False,
    #                        positive_negative_point_num_ratio=None)
    collater = SAMCollater(resize=1024,
                           positive_point_num_range=[1, 9],
                           negative_point_num_range=[1, 9],
                           batch_align_random_point_num=True,
                           positive_negative_point_num_ratio=1)
    train_loader = DataLoader(open_images_v7_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        print('3333', len(origin_images), len(origin_bboxs), len(origin_masks),
              len(origin_sizes))

        print('4444', len(input_images), len(input_boxs), len(input_masks),
              len(sizes))
        print('5555', input_images[0].shape, input_images[1].shape,
              input_boxs[0].shape, input_boxs[1].shape, input_masks[0].shape,
              input_masks[1].shape, sizes[0], sizes[1])
        print('6666', input_images[0].dtype, input_boxs[0].dtype,
              input_masks[0].dtype, sizes[0].dtype)

        print('7777', len(input_positive_prompt_points),
              len(input_negative_prompt_points), len(input_prompt_points))
        if input_positive_prompt_points[
                0] is not None and input_negative_prompt_points[0] is not None:
            print('8888', input_positive_prompt_points[0].shape,
                  input_positive_prompt_points[1].shape,
                  input_negative_prompt_points[0].shape,
                  input_negative_prompt_points[1].shape,
                  input_prompt_points[0].shape, input_prompt_points[1].shape)
            print('9999', input_positive_prompt_points[0].dtype,
                  input_negative_prompt_points[0].dtype,
                  input_prompt_points[0].dtype)

        print('9191', len(input_prompt_boxs), len(input_prompt_masks),
              batch_images.shape, batch_masks.shape, len(batch_prompts),
              torch.unique(batch_masks))
        print('9292', input_prompt_boxs[0].shape, input_prompt_boxs[1].shape,
              input_prompt_masks[0].shape, input_prompt_masks[1].shape)
        print('9393', input_prompt_boxs[0].dtype, input_prompt_masks[0].dtype)

        print('9494', batch_prompts[0]['prompt_point'].shape,
              batch_prompts[0]['prompt_box'].shape,
              batch_prompts[0]['prompt_mask'].shape)
        print('9595', batch_prompts[1]['prompt_point'].shape,
              batch_prompts[1]['prompt_box'].shape,
              batch_prompts[1]['prompt_mask'].shape)
        print('9696', batch_prompts[0]['prompt_point'].dtype,
              batch_prompts[0]['prompt_box'].dtype,
              batch_prompts[0]['prompt_mask'].dtype)

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # for i, (per_origin_image, per_origin_image_box,
        #         per_origin_image_mask) in enumerate(
        #             zip(origin_images, origin_bboxs, origin_masks)):
        #     origin_image = np.ascontiguousarray(per_origin_image,
        #                                         dtype=np.uint8)
        #     origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
        #     origin_bbox = per_origin_image_box
        #     origin_mask = per_origin_image_mask

        #     origin_image_for_box = copy.deepcopy(origin_image).astype('uint8')
        #     per_image_origin_box = (origin_bbox[0:4]).astype(np.int32)
        #     box_color = [int(np.random.choice(range(256))) for _ in range(3)]
        #     left_top, right_bottom = (per_image_origin_box[0],
        #                               per_image_origin_box[1]), (
        #                                   per_image_origin_box[2],
        #                                   per_image_origin_box[3])
        #     cv2.rectangle(origin_image_for_box,
        #                   left_top,
        #                   right_bottom,
        #                   color=box_color,
        #                   thickness=2,
        #                   lineType=cv2.LINE_AA)
        #     text = 'object_box'
        #     text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
        #     fill_right_bottom = (max(left_top[0] + text_size[0],
        #                              right_bottom[0]),
        #                          left_top[1] - text_size[1] - 3)
        #     cv2.rectangle(origin_image_for_box,
        #                   left_top,
        #                   fill_right_bottom,
        #                   color=box_color,
        #                   thickness=-1,
        #                   lineType=cv2.LINE_AA)
        #     cv2.putText(origin_image_for_box,
        #                 text, (left_top[0], left_top[1] - 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color=(0, 0, 0),
        #                 thickness=1,
        #                 lineType=cv2.LINE_AA)

        #     origin_image_for_mask = copy.deepcopy(origin_image).astype('uint8')
        #     origin_mask = origin_mask.astype('uint8')
        #     per_origin_image_mask = np.zeros(
        #         (origin_image_for_mask.shape[0],
        #          origin_image_for_mask.shape[1], 3))
        #     per_origin_image_contours = []
        #     origin_mask = np.nonzero(origin_mask == 1.)
        #     mask_color = [int(np.random.choice(range(256))) for _ in range(3)]
        #     if len(origin_mask[0]) > 0:
        #         per_origin_image_mask[origin_mask[0],
        #                               origin_mask[1]] = mask_color
        #     new_per_origin_image_mask = np.zeros(
        #         (origin_image_for_mask.shape[0],
        #          origin_image_for_mask.shape[1]))
        #     if len(origin_mask[0]) > 0:
        #         new_per_origin_image_mask[origin_mask[0], origin_mask[1]] = 255
        #     contours, _ = cv2.findContours(
        #         new_per_origin_image_mask.astype('uint8'), cv2.RETR_TREE,
        #         cv2.CHAIN_APPROX_SIMPLE)
        #     per_origin_image_contours.append(contours)
        #     per_origin_image_mask = per_origin_image_mask.astype('uint8')
        #     per_origin_image_mask = cv2.cvtColor(per_origin_image_mask,
        #                                          cv2.COLOR_RGBA2BGR)
        #     all_classes_mask = np.nonzero(per_origin_image_mask != 0)
        #     if len(all_classes_mask[0]) > 0:
        #         per_origin_image_mask[
        #             all_classes_mask[0],
        #             all_classes_mask[1]] = cv2.addWeighted(
        #                 origin_image_for_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 0.5,
        #                 per_origin_image_mask[all_classes_mask[0],
        #                                       all_classes_mask[1]], 1, 0)
        #     no_class_mask = np.nonzero(per_origin_image_mask == 0)
        #     if len(no_class_mask[0]) > 0:
        #         per_origin_image_mask[
        #             no_class_mask[0],
        #             no_class_mask[1]] = origin_image_for_mask[no_class_mask[0],
        #                                                       no_class_mask[1]]
        #     for contours in per_origin_image_contours:
        #         cv2.drawContours(per_origin_image_mask, contours, -1,
        #                          (255, 255, 255), 2)

        #     cv2.imencode('.jpg', origin_image_for_box)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count}_{i}_origin_image_with_box.jpg'))
        #     cv2.imencode('.jpg', per_origin_image_mask)[1].tofile(
        #         os.path.join(temp_dir,
        #                      f'idx_{count}_{i}_origin_image_with_mask.jpg'))

        # temp_dir = './temp2'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # for i, (per_image, per_image_box, per_image_mask) in enumerate(
        #         zip(input_images, input_boxs, input_masks)):
        #     per_image = per_image.permute(1, 2, 0).cpu().numpy()
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     per_image_box = per_image_box.cpu().numpy()
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

        # for i, (per_image, per_image_positive_prompt_points,
        #         per_image_negative_prompt_points, per_image_prompt_box,
        #         per_image_prompt_mask) in enumerate(
        #             zip(input_images, input_positive_prompt_points,
        #                 input_negative_prompt_points, input_prompt_boxs,
        #                 input_prompt_masks)):
        #     per_image = per_image.permute(1, 2, 0).cpu().numpy()
        #     per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
        #     per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

        #     if per_image_positive_prompt_points is not None:
        #         per_image_positive_prompt_points = per_image_positive_prompt_points.cpu(
        #         ).numpy()
        #     if per_image_negative_prompt_points is not None:
        #         per_image_negative_prompt_points = per_image_negative_prompt_points.cpu(
        #         ).numpy()
        #     per_image_prompt_box = per_image_prompt_box.cpu().numpy()
        #     per_image_prompt_mask = per_image_prompt_mask.cpu().numpy()

        #     per_image_prompt_mask = cv2.resize(
        #         per_image_prompt_mask, (per_image_prompt_mask.shape[1] * 4,
        #                                 per_image_prompt_mask.shape[0] * 4),
        #         interpolation=cv2.INTER_NEAREST)

        #     positive_prompt_point_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     negative_prompt_point_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     prompt_box_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]
        #     prompt_mask_color = [
        #         int(np.random.choice(range(256))) for _ in range(3)
        #     ]

        #     image_for_prompt_box = copy.deepcopy(per_image)

        #     if per_image_positive_prompt_points is not None:
        #         for per_point in per_image_positive_prompt_points:
        #             cv2.circle(image_for_prompt_box,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        positive_prompt_point_color, -1)

        #     if per_image_negative_prompt_points is not None:
        #         for per_point in per_image_negative_prompt_points:
        #             cv2.circle(image_for_prompt_box,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        negative_prompt_point_color, -1)

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

        #     if per_image_positive_prompt_points is not None:
        #         for per_point in per_image_positive_prompt_points:
        #             cv2.circle(image_for_prompt_mask,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        positive_prompt_point_color, -1)

        #     if per_image_negative_prompt_points is not None:
        #         for per_point in per_image_negative_prompt_points:
        #             cv2.circle(image_for_prompt_mask,
        #                        (int(per_point[0]), int(per_point[1])), 10,
        #                        negative_prompt_point_color, -1)

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
