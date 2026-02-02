import collections
import cv2
import os
import numpy as np

from tqdm import tqdm

CIHP_classes_idx_to_name_dict = {
    0: 'background',
    1: 'hat',
    2: 'hair',
    3: 'glove',
    4: 'sunglasses',
    5: 'upper_clothes',
    6: 'dress',
    7: 'coat',
    8: 'socks',
    9: 'pants',
    10: 'torso_skin',
    11: 'scarf',
    12: 'skirt',
    13: 'face',
    14: 'left_arm',
    15: 'right_arm',
    16: 'left_leg',
    17: 'right_leg',
    18: 'left_shoe',
    19: 'right_shoe',
}

LIP_all_class_labels = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
]

LIP_all_class_colors = [
    [0, 0, 0],
    [204, 0, 0],
    [76, 153, 0],
    [204, 204, 0],
    [51, 51, 255],
    [204, 0, 204],
    [0, 255, 255],
    [255, 204, 204],
    [102, 51, 0],
    [255, 0, 0],
    [102, 204, 0],
    [255, 255, 0],
    [0, 0, 153],
    [0, 0, 204],
    [255, 51, 153],
    [0, 204, 204],
    [0, 51, 0],
    [255, 153, 51],
    [0, 204, 0],
    [173, 144, 13],
]


def process_data(root_dataset_path, save_dataset_path, dataset_name):
    root_dataset_train_image_path = os.path.join(root_dataset_path, 'Training',
                                                 'Images')
    root_dataset_train_mask_path = os.path.join(root_dataset_path, 'Training',
                                                'Category_ids')

    save_train_dataset_path = os.path.join(save_dataset_path, dataset_name,
                                           'train')
    if not os.path.exists(save_train_dataset_path):
        os.makedirs(save_train_dataset_path)

    all_train_image_name_path_list = []
    for per_image_name in os.listdir(root_dataset_train_image_path):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(root_dataset_train_image_path,
                                          per_image_name)
            per_mask_name = per_image_name.split('.')[0] + '.png'
            per_mask_path = os.path.join(root_dataset_train_mask_path,
                                         per_mask_name)
            if not os.path.exists(per_image_path) or not os.path.join(
                    per_mask_path):
                continue
            all_train_image_name_path_list.append(
                [per_image_name, per_image_path, per_mask_name, per_mask_path])

    print('1111', len(all_train_image_name_path_list),
          all_train_image_name_path_list[0])

    for per_image_name, per_image_path, per_mask_name, per_mask_path in tqdm(
            all_train_image_name_path_list):
        if not os.path.exists(per_image_path) or not os.path.join(
                per_mask_path):
            continue

        per_image_name_prefix = per_image_name.split('.')[0]

        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        per_mask = cv2.imdecode(np.fromfile(per_mask_path, dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)

        if per_image.shape != per_mask.shape:
            per_mask = cv2.resize(per_mask,
                                  (per_image.shape[1], per_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        per_mask[per_mask >= 255] = 0
        per_mask[per_mask <= 0] = 0

        illegal_mask_flag = False
        exist_class_label_values = np.unique(per_mask).tolist()
        for per_label_value in exist_class_label_values:
            if per_label_value not in LIP_all_class_labels:
                print(f'标签中包含不在范围内的类别：{per_label_value}')
                illegal_mask_flag = True

        if illegal_mask_flag:
            print(f'不合法的mask,{per_image_name}')
            continue

        keep_image_name = f'{dataset_name}_{per_image_name_prefix}.jpg'
        keep_mask_name = f'{dataset_name}_{per_image_name_prefix}.png'

        save_image_path = os.path.join(save_train_dataset_path,
                                       keep_image_name)
        save_mask_path = os.path.join(save_train_dataset_path, keep_mask_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
        cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)

    root_dataset_val_image_path = os.path.join(root_dataset_path, 'Validation',
                                               'Images')
    root_dataset_val_mask_path = os.path.join(root_dataset_path, 'Validation',
                                              'Category_ids')

    save_val_dataset_path = os.path.join(save_dataset_path, dataset_name,
                                         'val')
    if not os.path.exists(save_val_dataset_path):
        os.makedirs(save_val_dataset_path)

    all_val_image_name_path_list = []
    for per_image_name in os.listdir(root_dataset_val_image_path):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(root_dataset_val_image_path,
                                          per_image_name)
            per_mask_name = per_image_name.split('.')[0] + '.png'
            per_mask_path = os.path.join(root_dataset_val_mask_path,
                                         per_mask_name)
            if not os.path.exists(per_image_path) or not os.path.join(
                    per_mask_path):
                continue
            all_val_image_name_path_list.append(
                [per_image_name, per_image_path, per_mask_name, per_mask_path])

    print('2222', len(all_val_image_name_path_list),
          all_val_image_name_path_list[0])

    for per_image_name, per_image_path, per_mask_name, per_mask_path in tqdm(
            all_val_image_name_path_list):
        if not os.path.exists(per_image_path) or not os.path.join(
                per_mask_path):
            continue

        per_image_name_prefix = per_image_name.split('.')[0]

        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        per_mask = cv2.imdecode(np.fromfile(per_mask_path, dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)

        if per_image.shape != per_mask.shape:
            per_mask = cv2.resize(per_mask,
                                  (per_image.shape[1], per_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        per_mask[per_mask >= 255] = 0
        per_mask[per_mask <= 0] = 0

        illegal_mask_flag = False
        exist_class_label_values = np.unique(per_mask).tolist()
        for per_label_value in exist_class_label_values:
            if per_label_value not in LIP_all_class_labels:
                print(f'标签中包含不在范围内的类别：{per_label_value}')
                illegal_mask_flag = True

        if illegal_mask_flag:
            print(f'不合法的mask,{per_image_name}')
            continue

        keep_image_name = f'{dataset_name}_{per_image_name_prefix}.jpg'
        keep_mask_name = f'{dataset_name}_{per_image_name_prefix}.png'

        save_image_path = os.path.join(save_val_dataset_path, keep_image_name)
        save_mask_path = os.path.join(save_val_dataset_path, keep_mask_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
        cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/human_parsing_dataset_origin/CIHP/'
    save_dataset_path = r'/root/autodl-tmp/human_parsing_dataset/'
    dataset_name = 'CIHP'
    process_data(root_dataset_path, save_dataset_path, dataset_name)
