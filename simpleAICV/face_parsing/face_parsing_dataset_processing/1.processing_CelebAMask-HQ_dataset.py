import collections
import cv2
import os
import numpy as np
import pandas as pd
import shutil

from tqdm import tqdm

CelebAMask_HQ_classes = [
    'skin',
    'nose',
    'eye_g',
    'l_eye',
    'r_eye',
    'l_brow',
    'r_brow',
    'l_ear',
    'r_ear',
    'mouth',
    'u_lip',
    'l_lip',
    'hair',
    'hat',
    'ear_r',
    'neck_l',
    'neck',
    'cloth',
]

CelebAMask_HQ_classes_idx_to_name_dict = {
    0: 'background',
    1: 'skin',
    2: 'nose',
    3: 'eye_g',
    4: 'l_eye',
    5: 'r_eye',
    6: 'l_brow',
    7: 'r_brow',
    8: 'l_ear',
    9: 'r_ear',
    10: 'mouth',
    11: 'u_lip',
    12: 'l_lip',
    13: 'hair',
    14: 'hat',
    15: 'ear_r',
    16: 'neck_l',
    17: 'neck',
    18: 'cloth',
}

CelebAMask_HQ_all_class_labels = [
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
]

CelebAMask_HQ_all_class_colors = [
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
]


def process_data(root_dataset_path, save_dataset_path, dataset_name):
    root_dataset_image_path = os.path.join(root_dataset_path, 'CelebA-HQ-img')
    root_dataset_annot_path = os.path.join(root_dataset_path,
                                           'CelebAMask-HQ-mask-anno')

    save_dataset_temp_path = os.path.join(save_dataset_path, dataset_name,
                                          'temp')
    if not os.path.exists(save_dataset_temp_path):
        os.makedirs(save_dataset_temp_path)

    mapping_txt_path = os.path.join(root_dataset_path,
                                    'CelebA-HQ-to-CelebA-mapping.txt')
    image_info_list = pd.read_csv(mapping_txt_path, sep=r'\s+', header=0)

    all_image_name_path_list = []
    for per_image_name in os.listdir(root_dataset_image_path):
        if '.jpg' in per_image_name:
            per_image_path = os.path.join(root_dataset_image_path,
                                          per_image_name)
            if not os.path.exists(per_image_path):
                continue
            all_image_name_path_list.append([per_image_name, per_image_path])

    print('1111', len(all_image_name_path_list), all_image_name_path_list[0])

    all_image_mask_dict = collections.OrderedDict()
    for per_image_name, per_image_path in tqdm(all_image_name_path_list):
        if not os.path.exists(per_image_path):
            continue

        per_image_name_prefix = per_image_name.split('.')[0]
        # 该图片的分割标签存放的目录，一共有15个目录，每个目录存了2000张分割结果（包含一张图片的面部各个组件分开的分割结果）
        folder_num = int(per_image_name_prefix) // 2000

        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        per_mask = np.zeros((per_image.shape[0], per_image.shape[1]))
        for part_idx, classes_label in enumerate(CelebAMask_HQ_classes):
            per_part_mask_path = os.path.join(
                root_dataset_annot_path, str(folder_num),
                per_image_name_prefix.rjust(5, '0') + '_' + classes_label +
                '.png')
            if os.path.exists(per_part_mask_path):
                per_part_mask = cv2.imdecode(
                    np.fromfile(per_part_mask_path, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE)
                if per_image.shape != per_part_mask.shape:
                    per_part_mask = cv2.resize(
                        per_part_mask,
                        (per_image.shape[1], per_image.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
                per_mask[per_part_mask != 0] = part_idx + 1

        per_mask[per_mask >= 255] = 0
        per_mask[per_mask <= 0] = 0

        illegal_mask_flag = False
        exist_class_label_values = np.unique(per_mask).tolist()
        for per_label_value in exist_class_label_values:
            if per_label_value not in CelebAMask_HQ_all_class_labels:
                print(f'标签中包含不在范围内的类别：{per_label_value}')
                illegal_mask_flag = True

        if illegal_mask_flag:
            print(f'不合法的mask,{per_image_name}')
            continue

        keep_image_name = f'{dataset_name}_{per_image_name_prefix}.jpg'
        keep_mask_name = f'{dataset_name}_{per_image_name_prefix}.png'

        save_image_path = os.path.join(save_dataset_temp_path, keep_image_name)
        save_mask_path = os.path.join(save_dataset_temp_path, keep_mask_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
        cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)

        all_image_mask_dict[per_image_name_prefix] = keep_image_name

    print('2222', len(all_image_mask_dict))
    for key, value in all_image_mask_dict.items():
        print(key, value)
        break

    save_dataset_train_path = os.path.join(save_dataset_path, dataset_name,
                                           'train')
    save_dataset_val_path = os.path.join(save_dataset_path, dataset_name,
                                         'val')
    save_dataset_test_path = os.path.join(save_dataset_path, dataset_name,
                                          'test')
    if not os.path.exists(save_dataset_train_path):
        os.makedirs(save_dataset_train_path)
    if not os.path.exists(save_dataset_val_path):
        os.makedirs(save_dataset_val_path)
    if not os.path.exists(save_dataset_test_path):
        os.makedirs(save_dataset_test_path)

    train_count, val_count, test_count = 0, 0, 0
    for _, row in image_info_list.iterrows():
        image_idx = row.iloc[0]
        origin_x = row.iloc[1]

        keep_image_name = f'{dataset_name}_{str(image_idx)}.jpg'
        keep_mask_name = f'{dataset_name}_{str(image_idx)}.png'

        if origin_x >= 162771 and origin_x < 182638:
            val_count += 1
            save_folder_path = save_dataset_val_path

        elif origin_x >= 182638:
            test_count += 1
            save_folder_path = save_dataset_test_path

        else:
            train_count += 1
            save_folder_path = save_dataset_train_path

        root_image_path = os.path.join(save_dataset_temp_path, keep_image_name)
        root_mask_path = os.path.join(save_dataset_temp_path, keep_mask_name)

        shutil.copy(root_image_path, save_folder_path)
        shutil.copy(root_mask_path, save_folder_path)

    print(train_count, val_count, test_count)


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/face_parsing_dataset_origin/CelebAMask-HQ/'
    save_dataset_path = r'/root/autodl-tmp/face_parsing_dataset/'
    dataset_name = 'CelebAMask-HQ'
    process_data(root_dataset_path, save_dataset_path, dataset_name)
