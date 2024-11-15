import collections
import cv2
import os
import numpy as np

from tqdm import tqdm

FaceSynthetics_classes_idx_to_name_dict = {
    0: 'background',
    1: 'skin',
    2: 'nose',
    3: 'right_eye',
    4: 'left_eye',
    5: 'right_brow',
    6: 'left_brow',
    7: 'right_ear',
    8: 'left_ear',
    9: 'mouth_interior',
    10: 'top_lip',
    11: 'bottom_lip',
    12: 'neck',
    13: 'hair',
    14: 'beard',
    15: 'clothing',
    16: 'glasses',
    17: 'headwear',
    18: 'facewear',
    255: 'ignore',
}

FaceSynthetics_all_class_labels = [
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

FaceSynthetics_all_class_colors = [
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
    root_dataset_image_annot_path = os.path.join(root_dataset_path,
                                                 'images_and_annots')

    save_dataset_path = os.path.join(save_dataset_path, dataset_name, 'train')
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    all_image_name_path_list = []
    for per_image_name in os.listdir(root_dataset_image_annot_path):
        if '.png' in per_image_name and '_seg' not in per_image_name:
            per_image_path = os.path.join(root_dataset_image_annot_path,
                                          per_image_name)
            per_mask_name = per_image_name.split('.')[0] + '_seg.png'
            per_mask_path = os.path.join(root_dataset_image_annot_path,
                                         per_mask_name)
            if not os.path.exists(per_image_path) or not os.path.join(
                    per_mask_path):
                continue
            all_image_name_path_list.append(
                [per_image_name, per_image_path, per_mask_name, per_mask_path])

    print('1111', len(all_image_name_path_list), all_image_name_path_list[0])

    for per_image_name, per_image_path, per_mask_name, per_mask_path in tqdm(
            all_image_name_path_list):
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
            if per_label_value not in FaceSynthetics_all_class_labels:
                print(f'标签中包含不在范围内的类别：{per_label_value}')
                illegal_mask_flag = True

        if illegal_mask_flag:
            print(f'不合法的mask,{per_image_name}')
            continue

        keep_image_name = f'{dataset_name}_{per_image_name_prefix}.jpg'
        keep_mask_name = f'{dataset_name}_{per_image_name_prefix}.png'

        save_image_path = os.path.join(save_dataset_path, keep_image_name)
        save_mask_path = os.path.join(save_dataset_path, keep_mask_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
        cv2.imencode('.png', per_mask)[1].tofile(save_mask_path)


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/face_parsing_dataset_origin/FaceSynthetics/'
    save_dataset_path = r'/root/autodl-tmp/face_parsing_dataset/'
    dataset_name = 'FaceSynthetics'
    process_data(root_dataset_path, save_dataset_path, dataset_name)
