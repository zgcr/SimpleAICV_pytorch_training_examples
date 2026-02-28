import os
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
import json
from multiprocessing import Pool


def process_single_video(args):
    """处理单个视频的函数"""
    per_video_name, per_video_dir_path, all_path_name, save_path = args

    save_per_video_dir = os.path.join(save_path, *all_path_name,
                                      per_video_name)
    os.makedirs(save_per_video_dir, exist_ok=True)

    # 读取所有图像文件
    per_video_all_image_names = []
    for file_name in sorted(os.listdir(per_video_dir_path)):
        if file_name.lower().endswith(('.jpg')):
            per_video_all_image_names.append(file_name)

    first_frame_path = os.path.join(per_video_dir_path,
                                    per_video_all_image_names[0])
    first_frame = cv2.imdecode(np.fromfile(first_frame_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)

    origin_height, origin_width = first_frame.shape[0], first_frame.shape[1]

    # 判断是否需要resize
    max_side = max(origin_height, origin_width)
    need_resize = max_side > 1080

    if need_resize:
        factor = 1080.0 / max(origin_height, origin_width)
        resize_h, resize_w = int(round(origin_height * factor)), int(
            round(origin_width * factor))
    else:
        resize_h = origin_height
        resize_w = origin_width

    # 处理JSON标注文件
    per_video_json_file_name = per_video_name + "_manual.json"
    per_video_json_path = os.path.join(per_video_dir_path,
                                       per_video_json_file_name)

    # 读取JSON文件
    with open(per_video_json_path, encoding='utf-8') as f:
        per_video_json_data = json.load(f)

    per_video_json_data['video_height'] = float(resize_h)
    per_video_json_data['video_width'] = float(resize_w)
    per_video_json_data['video_resolution'] = float(resize_h * resize_w)

    if need_resize:
        per_video_annotations = per_video_json_data['masklet']
        new_per_video_annotations = []

        for per_frame_annotations in per_video_annotations:
            new_per_frame_annotations = []
            for per_object_annotation in per_frame_annotations:
                per_mask = mask_utils.decode(per_object_annotation)
                resized_mask = cv2.resize(per_mask, (resize_w, resize_h),
                                          interpolation=cv2.INTER_NEAREST)
                resized_mask[resized_mask > 0] = 1
                resized_rle = mask_utils.encode(
                    np.asfortranarray(resized_mask))
                resized_rle['counts'] = resized_rle['counts'].decode('utf-8')

                new_per_frame_annotations.append(resized_rle)

            new_per_video_annotations.append(new_per_frame_annotations)

        per_video_json_data['masklet'] = new_per_video_annotations

    # 保存JSON
    save_per_video_json_path = os.path.join(save_per_video_dir,
                                            per_video_json_file_name)
    with open(save_per_video_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(per_video_json_data, json_file, ensure_ascii=False)

    # 处理所有图像
    for per_image_name in per_video_all_image_names:
        per_image_path = os.path.join(per_video_dir_path, per_image_name)
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        if need_resize:
            per_image = cv2.resize(per_image, (resize_w, resize_h))

        save_per_image_path = os.path.join(save_per_video_dir, per_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)


def process_videos(root_path, save_path):
    root_name = root_path.split('/')[-1]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_dataset_path = os.path.join(root_path, 'train')

    all_video_path_list = []
    for per_video_name in sorted(os.listdir(root_dataset_path)):
        if 'sav' in per_video_name:
            per_video_dir_path = os.path.join(root_dataset_path,
                                              per_video_name)

            all_path_name = per_video_dir_path.split('/')
            all_path_name = all_path_name[all_path_name.index(root_name) +
                                          1:-1]
            all_video_path_list.append(
                [per_video_name, per_video_dir_path, all_path_name])

    print('1111', len(all_video_path_list), all_video_path_list[0])

    # for per_video_name, per_video_dir_path, all_path_name in tqdm(
    #         all_video_path_list):
    #     save_per_video_dir = os.path.join(save_path, *all_path_name,
    #                                       per_video_name)
    #     os.makedirs(save_per_video_dir, exist_ok=True)

    #     # 读取所有图像文件
    #     per_video_all_image_names = []
    #     for file_name in sorted(os.listdir(per_video_dir_path)):
    #         if file_name.lower().endswith(('.jpg')):
    #             per_video_all_image_names.append(file_name)

    #     first_frame_path = os.path.join(per_video_dir_path,
    #                                     per_video_all_image_names[0])
    #     first_frame = cv2.imdecode(
    #         np.fromfile(first_frame_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    #     origin_height, origin_width = first_frame.shape[0], first_frame.shape[
    #         1]

    #     # 判断是否需要resize
    #     max_side = max(origin_height, origin_width)
    #     need_resize = max_side > 1080

    #     if need_resize:
    #         factor = 1080.0 / max(origin_height, origin_width)
    #         resize_h, resize_w = int(round(origin_height * factor)), int(
    #             round(origin_width * factor))
    #     else:
    #         resize_h = origin_height
    #         resize_w = origin_width

    #     # 处理JSON标注文件
    #     per_video_json_file_name = per_video_name + "_manual.json"
    #     per_video_json_path = os.path.join(per_video_dir_path,
    #                                        per_video_json_file_name)

    #     # 读取JSON文件
    #     with open(per_video_json_path, encoding='utf-8') as f:
    #         per_video_json_data = json.load(f)

    #     per_video_json_data['video_height'] = float(resize_h)
    #     per_video_json_data['video_width'] = float(resize_w)
    #     per_video_json_data['video_resolution'] = float(resize_h * resize_w)

    #     if need_resize:
    #         per_video_annotations = per_video_json_data['masklet']
    #         new_per_video_annotations = []

    #         for per_frame_annotations in tqdm(per_video_annotations):
    #             new_per_frame_annotations = []
    #             for per_object_annotation in per_frame_annotations:
    #                 per_mask = mask_utils.decode(per_object_annotation)
    #                 resized_mask = cv2.resize(per_mask, (resize_w, resize_h),
    #                                           interpolation=cv2.INTER_NEAREST)
    #                 resized_mask[resized_mask > 0] = 1
    #                 resized_rle = mask_utils.encode(
    #                     np.asfortranarray(resized_mask))
    #                 resized_rle['counts'] = resized_rle['counts'].decode(
    #                     'utf-8')

    #                 new_per_frame_annotations.append(resized_rle)

    #             new_per_video_annotations.append(new_per_frame_annotations)

    #         per_video_json_data['masklet'] = new_per_video_annotations

    #     # 保存JSON
    #     save_per_video_json_path = os.path.join(save_per_video_dir,
    #                                             per_video_json_file_name)
    #     with open(save_per_video_json_path, 'w',
    #               encoding='utf-8') as json_file:
    #         json.dump(per_video_json_data, json_file, ensure_ascii=False)

    #     # 处理所有图像
    #     for per_image_name in per_video_all_image_names:
    #         per_image_path = os.path.join(per_video_dir_path, per_image_name)
    #         per_image = cv2.imdecode(
    #             np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    #         if need_resize:
    #             per_image = cv2.resize(per_image, (resize_w, resize_h))

    #         save_per_image_path = os.path.join(save_per_video_dir,
    #                                            per_image_name)
    #         cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    # 准备多进程参数
    args_list = [(per_video_name, per_video_dir_path, all_path_name, save_path)
                 for per_video_name, per_video_dir_path, all_path_name in
                 all_video_path_list]

    # 使用多进程处理
    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_single_video, args_list),
                 total=len(args_list)))


if __name__ == "__main__":
    root_dir = r'/root/autodl-tmp/video_interactive_segmentation_dataset_new/sav_'
    save_dir = r'/root/autodl-tmp/video_interactive_segmentation_dataset_new2/sav_'

    for i in tqdm(range(10, 20)):
        houzhui = f'{i:03d}'

        root_path = root_dir + houzhui
        save_path = save_dir + houzhui

        print(root_path)
        print(save_path)

        process_videos(root_path=root_path, save_path=save_path)
