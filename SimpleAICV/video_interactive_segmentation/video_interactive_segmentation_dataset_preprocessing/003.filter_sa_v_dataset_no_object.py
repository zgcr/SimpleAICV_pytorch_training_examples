import os
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
import json
from multiprocessing import Pool
from functools import partial


def get_frame_invalid_flag(frame_annotations, min_ratio=0.0001, max_ratio=0.9):
    if len(frame_annotations) == 0:
        invalid_flag = True
        return invalid_flag

    invalid_flag = False

    # 检查每个object的mask是否有效
    valid_object_count = 0
    for annotation in frame_annotations:
        mask = mask_utils.decode(annotation)
        mask[mask > 0] = 1
        foreground_area = np.count_nonzero(mask)
        area_ratio = foreground_area / float(mask.shape[0] * mask.shape[1])

        if min_ratio <= area_ratio <= max_ratio:
            valid_object_count += 1

    if valid_object_count == 0:
        invalid_flag = True

    return invalid_flag


def process_single_video(args, save_path, root_name):
    per_video_name, per_video_dir_path, all_path_name = args

    # 读取所有图像文件
    per_video_all_image_names = []
    for file_name in sorted(os.listdir(per_video_dir_path)):
        if file_name.lower().endswith(('.jpg')):
            per_video_all_image_names.append(file_name)

    # 处理JSON标注文件
    per_video_json_file_name = per_video_name + "_manual.json"
    per_video_json_path = os.path.join(per_video_dir_path,
                                       per_video_json_file_name)

    # 读取JSON文件
    with open(per_video_json_path, encoding='utf-8') as f:
        per_video_json_data = json.load(f)

    per_video_annotations = per_video_json_data['masklet']

    assert len(per_video_all_image_names) == len(per_video_annotations)

    # 标记无效帧
    frames_invalid_flag = [False] * len(per_video_annotations)

    for idx in range(len(per_video_annotations)):
        per_frame_annotations = per_video_annotations[idx]
        per_frame_invalid_flag = get_frame_invalid_flag(per_frame_annotations,
                                                        min_ratio=0.0001,
                                                        max_ratio=0.9)
        frames_invalid_flag[idx] = per_frame_invalid_flag

    frame_nums = len(per_video_annotations)

    # 从头部删除连续的无效帧
    start_idx = 0
    while start_idx < frame_nums and frames_invalid_flag[start_idx]:
        start_idx += 1

    # 从尾部删除连续的无效帧
    end_idx = frame_nums - 1
    while end_idx >= 0 and frames_invalid_flag[end_idx]:
        end_idx -= 1

    # 检查剩余帧数是否满足要求
    if end_idx - start_idx + 1 < 8:
        return 0, 1  # (valid_count, invalid_count)

    # 更新annotations和image_names
    valid_annotations = per_video_annotations[start_idx:end_idx + 1]
    valid_image_names = per_video_all_image_names[start_idx:end_idx + 1]

    assert len(valid_annotations) == len(valid_image_names)

    if len(valid_annotations) < len(per_video_annotations):
        print(
            f'video_name:{per_video_name}, frame_num:{len(per_video_annotations)}, filtered_frame_num: {len(valid_annotations)}'
        )

    # 创建保存目录
    save_per_video_dir = os.path.join(save_path, *all_path_name,
                                      per_video_name)
    os.makedirs(save_per_video_dir, exist_ok=True)

    # 创建新字典
    new_video_json_data = {
        'video_id': per_video_json_data['video_id'],
        'video_height': per_video_json_data['video_height'],
        'video_width': per_video_json_data['video_width'],
        'video_resolution': per_video_json_data['video_resolution'],
        'video_environment': per_video_json_data['video_environment'],
        'video_split': per_video_json_data['video_split'],
        'masklet': valid_annotations,
    }

    # 保存新的JSON文件
    save_per_video_json_path = os.path.join(save_per_video_dir,
                                            per_video_json_file_name)
    with open(save_per_video_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(new_video_json_data, json_file, ensure_ascii=False)

    # 重新编号并保存图像
    for idx, image_name in enumerate(valid_image_names):
        per_image_path = os.path.join(per_video_dir_path, image_name)
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        image_name_prefix = image_name.rsplit('_', 1)[0]
        # 新的图像名称,从00000开始
        new_image_name = f'{image_name_prefix}_{idx:05d}.jpg'
        save_per_image_path = os.path.join(save_per_video_dir, new_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    return 1, 0


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

    # valid_video_count, invalid_video_count = 0, 0
    # for per_video_name, per_video_dir_path, all_path_name in tqdm(
    #         all_video_path_list):
    #     # 读取所有图像文件
    #     per_video_all_image_names = []
    #     for file_name in sorted(os.listdir(per_video_dir_path)):
    #         if file_name.lower().endswith(('.jpg')):
    #             per_video_all_image_names.append(file_name)

    #     # 处理JSON标注文件
    #     per_video_json_file_name = per_video_name + "_manual.json"
    #     per_video_json_path = os.path.join(per_video_dir_path,
    #                                        per_video_json_file_name)

    #     # 读取JSON文件
    #     with open(per_video_json_path, encoding='utf-8') as f:
    #         per_video_json_data = json.load(f)

    #     per_video_annotations = per_video_json_data['masklet']

    #     assert len(per_video_all_image_names) == len(per_video_annotations)

    #     # 标记无效帧
    #     frames_invalid_flag = [False] * len(per_video_annotations)

    #     for idx in range(len(per_video_annotations)):
    #         per_frame_annotations = per_video_annotations[idx]
    #         per_frame_invalid_flag = get_frame_invalid_flag(
    #             per_frame_annotations, min_ratio=0.0001, max_ratio=0.9)
    #         frames_invalid_flag[idx] = per_frame_invalid_flag

    #     frame_nums = len(per_video_annotations)

    #     # 从头部删除连续的无效帧
    #     start_idx = 0
    #     while start_idx < frame_nums and frames_invalid_flag[start_idx]:
    #         start_idx += 1

    #     # 从尾部删除连续的无效帧
    #     end_idx = frame_nums - 1
    #     while end_idx >= 0 and frames_invalid_flag[end_idx]:
    #         end_idx -= 1

    #     # 检查剩余帧数是否满足要求
    #     if end_idx - start_idx + 1 < 8:
    #         invalid_video_count += 1
    #         continue

    #     # 更新annotations和image_names
    #     valid_annotations = per_video_annotations[start_idx:end_idx + 1]
    #     valid_image_names = per_video_all_image_names[start_idx:end_idx + 1]

    #     assert len(valid_annotations) == len(valid_image_names)

    #     if len(valid_annotations) < len(per_video_annotations):
    #         print(
    #             f'video_name:{per_video_name}, frame_num:{len(per_video_annotations)}, filtered_frame_num: {len(valid_annotations)}'
    #         )

    #     # 创建保存目录
    #     save_per_video_dir = os.path.join(save_path, *all_path_name,
    #                                       per_video_name)
    #     os.makedirs(save_per_video_dir, exist_ok=True)

    #     # 创建新字典
    #     new_video_json_data = {
    #         'video_id': per_video_json_data['video_id'],
    #         'video_height': per_video_json_data['video_height'],
    #         'video_width': per_video_json_data['video_width'],
    #         'video_resolution': per_video_json_data['video_resolution'],
    #         'video_environment': per_video_json_data['video_environment'],
    #         'video_split': per_video_json_data['video_split'],
    #         'masklet': valid_annotations,
    #     }

    #     # 保存新的JSON文件
    #     save_per_video_json_path = os.path.join(save_per_video_dir,
    #                                             per_video_json_file_name)
    #     with open(save_per_video_json_path, 'w',
    #               encoding='utf-8') as json_file:
    #         json.dump(new_video_json_data, json_file, ensure_ascii=False)

    #     # 重新编号并保存图像
    #     for idx, image_name in enumerate(valid_image_names):
    #         per_image_path = os.path.join(per_video_dir_path, image_name)
    #         per_image = cv2.imdecode(
    #             np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    #         image_name_prefix = image_name.rsplit('_', 1)[0]
    #         # 新的图像名称,从00000开始
    #         new_image_name = f'{image_name_prefix}_{idx:05d}.jpg'
    #         save_per_image_path = os.path.join(save_per_video_dir,
    #                                            new_image_name)
    #         cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    #     valid_video_count += 1

    # print(valid_video_count, invalid_video_count)

    # 使用多进程处理
    with Pool(processes=16) as pool:
        process_func = partial(process_single_video,
                               save_path=save_path,
                               root_name=root_name)

        results = list(
            tqdm(pool.imap(process_func, all_video_path_list),
                 total=len(all_video_path_list),
                 desc="Processing videos"))

    # 汇总结果
    valid_video_count = sum(result[0] for result in results)
    invalid_video_count = sum(result[1] for result in results)

    print(valid_video_count, invalid_video_count)


if __name__ == "__main__":
    root_dir = r'/root/autodl-tmp/video_interactive_segmentation_dataset_new2/sav_'
    save_dir = r'/root/autodl-tmp/video_interactive_segmentation_dataset_new3/sav_'

    for i in tqdm(range(10, 20)):
        houzhui = f'{i:03d}'

        root_path = root_dir + houzhui
        save_path = save_dir + houzhui

        print(root_path)
        print(save_path)

        process_videos(root_path=root_path, save_path=save_path)
