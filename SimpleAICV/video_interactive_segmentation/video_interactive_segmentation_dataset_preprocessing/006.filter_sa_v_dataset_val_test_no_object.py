import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


def get_frame_invalid_flag(frame_mask_path_list,
                           min_ratio=0.0001,
                           max_ratio=0.9):
    if len(frame_mask_path_list) == 0:
        invalid_flag = True
        return invalid_flag

    invalid_flag = False
    valid_object_count = 0
    for per_mask_path in frame_mask_path_list:
        per_mask = np.array(Image.open(per_mask_path).convert('L'),
                            dtype=np.uint8)
        per_mask[per_mask > 0] = 1
        foreground_area = np.count_nonzero(per_mask)
        area_ratio = foreground_area / float(
            per_mask.shape[0] * per_mask.shape[1])

        if min_ratio <= area_ratio <= max_ratio:
            valid_object_count += 1

    if valid_object_count == 0:
        invalid_flag = True

    return invalid_flag


def process_single_video(args):
    """处理单个视频的函数，用于多进程"""
    per_video_name, per_video_dir_path, all_path_name, save_path = args

    per_video_image_dir_path = os.path.join(per_video_dir_path, 'image')
    per_video_object_mask_dir_path = os.path.join(per_video_dir_path, 'object')

    per_video_image_path_list = []
    for per_image_name in sorted(os.listdir(per_video_image_dir_path)):
        per_image_path = os.path.join(per_video_image_dir_path, per_image_name)
        if '.jpg' in per_image_name:
            per_video_image_path_list.append(per_image_path)

    assert len(per_video_image_path_list) >= 1

    per_video_object_mask_path_list = []
    for per_object_name in sorted(os.listdir(per_video_object_mask_dir_path)):
        per_object_mask_dir_path = os.path.join(per_video_object_mask_dir_path,
                                                per_object_name)
        per_object_mask_path_list = []
        for per_object_mask_name in sorted(
                os.listdir(per_object_mask_dir_path)):
            per_object_mask_path = os.path.join(per_object_mask_dir_path,
                                                per_object_mask_name)
            if '.png' in per_object_mask_name:
                per_object_mask_path_list.append(per_object_mask_path)

        per_video_object_mask_path_list.append(per_object_mask_path_list)

    assert len(per_video_object_mask_path_list) >= 1

    for per_object_mask_path_list in per_video_object_mask_path_list:
        assert len(per_object_mask_path_list) == len(per_video_image_path_list)

    # 标记无效帧
    frame_nums = len(per_video_image_path_list)
    frames_invalid_flag = [False] * frame_nums

    for idx in range(frame_nums):
        # 收集该帧所有对象的mask路径
        per_frame_mask_paths = []
        for per_object_mask_path_list in per_video_object_mask_path_list:
            per_frame_mask_paths.append(per_object_mask_path_list[idx])

        per_frame_invalid_flag = get_frame_invalid_flag(per_frame_mask_paths,
                                                        min_ratio=0.0001,
                                                        max_ratio=0.9)
        frames_invalid_flag[idx] = per_frame_invalid_flag

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
        return 0  # invalid video

    # 更新image_paths和mask_paths
    valid_image_path_list = per_video_image_path_list[start_idx:end_idx + 1]
    valid_object_mask_path_list = []
    for per_object_mask_path_list in per_video_object_mask_path_list:
        valid_object_mask_path_list.append(
            per_object_mask_path_list[start_idx:end_idx + 1])

    if len(valid_image_path_list) < len(per_video_image_path_list):
        print(
            f'video_name:{per_video_name}, frame_num:{len(per_video_image_path_list)}, filtered_frame_num: {len(valid_image_path_list)}'
        )

    # 创建保存目录
    save_per_video_dir = os.path.join(save_path, *all_path_name,
                                      per_video_name)
    save_per_video_image_dir = os.path.join(save_per_video_dir, 'image')
    save_per_video_object_dir = os.path.join(save_per_video_dir, 'object')

    os.makedirs(save_per_video_image_dir, exist_ok=True)
    os.makedirs(save_per_video_object_dir, exist_ok=True)

    # 重新编号并保存图像
    for idx, image_path in enumerate(valid_image_path_list):
        per_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)

        image_name = os.path.basename(image_path)
        image_name_prefix = image_name.rsplit('_', 1)[0]
        # 新的图像名称,从00000开始
        new_image_name = f'{image_name_prefix}_{idx:05d}.jpg'
        save_per_image_path = os.path.join(save_per_video_image_dir,
                                           new_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    # 重新编号并保存所有对象的mask
    for obj_idx, per_object_valid_mask_paths in enumerate(
            valid_object_mask_path_list):
        # 获取对象名称
        object_name = os.path.basename(
            os.path.dirname(per_object_valid_mask_paths[0]))
        save_per_object_dir = os.path.join(save_per_video_object_dir,
                                           object_name)
        os.makedirs(save_per_object_dir, exist_ok=True)

        for idx, mask_path in enumerate(per_object_valid_mask_paths):
            per_mask = np.array(Image.open(mask_path).convert('L'),
                                dtype=np.uint8)
            per_mask[per_mask > 0] = 255

            mask_name = os.path.basename(mask_path)
            # 修改: 分离出 video_name, frame_idx, object_id
            # 例如: sav_000208_00004_000.png -> sav_000208, 00004, 000
            mask_name_without_ext = mask_name.rsplit('.', 1)[0]
            parts = mask_name_without_ext.rsplit('_', 2)
            # parts = [sav_000208, 00004, 000]
            video_prefix = parts[0]  # sav_000208
            object_id = parts[2]  # 000

            # 新的mask名称: video_name_新帧号_object_id.png
            new_mask_name = f'{video_prefix}_{idx:05d}_{object_id}.png'
            save_per_mask_path = os.path.join(save_per_object_dir,
                                              new_mask_name)
            cv2.imencode('.png', per_mask)[1].tofile(save_per_mask_path)

    return 1


def process_videos(root_path, save_path):
    root_name = root_path.split('/')[-1]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_video_path_list = []
    for per_video_name in sorted(os.listdir(root_path)):
        if 'sav' in per_video_name:
            per_video_dir_path = os.path.join(root_path, per_video_name)

            all_path_name = per_video_dir_path.split('/')
            all_path_name = all_path_name[all_path_name.index(root_name) +
                                          1:-1]
            all_video_path_list.append(
                [per_video_name, per_video_dir_path, all_path_name])

    print('1111', len(all_video_path_list), all_video_path_list[0])

    # valid_video_count, invalid_video_count = 0, 0
    # for per_video_name, per_video_dir_path, all_path_name in tqdm(
    #         all_video_path_list):
    #     per_video_image_dir_path = os.path.join(per_video_dir_path, 'image')
    #     per_video_object_mask_dir_path = os.path.join(per_video_dir_path,
    #                                                   'object')

    #     per_video_image_path_list = []
    #     for per_image_name in sorted(os.listdir(per_video_image_dir_path)):
    #         per_image_path = os.path.join(per_video_image_dir_path,
    #                                       per_image_name)
    #         if '.jpg' in per_image_name:
    #             per_video_image_path_list.append(per_image_path)

    #     assert len(per_video_image_path_list) >= 1

    #     per_video_object_mask_path_list = []
    #     for per_object_name in sorted(
    #             os.listdir(per_video_object_mask_dir_path)):
    #         per_object_mask_dir_path = os.path.join(
    #             per_video_object_mask_dir_path, per_object_name)
    #         per_object_mask_path_list = []
    #         for per_object_mask_name in sorted(
    #                 os.listdir(per_object_mask_dir_path)):
    #             per_object_mask_path = os.path.join(per_object_mask_dir_path,
    #                                                 per_object_mask_name)
    #             if '.png' in per_object_mask_name:
    #                 per_object_mask_path_list.append(per_object_mask_path)

    #         per_video_object_mask_path_list.append(per_object_mask_path_list)

    #     assert len(per_video_object_mask_path_list) >= 1

    #     for per_object_mask_path_list in per_video_object_mask_path_list:
    #         assert len(per_object_mask_path_list) == len(
    #             per_video_image_path_list)

    #     # 标记无效帧
    #     frame_nums = len(per_video_image_path_list)
    #     frames_invalid_flag = [False] * frame_nums

    #     for idx in range(frame_nums):
    #         # 收集该帧所有对象的mask路径
    #         per_frame_mask_paths = []
    #         for per_object_mask_path_list in per_video_object_mask_path_list:
    #             per_frame_mask_paths.append(per_object_mask_path_list[idx])

    #         per_frame_invalid_flag = get_frame_invalid_flag(
    #             per_frame_mask_paths, min_ratio=0.0001, max_ratio=0.9)
    #         frames_invalid_flag[idx] = per_frame_invalid_flag

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

    #     # 更新image_paths和mask_paths
    #     valid_image_path_list = per_video_image_path_list[start_idx:end_idx +
    #                                                       1]
    #     valid_object_mask_path_list = []
    #     for per_object_mask_path_list in per_video_object_mask_path_list:
    #         valid_object_mask_path_list.append(
    #             per_object_mask_path_list[start_idx:end_idx + 1])

    #     if len(valid_image_path_list) < len(per_video_image_path_list):
    #         print(
    #             f'video_name:{per_video_name}, frame_num:{len(per_video_image_path_list)}, filtered_frame_num: {len(valid_image_path_list)}'
    #         )

    #     # 创建保存目录
    #     save_per_video_dir = os.path.join(save_path, *all_path_name,
    #                                       per_video_name)
    #     save_per_video_image_dir = os.path.join(save_per_video_dir, 'image')
    #     save_per_video_object_dir = os.path.join(save_per_video_dir, 'object')

    #     os.makedirs(save_per_video_image_dir, exist_ok=True)
    #     os.makedirs(save_per_video_object_dir, exist_ok=True)

    #     # 重新编号并保存图像
    #     for idx, image_path in enumerate(valid_image_path_list):
    #         per_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
    #                                  cv2.IMREAD_COLOR)

    #         image_name = os.path.basename(image_path)
    #         image_name_prefix = image_name.rsplit('_', 1)[0]
    #         # 新的图像名称,从00000开始
    #         new_image_name = f'{image_name_prefix}_{idx:05d}.jpg'
    #         save_per_image_path = os.path.join(save_per_video_image_dir,
    #                                            new_image_name)
    #         cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    #     # 重新编号并保存所有对象的mask
    #     for obj_idx, per_object_valid_mask_paths in enumerate(
    #             valid_object_mask_path_list):
    #         # 获取对象名称
    #         object_name = os.path.basename(
    #             os.path.dirname(per_object_valid_mask_paths[0]))
    #         save_per_object_dir = os.path.join(save_per_video_object_dir,
    #                                            object_name)
    #         os.makedirs(save_per_object_dir, exist_ok=True)

    #         for idx, mask_path in enumerate(per_object_valid_mask_paths):
    #             per_mask = np.array(Image.open(mask_path).convert('L'),
    #                                 dtype=np.uint8)
    #             per_mask[per_mask > 0] = 255

    #             mask_name = os.path.basename(mask_path)
    #             # 修改: 分离出 video_name, frame_idx, object_id
    #             # 例如: sav_000208_00004_000.png -> sav_000208, 00004, 000
    #             mask_name_without_ext = mask_name.rsplit('.', 1)[0]
    #             parts = mask_name_without_ext.rsplit('_', 2)
    #             # parts = [sav_000208, 00004, 000]
    #             video_prefix = parts[0]  # sav_000208
    #             object_id = parts[2]  # 000

    #             # 新的mask名称: video_name_新帧号_object_id.png
    #             new_mask_name = f'{video_prefix}_{idx:05d}_{object_id}.png'
    #             save_per_mask_path = os.path.join(save_per_object_dir,
    #                                               new_mask_name)
    #             cv2.imencode('.png', per_mask)[1].tofile(save_per_mask_path)

    #     valid_video_count += 1

    # print(valid_video_count, invalid_video_count)

    args_list = [(per_video_name, per_video_dir_path, all_path_name, save_path)
                 for per_video_name, per_video_dir_path, all_path_name in
                 all_video_path_list]

    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(process_single_video, args_list),
                 total=len(args_list)))

    # 统计结果
    valid_video_count = sum(results)
    invalid_video_count = len(results) - valid_video_count

    print(valid_video_count, invalid_video_count)


if __name__ == "__main__":
    root_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_val_new'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_val_new2'

    print(root_path)
    print(save_path)

    process_videos(root_path=root_path, save_path=save_path)

    root_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_test_new'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_test_new2'

    print(root_path)
    print(save_path)

    process_videos(root_path=root_path, save_path=save_path)
