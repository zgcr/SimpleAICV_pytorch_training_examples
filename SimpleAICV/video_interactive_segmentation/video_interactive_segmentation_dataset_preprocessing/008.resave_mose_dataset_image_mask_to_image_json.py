import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils
from multiprocessing import Pool


def get_frame_invalid_flag(frame_masks, min_ratio=0.0001, max_ratio=0.9):
    if len(frame_masks) == 0:
        invalid_flag = True
        return invalid_flag

    invalid_flag = False
    valid_object_count = 0
    for per_mask in frame_masks:
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
    """处理单个视频的函数"""
    per_video_name, per_video_image_dir_path, per_video_mask_dir_path, per_video_image_path_list, per_video_mask_path_list, save_path, root_name = args

    try:
        assert len(per_video_image_path_list) == len(per_video_mask_path_list)

        first_frame_mask_path = per_video_mask_path_list[0]
        first_frame_mask = np.array(
            Image.open(first_frame_mask_path).convert('P'), dtype=np.uint8)

        all_object_ids = np.unique(first_frame_mask)
        all_object_ids = all_object_ids.tolist()
        # 0为背景,去掉0 id
        while 0 in all_object_ids:
            all_object_ids.remove(0)

        per_video_all_object_masks = []
        for per_frame_mask_path in per_video_mask_path_list:
            per_frame_all_object_masks = []
            per_frame_mask = np.array(
                Image.open(per_frame_mask_path).convert('P'), dtype=np.uint8)
            for per_object_id in all_object_ids:
                per_object_mask = (per_frame_mask == per_object_id).astype(
                    np.uint8)
                per_frame_all_object_masks.append(per_object_mask)

            per_video_all_object_masks.append(per_frame_all_object_masks)

        assert len(per_video_all_object_masks) == len(
            per_video_image_path_list)
        assert len(per_video_all_object_masks[0]) == len(all_object_ids)

        # 标记无效帧
        frame_nums = len(per_video_image_path_list)
        frames_invalid_flag = [False] * frame_nums

        for idx in range(frame_nums):
            per_frame_all_object_masks = per_video_all_object_masks[idx]
            per_frame_invalid_flag = get_frame_invalid_flag(
                per_frame_all_object_masks, min_ratio=0.0001, max_ratio=0.9)
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
            return 0, 1  # invalid_video

        # 更新image_paths和mask_paths
        valid_image_path_list = per_video_image_path_list[start_idx:end_idx +
                                                          1]
        valid_per_video_all_object_masks = per_video_all_object_masks[
            start_idx:end_idx + 1]

        if len(valid_image_path_list) < len(per_video_image_path_list):
            print(
                f'video_name:{per_video_name}, frame_num:{len(per_video_image_path_list)}, filtered_frame_num: {len(valid_image_path_list)}'
            )

        # 创建保存目录
        save_per_video_dir = os.path.join(save_path, per_video_name)
        os.makedirs(save_per_video_dir, exist_ok=True)

        first_frame_image_path = valid_image_path_list[0]
        first_frame_image = cv2.imdecode(
            np.fromfile(first_frame_image_path, dtype=np.uint8),
            cv2.IMREAD_COLOR)

        origin_height, origin_width = first_frame_image.shape[
            0], first_frame_image.shape[1]

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

        # 重新编号并保存图像
        for idx, image_path in enumerate(valid_image_path_list):
            per_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)

            if need_resize:
                per_image = cv2.resize(per_image, (resize_w, resize_h))

            # 新的图像名称,从00000开始
            new_image_name = f'{per_video_name}_{idx:05d}.jpg'
            save_per_image_path = os.path.join(save_per_video_dir,
                                               new_image_name)
            cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

        masklet = []
        for frame_idx, per_frame_masks in enumerate(
                valid_per_video_all_object_masks):
            per_frame_rle_masks_list = []
            for per_object_mask in per_frame_masks:
                if need_resize:
                    resized_per_object_mask = cv2.resize(
                        per_object_mask, (resize_w, resize_h),
                        interpolation=cv2.INTER_NEAREST)
                else:
                    resized_per_object_mask = per_object_mask

                resized_rle = mask_utils.encode(
                    np.asfortranarray(resized_per_object_mask))
                resized_rle['counts'] = resized_rle['counts'].decode('utf-8')
                per_frame_rle_masks_list.append(resized_rle)
            masklet.append(per_frame_rle_masks_list)

        # 创建JSON数据结构
        per_video_json_data = {
            'video_id': per_video_name,
            'video_height': float(resize_h),
            'video_width': float(resize_w),
            'video_resolution': float(resize_h * resize_w),
            'video_environment': root_name,
            'video_split': 'train',
            'masklet': masklet,
        }

        # 保存JSON文件
        json_filename = f'{per_video_name}_manual.json'
        save_per_video_json_path = os.path.join(save_per_video_dir,
                                                json_filename)
        with open(save_per_video_json_path, 'w',
                  encoding='utf-8') as json_file:
            json.dump(per_video_json_data, json_file, ensure_ascii=False)

        # valid_video
        return 1, 0
    except Exception as e:
        print(f"Error processing {per_video_name}: {str(e)}")

        # invalid_video
        return 0, 1


def process_videos(root_path, save_path):
    root_name = root_path.split('/')[-2]

    save_path = os.path.join(save_path, 'train')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_dataset_path = os.path.join(root_path, 'JPEGImages')

    all_video_path_list = []
    for per_video_name in sorted(os.listdir(root_dataset_path)):
        per_video_image_dir_path = os.path.join(root_dataset_path,
                                                per_video_name)

        mask_dataset_path = os.path.join(root_path, 'Annotations')
        per_video_mask_dir_path = os.path.join(mask_dataset_path,
                                               per_video_name)

        per_video_image_path_list = []
        for per_image_name in sorted(os.listdir(per_video_image_dir_path)):
            if '.jpg' in per_image_name:
                per_image_path = os.path.join(per_video_image_dir_path,
                                              per_image_name)
                per_video_image_path_list.append(per_image_path)

        per_video_mask_path_list = []
        for per_mask_name in sorted(os.listdir(per_video_mask_dir_path)):
            if '.png' in per_mask_name:
                per_mask_path = os.path.join(per_video_mask_dir_path,
                                             per_mask_name)
                per_video_mask_path_list.append(per_mask_path)

        assert len(per_video_image_path_list) == len(
            per_video_mask_path_list) >= 1

        all_video_path_list.append([
            per_video_name,
            per_video_image_dir_path,
            per_video_mask_dir_path,
            per_video_image_path_list,
            per_video_mask_path_list,
        ])

    print('1111', len(all_video_path_list), all_video_path_list[0])

    # valid_video_count, invalid_video_count = 0, 0
    # for per_video_name, per_video_image_dir_path, per_video_mask_dir_path, per_video_image_path_list, per_video_mask_path_list in tqdm(
    #         all_video_path_list):
    #     assert len(per_video_image_path_list) == len(per_video_mask_path_list)

    #     first_frame_mask_path = per_video_mask_path_list[0]
    #     first_frame_mask = np.array(
    #         Image.open(first_frame_mask_path).convert('P'), dtype=np.uint8)

    #     all_object_ids = np.unique(first_frame_mask)
    #     all_object_ids = all_object_ids.tolist()
    #     # 0为背景,去掉0 id
    #     while 0 in all_object_ids:
    #         all_object_ids.remove(0)

    #     per_video_all_object_masks = []
    #     for per_frame_mask_path in per_video_mask_path_list:
    #         per_frame_all_object_masks = []
    #         per_frame_mask = np.array(
    #             Image.open(per_frame_mask_path).convert('P'), dtype=np.uint8)
    #         for per_object_id in all_object_ids:
    #             per_object_mask = (per_frame_mask == per_object_id).astype(
    #                 np.uint8)
    #             per_frame_all_object_masks.append(per_object_mask)

    #         per_video_all_object_masks.append(per_frame_all_object_masks)

    #     assert len(per_video_all_object_masks) == len(
    #         per_video_image_path_list)
    #     assert len(per_video_all_object_masks[0]) == len(all_object_ids)

    #     # 标记无效帧
    #     frame_nums = len(per_video_image_path_list)
    #     frames_invalid_flag = [False] * frame_nums

    #     for idx in range(frame_nums):
    #         per_frame_all_object_masks = per_video_all_object_masks[idx]
    #         per_frame_invalid_flag = get_frame_invalid_flag(
    #             per_frame_all_object_masks, min_ratio=0.0001, max_ratio=0.9)
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
    #     valid_per_video_all_object_masks = per_video_all_object_masks[
    #         start_idx:end_idx + 1]

    #     if len(valid_image_path_list) < len(per_video_image_path_list):
    #         print(
    #             f'video_name:{per_video_name}, frame_num:{len(per_video_image_path_list)}, filtered_frame_num: {len(valid_image_path_list)}'
    #         )

    #     # 创建保存目录
    #     save_per_video_dir = os.path.join(save_path, per_video_name)
    #     os.makedirs(save_per_video_dir, exist_ok=True)

    #     first_frame_image_path = valid_image_path_list[0]
    #     first_frame_image = cv2.imdecode(
    #         np.fromfile(first_frame_image_path, dtype=np.uint8),
    #         cv2.IMREAD_COLOR)

    #     origin_height, origin_width = first_frame_image.shape[
    #         0], first_frame_image.shape[1]

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

    #     # 重新编号并保存图像
    #     for idx, image_path in enumerate(valid_image_path_list):
    #         per_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
    #                                  cv2.IMREAD_COLOR)

    #         if need_resize:
    #             per_image = cv2.resize(per_image, (resize_w, resize_h))

    #         # 新的图像名称,从00000开始
    #         new_image_name = f'{per_video_name}_{idx:05d}.jpg'
    #         save_per_image_path = os.path.join(save_per_video_dir,
    #                                            new_image_name)
    #         cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    #     masklet = []
    #     for frame_idx, per_frame_masks in enumerate(
    #             valid_per_video_all_object_masks):
    #         per_frame_rle_masks_list = []
    #         for per_object_mask in per_frame_masks:
    #             if need_resize:
    #                 resized_per_object_mask = cv2.resize(
    #                     per_object_mask, (resize_w, resize_h),
    #                     interpolation=cv2.INTER_NEAREST)
    #             else:
    #                 resized_per_object_mask = per_object_mask

    #             resized_rle = mask_utils.encode(
    #                 np.asfortranarray(resized_per_object_mask))
    #             resized_rle['counts'] = resized_rle['counts'].decode('utf-8')
    #             per_frame_rle_masks_list.append(resized_rle)
    #         masklet.append(per_frame_rle_masks_list)

    #     # 创建JSON数据结构
    #     per_video_json_data = {
    #         'video_id': per_video_name,
    #         'video_height': float(resize_h),
    #         'video_width': float(resize_w),
    #         'video_resolution': float(resize_h * resize_w),
    #         'video_environment': root_name,
    #         'video_split': 'train',
    #         'masklet': masklet,
    #     }

    #     # 保存JSON文件
    #     json_filename = f'{per_video_name}_manual.json'
    #     save_per_video_json_path = os.path.join(save_per_video_dir,
    #                                             json_filename)
    #     with open(save_per_video_json_path, 'w',
    #               encoding='utf-8') as json_file:
    #         json.dump(per_video_json_data, json_file, ensure_ascii=False)

    #     valid_video_count += 1

    # print(valid_video_count, invalid_video_count)

    process_args = []
    for per_video_name, per_video_image_dir_path, per_video_mask_dir_path, per_video_image_path_list, per_video_mask_path_list in all_video_path_list:
        process_args.append(
            (per_video_name, per_video_image_dir_path, per_video_mask_dir_path,
             per_video_image_path_list, per_video_mask_path_list, save_path,
             root_name))

    valid_video_count, invalid_video_count = 0, 0
    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(process_single_video, process_args),
                 total=len(process_args)))

    for valid, invalid in results:
        valid_video_count += valid
        invalid_video_count += invalid

    print(valid_video_count, invalid_video_count)


if __name__ == "__main__":
    root_path = r'/root/autodl-tmp/MOSEv1/train'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/MOSEv1'
    process_videos(root_path=root_path, save_path=save_path)

    root_path = r'/root/autodl-tmp/MOSEv2/train'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/MOSEv2'
    process_videos(root_path=root_path, save_path=save_path)
