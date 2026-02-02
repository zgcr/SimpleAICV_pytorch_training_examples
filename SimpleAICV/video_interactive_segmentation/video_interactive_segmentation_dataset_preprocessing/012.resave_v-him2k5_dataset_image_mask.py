import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


def get_frame_invalid_flag(per_frame_mask, min_ratio=0.0001, max_ratio=0.9):
    per_frame_mask = (per_frame_mask > 0.5).astype(np.uint8)

    if np.count_nonzero(per_frame_mask) == 0:
        invalid_flag = True
        return invalid_flag

    invalid_flag = False
    foreground_area = np.count_nonzero(per_frame_mask)
    area_ratio = foreground_area / float(
        per_frame_mask.shape[0] * per_frame_mask.shape[1])

    if area_ratio <= min_ratio or area_ratio >= max_ratio:
        invalid_flag = True

    return invalid_flag

def process_single_video(args):
    per_video_name, per_video_image_dir_path, per_video_image_path_list, per_video_mask_dir_path, per_video_mask_dir_path_list, object_mask_name_list, save_path, root_name = args
    
    local_valid_count = 0
    local_invalid_count = 0
    
    assert len(per_video_image_path_list) == len(
        per_video_mask_dir_path_list)

    for object_idx, per_object_mask_name in enumerate(
            object_mask_name_list):
        first_frame_image_path = per_video_image_path_list[0]
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

        per_video_all_images = []
        for per_frame_image_path in per_video_image_path_list:
            per_frame_image = cv2.imdecode(
                np.fromfile(per_frame_image_path, dtype=np.uint8),
                cv2.IMREAD_COLOR)

            if need_resize:
                per_frame_image = cv2.resize(per_frame_image,
                                             (resize_w, resize_h))

            per_video_all_images.append(per_frame_image)

        per_video_all_masks = []
        for per_frame_mask_dir_path in per_video_mask_dir_path_list:
            per_object_frame_mask_path = os.path.join(
                per_frame_mask_dir_path, per_object_mask_name)
            per_mask = np.array(
                Image.open(per_object_frame_mask_path).convert('L'),
                dtype=np.uint8)

            if need_resize:
                per_mask = cv2.resize(per_mask, (resize_w, resize_h),
                                      interpolation=cv2.INTER_NEAREST)

            # 0.9*255
            per_mask[per_mask >= 230] = 255
            # 0.1*255
            per_mask[per_mask <= 25] = 0
            per_mask = per_mask / 255.
            per_video_all_masks.append(per_mask)

        assert len(per_video_all_images) == len(per_video_all_masks)

        # 标记无效帧
        frame_nums = len(per_video_all_images)
        frames_invalid_flag = [False] * frame_nums

        for idx in range(frame_nums):
            per_frame_mask = per_video_all_masks[idx]
            per_frame_invalid_flag = get_frame_invalid_flag(
                per_frame_mask, min_ratio=0.0001, max_ratio=0.9)
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
            local_invalid_count += 1
            continue

        # 更新images和masks
        valid_per_video_all_images = per_video_all_images[
            start_idx:end_idx + 1]
        valid_per_video_all_masks = per_video_all_masks[start_idx:end_idx +
                                                        1]

        if len(valid_per_video_all_images) < len(per_video_all_images):
            print(
                f'video_name:{per_video_name}, object_mask_name:{per_object_mask_name}, frame_num:{len(per_video_all_images)}, filtered_frame_num: {len(valid_per_video_all_images)}'
            )

        # 创建保存目录
        save_per_video_dir = os.path.join(save_path,
                                          f'{root_name}_{per_video_name}_{per_object_mask_name.split('.')[0]}')
        os.makedirs(save_per_video_dir, exist_ok=True)

        save_per_video_image_dir = os.path.join(save_per_video_dir,
                                                'image')
        save_per_video_mask_dir = os.path.join(save_per_video_dir, 'mask')

        os.makedirs(save_per_video_image_dir, exist_ok=True)
        os.makedirs(save_per_video_mask_dir, exist_ok=True)

        # 重新编号并保存图像
        for idx, per_image in enumerate(valid_per_video_all_images):
            # 新的图像名称,从00000开始
            new_image_name = f'{root_name}_{per_video_name}_{per_object_mask_name.split('.')[0]}_{idx:05d}.jpg'
            save_per_image_path = os.path.join(save_per_video_image_dir,
                                               new_image_name)
            cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

        # 重新编号并保存mask
        for idx, per_mask in enumerate(valid_per_video_all_masks):
            # 新的mask名称,从00000开始
            per_mask = (per_mask * 255).astype(np.uint8)
            new_mask_name = f'{root_name}_{per_video_name}_{per_object_mask_name.split('.')[0]}_{idx:05d}.png'
            save_per_mask_path = os.path.join(save_per_video_mask_dir,
                                              new_mask_name)
            cv2.imencode('.png', per_mask)[1].tofile(save_per_mask_path)

        local_valid_count += 1
    
    return local_valid_count, local_invalid_count


def process_videos(root_path, save_path):
    root_name = root_path.split('/')[-1]

    save_path = os.path.join(save_path, 'train')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_dataset_path = os.path.join(root_path, 'fgr')

    all_video_path_list = []
    for per_video_name in sorted(os.listdir(root_dataset_path)):
        per_video_image_dir_path = os.path.join(root_dataset_path,
                                                per_video_name)

        per_video_image_path_list = []
        for per_image_name in sorted(os.listdir(per_video_image_dir_path)):
            if '.jpg' in per_image_name:
                per_image_path = os.path.join(per_video_image_dir_path,
                                              per_image_name)
                per_video_image_path_list.append(per_image_path)

        mask_dataset_path = os.path.join(root_path, 'pha')
        per_video_mask_dir_path = os.path.join(mask_dataset_path,
                                               per_video_name)

        per_video_mask_dir_path_list = []
        for per_mask_dir_name in sorted(os.listdir(per_video_mask_dir_path)):
            per_mask_dir_path = os.path.join(per_video_mask_dir_path,
                                             per_mask_dir_name)
            if os.path.isdir(per_mask_dir_path):
                per_video_mask_dir_path_list.append(per_mask_dir_path)

        assert len(per_video_image_path_list) == len(
            per_video_mask_dir_path_list) >= 1

        object_mask_name_list = []
        for per_object_mask_name in sorted(
                os.listdir(per_video_mask_dir_path_list[0])):
            object_mask_name_list.append(per_object_mask_name)

        assert len(object_mask_name_list) >= 1

        all_video_path_list.append([
            per_video_name,
            per_video_image_dir_path,
            per_video_image_path_list,
            per_video_mask_dir_path,
            per_video_mask_dir_path_list,
            object_mask_name_list,
        ])

    print(
        '1111',
        len(all_video_path_list),
        all_video_path_list[0][0],
        all_video_path_list[0][1],
        len(all_video_path_list[0][2]),
        all_video_path_list[0][3],
        len(all_video_path_list[0][4]),
        object_mask_name_list,
    )

    # valid_video_count, invalid_video_count = 0, 0
    # for per_video_name, per_video_image_dir_path, per_video_image_path_list, per_video_mask_dir_path, per_video_mask_dir_path_list, object_mask_name_list in tqdm(
    #         all_video_path_list):
    #     assert len(per_video_image_path_list) == len(
    #         per_video_mask_dir_path_list)

    #     for object_idx, per_object_mask_name in enumerate(
    #             object_mask_name_list):
    #         first_frame_image_path = per_video_image_path_list[0]
    #         first_frame_image = cv2.imdecode(
    #             np.fromfile(first_frame_image_path, dtype=np.uint8),
    #             cv2.IMREAD_COLOR)

    #         origin_height, origin_width = first_frame_image.shape[
    #             0], first_frame_image.shape[1]

    #         # 判断是否需要resize
    #         max_side = max(origin_height, origin_width)
    #         need_resize = max_side > 1080

    #         if need_resize:
    #             factor = 1080.0 / max(origin_height, origin_width)
    #             resize_h, resize_w = int(round(origin_height * factor)), int(
    #                 round(origin_width * factor))
    #         else:
    #             resize_h = origin_height
    #             resize_w = origin_width

    #         per_video_all_images = []
    #         for per_frame_image_path in per_video_image_path_list:
    #             per_frame_image = cv2.imdecode(
    #                 np.fromfile(per_frame_image_path, dtype=np.uint8),
    #                 cv2.IMREAD_COLOR)

    #             if need_resize:
    #                 per_frame_image = cv2.resize(per_frame_image,
    #                                              (resize_w, resize_h))

    #             per_video_all_images.append(per_frame_image)

    #         per_video_all_masks = []
    #         for per_frame_mask_dir_path in per_video_mask_dir_path_list:
    #             per_object_frame_mask_path = os.path.join(
    #                 per_frame_mask_dir_path, per_object_mask_name)
    #             per_mask = np.array(
    #                 Image.open(per_object_frame_mask_path).convert('L'),
    #                 dtype=np.uint8)

    #             if need_resize:
    #                 per_mask = cv2.resize(per_mask, (resize_w, resize_h),
    #                                       interpolation=cv2.INTER_NEAREST)

    #             # 0.9*255
    #             per_mask[per_mask >= 230] = 255
    #             # 0.1*255
    #             per_mask[per_mask <= 25] = 0
    #             per_mask = per_mask / 255.
    #             per_video_all_masks.append(per_mask)

    #         assert len(per_video_all_images) == len(per_video_all_masks)

    #         # 标记无效帧
    #         frame_nums = len(per_video_all_images)
    #         frames_invalid_flag = [False] * frame_nums

    #         for idx in range(frame_nums):
    #             per_frame_mask = per_video_all_masks[idx]
    #             per_frame_invalid_flag = get_frame_invalid_flag(
    #                 per_frame_mask, min_ratio=0.0001, max_ratio=0.9)
    #             frames_invalid_flag[idx] = per_frame_invalid_flag

    #         # 从头部删除连续的无效帧
    #         start_idx = 0
    #         while start_idx < frame_nums and frames_invalid_flag[start_idx]:
    #             start_idx += 1

    #         # 从尾部删除连续的无效帧
    #         end_idx = frame_nums - 1
    #         while end_idx >= 0 and frames_invalid_flag[end_idx]:
    #             end_idx -= 1

    #         # 检查剩余帧数是否满足要求
    #         if end_idx - start_idx + 1 < 8:
    #             invalid_video_count += 1
    #             continue

    #         # 更新images和masks
    #         valid_per_video_all_images = per_video_all_images[
    #             start_idx:end_idx + 1]
    #         valid_per_video_all_masks = per_video_all_masks[start_idx:end_idx +
    #                                                         1]

    #         if len(valid_per_video_all_images) < len(per_video_all_images):
    #             print(
    #                 f'video_name:{per_video_name}, object_mask_name:{per_object_mask_name}, frame_num:{len(per_video_all_images)}, filtered_frame_num: {len(valid_per_video_all_images)}'
    #             )

    #         # 创建保存目录
    #         save_per_video_dir = os.path.join(save_path,
    #                                           f'{root_name}_{per_video_name}_{per_object_mask_name.split('.')[0]}')
    #         os.makedirs(save_per_video_dir, exist_ok=True)

    #         save_per_video_image_dir = os.path.join(save_per_video_dir,
    #                                                 'image')
    #         save_per_video_mask_dir = os.path.join(save_per_video_dir, 'mask')

    #         os.makedirs(save_per_video_image_dir, exist_ok=True)
    #         os.makedirs(save_per_video_mask_dir, exist_ok=True)

    #         # 重新编号并保存图像
    #         for idx, per_image in enumerate(valid_per_video_all_images):
    #             # 新的图像名称,从00000开始
    #             new_image_name = f'{root_name}_{per_video_name}_{per_object_mask_name.split('.')[0]}_{idx:05d}.jpg'
    #             save_per_image_path = os.path.join(save_per_video_image_dir,
    #                                                new_image_name)
    #             cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    #         # 重新编号并保存mask
    #         for idx, per_mask in enumerate(valid_per_video_all_masks):
    #             # 新的mask名称,从00000开始
    #             per_mask = (per_mask * 255).astype(np.uint8)
    #             new_mask_name = f'{root_name}_{per_video_name}_{per_object_mask_name.split('.')[0]}_{idx:05d}.png'
    #             save_per_mask_path = os.path.join(save_per_video_mask_dir,
    #                                               new_mask_name)
    #             cv2.imencode('.png', per_mask)[1].tofile(save_per_mask_path)

    #         valid_video_count += 1

    # print(valid_video_count, invalid_video_count)


    process_args = []
    for video_info in all_video_path_list:
        process_args.append((*video_info, save_path, root_name))
    
    with Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(process_single_video, process_args), 
                           total=len(process_args)))

    valid_video_count = sum(r[0] for r in results)
    invalid_video_count = sum(r[1] for r in results)

    print(valid_video_count, invalid_video_count)


if __name__ == "__main__":
    root_path = r'/root/autodl-tmp/MaGGIe-HIM/V-HIM2K5'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/V-HIM2K5'
    process_videos(root_path=root_path, save_path=save_path)
