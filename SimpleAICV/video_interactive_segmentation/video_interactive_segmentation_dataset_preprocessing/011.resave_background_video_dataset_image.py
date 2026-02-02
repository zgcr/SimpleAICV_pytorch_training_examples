import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def process_single_video(args):
    """处理单个视频的函数"""
    per_video_name, per_video_image_dir_path, per_video_image_path_list, save_path, root_name, set_type = args

    per_video_all_images = []
    for per_frame_image_path in per_video_image_path_list:
        per_frame_image = cv2.imdecode(
            np.fromfile(per_frame_image_path, dtype=np.uint8),
            cv2.IMREAD_COLOR)
        per_video_all_images.append(per_frame_image)

    # 创建保存目录
    save_per_video_dir = os.path.join(
        save_path, f'{root_name}_{set_type}_{per_video_name}')
    os.makedirs(save_per_video_dir, exist_ok=True)

    # 重新编号并保存图像
    for idx, per_image in enumerate(per_video_all_images):
        # 新的图像名称,从00000开始
        new_image_name = f'{root_name}_{set_type}_{per_video_name}_{idx:05d}.jpg'
        save_per_image_path = os.path.join(save_per_video_dir, new_image_name)
        cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)


def process_videos(root_path, save_path, set_type):
    root_name = root_path.split('/')[-1]

    save_path = os.path.join(save_path, set_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_dataset_path = os.path.join(root_path, set_type)

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

        all_video_path_list.append([
            per_video_name,
            per_video_image_dir_path,
            per_video_image_path_list,
        ])

    print('1111', len(all_video_path_list), all_video_path_list[0])

    # for per_video_name, per_video_image_dir_path, per_video_image_path_list in tqdm(
    #         all_video_path_list):
    #     per_video_all_images = []
    #     for per_frame_image_path in per_video_image_path_list:
    #         per_frame_image = cv2.imdecode(
    #             np.fromfile(per_frame_image_path, dtype=np.uint8),
    #             cv2.IMREAD_COLOR)
    #         per_video_all_images.append(per_frame_image)

    #     # 创建保存目录
    #     save_per_video_dir = os.path.join(
    #         save_path, f'{root_name}_{set_type}_{per_video_name}')
    #     os.makedirs(save_per_video_dir, exist_ok=True)

    #     # 重新编号并保存图像
    #     for idx, per_image in enumerate(per_video_all_images):
    #         # 新的图像名称,从00000开始
    #         new_image_name = f'{root_name}_{set_type}_{per_video_name}_{idx:05d}.jpg'
    #         save_per_image_path = os.path.join(save_per_video_dir,
    #                                            new_image_name)
    #         cv2.imencode('.jpg', per_image)[1].tofile(save_per_image_path)

    process_args = []
    for per_video_name, per_video_image_dir_path, per_video_image_path_list in all_video_path_list:
        process_args.append(
            (per_video_name, per_video_image_dir_path,
             per_video_image_path_list, save_path, root_name, set_type))

    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_single_video, process_args),
                 total=len(process_args)))


if __name__ == "__main__":
    root_path = r'/root/autodl-tmp/Background_Video_Datasets'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/Background_Video_Datasets'
    set_type = 'test'
    process_videos(root_path=root_path, save_path=save_path, set_type=set_type)

    root_path = r'/root/autodl-tmp/Background_Video_Datasets'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/Background_Video_Datasets'
    set_type = 'train'
    process_videos(root_path=root_path, save_path=save_path, set_type=set_type)
