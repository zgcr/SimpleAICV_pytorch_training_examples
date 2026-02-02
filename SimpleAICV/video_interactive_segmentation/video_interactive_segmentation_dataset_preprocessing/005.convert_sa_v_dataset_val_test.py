import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


def process_single_video(args):
    dir_name, dir_path, video_image_path, save_path = args

    per_video_save_path = os.path.join(save_path, dir_name, 'image')
    if not os.path.exists(per_video_save_path):
        os.makedirs(per_video_save_path)

    for per_image_name in sorted(os.listdir(video_image_path)):
        if '.jpg' in per_image_name:
            per_image_name_prefix = per_image_name.split('.')[0]
            per_image_name_int = int(per_image_name_prefix)
            if per_image_name_int % 4 == 0:
                per_image_name_prefix = f'{per_image_name_int:05d}'
                per_image_name = per_image_name_prefix + '.jpg'
                per_image_path = os.path.join(video_image_path, per_image_name)
                per_image = cv2.imdecode(
                    np.fromfile(per_image_path, dtype=np.uint8),
                    cv2.IMREAD_COLOR)

                save_per_image_name = dir_name + '_' + per_image_name
                per_image_save_path = os.path.join(per_video_save_path,
                                                   save_per_image_name)
                cv2.imencode('.jpg', per_image)[1].tofile(per_image_save_path)

    object_dir_list = []
    for per_object_name in sorted(os.listdir(dir_path)):
        per_object_path = os.path.join(dir_path, per_object_name)
        object_dir_list.append([per_object_name, per_object_path])

    per_video_objects_save_path = os.path.join(save_path, dir_name, 'object')
    if not os.path.exists(per_video_objects_save_path):
        os.makedirs(per_video_objects_save_path)

    for per_object_name, per_object_path in object_dir_list:
        per_video_per_object_save_path = os.path.join(
            per_video_objects_save_path, per_object_name)
        if not os.path.exists(per_video_per_object_save_path):
            os.makedirs(per_video_per_object_save_path)

        for per_mask_name in sorted(os.listdir(per_object_path)):
            if '.png' in per_mask_name:
                per_mask_name_prefix = per_mask_name.split('.')[0]
                per_mask_name_int = int(per_mask_name_prefix)
                if per_mask_name_int % 4 == 0:
                    per_mask_name_prefix = f'{per_mask_name_int:05d}'
                    per_mask_name = per_mask_name_prefix + '.png'
                    per_mask_path = os.path.join(per_object_path,
                                                 per_mask_name)
                    per_mask = np.array(Image.open(per_mask_path).convert('L'),
                                        dtype=np.uint8)
                    per_mask[per_mask > 0] = 255

                    save_per_mask_name = dir_name + '_' + per_mask_name_prefix + '_' + per_object_name + '.png'
                    per_mask_save_path = os.path.join(
                        per_video_per_object_save_path, save_per_mask_name)
                    cv2.imencode('.png',
                                 per_mask)[1].tofile(per_mask_save_path)


def process_videos(root_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    annotation_path = os.path.join(root_path, 'Annotations_6fps')

    all_video_path_list = []
    for per_video_name in sorted(os.listdir(annotation_path)):
        if 'sav' in per_video_name:
            annot_path = os.path.join(annotation_path, per_video_name)
            video_image_path = annot_path.replace('Annotations_6fps',
                                                  'JPEGImages_24fps')
            all_video_path_list.append(
                [per_video_name, annot_path, video_image_path])
    all_video_path_list = sorted(all_video_path_list)

    print('1111', len(all_video_path_list), all_video_path_list[0])

    # for dir_name, dir_path, video_image_path in tqdm(all_video_path_list):
    #     per_video_save_path = os.path.join(save_path, dir_name, 'image')
    #     if not os.path.exists(per_video_save_path):
    #         os.makedirs(per_video_save_path)

    #     for per_image_name in tqdm(sorted(os.listdir(video_image_path))):
    #         if '.jpg' in per_image_name:
    #             per_image_name_prefix = per_image_name.split('.')[0]
    #             per_image_name_int = int(per_image_name_prefix)
    #             if per_image_name_int % 4 == 0:
    #                 per_image_name_prefix = f'{per_image_name_int:05d}'
    #                 per_image_name = per_image_name_prefix + '.jpg'
    #                 per_image_path = os.path.join(video_image_path,
    #                                               per_image_name)
    #                 per_image = cv2.imdecode(
    #                     np.fromfile(per_image_path, dtype=np.uint8),
    #                     cv2.IMREAD_COLOR)

    #                 save_per_image_name = dir_name + '_' + per_image_name
    #                 per_image_save_path = os.path.join(per_video_save_path,
    #                                                    save_per_image_name)
    #                 cv2.imencode('.jpg',
    #                              per_image)[1].tofile(per_image_save_path)

    #     object_dir_list = []
    #     for per_object_name in sorted(os.listdir(dir_path)):
    #         per_object_path = os.path.join(dir_path, per_object_name)
    #         object_dir_list.append([per_object_name, per_object_path])

    #     per_video_objects_save_path = os.path.join(save_path, dir_name,
    #                                                'object')
    #     if not os.path.exists(per_video_objects_save_path):
    #         os.makedirs(per_video_objects_save_path)

    #     for per_object_name, per_object_path in tqdm(object_dir_list):
    #         per_video_per_object_save_path = os.path.join(
    #             per_video_objects_save_path, per_object_name)
    #         if not os.path.exists(per_video_per_object_save_path):
    #             os.makedirs(per_video_per_object_save_path)

    #         for per_mask_name in sorted(os.listdir(per_object_path)):
    #             if '.png' in per_mask_name:
    #                 per_mask_name_prefix = per_mask_name.split('.')[0]
    #                 per_mask_name_int = int(per_mask_name_prefix)
    #                 if per_mask_name_int % 4 == 0:
    #                     per_mask_name_prefix = f'{per_mask_name_int:05d}'
    #                     per_mask_name = per_mask_name_prefix + '.png'
    #                     per_mask_path = os.path.join(per_object_path,
    #                                                  per_mask_name)
    #                     per_mask = np.array(
    #                         Image.open(per_mask_path).convert('L'),
    #                         dtype=np.uint8)
    #                     per_mask[per_mask > 0] = 255

    #                     save_per_mask_name = dir_name + '_' + per_mask_name_prefix + '_' + per_object_name + '.png'
    #                     per_mask_save_path = os.path.join(
    #                         per_video_per_object_save_path, save_per_mask_name)
    #                     cv2.imencode('.png',
    #                                  per_mask)[1].tofile(per_mask_save_path)

    args_list = [
        (dir_name, dir_path, video_image_path, save_path)
        for dir_name, dir_path, video_image_path in all_video_path_list
    ]

    # 使用多进程处理
    with Pool(processes=16) as pool:
        list(
            tqdm(pool.imap(process_single_video, args_list),
                 total=len(args_list)))


if __name__ == "__main__":
    root_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_val'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_val_new'

    print(root_path)
    print(save_path)

    process_videos(root_path=root_path, save_path=save_path)

    root_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_test'
    save_path = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_test_new'

    print(root_path)
    print(save_path)

    process_videos(root_path=root_path, save_path=save_path)
