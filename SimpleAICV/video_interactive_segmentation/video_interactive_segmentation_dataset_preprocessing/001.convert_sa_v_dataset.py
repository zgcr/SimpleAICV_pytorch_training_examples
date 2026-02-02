import os
import cv2
import shutil

from tqdm import tqdm

from pathlib import Path
from multiprocessing import Pool


def decode_video(video_path: str):
    assert os.path.exists(video_path)
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            video_frames.append(frame)
        else:
            break
    video.release()

    return video_frames


def process_single_video(args):
    path, save_root, sample_rate = args
    video_name = Path(path).stem
    frames = decode_video(path)
    frames = frames[::sample_rate]
    output_folder = os.path.join(save_root, video_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for fid, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{video_name}_{fid:05d}.jpg")
        cv2.imwrite(frame_path, frame)

    # 复制标注文件
    video_stem = Path(path).stem
    annotation_suffixes = ['_auto.json', '_manual.json']
    for suffix in annotation_suffixes:
        annotation_filename = video_stem + suffix
        annotation_path = os.path.join(Path(path).parent, annotation_filename)
        if os.path.exists(annotation_path):
            shutil.copy(annotation_path, output_folder)
        else:
            print(
                f"Annotation file {annotation_filename} not found for video {path}"
            )

    return True


def process_videos(video_paths, save_root, sample_rate):
    save_root = os.path.join(save_root, 'train')

    # 准备多进程参数
    args_list = [(path, save_root, sample_rate) for path in video_paths]

    # 使用多进程池处理
    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(process_single_video, args_list),
                 total=len(args_list),
                 desc="Processing videos"))

    print(f"Saved output to {save_root}")


if __name__ == "__main__":
    root_dir = r'/root/autodl-tmp/video_interactive_segmentation_dataset/sav_'
    save_dir = r'/root/autodl-tmp/video_interactive_segmentation_dataset_new/sav_'

    for i in tqdm(range(10, 20)):
        houzhui = f'{i:03d}'

        sa_v_dataset_root_dir = root_dir + houzhui
        sa_v_dataset_save_dir = save_dir + houzhui
        sample_rate = 4

        print(sa_v_dataset_root_dir)
        print(sa_v_dataset_save_dir)

        mp4_files = sorted(
            [str(p) for p in Path(sa_v_dataset_root_dir).glob("*.mp4")])
        print(f"Processing videos in: {sa_v_dataset_root_dir}")
        print(f"Processing {len(mp4_files)} files")

        process_videos(video_paths=mp4_files,
                       save_root=sa_v_dataset_save_dir,
                       sample_rate=sample_rate)
