import os
import numpy as np
import cv2
import json

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from tqdm import tqdm


def load_single_image(coco, image_dir, image_id):
    file_name = coco.loadImgs(image_id)[0]['coco_url'].split('/')[-1]
    image = cv2.imdecode(
        np.fromfile(os.path.join(image_dir, file_name), dtype=np.uint8),
        cv2.IMREAD_COLOR)

    return image


def load_single_image_mask(coco, cat_ids, image_id):
    annot_ids = coco.getAnnIds(imgIds=image_id)
    annots = coco.loadAnns(annot_ids)

    image_info = coco.loadImgs(image_id)[0]
    image_h, image_w = image_info['height'], image_info['width']

    target_masks = np.zeros((image_h, image_w, 0))

    # filter annots
    for annot in annots:
        if 'ignore' in annot.keys():
            continue
        # bbox format:[x_min, y_min, w, h]
        bbox = annot['bbox']

        inter_w = max(0, min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
        inter_h = max(0, min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
        if inter_w * inter_h == 0:
            continue
        if bbox[2] * bbox[3] <= 1 or bbox[2] <= 1 or bbox[3] <= 1:
            continue
        if annot['category_id'] not in cat_ids:
            continue

        target_mask = np.zeros((image_h, image_w, 1))
        annot_mask = coco.annToMask(annot)
        target_mask[:, :, 0] = annot_mask
        target_masks = np.append(target_masks, target_mask, axis=2)

    return target_masks.astype(np.float32)


def get_random_foreground_point(mask):
    """从mask的前景区域随机选择一个点"""
    foreground_indices = np.argwhere(mask > 0)
    if len(foreground_indices) == 0:
        # 如果没有前景点，返回mask中心
        h, w = mask.shape
        return [w // 2, h // 2]

    random_idx = np.random.randint(0, len(foreground_indices))
    y, x = foreground_indices[random_idx]

    return [float(x), float(y)]


def preprocess_image(root_dataset_path,
                     save_dataset_path,
                     set_name='train2017'):
    os.makedirs(save_dataset_path, exist_ok=True)

    image_dir = os.path.join(root_dataset_path, 'images')
    annot_type = set_name[:-4]
    annot_dir = os.path.join(root_dataset_path, 'annotations',
                             f'lvis_v1_{annot_type}.json')
    coco = COCO(annot_dir)

    image_ids = sorted(coco.getImgIds())

    # filter image id without annotation,from 118287 ids to 117266 ids
    ids = []
    for image_id in image_ids:
        annot_ids = coco.getAnnIds(imgIds=image_id)
        annots = coco.loadAnns(annot_ids)
        if len(annots) == 0:
            continue
        ids.append(image_id)
    image_ids = ids
    image_ids = sorted(image_ids)

    cat_ids = coco.getCatIds()

    print('1111', len(image_ids), image_ids[0])

    for per_image_id in tqdm(ids):
        per_image = load_single_image(coco, image_dir, per_image_id)

        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

        per_image_instance_masks = load_single_image_mask(
            coco, cat_ids, per_image_id)

        if per_image_instance_masks.shape[-1] == 0:
            continue

        per_mask_h, per_mask_w = per_image_instance_masks.shape[
            0], per_image_instance_masks.shape[1]

        assert per_image_h == per_mask_h and per_image_w == per_mask_w

        file_name = coco.loadImgs(per_image_id)[0]['coco_url'].split('/')[-1]

        new_file_name = 'lvis_v1_' + file_name
        new_json_name = 'lvis_v1_' + file_name.split('.jpg')[0] + '.json'

        annotations = []
        for idx in range(per_image_instance_masks.shape[-1]):
            per_mask = (per_image_instance_masks[:, :,
                                                 idx]).copy().astype(np.uint8)
            # 计算bbox
            rows, cols = np.where(per_mask > 0)
            if len(rows) == 0 or len(cols) == 0:
                continue

            x_min = float(np.min(cols))
            y_min = float(np.min(rows))
            x_max = float(np.max(cols))
            y_max = float(np.max(rows))
            # [x_min, y_min, w, h]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # 计算area
            area = float(np.count_nonzero(per_mask))

            # 转换为RLE格式
            rle = mask_utils.encode(np.asfortranarray(per_mask))
            segmentation = {
                "size": [per_image_h, per_image_w],
                "counts": rle['counts'].decode('utf-8')
            }

            # 随机前景点
            point_coords = [get_random_foreground_point(per_mask)]

            # crop_box (使用与bbox相同的值)
            crop_box = [x_min, y_min, x_max - x_min, y_max - y_min]

            # 生成唯一ID
            annot_id = int(image_id) + 200000 + idx

            annotation = {
                "bbox": bbox,
                "area": area,
                "segmentation": segmentation,
                "predicted_iou": 1.0,
                "point_coords": point_coords,
                "crop_box": crop_box,
                "id": annot_id,
                "stability_score": 1.0,
            }
            annotations.append(annotation)

        if len(annotations) == 0:
            continue

        per_image_json_data = {
            "image": {
                "image_id": int(image_id),
                "width": per_image_w,
                "height": per_image_h,
                "file_name": new_file_name,
            },
            "annotations": annotations,
        }

        if 'train' in set_name:
            save_set_name = 'train'
        elif 'val' in set_name:
            save_set_name = 'val'

        save_image_path = os.path.join(save_dataset_path, save_set_name,
                                       new_file_name)
        save_json_path = os.path.join(save_dataset_path, save_set_name,
                                      new_json_name)
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)

        cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

        with open(save_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(per_image_json_data, json_file, ensure_ascii=False)


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/lvisv1.0'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/lvisv1.0'
    set_name = 'train2017'
    preprocess_image(root_dataset_path, save_dataset_path, set_name=set_name)

    root_dataset_path = r'/root/autodl-tmp/lvisv1.0'
    save_dataset_path = r'/root/autodl-tmp/interactive_segmentation_dataset/lvisv1.0'
    set_name = 'val2017'
    preprocess_image(root_dataset_path, save_dataset_path, set_name=set_name)
