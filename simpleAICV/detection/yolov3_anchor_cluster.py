'''
Original generate anchor script
https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py
'''
import numpy as np


def compute_ious(boxes, clusters):
    '''
    computer IoUs between N boxes and k clusters
    boxes:[N,2],2[w,h]
    clusters:[k,2],2:[w,h]
    '''

    boxes_area = boxes[:, 0] * boxes[:, 1]
    clusters_area = clusters[:, 0] * clusters[:, 1]

    overlaps_area_w = np.minimum(np.expand_dims(boxes[:, 0], axis=-1),
                                 np.expand_dims(clusters[:, 0], axis=0))
    overlaps_area_h = np.minimum(np.expand_dims(boxes[:, 1], axis=-1),
                                 np.expand_dims(clusters[:, 1], axis=0))

    # overlaps_area w/h must >0
    if np.count_nonzero(overlaps_area_w == 0) > 0 or np.count_nonzero(
            overlaps_area_h == 0) > 0:
        raise ValueError('Box has no area')

    overlaps_area = overlaps_area_w * overlaps_area_h
    ious = overlaps_area / (np.expand_dims(boxes_area, axis=-1) +
                            np.expand_dims(clusters_area, axis=0) -
                            overlaps_area)

    return ious


def compute_avg_iou(boxes, clusters):
    '''
    compute average iou
    boxes:[N,2],2[w,h]
    clusters:[k,2],2:[w,h]
    '''
    ious = compute_ious(boxes, clusters)
    boxes_iou_max = np.max(ious, axis=1)
    avg_iou = np.average(boxes_iou_max)

    return avg_iou


def kmeans_cluster(boxes, k, seed, resize, dist=np.average):
    '''
    K-means clustering(Using IoU as distance)
    boxes:[N,2],2:[w,h]
    k:number of cluster centers
    anchors:[k,2],2[w/resize,h/resize]
    '''
    sample_nums = boxes.shape[0]
    # for each sample,record all distance from each sample to each cluster
    distances = np.empty((sample_nums, k))
    # for each sample,record the smallest distance cluster
    last_clusters = np.zeros((sample_nums, ))
    np.random.seed(seed)

    # random choice k clusters
    clusters = boxes[np.random.choice(sample_nums, k, replace=False)]
    # start kmeans clustering
    while True:
        # computer distance between all samples and k clusters,distance is 1-IOU(box,anchor)
        distances = 1 - compute_ious(boxes, clusters)

        # for each sample,find nearest cluster index
        nearest_clusters = np.argmin(distances, axis=1)
        # if all sample nearset clusters doesn't change,end clustering
        if (last_clusters == nearest_clusters).all():
            break
        # update clusters(using median value)
        for index in range(k):
            clusters[index] = dist(boxes[nearest_clusters == index], axis=0)
        # update last_clusters
        last_clusters = nearest_clusters

    clusters = np.around(clusters, decimals=0).astype(np.int32)
    clusters_area = clusters[:, 0] * clusters[:, 1]
    clusters_indexes = sorted(range(len(clusters_area)),
                              key=lambda k: clusters_area[k])
    clusters = clusters[clusters_indexes]

    return clusters


if __name__ == '__main__':
    anchor_nums = 9
    seed = 0
    resize = 640

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from tools.path import COCO2017_path, VOCdataset_path

    from tqdm import tqdm
    import torchvision.transforms as transforms
    from simpleAICV.detection.datasets.cocodataset import CocoDetection
    from simpleAICV.detection.datasets.vocdataset import VocDetection
    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize, DetectionCollater

    coco = CocoDetection(
        COCO2017_path,
        set_name='train2017',
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            # RandomCrop(prob=0.5),
            # RandomTranslate(prob=0.5),
            YoloStyleResize(resize=resize,
                            divisor=32,
                            stride=32,
                            multi_scale=False,
                            multi_scale_range=[0.5, 1.0]),
            # RetinaStyleResize(resize=resize, multi_scale=True),
            Normalize(),
        ]))

    voc = VocDetection(
        root_dir=VOCdataset_path,
        image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            # RandomCrop(prob=0.5),
            # RandomTranslate(prob=0.5),
            YoloStyleResize(resize=resize,
                            divisor=32,
                            stride=32,
                            multi_scale=False,
                            multi_scale_range=[0.5, 1.0]),
            # RetinaStyleResize(resize=resize, multi_scale=True),
            Normalize(),
        ]),
        keep_difficult=True)

    boxes_wh = []
    for index in tqdm(range(len(voc))):
        per_image_boxes = voc[index]['annots'][:, 0:4]
        per_image_boxes_wh = per_image_boxes[:, 2:4] - per_image_boxes[:, 0:2]
        per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 0] > 0]
        per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 1] > 0]
        per_image_boxes_wh = np.array(per_image_boxes_wh)
        boxes_wh.append(per_image_boxes_wh)

    # for index in tqdm(range(len(coco))):
    #     per_image_boxes = coco[index]['annots'][:, 0:4]
    #     per_image_boxes_wh = per_image_boxes[:, 2:4] - per_image_boxes[:, 0:2]
    #     per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 0] > 0]
    #     per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 1] > 0]
    #     per_image_boxes_wh = np.array(per_image_boxes_wh)
    #     boxes_wh.append(per_image_boxes_wh)

    boxes_wh = np.concatenate(boxes_wh, axis=0)
    print(boxes_wh.shape)
    # dist=np.average or dist=np.median
    anchors = kmeans_cluster(boxes_wh,
                             anchor_nums,
                             seed,
                             resize,
                             dist=np.median)
    print(anchors)

    avg_iou = compute_avg_iou(boxes_wh, anchors)
    print(avg_iou)

# YoloStyleResize=640,voc2007+2012
# np.median
# [[22, 27], [38, 60], [95, 68], [64, 135], [180, 137], [122, 239], [230, 350],
#  [403, 215], [497, 426]]
# avg_iou:0.6657713134621938

# YoloStyleResize=640,coco2017
# np.median
# [[8, 10], [17, 23], [49, 30], [27, 55], [53, 106], [99, 65], [111, 207],
#  [213, 141], [366, 363]]
# avg_iou:0.6070449695289617