'''
Original generate anchor script：
https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py
'''
from tqdm import tqdm
import operator
import numpy as np
import xml.etree.ElementTree as ET

import torchvision.transforms as transforms
from cocodataset import CocoDetection, Resize
from vocdataset import VocDetection


def compute_ious(boxes, clusters):
    """
    computer IoUs between N boxes and k clusters
    boxes:[N,2],2[w,h]
    clusters:[k,2],2:[w,h]
    """

    boxes_area = boxes[:, 0] * boxes[:, 1]
    clusters_area = clusters[:, 0] * clusters[:, 1]

    overlaps_area_w = np.minimum(np.expand_dims(boxes[:, 0], axis=-1),
                                 np.expand_dims(clusters[:, 0], axis=0))
    overlaps_area_h = np.minimum(np.expand_dims(boxes[:, 1], axis=-1),
                                 np.expand_dims(clusters[:, 1], axis=0))

    # overlaps_area w/h must >0
    if np.count_nonzero(overlaps_area_w == 0) > 0 or np.count_nonzero(
            overlaps_area_h == 0) > 0:
        raise ValueError("Box has no area")

    overlaps_area = overlaps_area_w * overlaps_area_h
    ious = overlaps_area / (np.expand_dims(boxes_area, axis=-1) +
                            np.expand_dims(clusters_area, axis=0) -
                            overlaps_area)

    return ious


def compute_avg_iou(boxes, clusters):
    """
    compute average iou
    boxes:[N,2],2[w,h]
    clusters:[k,2],2:[w,h]
    """
    ious = compute_ious(boxes, clusters)
    boxes_iou_max = np.max(ious, axis=1)
    avg_iou = np.average(boxes_iou_max)

    return avg_iou


def kmeans_cluster(boxes, k, seed, resize, dist=np.average):
    """
    K-means clustering(Using IoU as distance)
    boxes:[N,2],2:[w,h]
    k:number of cluster centers
    anchors:[k,2],2[w/resize,h/resize]
    """
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
    resize = 416

    coco = CocoDetection(
        image_root_dir=
        '/home/zgcr/Downloads/datasets/COCO2017/images/train2017/',
        annotation_root_dir=
        "/home/zgcr/Downloads/datasets/COCO2017/annotations/",
        set='train2017',
        transform=transforms.Compose([
            Resize(resize=resize),
        ]))
    print(len(coco))

    # voc = VocDetection(root_dir='/home/zgcr/Downloads/datasets/VOCdataset',
    #                    image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
    #                    transform=transforms.Compose([
    #                        Resize(resize=resize),
    #                    ]),
    #                    keep_difficult=False)
    # print(len(voc))

    boxes_wh = []
    # for index in tqdm(range(len(voc))):
    #     per_image_boxes = voc[index]['annot'][:, 0:4]
    #     per_image_boxes_wh = per_image_boxes[:, 2:4] - per_image_boxes[:, 0:2]
    #     per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 0] > 0]
    #     per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 1] > 0]
    #     per_image_boxes_wh = np.array(per_image_boxes_wh)
    #     boxes_wh.append(per_image_boxes_wh)

    for index in tqdm(range(len(coco))):
        per_image_boxes = coco[index]['annot'][:, 0:4]
        per_image_boxes_wh = per_image_boxes[:, 2:4] - per_image_boxes[:, 0:2]
        per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 0] > 0]
        per_image_boxes_wh = per_image_boxes_wh[per_image_boxes_wh[:, 1] > 0]
        per_image_boxes_wh = np.array(per_image_boxes_wh)
        boxes_wh.append(per_image_boxes_wh)

    boxes_wh = np.concatenate(boxes_wh, axis=0)
    print(boxes_wh.shape)
    # dist=np.average or dist=np.median
    anchors = kmeans_cluster(boxes_wh,
                             anchor_nums,
                             seed,
                             resize,
                             dist=np.average)
    print(anchors)

    avg_iou = compute_avg_iou(boxes_wh, anchors)
    print(avg_iou)

# resize=416,voc2007
# np.median
# [[22, 33], [58, 45], [34, 79], [69, 110], [137, 94], [103, 186], [268, 146], [180, 242], [329, 268]]
# avg_iou:0.689093497797444
# np.average
# [[33, 41], [49, 94], [113, 71], [86, 166], [173, 128], [140, 236], [318, 153],
#  [228, 276], [358, 274]]
# avg_iou:0.6753206003932308

# resize=416,voc2007+2012
# np.median
# [[20, 27], [29, 64], [57, 42], [55, 108], [113, 82], [98, 171], [245, 134], [167, 237], [319, 269]]
# avg_iou:0.6833950629104503
# np.average
# [[28, 41][74, 58][57, 119][151, 104][103, 195][174, 238][298, 146][256,313][369,258]]
# avg_iou:0.6737023918552011

# resize=416,coco2017
# np.median
# [[5, 7], [10, 20], [19, 12], [19, 40], [38, 25], [39, 80], [75, 50],
#  [103, 129], [246, 228]]
# avg_iou:0.6106945462262936
# np.average
# [[12, 14], [29, 35], [87, 44], [45, 87], [130, 97], [78, 167], [273, 134],
#      [158, 236], [328, 284]]
# avg_iou:0.5731441860234409
