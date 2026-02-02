import math
import numpy as np

import torch.nn as nn


class RetinaFaceAnchors:

    def __init__(self,
                 anchor_sizes=[[16, 32], [64, 128], [256, 512]],
                 strides=[8, 16, 32]):
        self.anchor_sizes = np.array(anchor_sizes, dtype=np.float32)
        self.strides = np.array(strides, dtype=np.float32)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image anchors
        '''
        one_image_anchors = []
        for index, per_level_anchor_size in enumerate(self.anchor_sizes):
            # per_level_feature_map_per_position_anchor_num:len(per_level_anchor_size)
            # anchor wh ratio set to 1
            base_anchors = self.generate_base_anchors(per_level_anchor_size)
            feature_anchors = self.generate_anchors_on_feature_map(
                base_anchors, fpn_feature_sizes[index], self.strides[index])
            one_image_anchors.append(feature_anchors)

        # if input size:[640,640]
        # one_image_anchors shape:[[80,80,2,4],[40,40,2,4],[20,20,2,4],[10,10,2,4],[5,5,2,4]]
        # per anchor format:[x_min,y_min,x_max,y_max]
        return one_image_anchors

    def generate_base_anchors(self, per_level_anchor_size):
        '''
        generate base anchor
        '''
        # base anchor for each position on feature map,shape[2,4]
        base_anchors = np.zeros((len(per_level_anchor_size), 4),
                                dtype=np.float32)

        # compute base anchor w\h,shape[2,2]
        base_w_h = []
        for per_size in per_level_anchor_size:
            base_w_h.append([per_size, per_size])
        base_w_h = np.array(base_w_h, dtype=np.float32)
        base_anchors[:, 2:] += base_w_h

        # base_anchors format: [x_min,y_min,x_max,y_max],center point:[0,0],shape[2,4],2 is anchor num
        base_anchors[:, 0] -= base_anchors[:, 2] / 2
        base_anchors[:, 1] -= base_anchors[:, 3] / 2
        base_anchors[:, 2] /= 2
        base_anchors[:, 3] /= 2

        return base_anchors

    def generate_anchors_on_feature_map(self, base_anchors, feature_map_size,
                                        stride):
        '''
        generate one feature map anchors
        '''
        # shifts_x shape:[w],shifts_y shape:[h]
        shifts_x = (np.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (np.arange(0, feature_map_size[1]) + 0.5) * stride

        # shifts shape:[w,h,2] -> [w,h,4] -> [w,h,1,4]
        # 4:[x_min,y_min,x_min,y_min]
        shifts = np.array([[[shift_x, shift_y] for shift_y in shifts_y]
                           for shift_x in shifts_x],
                          dtype=np.float32)
        shifts = np.expand_dims(np.tile(shifts, (1, 1, 2)), axis=2)

        # base anchors shape:[2,4] -> [1,1,2,4]
        base_anchors = np.expand_dims(base_anchors, axis=0)
        base_anchors = np.expand_dims(base_anchors, axis=0)

        # generate all featrue map anchors on each feature map points
        # featrue map anchors shape:[w,h,2,4] -> [h,w,2,4]
        feature_map_anchors = np.transpose(base_anchors + shifts,
                                           axes=(1, 0, 2, 3))
        feature_map_anchors = np.ascontiguousarray(feature_map_anchors,
                                                   dtype=np.float32)

        # feature_map_anchors format: [h,w,2,4],4:[x_min,y_min,x_max,y_max]
        return feature_map_anchors


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    anchor_sizes = [[8, 16, 32], [32, 64, 128], [128, 256, 512]]
    strides = [8, 16, 32]
    image_w, image_h = 1024, 1024
    fpn_feature_sizes = [[
        math.ceil(image_w / stride),
        math.ceil(image_h / stride)
    ] for stride in strides]

    anchors = RetinaFaceAnchors(anchor_sizes=anchor_sizes, strides=strides)
    one_image_anchors = anchors(fpn_feature_sizes)

    for per_level_anchors in one_image_anchors:
        print('1111', per_level_anchors.shape)
