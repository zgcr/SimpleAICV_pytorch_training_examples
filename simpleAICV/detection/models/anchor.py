import math
import numpy as np

import torch.nn as nn


class RetinaAnchors:

    def __init__(self,
                 areas=[[32, 32], [64, 64], [128, 128], [256, 256], [512,
                                                                     512]],
                 ratios=[0.5, 1, 2],
                 scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
                 strides=[8, 16, 32, 64, 128]):
        self.areas = np.array(areas, dtype=np.float32)
        self.ratios = np.array(ratios, dtype=np.float32)
        self.scales = np.array(scales, dtype=np.float32)
        self.strides = np.array(strides, dtype=np.float32)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image anchors
        '''
        one_image_anchors = []
        for index, area in enumerate(self.areas):
            base_anchors = self.generate_base_anchors(area, self.scales,
                                                      self.ratios)
            feature_anchors = self.generate_anchors_on_feature_map(
                base_anchors, fpn_feature_sizes[index], self.strides[index])
            one_image_anchors.append(feature_anchors)

        # if input size:[640,640]
        # one_image_anchors shape:[[80,80,9,4],[40,40,9,4],[20,20,9,4],[10,10,9,4],[5,5,9,4]]
        # per anchor format:[x_min,y_min,x_max,y_max]
        return one_image_anchors

    def generate_base_anchors(self, area, scales, ratios):
        '''
        generate base anchor
        '''
        # get w,h aspect ratio,shape:[9,2]
        aspects = np.array([[[s * math.sqrt(r), s * math.sqrt(1 / r)]
                             for s in scales] for r in ratios],
                           dtype=np.float32).reshape(-1, 2)
        # base anchor for each position on feature map,shape[9,4]
        base_anchors = np.zeros((len(scales) * len(ratios), 4),
                                dtype=np.float32)

        # compute aspect w\h,shape[9,2]
        base_w_h = area * aspects
        base_anchors[:, 2:] += base_w_h

        # base_anchors format: [x_min,y_min,x_max,y_max],center point:[0,0],shape[9,4]
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
        shifts = np.array([[[shift_x, shift_y] for shift_y in shifts_y]
                           for shift_x in shifts_x],
                          dtype=np.float32)
        shifts = np.expand_dims(np.tile(shifts, (1, 1, 2)), axis=2)

        # base anchors shape:[9,4] -> [1,1,9,4]
        base_anchors = np.expand_dims(base_anchors, axis=0)
        base_anchors = np.expand_dims(base_anchors, axis=0)

        # generate all featrue map anchors on each feature map points
        # featrue map anchors shape:[w,h,9,4] -> [h,w,9,4]
        feature_map_anchors = np.transpose(base_anchors + shifts,
                                           axes=(1, 0, 2, 3))
        feature_map_anchors = np.ascontiguousarray(feature_map_anchors,
                                                   dtype=np.float32)

        # feature_map_anchors format: [h,w,9,4],4:[x_min,y_min,x_max,y_max]
        return feature_map_anchors


class FCOSPositions:

    def __init__(self, strides=[8, 16, 32, 64, 128]):
        self.strides = np.array(strides, dtype=np.float32)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image positions
        '''

        one_image_positions = []
        for stride, fpn_feature_size in zip(self.strides, fpn_feature_sizes):
            featrue_positions = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            one_image_positions.append(featrue_positions)

        # if input size:[640,640]
        # one_image_positions shape:[[80, 80, 2],[40, 40, 2],[20, 20, 2],[10, 10, 2],[5, 5, 2]]
        # per position format:[x_center,y_center]
        return one_image_positions

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        '''
        generate one feature map positions
        '''

        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (np.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (np.arange(0, feature_map_size[1]) + 0.5) * stride

        # feature_map_positions shape:[w,h,2] -> [h,w,2]
        feature_map_positions = np.array([[[shift_x, shift_y]
                                           for shift_y in shifts_y]
                                          for shift_x in shifts_x],
                                         dtype=np.float32)
        feature_map_positions = np.transpose(feature_map_positions,
                                             axes=(1, 0, 2))
        feature_map_positions = np.ascontiguousarray(feature_map_positions,
                                                     dtype=np.float32)

        # feature_map_positions format: [point_nums,2],2:[x_center,y_center]
        return feature_map_positions


class Yolov3Anchors:

    def __init__(self,
                 anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]],
                 strides=[8, 16, 32],
                 per_level_num_anchors=3):
        assert len(strides) == (len(anchor_sizes) // per_level_num_anchors)

        self.anchor_sizes = np.array(anchor_sizes, dtype=np.float32)
        self.strides = np.array(strides, dtype=np.float32)
        self.per_level_num_anchors = per_level_num_anchors
        self.per_level_anchor_sizes = self.anchor_sizes.reshape(
            len(self.anchor_sizes) // self.per_level_num_anchors,
            self.per_level_num_anchors, 2)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image anchors
        '''
        one_image_anchors = []
        for index, per_level_anchors in enumerate(self.per_level_anchor_sizes):
            feature_map_anchors = self.generate_anchors_on_feature_map(
                per_level_anchors, fpn_feature_sizes[index],
                self.strides[index])
            one_image_anchors.append(feature_map_anchors)

        # if input size:[416,416]
        # one_image_anchors shape:[[52, 52, 3, 5],[26, 26, 3, 5],[13, 13, 3, 5]]
        # per anchor format:[grids_x_index,grids_y_index,relative_anchor_w,relative_anchor_h,stride]
        return one_image_anchors

    def generate_anchors_on_feature_map(self, per_level_anchors,
                                        feature_map_size, stride):
        '''
        generate one feature map anchors
        '''
        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (np.arange(0, feature_map_size[0]))
        shifts_y = (np.arange(0, feature_map_size[1]))

        # shifts shape:[w,h,2] -> [w,h,1,2] -> [w,h,3,2] -> [h,w,3,2]
        shifts = np.array([[[shift_x, shift_y] for shift_y in shifts_y]
                           for shift_x in shifts_x],
                          dtype=np.float32)
        shifts = np.expand_dims(shifts, axis=2)
        shifts = np.tile(shifts, (1, 1, self.per_level_num_anchors, 1))
        shifts = np.transpose(shifts, axes=(1, 0, 2, 3))

        # per_level_anchors shape:[3,2] -> [1,1,3,2] -> [h,w,3,2]
        all_anchors_wh = np.expand_dims(np.expand_dims(per_level_anchors,
                                                       axis=0),
                                        axis=0)
        all_anchors_wh = np.tile(all_anchors_wh,
                                 (shifts.shape[0], shifts.shape[1], 1, 1))

        # all_strides shape:[] -> [1,1,1,1] -> [h,w,3,1]
        all_strides = np.expand_dims(np.expand_dims(stride, axis=0), axis=0)
        all_strides = np.expand_dims(np.expand_dims(all_strides, axis=0),
                                     axis=0)
        all_strides = np.tile(
            all_strides,
            (shifts.shape[0], shifts.shape[1], shifts.shape[2], 1))

        # anchors_wh is relative wh on each feature map
        all_anchors_wh = all_anchors_wh / all_strides

        feature_map_anchors = np.concatenate(
            (shifts, all_anchors_wh, all_strides), axis=-1)

        # feature_map_anchors format: [h,w,3,5],3:self.per_level_num_anchors,5:[grids_x_index,grids_y_index,relative_anchor_w,relative_anchor_h,stride]
        return feature_map_anchors


class YoloxAnchors:

    def __init__(self, strides=[8, 16, 32]):
        self.strides = np.array(strides, dtype=np.float32)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image grid strides
        '''
        one_image_grid_strides = []
        for stride, fpn_feature_size in zip(self.strides, fpn_feature_sizes):
            feature_map_grid_strides = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            one_image_grid_strides.append(feature_map_grid_strides)

        # if input size:[640,640]
        # one_image_positions shape:[[80, 80, 3],[40, 40, 3],[20, 20, 3]]
        # per position format:[grids_x_index,grids_y_index,stride]
        return one_image_grid_strides

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        '''
        generate one feature map positions
        '''
        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = np.arange(0, feature_map_size[0]) + 0.5
        shifts_y = np.arange(0, feature_map_size[1]) + 0.5

        # feature_map_grids shape:[w,h,2] -> [h,w,2]
        feature_map_grid_centers = np.array([[[shift_x, shift_y]
                                              for shift_y in shifts_y]
                                             for shift_x in shifts_x],
                                            dtype=np.float32)
        feature_map_grid_centers = np.transpose(feature_map_grid_centers,
                                                axes=(1, 0, 2))
        feature_map_grid_centers = np.ascontiguousarray(
            feature_map_grid_centers, dtype=np.float32)

        # feature_map_strides shape:[] -> [1,1,1] -> [h,w,1]
        feature_map_strides = np.expand_dims(np.expand_dims(np.expand_dims(
            stride, axis=0),
                                                            axis=0),
                                             axis=0)

        feature_map_strides = np.tile(feature_map_strides,
                                      (feature_map_grid_centers.shape[0],
                                       feature_map_grid_centers.shape[1], 1))

        feature_map_grid_center_strides = np.concatenate(
            (feature_map_grid_centers, feature_map_strides), axis=-1)

        # feature_map_grid_center_strides format: [point_nums,3],3:[scale_grid_x_center,scale_grid_y_center,stride]
        return feature_map_grid_center_strides


class TTFNetPositions:

    def __init__(self):
        pass

    def __call__(self, feature_map_size):
        '''
        generate one image positions
        '''

        one_image_positions = self.generate_positions_on_feature_map(
            feature_map_size)

        # if input size:[640,640]
        # one_image_positions shape:[160, 160, 2]
        # per position format:[x_center,y_center]
        return one_image_positions

    def generate_positions_on_feature_map(self, feature_map_size):
        '''
        generate one feature map positions
        '''
        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (np.arange(0, feature_map_size[0]) + 0.5)
        shifts_y = (np.arange(0, feature_map_size[1]) + 0.5)

        # feature_map_positions shape:[w,h,2] -> [h,w,2]
        feature_map_positions = np.array([[[shift_x, shift_y]
                                           for shift_y in shifts_y]
                                          for shift_x in shifts_x],
                                         dtype=np.float32)
        feature_map_positions = np.transpose(feature_map_positions,
                                             axes=(1, 0, 2))
        feature_map_positions = np.ascontiguousarray(feature_map_positions,
                                                     dtype=np.float32)

        # feature_map_positions format: [point_nums,2],2:[x_center,y_center]
        return feature_map_positions


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

    areas = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]
    ratios = [0.5, 1, 2]
    scales = [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)]
    strides = [8, 16, 32, 64, 128]
    image_w, image_h = 640, 640
    fpn_feature_sizes = [[
        math.ceil(image_w / stride),
        math.ceil(image_h / stride)
    ] for stride in strides]

    anchors = RetinaAnchors(areas=areas,
                            ratios=ratios,
                            scales=scales,
                            strides=strides)
    one_image_anchors = anchors(fpn_feature_sizes)

    for per_level_anchors in one_image_anchors:
        print('1111', per_level_anchors.shape)

    strides = [8, 16, 32, 64, 128]
    image_w, image_h = 640, 640
    fpn_feature_sizes = [[
        math.ceil(image_w / stride),
        math.ceil(image_h / stride)
    ] for stride in strides]

    positions = FCOSPositions(strides)
    one_image_positions = positions(fpn_feature_sizes)

    for per_level_positions in one_image_positions:
        print('2222', per_level_positions.shape)

    anchor_sizes = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                    [59, 119], [116, 90], [156, 198], [373, 326]]
    strides = [8, 16, 32]
    per_level_num_anchors = 3
    image_w, image_h = 416, 416
    fpn_feature_sizes = [[
        math.ceil(image_w / stride),
        math.ceil(image_h / stride)
    ] for stride in strides]

    anchors = Yolov3Anchors(anchor_sizes=anchor_sizes,
                            strides=strides,
                            per_level_num_anchors=per_level_num_anchors)
    one_image_anchors = anchors(fpn_feature_sizes)

    for per_level_anchors in one_image_anchors:
        print('3333', per_level_anchors.shape)

    strides = [8, 16, 32]
    image_w, image_h = 640, 640
    fpn_feature_sizes = [[
        math.ceil(image_w / stride),
        math.ceil(image_h / stride)
    ] for stride in strides]

    grid_strides = YoloxAnchors(strides)
    one_image_grid_center_strides = grid_strides(fpn_feature_sizes)

    for per_level_grid_center_strides in one_image_grid_center_strides:
        print('4444', per_level_grid_center_strides.shape)

    image_w, image_h = 640, 640
    feature_map_size = [160, 160]

    positions = TTFNetPositions()
    one_image_positions = positions(feature_map_size)
    print('5555', one_image_positions.shape)