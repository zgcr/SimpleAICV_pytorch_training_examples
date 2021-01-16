import math
import numpy as np
import torch
import torch.nn as nn


class RetinaAnchors(nn.Module):
    def __init__(self, areas, ratios, scales, strides):
        super(RetinaAnchors, self).__init__()
        self.areas = areas
        self.ratios = ratios
        self.scales = scales
        self.strides = strides

    def forward(self, batch_size, fpn_feature_sizes):
        """
        generate batch anchors
        """
        device = fpn_feature_sizes.device
        one_sample_anchors = []
        for index, area in enumerate(self.areas):
            base_anchors = self.generate_base_anchors(area, self.scales,
                                                      self.ratios)
            featrue_anchors = self.generate_anchors_on_feature_map(
                base_anchors, fpn_feature_sizes[index], self.strides[index])
            featrue_anchors = featrue_anchors.to(device)
            one_sample_anchors.append(featrue_anchors)

        batch_anchors = []
        for per_level_featrue_anchors in one_sample_anchors:
            per_level_featrue_anchors = per_level_featrue_anchors.unsqueeze(
                0).repeat(batch_size, 1, 1)
            batch_anchors.append(per_level_featrue_anchors)

        # if input size:[B,3,640,640]
        # batch_anchors shape:[[B, 57600, 4],[B, 14400, 4],[B, 3600, 4],[B, 900, 4],[B, 225, 4]]
        # per anchor format:[x_min,y_min,x_max,y_max]
        return batch_anchors

    def generate_base_anchors(self, area, scales, ratios):
        """
        generate base anchor
        """
        # get w,h aspect ratio,shape:[9,2]
        aspects = torch.tensor([[[s * math.sqrt(r), s * math.sqrt(1 / r)]
                                 for s in scales]
                                for r in ratios]).view(-1, 2)
        # base anchor for each position on feature map,shape[9,4]
        base_anchors = torch.zeros((len(scales) * len(ratios), 4))

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
        """
        generate all anchors on a feature map
        """
        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (torch.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (torch.arange(0, feature_map_size[1]) + 0.5) * stride

        # shifts shape:[w,h,2] -> [w,h,4] -> [w,h,1,4]
        shifts = torch.tensor([[[shift_x, shift_y] for shift_y in shifts_y]
                               for shift_x in shifts_x]).repeat(1, 1,
                                                                2).unsqueeze(2)

        # base anchors shape:[9,4] -> [1,1,9,4]
        base_anchors = base_anchors.unsqueeze(0).unsqueeze(0)
        # generate all featrue map anchors on each feature map points
        # featrue map anchors shape:[w,h,9,4] -> [h,w,9,4] -> [h*w*9,4]
        feature_map_anchors = (base_anchors + shifts).permute(
            1, 0, 2, 3).contiguous().view(-1, 4)

        # feature_map_anchors format: [anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        return feature_map_anchors


class FCOSPositions(nn.Module):
    def __init__(self, strides):
        super(FCOSPositions, self).__init__()
        self.strides = strides

    def forward(self, batch_size, fpn_feature_sizes):
        """
        generate batch positions
        """
        device = fpn_feature_sizes.device

        one_sample_positions = []
        for stride, fpn_feature_size in zip(self.strides, fpn_feature_sizes):
            featrue_positions = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            featrue_positions = featrue_positions.to(device)
            one_sample_positions.append(featrue_positions)

        batch_positions = []
        for per_level_featrue_positions in one_sample_positions:
            per_level_featrue_positions = per_level_featrue_positions.unsqueeze(
                0).repeat(batch_size, 1, 1, 1)
            batch_positions.append(per_level_featrue_positions)

        # if input size:[B,3,640,640]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]
        # per position format:[x_center,y_center]
        return batch_positions

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        """
        generate all positions on a feature map
        """

        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (torch.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (torch.arange(0, feature_map_size[1]) + 0.5) * stride

        # feature_map_positions shape:[w,h,2] -> [h,w,2] -> [h*w,2]
        feature_map_positions = torch.tensor([[[shift_x, shift_y]
                                               for shift_y in shifts_y]
                                              for shift_x in shifts_x
                                              ]).permute(1, 0, 2).contiguous()

        # feature_map_positions format: [point_nums,2],2:[x_center,y_center]
        return feature_map_positions


class YOLOV3Anchors(nn.Module):
    def __init__(self, anchor_sizes, per_level_num_anchors, strides):
        super(YOLOV3Anchors, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.per_level_num_anchors = per_level_num_anchors
        self.strides = strides

    def forward(self, batch_size, fpn_feature_sizes):
        """
        generate batch anchors
        """
        assert len(self.anchor_sizes
                   ) == self.per_level_num_anchors * len(fpn_feature_sizes)

        self.per_level_anchor_sizes = self.anchor_sizes.view(
            self.per_level_num_anchors,
            len(self.anchor_sizes) // self.per_level_num_anchors, 2)

        device = fpn_feature_sizes.device
        one_sample_anchors = []
        for index, per_level_anchors in enumerate(self.per_level_anchor_sizes):
            feature_map_anchors = self.generate_anchors_on_feature_map(
                per_level_anchors, fpn_feature_sizes[index],
                self.strides[index])
            feature_map_anchors = feature_map_anchors.to(device)
            one_sample_anchors.append(feature_map_anchors)

        batch_anchors = []
        for per_level_featrue_anchors in one_sample_anchors:
            per_level_featrue_anchors = per_level_featrue_anchors.unsqueeze(
                0).repeat(batch_size, 1, 1, 1, 1)
            batch_anchors.append(per_level_featrue_anchors)

        # if input size:[B,3,416,416]
        # batch_anchors shape:[[B, 52, 52, 3, 5],[B, 26, 26, 3, 5],[B, 13, 13, 3, 5]]
        # per anchor format:[grids_x_index,grids_y_index,anchor_w,anchor_h,stride]
        return batch_anchors

    def generate_anchors_on_feature_map(self, per_level_anchors,
                                        feature_map_size, stride):
        """
        generate all anchors on a feature map
        """
        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (torch.arange(0, feature_map_size[0]))
        shifts_y = (torch.arange(0, feature_map_size[1]))

        # shifts shape:[w,h,2] -> [w,h,1,2] -> [w,h,3,2] -> [h,w,3,2]
        shifts = torch.tensor([[[shift_x, shift_y] for shift_y in shifts_y]
                               for shift_x in shifts_x]).unsqueeze(2).repeat(
                                   1, 1, self.per_level_num_anchors,
                                   1).permute(1, 0, 2, 3)

        # per_level_anchors shape:[3,2] -> [1,1,3,2] -> [h,w,3,2]
        all_anchors_wh = per_level_anchors.unsqueeze(0).unsqueeze(0).repeat(
            shifts.shape[0], shifts.shape[1], 1, 1).type_as(shifts)

        # all_strides shape:[] -> [1,1,1,1] -> [h,w,3,1]
        all_strides = stride.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(
            0).repeat(shifts.shape[0], shifts.shape[1], shifts.shape[2],
                      1).type_as(shifts)

        feature_map_anchors = torch.cat([shifts, all_anchors_wh, all_strides],
                                        axis=-1)

        # feature_map_anchors format: [h,w,3,5],3:self.per_level_num_anchors,5:[grids_x_index,grids_y_index,anchor_w,anchor_h,stride]
        return feature_map_anchors


if __name__ == '__main__':
    areas = torch.tensor([[32, 32], [64, 64], [128, 128], [256, 256],
                          [512, 512]])
    ratios = torch.tensor([0.5, 1, 2])
    scales = torch.tensor([2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)])
    strides = torch.tensor([8, 16, 32, 64, 128], dtype=torch.float)
    image_w, image_h = 640, 640
    fpn_feature_sizes = torch.tensor(
        [[torch.ceil(image_w / stride),
          torch.ceil(image_h / stride)] for stride in strides])

    anchors = RetinaAnchors(areas, ratios, scales, strides)
    anchors = anchors(1, fpn_feature_sizes)

    for per_level_anchors in anchors:
        print("1111", per_level_anchors.shape)

    strides = torch.tensor([8, 16, 32, 64, 128], dtype=torch.float)
    image_w, image_h = 640, 640
    fpn_feature_sizes = torch.tensor(
        [[torch.ceil(image_w / stride),
          torch.ceil(image_h / stride)] for stride in strides])
    positions = FCOSPositions(strides)
    positions = positions(1, fpn_feature_sizes)
    for per_level_positions in positions:
        print("2222", per_level_positions.shape)

    anchor_sizes = torch.tensor(
        [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
         [116, 90], [156, 198], [373, 326]],
        dtype=torch.float)
    strides = torch.tensor([8, 16, 32], dtype=torch.float)
    image_w, image_h = 416, 416
    fpn_feature_sizes = torch.tensor(
        [[torch.ceil(image_w / stride),
          torch.ceil(image_h / stride)] for stride in strides])

    anchors = YOLOV3Anchors(anchor_sizes=anchor_sizes,
                            per_level_num_anchors=3,
                            strides=strides)
    all_anchors = anchors(3, fpn_feature_sizes)
    for per_level_anchors in all_anchors:
        print("3333", per_level_anchors.shape)
