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
