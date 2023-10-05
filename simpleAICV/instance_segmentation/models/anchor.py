import numpy as np
from itertools import product


class YOLACTAnchors:

    def __init__(self,
                 resize=544,
                 scales=[24, 48, 96, 192, 384],
                 ratios=[1, 1 / 2, 2],
                 strides=[8, 16, 32, 64, 128]):
        self.resize = resize
        self.ratios = np.array(ratios, dtype=np.float32)
        self.scales = np.array(scales, dtype=np.float32)
        self.strides = np.array(strides, dtype=np.float32)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image anchors
        '''
        assert len(self.scales) == len(self.strides) == len(fpn_feature_sizes)

        one_image_anchors = []
        for index, (per_feature_size_w,
                    per_feature_size_h) in enumerate(fpn_feature_sizes):
            feature_anchors = self.generate_anchors_on_feature_map(
                self.ratios, self.resize, per_feature_size_h,
                per_feature_size_w, self.scales[index])
            one_image_anchors.append(feature_anchors)

        return one_image_anchors

    def generate_anchors_on_feature_map(self, ratios, resize, feature_map_h,
                                        feature_map_w, scale):
        feature_anchors = []
        for j, i in product(range(feature_map_h), range(feature_map_w)):
            # + 0.5 because priors are in center
            x = (i + 0.5) / feature_map_w
            y = (j + 0.5) / feature_map_h

            for per_ratio in ratios:
                per_ratio = np.sqrt(per_ratio)
                w = scale * per_ratio / resize
                h = scale / per_ratio / resize

                feature_anchors.append([x, y, w, h])

        feature_anchors = np.array(feature_anchors, dtype=np.float32)

        return feature_anchors