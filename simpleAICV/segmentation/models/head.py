import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Solov2InsHead(nn.Module):
    def __init__(self,
                 inplanes,
                 planes=512,
                 num_classes=80,
                 num_kernels=256,
                 num_grids=[40, 36, 24, 16, 12],
                 num_layers=4,
                 prior=0.01,
                 use_gn=True):
        super(Solov2InsHead, self).__init__()
        self.num_grids = num_grids
        cate_layers = []
        for i in range(num_layers):
            cate_layers.append(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_gn is False) if i ==
                0 else nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=use_gn is False))
            if use_gn:
                cate_layers.append(nn.GroupNorm(32, planes))
            cate_layers.append(nn.ReLU(inplace=True))
        self.cate_head = nn.Sequential(*cate_layers)
        self.cate_out = nn.Conv2d(planes,
                                  num_classes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=True)

        kernel_layers = []
        for i in range(num_layers):
            kernel_layers.append(
                nn.Conv2d(inplanes + 2,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_gn is False) if i ==
                0 else nn.Conv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=use_gn is False))
            if use_gn:
                kernel_layers.append(nn.GroupNorm(32, planes))
            kernel_layers.append(nn.ReLU(inplace=True))
        self.kernel_head = nn.Sequential(*kernel_layers)
        self.kernel_out = nn.Conv2d(planes,
                                    num_kernels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cate_out.bias.data.fill_(b)

    def forward(self, inputs):
        assert len(inputs) == len(self.num_grids), 'wrong inputs or num_grids!'

        [P2, P3, P4, P5, P6] = inputs
        P2 = F.interpolate(P2, size=(P3.shape[2], P3.shape[3]), mode='nearest')
        P6 = F.interpolate(P6, size=(P5.shape[2], P5.shape[3]), mode='nearest')

        kernel_outs, cate_outs = [], []
        for i, feature in enumerate(inputs):
            feature = self.coord_feat(feature)
            kernel_feature = F.interpolate(feature,
                                           size=(self.num_grids[i],
                                                 self.num_grids[i]),
                                           mode='nearest')
            cate_feature = kernel_feature[:, :-2, :, :]

            cate_feature = self.cate_head(cate_feature)
            cate_out = self.cate_out(cate_feature)
            kernel_feature = self.kernel_head(kernel_feature)
            kernel_out = self.kernel_out(kernel_feature)

            cate_out = cate_out.permute(0, 2, 3, 1).contiguous()
            cate_out = torch.sigmoid(cate_out)
            kernel_out = kernel_out.permute(0, 2, 3, 1).contiguous()

            cate_outs.append(cate_out)
            kernel_outs.append(kernel_out)

        return cate_outs, kernel_outs

    def coord_feat(self, feature):
        x_range = torch.linspace(-1,
                                 1,
                                 feature.shape[-1],
                                 device=feature.device,
                                 dtype=feature.dtype)
        y_range = torch.linspace(-1,
                                 1,
                                 feature.shape[-2],
                                 device=feature.device,
                                 dtype=feature.dtype)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feature.shape[0], 1, -1, -1])
        x = x.expand([feature.shape[0], 1, -1, -1])
        coord_feature = torch.cat([x, y], 1)

        return torch.cat([feature, coord_feature], 1)


class Solov2MaskHead(nn.Module):
    def __init__(self,
                 inplanes,
                 planes=128,
                 num_masks=256,
                 num_layers=4,
                 use_gn=True):
        super(Solov2MaskHead, self).__init__()
        self.P2_1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_gn is False),
            nn.GroupNorm(32, planes) if use_gn else nn.Sequential(), nn.ReLU())
        self.P3_1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_gn is False),
            nn.GroupNorm(32, planes) if use_gn else nn.Sequential(), nn.ReLU())
        self.P4_1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_gn is False),
            nn.GroupNorm(32, planes) if use_gn else nn.Sequential(), nn.ReLU())
        self.P4_2 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_gn is False),
            nn.GroupNorm(32, planes) if use_gn else nn.Sequential(), nn.ReLU())
        self.P5_1 = nn.Sequential(
            nn.Conv2d(inplanes + 2,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_gn is False),
            nn.GroupNorm(32, planes) if use_gn else nn.Sequential(), nn.ReLU())
        self.P5_2 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_gn is False),
            nn.GroupNorm(32, planes) if use_gn else nn.Sequential(), nn.ReLU())
        self.P5_3 = nn.Sequential(
            nn.Conv2d(planes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_gn is False),
            nn.GroupNorm(32, planes) if use_gn else nn.Sequential(), nn.ReLU())

        self.mask_out = nn.Sequential(
            nn.Conv2d(planes,
                      num_masks,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=use_gn is False),
            nn.GroupNorm(32, num_masks) if use_gn else nn.Sequential(),
            nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, inputs):
        [P2, P3, P4, P5] = inputs

        P2 = self.P2_1(P2)

        P5 = self.coord_feat(P5)
        P5 = self.P5_1(P5)
        P5 = F.interpolate(P5, size=(P4.shape[2], P4.shape[3]), mode='nearest')
        P5 = self.P5_2(P5)
        P5 = F.interpolate(P5, size=(P3.shape[2], P3.shape[3]), mode='nearest')
        P5 = self.P5_3(P5)
        P2 = F.interpolate(P5, size=(P2.shape[2], P2.shape[3]),
                           mode='nearest') + P2

        del P5

        P4 = self.P4_1(P4)
        P4 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]), mode='nearest')
        P4 = self.P4_2(P4)
        P2 = F.interpolate(P4, size=(P2.shape[2], P2.shape[3]),
                           mode='nearest') + P2

        del P4

        P3 = self.P3_1(P3)
        P2 = F.interpolate(P3, size=(P2.shape[2], P2.shape[3]),
                           mode='nearest') + P2

        del P3

        mask_out = self.mask_out(P2)

        return mask_out

    def coord_feat(self, feature):
        x_range = torch.linspace(-1,
                                 1,
                                 feature.shape[-1],
                                 device=feature.device,
                                 dtype=feature.dtype)
        y_range = torch.linspace(-1,
                                 1,
                                 feature.shape[-2],
                                 device=feature.device,
                                 dtype=feature.dtype)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feature.shape[0], 1, -1, -1])
        x = x.expand([feature.shape[0], 1, -1, -1])
        coord_feature = torch.cat([x, y], 1)

        return torch.cat([feature, coord_feature], 1)


class CondInstPublicHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_classes,
                 fcn_head_layers=3,
                 num_masks=8,
                 num_layers=4,
                 prior=0.01,
                 use_gn=True):
        super(CondInstPublicHead, self).__init__()
        cls_layers = []
        for _ in range(num_layers):
            cls_layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_gn is False))
            if use_gn:
                cls_layers.append(nn.GroupNorm(32, inplanes))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*cls_layers)

        reg_layers = []
        for _ in range(num_layers):
            reg_layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_gn is False))
            if use_gn:
                reg_layers.append(nn.GroupNorm(32, inplanes))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*reg_layers)

        self.cls_out = nn.Conv2d(inplanes,
                                 num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.reg_out = nn.Conv2d(inplanes,
                                 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.center_out = nn.Conv2d(inplanes,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.scales = nn.Parameter(
            torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32))

        controller_planes = 0
        for i in range(fcn_head_layers):
            if i == 0:
                # add weight nums
                controller_planes += int((num_masks + 2) * num_masks)
                # add bias nums
                controller_planes += int(num_masks)
            elif i == fcn_head_layers - 1:
                # add weight nums
                controller_planes += int(num_masks)
                # add bias nums
                controller_planes += 1
            else:
                # add weight nums
                controller_planes += int(num_masks * num_masks)
                # add bias nums
                controller_planes += int(num_masks)

        # controller_planes = 169,controller_planes num is related to mask loss
        self.controller_out = nn.Conv2d(inplanes,
                                        controller_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_out.bias.data.fill_(b)

    def forward(self, inputs):
        cls_heads, reg_heads, center_heads, controllers_heads = [], [], [], []
        for feature, scale in zip(inputs, self.scales):
            cls_feature = self.cls_head(feature)
            reg_feature = self.reg_head(feature)

            del feature

            cls_out = self.cls_out(cls_feature)
            reg_out = self.reg_out(reg_feature)
            center_out = self.center_out(reg_feature)
            controller_out = self.controller_out(reg_feature)

            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = torch.sigmoid(cls_out)
            # [N,4,H,W] -> [N,H,W,4]
            reg_out = reg_out.permute(0, 2, 3, 1).contiguous()
            reg_out = torch.exp(reg_out) * torch.exp(scale)
            # [N,1,H,W] -> [N,H,W,1]
            center_out = center_out.permute(0, 2, 3, 1).contiguous()
            center_out = torch.sigmoid(center_out)
            # [N,169,H,W] -> [N,H,W,169]
            controller_out = controller_out.permute(0, 2, 3, 1).contiguous()

            cls_heads.append(cls_out)
            reg_heads.append(reg_out)
            center_heads.append(center_out)
            controllers_heads.append(controller_out)

        del inputs

        return cls_heads, reg_heads, center_heads, controllers_heads


class CondInstMaskBranch(nn.Module):
    def __init__(self, inplanes, planes=128, num_layers=4, num_masks=8):
        super(CondInstMaskBranch, self).__init__()
        self.P3_1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.P4_1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.P5_1 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))

        mask_layers = []
        for _ in range(num_layers):
            mask_layers.append(
                nn.Conv2d(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            mask_layers.append(nn.BatchNorm2d(planes))
            mask_layers.append(nn.ReLU(inplace=True))
        self.mask_head = nn.Sequential(*mask_layers)
        self.mask_out = nn.Conv2d(planes,
                                  num_masks,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, inputs):
        [P3, P4, P5] = inputs

        P3 = self.P3_1(P3)
        P4 = self.P4_1(P4)
        P3 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3

        del P4

        P5 = self.P5_1(P5)
        P3 = F.interpolate(P5, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3

        del P5

        P3 = self.mask_head(P3)
        mask_out = self.mask_out(P3)

        del P3

        # [N,8,H,W] -> [N,H,W,8]
        mask_out = mask_out.permute(0, 2, 3, 1).contiguous()

        return mask_out


if __name__ == '__main__':
    image_h, image_w = 640, 640
    from fpn import Solov2FPN
    image_h, image_w = 640, 640
    fpn = Solov2FPN(256, 512, 1024, 2048, 256)
    C2, C3, C4, C5 = torch.randn(3, 256, 160, 160), torch.randn(
        3, 512, 80, 80), torch.randn(3, 1024, 40,
                                     40), torch.randn(3, 2048, 20, 20)
    features = fpn([C2, C3, C4, C5])

    for f in features:
        print('1111', f.shape)

    model1 = Solov2InsHead(256,
                           planes=512,
                           num_classes=80,
                           num_kernels=256,
                           num_grids=[40, 36, 24, 16, 12],
                           num_layers=4,
                           prior=0.01,
                           use_gn=True)

    cate_outs, kernel_outs = model1(features)

    for x, y in zip(cate_outs, kernel_outs):
        print('2222', x.shape, y.shape)

    model2 = Solov2MaskHead(256,
                            planes=128,
                            num_masks=256,
                            num_layers=4,
                            use_gn=True)

    mask_out = model2(features[:-1])

    print('3333', mask_out.shape)

    model3 = CondInstPublicHead(256,
                                80,
                                fcn_head_layers=3,
                                num_masks=8,
                                num_layers=4,
                                prior=0.01,
                                use_gn=True)

    cls_heads, reg_heads, center_heads, controllers_heads = model3(features)

    for x, y, z, w in zip(cls_heads, reg_heads, center_heads,
                          controllers_heads):
        print("4444", x.shape, y.shape, z.shape, w.shape)

    model4 = CondInstMaskBranch(256, planes=128, num_layers=4, num_masks=8)

    mask_out = model4(features[0:3])
    print("5555", mask_out.shape)
