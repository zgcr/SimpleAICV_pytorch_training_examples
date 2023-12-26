import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'transxnet_t',
    'transxnet_s',
    'transxnet_b',
]


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 dilation=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=768,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 has_bn=True,
                 has_act=False):
        super(PatchEmbed, self).__init__()
        self.proj = ConvBnActBlock(inplanes,
                                   planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=1,
                                   dilation=1,
                                   has_bn=has_bn,
                                   has_act=has_act)

    def forward(self, x):
        x = self.proj(x)

        return x


class Attention(nn.Module):

    def __init__(self, inplanes, head_nums=1, sr_ratio=1):
        super(Attention, self).__init__()
        assert inplanes % head_nums == 0
        self.head_nums = head_nums

        head_inplanes = inplanes // head_nums
        self.scale = head_inplanes**-0.5

        self.q = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.kv = nn.Conv2d(inplanes, inplanes * 2, kernel_size=1)

        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvBnActBlock(inplanes,
                               inplanes,
                               kernel_size=sr_ratio + 3,
                               stride=sr_ratio,
                               padding=(sr_ratio + 3) // 2,
                               groups=inplanes,
                               dilation=1,
                               has_bn=True,
                               has_act=True),
                ConvBnActBlock(inplanes,
                               inplanes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=inplanes,
                               dilation=1,
                               has_bn=True,
                               has_act=False),
            )
        else:
            self.sr = nn.Identity()

        self.local_conv = nn.Conv2d(inplanes,
                                    inplanes,
                                    kernel_size=3,
                                    padding=1,
                                    groups=inplanes)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape

        q = self.q(x)
        q = q.reshape(B, self.head_nums, C // self.head_nums,
                      -1).transpose(-1, -2)

        kv = self.sr(x)
        kv = self.local_conv(kv) + kv

        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.head_nums, C // self.head_nums, -1)
        v = v.reshape(B, self.head_nums, C // self.head_nums,
                      -1).transpose(-1, -2)

        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc,
                                                 size=attn.shape[2:],
                                                 mode='nearest')
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)

        x = (attn @ v).transpose(-1, -2)
        x = x.reshape(B, C, H, W)

        return x


class DynamicConv2d(nn.Module):
    '''
    IDConv
    '''

    def __init__(self,
                 inplanes,
                 kernel_size=3,
                 num_groups=2,
                 reduction_ratio=4):
        super(DynamicConv2d, self).__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."

        self.num_groups = num_groups
        self.K = kernel_size

        self.weight = nn.Parameter(torch.zeros(num_groups, inplanes,
                                               kernel_size, kernel_size),
                                   requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size,
                                                      kernel_size))
        self.proj = nn.Sequential(
            ConvBnActBlock(inplanes,
                           inplanes // reduction_ratio,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           dilation=1,
                           has_bn=True,
                           has_act=True),
            nn.Conv2d(inplanes // reduction_ratio,
                      inplanes * num_groups,
                      kernel_size=1),
        )

        self.bias = nn.Parameter(torch.zeros(num_groups, inplanes),
                                 requires_grad=True)

        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape

        scale = self.proj(self.pool(x))
        scale = scale.reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)

        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
        scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)

        bias = scale * self.bias.unsqueeze(0)
        bias = torch.sum(bias, dim=1).flatten(0)

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)
        x = x.reshape(B, C, H, W)

        return x


class HybridTokenMixer(nn.Module):
    '''
    D-Mixer
    '''

    def __init__(self,
                 inplanes,
                 kernel_size=3,
                 num_groups=2,
                 head_nums=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super(HybridTokenMixer, self).__init__()
        assert inplanes % 2 == 0

        self.local_unit = DynamicConv2d(inplanes=inplanes // 2,
                                        kernel_size=kernel_size,
                                        num_groups=num_groups)
        self.global_unit = Attention(inplanes=inplanes // 2,
                                     head_nums=head_nums,
                                     sr_ratio=sr_ratio)

        inter_planes = max(16, inplanes // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(inplanes,
                      inplanes,
                      kernel_size=3,
                      padding=1,
                      groups=inplanes),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, inter_planes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inter_planes),
            nn.Conv2d(inter_planes, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)

        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x

        return x


class MultiScaleDWConv(nn.Module):

    def __init__(self, inplanes, scales=(1, 3, 5, 7)):
        super(MultiScaleDWConv, self).__init__()
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scales)):
            if i == 0:
                channels = inplanes - inplanes // len(scales) * (len(scales) -
                                                                 1)
            else:
                channels = inplanes // len(scales)

            conv = nn.Conv2d(channels,
                             channels,
                             kernel_size=scales[i],
                             padding=scales[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)

        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)

        return x


class Mlp(nn.Module):
    """
    MS FFN
    """

    def __init__(self, inplanes, hidden_planes, dropout_prob=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(inplanes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_planes),
        )
        self.dwconv = MultiScaleDWConv(inplanes=hidden_planes,
                                       scales=(1, 3, 5, 7))
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_planes, inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes),
        )
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DropPathBlock(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    if drop_path_prob = 0. ,not use DropPath
    """

    def __init__(self, drop_path_prob=0., scale_by_keep=True):
        super(DropPathBlock, self).__init__()
        assert drop_path_prob >= 0.

        self.drop_path_prob = drop_path_prob
        self.keep_path_prob = 1 - drop_path_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_path_prob == 0. or not self.training:
            return x

        b = x.shape[0]
        device = x.device

        # work with diff dim tensors, not just 2D ConvNets
        shape = (b, ) + (1, ) * (len(x.shape) - 1)
        random_weight = torch.empty(shape).to(device).bernoulli_(
            self.keep_path_prob)

        if self.keep_path_prob > 0. and self.scale_by_keep:
            random_weight.div_(self.keep_path_prob)

        x = random_weight * x

        return x


class LayerScale(nn.Module):

    def __init__(self, inplanes, init_value=1e-5):
        super(LayerScale, self).__init__()
        self.weight = nn.Parameter(torch.ones(inplanes, 1, 1, 1) * init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(inplanes), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])

        return x


class Block(nn.Module):

    def __init__(self,
                 inplanes,
                 hybridtokenmixer_kernel_size=3,
                 num_groups=2,
                 head_nums=1,
                 mlp_ratio=4,
                 sr_ratio=1,
                 dropout_prob=0.,
                 drop_path_prob=0.):

        super(Block, self).__init__()
        mlp_hidden_planes = int(inplanes * mlp_ratio)

        self.pos_embed = nn.Conv2d(inplanes,
                                   inplanes,
                                   kernel_size=7,
                                   padding=3,
                                   groups=inplanes)
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.token_mixer = HybridTokenMixer(
            inplanes,
            kernel_size=hybridtokenmixer_kernel_size,
            num_groups=num_groups,
            head_nums=head_nums,
            sr_ratio=sr_ratio)
        self.norm2 = nn.BatchNorm2d(inplanes)
        self.mlp = Mlp(inplanes=inplanes,
                       hidden_planes=mlp_hidden_planes,
                       dropout_prob=dropout_prob)

        self.layer_scale_1 = LayerScale(inplanes, 1e-5)
        self.layer_scale_2 = LayerScale(inplanes, 1e-5)

        # if test model,drop_path must set to 0.
        self.drop_path = DropPathBlock(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x, relative_pos_enc=None):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(
            self.layer_scale_1(
                self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x


class TransXNet(nn.Module):

    def __init__(self,
                 image_size=224,
                 inplanes=3,
                 layers=[3, 3, 9, 3],
                 embedding_planes=[48, 96, 224, 448],
                 kernel_size=[7, 7, 7, 7],
                 num_groups=[2, 2, 2, 2],
                 sr_ratio=[8, 4, 2, 1],
                 head_nums=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 dropout_prob=0.,
                 drop_path_prob=0.,
                 num_classes=1000):

        super(TransXNet, self).__init__()
        in_stride = 4

        self.relative_pos_enc = []
        image_size = (image_size, image_size)
        image_size = [
            math.ceil(image_size[0] / in_stride),
            math.ceil(image_size[1] / in_stride)
        ]
        for i in range(4):
            num_patches = image_size[0] * image_size[1]
            sr_patches = math.ceil(image_size[0] / sr_ratio[i]) * math.ceil(
                image_size[1] / sr_ratio[i])
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, head_nums[i], num_patches,
                                         sr_patches),
                             requires_grad=True))
            image_size = [
                math.ceil(image_size[0] / 2),
                math.ceil(image_size[1] / 2)
            ]
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)

        self.patch_embed = PatchEmbed(inplanes=inplanes,
                                      planes=embedding_planes[0],
                                      kernel_size=7,
                                      stride=in_stride,
                                      padding=3,
                                      has_bn=True,
                                      has_act=False)

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = nn.ModuleList()
            for _ in range(layers[i]):
                stage.append(
                    Block(inplanes=embedding_planes[i],
                          hybridtokenmixer_kernel_size=kernel_size[i],
                          num_groups=num_groups[i],
                          head_nums=head_nums[i],
                          mlp_ratio=mlp_ratios[i],
                          sr_ratio=sr_ratio[i],
                          dropout_prob=dropout_prob,
                          drop_path_prob=drop_path_prob))
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embedding_planes[i] != embedding_planes[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(inplanes=embedding_planes[i],
                               planes=embedding_planes[i + 1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               has_bn=True,
                               has_act=False))
        self.network = nn.ModuleList(network)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embedding_planes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)

        pos_idx = 0
        for idx in range(len(self.network)):
            if idx in [0, 2, 4, 6]:
                for blk in self.network[idx]:
                    x = blk(x, self.relative_pos_enc[pos_idx])
                pos_idx += 1
            else:
                x = self.network[idx](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.head(x)

        return x


def _transxnet(layers, embedding_planes, kernel_size, num_groups, sr_ratio,
               head_nums, mlp_ratios, **kwargs):
    model = TransXNet(layers=layers,
                      embedding_planes=embedding_planes,
                      kernel_size=kernel_size,
                      num_groups=num_groups,
                      sr_ratio=sr_ratio,
                      head_nums=head_nums,
                      mlp_ratios=mlp_ratios,
                      **kwargs)

    return model


def transxnet_t(**kwargs):
    model = _transxnet(layers=[3, 3, 9, 3],
                       embedding_planes=[48, 96, 224, 448],
                       kernel_size=[7, 7, 7, 7],
                       num_groups=[2, 2, 2, 2],
                       sr_ratio=[8, 4, 2, 1],
                       head_nums=[1, 2, 4, 8],
                       mlp_ratios=[4, 4, 4, 4],
                       **kwargs)

    return model


def transxnet_s(**kwargs):
    model = _transxnet(layers=[4, 4, 12, 4],
                       embedding_planes=[64, 128, 320, 512],
                       kernel_size=[7, 7, 7, 7],
                       num_groups=[2, 2, 3, 4],
                       sr_ratio=[8, 4, 2, 1],
                       head_nums=[1, 2, 5, 8],
                       mlp_ratios=[6, 6, 4, 4],
                       **kwargs)

    return model


def transxnet_b(**kwargs):
    model = _transxnet(layers=[4, 4, 21, 4],
                       embedding_planes=[76, 152, 336, 672],
                       kernel_size=[7, 7, 7, 7],
                       num_groups=[2, 2, 4, 4],
                       sr_ratio=[8, 4, 2, 1],
                       head_nums=[2, 4, 8, 16],
                       mlp_ratios=[8, 8, 4, 4],
                       **kwargs)

    return model


def load_state_dict(saved_state_dict,
                    model,
                    excluded_layer_name=(),
                    loading_new_input_size_position_encoding_weight=False):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    loading_new_input_size_position_encoding_weight: default False, for vit net, loading a position encoding layer with new input size, set True
    only load layer parameters which has same layer name and same layer weight shape
    '''
    not_loaded_save_state_dict = []
    filtered_state_dict = {}
    for name, weight in saved_state_dict.items():
        if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name
        ) and weight.shape == model.state_dict()[name].shape:
            filtered_state_dict[name] = weight
        else:
            not_loaded_save_state_dict.append(name)

    position_encoding_already_loaded = False
    if 'position_encoding' in filtered_state_dict.keys():
        position_encoding_already_loaded = True

    # for vit net, loading a position encoding layer with new input size
    if loading_new_input_size_position_encoding_weight and not position_encoding_already_loaded:
        # assert position_encoding_layer name are unchanged for model and saved_model
        # assert class_token num are unchanged for model and saved_model
        # assert embedding_planes are unchanged for model and saved_model
        model_num_cls_token = model.cls_token.shape[1]
        model_embedding_planes = model.position_encoding.shape[2]
        model_encoding_shape = int(
            (model.position_encoding.shape[1] - model_num_cls_token)**0.5)
        encoding_layer_name, encoding_layer_weight = None, None
        for name, weight in saved_state_dict.items():
            if 'position_encoding' in name:
                encoding_layer_name = name
                encoding_layer_weight = weight
                break
        save_model_encoding_shape = int(
            (encoding_layer_weight.shape[1] - model_num_cls_token)**0.5)

        save_model_cls_token_weight = encoding_layer_weight[:, 0:
                                                            model_num_cls_token, :]
        save_model_position_weight = encoding_layer_weight[:,
                                                           model_num_cls_token:, :]
        save_model_position_weight = save_model_position_weight.reshape(
            -1, save_model_encoding_shape, save_model_encoding_shape,
            model_embedding_planes).permute(0, 3, 1, 2)
        save_model_position_weight = F.interpolate(save_model_position_weight,
                                                   size=(model_encoding_shape,
                                                         model_encoding_shape),
                                                   mode='bicubic',
                                                   align_corners=False)
        save_model_position_weight = save_model_position_weight.permute(
            0, 2, 3, 1).flatten(1, 2)
        model_encoding_layer_weight = torch.cat(
            (save_model_cls_token_weight, save_model_position_weight), dim=1)

        filtered_state_dict[encoding_layer_name] = model_encoding_layer_weight
        not_loaded_save_state_dict.remove('position_encoding')

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return


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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    # net = transxnet_t()
    # image_h, image_w = 224, 224
    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net,
    #                        inputs=(torch.randn(1, 3, image_h, image_w), ),
    #                        verbose=False)
    # macs, params = clever_format([macs, params], '%.3f')
    # outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    # print(f'1111, macs: {macs}, params: {params}')
    # for per_out in outs:
    #     print('1111', per_out.shape)

    # net = transxnet_s()
    # image_h, image_w = 224, 224
    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net,
    #                        inputs=(torch.randn(1, 3, image_h, image_w), ),
    #                        verbose=False)
    # macs, params = clever_format([macs, params], '%.3f')
    # outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    # print(f'1111, macs: {macs}, params: {params}')
    # for per_out in outs:
    #     print('1111', per_out.shape)

    # net = transxnet_b()
    # image_h, image_w = 224, 224
    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net,
    #                        inputs=(torch.randn(1, 3, image_h, image_w), ),
    #                        verbose=False)
    # macs, params = clever_format([macs, params], '%.3f')
    # outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    # print(f'1111, macs: {macs}, params: {params}')
    # for per_out in outs:
    #     print('1111', per_out.shape)

    # net = transxnet_t()
    # saved_state_dict = torch.load(
    #     f'/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/transxnet_pytorch_official_weights/transx-t.pth.tar',
    #     map_location=torch.device('cpu'))

    # replace_name_save_state_dict = {}
    # for name, weight in saved_state_dict.items():
    #     if 'proj.conv.' in name:
    #         name = name.replace('proj.conv.', 'proj.layer.0.')
    #     elif 'proj.bn.' in name:
    #         name = name.replace('proj.bn.', 'proj.layer.1.')
    #     elif 'token_mixer.local_unit.weight' in name:
    #         name = name.replace('proj.bn.', 'proj.layer.1.')
    #     elif 'local_unit.proj.0.conv.weight' in name:
    #         name = name.replace('local_unit.proj.0.conv.weight',
    #                             'local_unit.proj.0.layer.0.weight')
    #     elif 'local_unit.proj.0.bn.weight' in name:
    #         name = name.replace('local_unit.proj.0.bn.weight',
    #                             'local_unit.proj.0.layer.1.weight')
    #     elif 'local_unit.proj.0.bn.bias' in name:
    #         name = name.replace('local_unit.proj.0.bn.bias',
    #                             'local_unit.proj.0.layer.1.bias')
    #     elif 'local_unit.proj.0.bn.running_mean' in name:
    #         name = name.replace('local_unit.proj.0.bn.running_mean',
    #                             'local_unit.proj.0.layer.1.running_mean')
    #     elif 'local_unit.proj.0.bn.running_var' in name:
    #         name = name.replace('local_unit.proj.0.bn.running_var',
    #                             'local_unit.proj.0.layer.1.running_var')
    #     elif 'local_unit.proj.0.bn.num_batches_tracked' in name:
    #         name = name.replace(
    #             'local_unit.proj.0.bn.num_batches_tracked',
    #             'local_unit.proj.0.layer.1.num_batches_tracked')
    #     elif 'global_unit.sr.0.conv.weight' in name:
    #         name = name.replace('global_unit.sr.0.conv.weight',
    #                             'global_unit.sr.0.layer.0.weight')
    #     elif 'global_unit.sr.0.bn.weight' in name:
    #         name = name.replace('global_unit.sr.0.bn.weight',
    #                             'global_unit.sr.0.layer.1.weight')
    #     elif 'global_unit.sr.0.bn.bias' in name:
    #         name = name.replace('global_unit.sr.0.bn.bias',
    #                             'global_unit.sr.0.layer.1.bias')
    #     elif 'global_unit.sr.0.bn.running_mean' in name:
    #         name = name.replace('global_unit.sr.0.bn.running_mean',
    #                             'global_unit.sr.0.layer.1.running_mean')
    #     elif 'global_unit.sr.0.bn.running_var' in name:
    #         name = name.replace('global_unit.sr.0.bn.running_var',
    #                             'global_unit.sr.0.layer.1.running_var')
    #     elif 'global_unit.sr.0.bn.num_batches_tracked' in name:
    #         name = name.replace(
    #             'global_unit.sr.0.bn.num_batches_tracked',
    #             'global_unit.sr.0.layer.1.num_batches_tracked')
    #     elif 'global_unit.sr.1.conv.weight' in name:
    #         name = name.replace('global_unit.sr.1.conv.weight',
    #                             'global_unit.sr.1.layer.0.weight')
    #     elif 'global_unit.sr.1.bn.weight' in name:
    #         name = name.replace('global_unit.sr.1.bn.weight',
    #                             'global_unit.sr.1.layer.1.weight')
    #     elif 'global_unit.sr.1.bn.bias' in name:
    #         name = name.replace('global_unit.sr.1.bn.bias',
    #                             'global_unit.sr.1.layer.1.bias')
    #     elif 'global_unit.sr.1.bn.running_mean' in name:
    #         name = name.replace('global_unit.sr.1.bn.running_mean',
    #                             'global_unit.sr.1.layer.1.running_mean')
    #     elif 'global_unit.sr.1.bn.running_var' in name:
    #         name = name.replace('global_unit.sr.1.bn.running_var',
    #                             'global_unit.sr.1.layer.1.running_var')
    #     elif 'global_unit.sr.1.bn.num_batches_tracked' in name:
    #         name = name.replace(
    #             'global_unit.sr.1.bn.num_batches_tracked',
    #             'global_unit.sr.1.layer.1.num_batches_tracked')
    #     replace_name_save_state_dict[name] = weight

    # load_state_dict(replace_name_save_state_dict, net)
    # torch.save(net.state_dict(), 'transx-t-pretrained-convert.pth')

    # net = transxnet_s()
    # saved_state_dict = torch.load(
    #     f'/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/transxnet_pytorch_official_weights/transx-s.pth.tar',
    #     map_location=torch.device('cpu'))

    # replace_name_save_state_dict = {}
    # for name, weight in saved_state_dict.items():
    #     if 'proj.conv.' in name:
    #         name = name.replace('proj.conv.', 'proj.layer.0.')
    #     elif 'proj.bn.' in name:
    #         name = name.replace('proj.bn.', 'proj.layer.1.')
    #     elif 'token_mixer.local_unit.weight' in name:
    #         name = name.replace('proj.bn.', 'proj.layer.1.')
    #     elif 'local_unit.proj.0.conv.weight' in name:
    #         name = name.replace('local_unit.proj.0.conv.weight',
    #                             'local_unit.proj.0.layer.0.weight')
    #     elif 'local_unit.proj.0.bn.weight' in name:
    #         name = name.replace('local_unit.proj.0.bn.weight',
    #                             'local_unit.proj.0.layer.1.weight')
    #     elif 'local_unit.proj.0.bn.bias' in name:
    #         name = name.replace('local_unit.proj.0.bn.bias',
    #                             'local_unit.proj.0.layer.1.bias')
    #     elif 'local_unit.proj.0.bn.running_mean' in name:
    #         name = name.replace('local_unit.proj.0.bn.running_mean',
    #                             'local_unit.proj.0.layer.1.running_mean')
    #     elif 'local_unit.proj.0.bn.running_var' in name:
    #         name = name.replace('local_unit.proj.0.bn.running_var',
    #                             'local_unit.proj.0.layer.1.running_var')
    #     elif 'local_unit.proj.0.bn.num_batches_tracked' in name:
    #         name = name.replace(
    #             'local_unit.proj.0.bn.num_batches_tracked',
    #             'local_unit.proj.0.layer.1.num_batches_tracked')
    #     elif 'global_unit.sr.0.conv.weight' in name:
    #         name = name.replace('global_unit.sr.0.conv.weight',
    #                             'global_unit.sr.0.layer.0.weight')
    #     elif 'global_unit.sr.0.bn.weight' in name:
    #         name = name.replace('global_unit.sr.0.bn.weight',
    #                             'global_unit.sr.0.layer.1.weight')
    #     elif 'global_unit.sr.0.bn.bias' in name:
    #         name = name.replace('global_unit.sr.0.bn.bias',
    #                             'global_unit.sr.0.layer.1.bias')
    #     elif 'global_unit.sr.0.bn.running_mean' in name:
    #         name = name.replace('global_unit.sr.0.bn.running_mean',
    #                             'global_unit.sr.0.layer.1.running_mean')
    #     elif 'global_unit.sr.0.bn.running_var' in name:
    #         name = name.replace('global_unit.sr.0.bn.running_var',
    #                             'global_unit.sr.0.layer.1.running_var')
    #     elif 'global_unit.sr.0.bn.num_batches_tracked' in name:
    #         name = name.replace(
    #             'global_unit.sr.0.bn.num_batches_tracked',
    #             'global_unit.sr.0.layer.1.num_batches_tracked')
    #     elif 'global_unit.sr.1.conv.weight' in name:
    #         name = name.replace('global_unit.sr.1.conv.weight',
    #                             'global_unit.sr.1.layer.0.weight')
    #     elif 'global_unit.sr.1.bn.weight' in name:
    #         name = name.replace('global_unit.sr.1.bn.weight',
    #                             'global_unit.sr.1.layer.1.weight')
    #     elif 'global_unit.sr.1.bn.bias' in name:
    #         name = name.replace('global_unit.sr.1.bn.bias',
    #                             'global_unit.sr.1.layer.1.bias')
    #     elif 'global_unit.sr.1.bn.running_mean' in name:
    #         name = name.replace('global_unit.sr.1.bn.running_mean',
    #                             'global_unit.sr.1.layer.1.running_mean')
    #     elif 'global_unit.sr.1.bn.running_var' in name:
    #         name = name.replace('global_unit.sr.1.bn.running_var',
    #                             'global_unit.sr.1.layer.1.running_var')
    #     elif 'global_unit.sr.1.bn.num_batches_tracked' in name:
    #         name = name.replace(
    #             'global_unit.sr.1.bn.num_batches_tracked',
    #             'global_unit.sr.1.layer.1.num_batches_tracked')
    #     replace_name_save_state_dict[name] = weight

    # load_state_dict(replace_name_save_state_dict, net)
    # torch.save(net.state_dict(), 'transx-s-pretrained-convert.pth')

    net = transxnet_b()
    saved_state_dict = torch.load(
        f'/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/transxnet_pytorch_official_weights/transx-b.pth.tar',
        map_location=torch.device('cpu'))

    replace_name_save_state_dict = {}
    for name, weight in saved_state_dict.items():
        if 'proj.conv.' in name:
            name = name.replace('proj.conv.', 'proj.layer.0.')
        elif 'proj.bn.' in name:
            name = name.replace('proj.bn.', 'proj.layer.1.')
        elif 'token_mixer.local_unit.weight' in name:
            name = name.replace('proj.bn.', 'proj.layer.1.')
        elif 'local_unit.proj.0.conv.weight' in name:
            name = name.replace('local_unit.proj.0.conv.weight',
                                'local_unit.proj.0.layer.0.weight')
        elif 'local_unit.proj.0.bn.weight' in name:
            name = name.replace('local_unit.proj.0.bn.weight',
                                'local_unit.proj.0.layer.1.weight')
        elif 'local_unit.proj.0.bn.bias' in name:
            name = name.replace('local_unit.proj.0.bn.bias',
                                'local_unit.proj.0.layer.1.bias')
        elif 'local_unit.proj.0.bn.running_mean' in name:
            name = name.replace('local_unit.proj.0.bn.running_mean',
                                'local_unit.proj.0.layer.1.running_mean')
        elif 'local_unit.proj.0.bn.running_var' in name:
            name = name.replace('local_unit.proj.0.bn.running_var',
                                'local_unit.proj.0.layer.1.running_var')
        elif 'local_unit.proj.0.bn.num_batches_tracked' in name:
            name = name.replace(
                'local_unit.proj.0.bn.num_batches_tracked',
                'local_unit.proj.0.layer.1.num_batches_tracked')
        elif 'global_unit.sr.0.conv.weight' in name:
            name = name.replace('global_unit.sr.0.conv.weight',
                                'global_unit.sr.0.layer.0.weight')
        elif 'global_unit.sr.0.bn.weight' in name:
            name = name.replace('global_unit.sr.0.bn.weight',
                                'global_unit.sr.0.layer.1.weight')
        elif 'global_unit.sr.0.bn.bias' in name:
            name = name.replace('global_unit.sr.0.bn.bias',
                                'global_unit.sr.0.layer.1.bias')
        elif 'global_unit.sr.0.bn.running_mean' in name:
            name = name.replace('global_unit.sr.0.bn.running_mean',
                                'global_unit.sr.0.layer.1.running_mean')
        elif 'global_unit.sr.0.bn.running_var' in name:
            name = name.replace('global_unit.sr.0.bn.running_var',
                                'global_unit.sr.0.layer.1.running_var')
        elif 'global_unit.sr.0.bn.num_batches_tracked' in name:
            name = name.replace(
                'global_unit.sr.0.bn.num_batches_tracked',
                'global_unit.sr.0.layer.1.num_batches_tracked')
        elif 'global_unit.sr.1.conv.weight' in name:
            name = name.replace('global_unit.sr.1.conv.weight',
                                'global_unit.sr.1.layer.0.weight')
        elif 'global_unit.sr.1.bn.weight' in name:
            name = name.replace('global_unit.sr.1.bn.weight',
                                'global_unit.sr.1.layer.1.weight')
        elif 'global_unit.sr.1.bn.bias' in name:
            name = name.replace('global_unit.sr.1.bn.bias',
                                'global_unit.sr.1.layer.1.bias')
        elif 'global_unit.sr.1.bn.running_mean' in name:
            name = name.replace('global_unit.sr.1.bn.running_mean',
                                'global_unit.sr.1.layer.1.running_mean')
        elif 'global_unit.sr.1.bn.running_var' in name:
            name = name.replace('global_unit.sr.1.bn.running_var',
                                'global_unit.sr.1.layer.1.running_var')
        elif 'global_unit.sr.1.bn.num_batches_tracked' in name:
            name = name.replace(
                'global_unit.sr.1.bn.num_batches_tracked',
                'global_unit.sr.1.layer.1.num_batches_tracked')
        replace_name_save_state_dict[name] = weight

    load_state_dict(replace_name_save_state_dict, net)
    torch.save(net.state_dict(), 'transx-b-pretrained-convert.pth')
