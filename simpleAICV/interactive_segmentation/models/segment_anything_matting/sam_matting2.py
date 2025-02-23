import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleAICV.interactive_segmentation.models.segment_anything.image_encoder import ViTImageEncoder
from simpleAICV.interactive_segmentation.models.segment_anything.prompt_encoder import PromptEncoder
from simpleAICV.interactive_segmentation.models.segment_anything_matting.mask_decoder_matting import MaskDecoderMatting

__all__ = [
    'sam_b_matting2',
    'sam_l_matting2',
    'sam_h_matting2',
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


class ConvTransposeBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvTransposeBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(inplanes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               groups=groups,
                               bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class FUSION(nn.Module):

    def __init__(self, planes=[32, 256], cpfe_planes=32):
        super(FUSION, self).__init__()
        ##############################################################################
        self.global_feat3_reduce_conv = ConvBnActBlock(planes[-1],
                                                       cpfe_planes,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0,
                                                       groups=1,
                                                       dilation=1,
                                                       has_bn=True,
                                                       has_act=True)
        self.global_feat1_reduce_conv = ConvBnActBlock(planes[-2],
                                                       cpfe_planes,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0,
                                                       groups=1,
                                                       dilation=1,
                                                       has_bn=True,
                                                       has_act=True)
        self.global_combine_conv = ConvBnActBlock(int(2 * cpfe_planes) + 1,
                                                  cpfe_planes,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  groups=1,
                                                  dilation=1,
                                                  has_bn=True,
                                                  has_act=False)
        self.global_reduce_conv = ConvBnActBlock(cpfe_planes,
                                                 cpfe_planes,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 groups=1,
                                                 dilation=1,
                                                 has_bn=True,
                                                 has_act=True)
        self.global_upsample_conv1 = ConvTransposeBnActBlock(cpfe_planes,
                                                             cpfe_planes,
                                                             kernel_size=2,
                                                             stride=2,
                                                             groups=1,
                                                             has_bn=True,
                                                             has_act=True)
        self.global_upsample_conv2 = ConvBnActBlock(cpfe_planes,
                                                    cpfe_planes,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    groups=1,
                                                    dilation=1,
                                                    has_bn=True,
                                                    has_act=True)
        self.global_upsample_conv3 = ConvTransposeBnActBlock(cpfe_planes,
                                                             cpfe_planes,
                                                             kernel_size=2,
                                                             stride=2,
                                                             groups=1,
                                                             has_bn=True,
                                                             has_act=True)
        self.global_pred_conv = nn.Conv2d(cpfe_planes,
                                          3,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)

        ##########################################################################
        self.local_feat3_reduce_conv = ConvBnActBlock(planes[-1],
                                                      cpfe_planes,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0,
                                                      groups=1,
                                                      dilation=1,
                                                      has_bn=True,
                                                      has_act=True)
        self.local_feat1_reduce_conv = ConvBnActBlock(planes[-2],
                                                      cpfe_planes,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0,
                                                      groups=1,
                                                      dilation=1,
                                                      has_bn=True,
                                                      has_act=True)
        self.local_combine_conv = ConvBnActBlock(int(4 * cpfe_planes) + 1,
                                                 cpfe_planes,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 groups=1,
                                                 dilation=1,
                                                 has_bn=True,
                                                 has_act=False)
        self.local_reduce_conv = ConvBnActBlock(cpfe_planes,
                                                cpfe_planes,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                groups=1,
                                                dilation=1,
                                                has_bn=True,
                                                has_act=True)
        self.local_upsample_conv1 = ConvTransposeBnActBlock(cpfe_planes,
                                                            cpfe_planes,
                                                            kernel_size=2,
                                                            stride=2,
                                                            groups=1,
                                                            has_bn=True,
                                                            has_act=True)
        self.local_upsample_conv2 = ConvBnActBlock(cpfe_planes,
                                                   cpfe_planes,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   groups=1,
                                                   dilation=1,
                                                   has_bn=True,
                                                   has_act=True)
        self.local_upsample_conv3 = ConvTransposeBnActBlock(cpfe_planes,
                                                            cpfe_planes,
                                                            kernel_size=2,
                                                            stride=2,
                                                            groups=1,
                                                            has_bn=True,
                                                            has_act=True)
        self.local_pred_conv = nn.Conv2d(cpfe_planes,
                                         1,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_masks, batch_feat3, batch_feat1):
        # batch_masks [B,4,256,256]
        # batch_feat3 [B,256,64,64]
        # batch_feat1 [B,32,256,256]

        ##########################
        ### Decoder part - Global
        ##########################
        batch_feat3_global = self.global_feat3_reduce_conv(batch_feat3)
        # torch.Size([1, 32, 64, 64])
        batch_feat3_global = F.interpolate(batch_feat3_global,
                                           size=(batch_feat1.shape[2],
                                                 batch_feat1.shape[3]),
                                           mode='bilinear',
                                           align_corners=True)
        batch_feat1_global = self.global_feat1_reduce_conv(batch_feat1)

        # torch.Size([1, 32, 256, 256])
        conv_0_feats_g = torch.cat(
            (batch_feat1_global, batch_feat3_global, batch_masks), dim=1)

        # torch.Size([1, 64, 256, 256])
        conv_0_feats_g = self.global_combine_conv(conv_0_feats_g)
        # torch.Size([1, 32, 256, 256])
        conv_0_feats_g = self.global_reduce_conv(conv_0_feats_g)
        # torch.Size([1, 32, 256, 256])
        conv_0_feats_g = self.global_upsample_conv1(conv_0_feats_g)
        # torch.Size([1, 32, 512, 512])
        conv_0_feats_g = self.global_upsample_conv2(conv_0_feats_g)
        # torch.Size([1, 32, 512, 512])
        conv_0_feats_g = self.global_upsample_conv3(conv_0_feats_g)
        # [1, 32, 1024, 1024]
        global_pred = self.global_pred_conv(conv_0_feats_g)
        # global_pred:[1, 3, 1024, 1024],3:0为背景区域，1为local区域，2为global区域

        ##########################
        ### Decoder part - Local
        ##########################
        # batch_feat3_local:[B,256,64,64]
        # batch_feat1_local:[B,32,256,256]
        batch_feat3_local = self.local_feat3_reduce_conv(batch_feat3)
        # torch.Size([1, 32, 64, 64])
        batch_feat3_local = F.interpolate(batch_feat3_local,
                                          size=(batch_feat1.shape[2],
                                                batch_feat1.shape[3]),
                                          mode='bilinear',
                                          align_corners=True)
        batch_feat1_local = self.local_feat1_reduce_conv(batch_feat1)

        # torch.Size([1, 32, 256, 256])
        conv_0_feats_f = torch.cat(
            (batch_feat1_local, batch_feat3_local, batch_feat1_global,
             batch_feat3_global, batch_masks),
            dim=1)

        # torch.Size([1, 129, 256, 256])
        conv_0_feats_f = self.local_combine_conv(conv_0_feats_f)
        # torch.Size([1, 32, 256, 256])
        conv_0_feats_f = self.local_reduce_conv(conv_0_feats_f)
        # torch.Size([1, 32, 256, 256])
        conv_0_feats_f = self.local_upsample_conv1(conv_0_feats_f)
        # torch.Size([1, 32, 512, 512])
        conv_0_feats_f = self.local_upsample_conv2(conv_0_feats_f)
        # torch.Size([1, 32, 512, 512])
        conv_0_feats_f = self.local_upsample_conv3(conv_0_feats_f)
        # [1, 32, 1024, 1024]
        local_pred = self.local_pred_conv(conv_0_feats_f)
        # torch.Size([1, 1, 1024, 1024])
        # local_pred:[1, 1, 1024, 1024]

        global_pred = global_pred.float()
        local_pred = local_pred.float()
        global_pred = self.sigmoid(global_pred)
        local_pred = self.sigmoid(local_pred)

        return global_pred, local_pred


class SAMMATTING(nn.Module):

    def __init__(self,
                 image_size=1024,
                 patch_size=16,
                 inplanes=3,
                 image_encoder_embedding_planes=768,
                 image_encoder_block_nums=12,
                 image_encoder_head_nums=12,
                 image_encoder_mlp_ratio=4,
                 image_encoder_window_size=14,
                 image_encoder_global_attn_indexes=[2, 5, 8, 11],
                 prompt_encoder_embedding_planes=256,
                 prompt_encoder_mask_inter_planes=16,
                 mask_decoder_num_multimask_outputs=3,
                 mask_decoder_iou_prediction_head_block_nums=3,
                 mask_decoder_iou_prediction_head_hidden_planes=256,
                 use_gradient_checkpoint=False,
                 frozen_image_encoder=False,
                 frozen_prompt_encoder=False,
                 frozen_mask_decoder=False,
                 matting_planes=[32, 256],
                 matting_cpfe_planes=32):
        super(SAMMATTING, self).__init__()
        self.image_encoder = ViTImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            inplanes=inplanes,
            embedding_planes=image_encoder_embedding_planes,
            block_nums=image_encoder_block_nums,
            head_nums=image_encoder_head_nums,
            mlp_ratio=image_encoder_mlp_ratio,
            out_planes=prompt_encoder_embedding_planes,
            window_size=image_encoder_window_size,
            global_attn_indexes=image_encoder_global_attn_indexes,
            use_gradient_checkpoint=use_gradient_checkpoint)
        self.prompt_encoder = PromptEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embedding_planes=prompt_encoder_embedding_planes,
            mask_inter_planes=prompt_encoder_mask_inter_planes)
        self.mask_decoder = MaskDecoderMatting(
            inplanes=prompt_encoder_embedding_planes,
            num_multimask_outputs=mask_decoder_num_multimask_outputs,
            iou_prediction_head_block_nums=
            mask_decoder_iou_prediction_head_block_nums,
            iou_prediction_head_hidden_planes=
            mask_decoder_iou_prediction_head_hidden_planes)

        self.num_mask_tokens = mask_decoder_num_multimask_outputs + 1
        self.fusion_pred_list = nn.ModuleList()
        for _ in range(self.num_mask_tokens):
            self.fusion_pred_list.append(
                FUSION(planes=matting_planes, cpfe_planes=matting_cpfe_planes))

        self.sigmoid = nn.Sigmoid()

        if frozen_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if frozen_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
        if frozen_mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, batch_images, batch_prompts, mask_out_idxs=[0, 1, 2, 3]):
        device = batch_images.device

        batch_image_embeddings = self.image_encoder(batch_images)

        prompt_points = None
        if batch_prompts['prompt_point'] is not None:
            prompt_points = batch_prompts['prompt_point']
            prompt_points = prompt_points.to(device)

        prompt_boxes = None
        if batch_prompts['prompt_box'] is not None:
            prompt_boxes = batch_prompts['prompt_box']
            prompt_boxes = prompt_boxes.to(device)

        prompt_mask = None
        if batch_prompts['prompt_mask'] is not None:
            prompt_mask = batch_prompts['prompt_mask']
            prompt_mask = prompt_mask.to(device)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=prompt_points, boxes=prompt_boxes, masks=prompt_mask)

        masks, iou_predictions, feat3, feat1 = self.mask_decoder(
            image_embeddings=batch_image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            mask_out_idxs=mask_out_idxs)

        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds = [], [], []
        for idx, mask_out_idx in enumerate(mask_out_idxs):
            batch_masks_per_layer = masks[:, idx:idx + 1, :, :]
            mask_global_pred_per_layer, mask_local_pred_per_layer = self.fusion_pred_list[
                mask_out_idx](batch_masks_per_layer, feat3, feat1)
            mask_fused_pred_per_layer = self.collaborative_matting(
                mask_global_pred_per_layer, mask_local_pred_per_layer)

            batch_masks_global_preds.append(mask_global_pred_per_layer)
            batch_masks_local_preds.append(mask_local_pred_per_layer)
            batch_masks_fused_preds.append(mask_fused_pred_per_layer)

        # torch.Size([2, 4, 3, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024]) torch.Size([2, 4, 1, 1024, 1024])
        batch_masks_global_preds = torch.stack(batch_masks_global_preds, dim=1)
        batch_masks_local_preds = torch.stack(batch_masks_local_preds, dim=1)
        batch_masks_fused_preds = torch.stack(batch_masks_fused_preds, dim=1)

        batch_iou_preds = iou_predictions.float()
        batch_iou_preds = self.sigmoid(batch_iou_preds)

        return batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds

    def collaborative_matting(self, global_pred, local_pred):
        # 0为背景区域，1为local区域，2为global区域
        device = global_pred.device
        # max_cls_idxs <===> [0, 1, 2]
        # max_cls_idxs:[b,h,w] -> [b,1,h,w]
        _, max_cls_idxs = torch.max(global_pred, dim=1)
        max_cls_idxs = torch.unsqueeze(max_cls_idxs.float(), dim=1)

        # trimap_mask:[0, 1, 2] ===> [0, 1, 0],保留local区域
        trimap_mask = max_cls_idxs.clone().to(device)
        trimap_mask[trimap_mask == 2] = 0

        # fg_mask: [0, 1, 2] ===> [0, 0, 1]，保留global区域
        fg_mask = max_cls_idxs.clone().to(device)
        fg_mask[fg_mask == 1] = 0
        fg_mask[fg_mask == 2] = 1

        # fused_pred只保留预测为128区域
        fused_pred = local_pred * trimap_mask + fg_mask

        return fused_pred


def _sam_matting2(image_size, patch_size, image_encoder_embedding_planes,
                  image_encoder_block_nums, image_encoder_head_nums,
                  image_encoder_global_attn_indexes,
                  prompt_encoder_embedding_planes, **kwargs):
    model = SAMMATTING(
        image_size=image_size,
        patch_size=patch_size,
        image_encoder_embedding_planes=image_encoder_embedding_planes,
        image_encoder_block_nums=image_encoder_block_nums,
        image_encoder_head_nums=image_encoder_head_nums,
        image_encoder_global_attn_indexes=image_encoder_global_attn_indexes,
        prompt_encoder_embedding_planes=prompt_encoder_embedding_planes,
        **kwargs)

    return model


def sam_b_matting2(image_size=1024, patch_size=16, **kwargs):
    return _sam_matting2(image_size=image_size,
                         patch_size=patch_size,
                         image_encoder_embedding_planes=768,
                         image_encoder_block_nums=12,
                         image_encoder_head_nums=12,
                         image_encoder_global_attn_indexes=[2, 5, 8, 11],
                         prompt_encoder_embedding_planes=256,
                         **kwargs)


def sam_l_matting2(image_size=1024, patch_size=16, **kwargs):
    return _sam_matting2(image_size=image_size,
                         patch_size=patch_size,
                         image_encoder_embedding_planes=1024,
                         image_encoder_block_nums=24,
                         image_encoder_head_nums=16,
                         image_encoder_global_attn_indexes=[5, 11, 17, 23],
                         prompt_encoder_embedding_planes=256,
                         **kwargs)


def sam_h_matting2(image_size=1024, patch_size=16, **kwargs):
    return _sam_matting2(image_size=image_size,
                         patch_size=patch_size,
                         image_encoder_embedding_planes=1280,
                         image_encoder_block_nums=32,
                         image_encoder_head_nums=16,
                         image_encoder_global_attn_indexes=[7, 15, 23, 31],
                         prompt_encoder_embedding_planes=256,
                         **kwargs)


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
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    sys.path.append(BASE_DIR)

    from tools.path import interactive_segmentation_dataset_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.interactive_segmentation.datasets.sam_matting_dataset import SAMMattingDataset
    from simpleAICV.interactive_segmentation.common_matting import SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingBatchCollater, load_state_dict

    samdataset = SAMMattingDataset(
        interactive_segmentation_dataset_path,
        set_name=[
            'DIS5K',
            'sa_000020',
        ],
        set_type='train',
        max_side=2048,
        kernel_size_range=[10, 15],
        per_set_image_choose_max_num={
            'DIS5K': 10000000,
            'sa_000020': 1000000,
        },
        per_image_mask_chosse_max_num=1,
        positive_points_num=9,
        negative_points_num=9,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        resample_num=1,
        transform=transforms.Compose([
            SAMMattingResize(resize=1024),
            SAMMattingRandomHorizontalFlip(prob=0.5),
            SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAMMattingBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    net = sam_b_matting2(image_size=1024,
                         frozen_image_encoder=False,
                         frozen_prompt_encoder=False,
                         frozen_mask_decoder=False,
                         use_gradient_checkpoint=True)
    load_state_dict('', net)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
            'bg_map']

        net = net.cuda()
        input_images = input_images.cuda()
        input_masks = input_masks.cuda()
        print('1111', input_images.shape, input_masks.shape)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        print('2222', input_prompt_points.shape, input_prompt_boxs.shape,
              input_prompt_masks.shape)

        input_prompts = {
            'prompt_point': input_prompt_points,
            'prompt_box': input_prompt_boxs,
            'prompt_mask': input_prompt_masks,
        }

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        batch_masks_global_preds, batch_masks_local_preds, batch_masks_fused_preds, batch_iou_preds = preds
        print('3333', batch_masks_global_preds.shape,
              batch_masks_local_preds.shape, batch_masks_fused_preds.shape,
              batch_iou_preds.shape)

        break
