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

from torch.utils.checkpoint import checkpoint

from SimpleAICV.video_interactive_segmentation.models.dinov3_segment_anything2.dinov3_image_encoder import DINOV3ViTImageEncoder
from SimpleAICV.video_interactive_segmentation.models.segment_anything2.prompt_encoder import PromptEncoder
from SimpleAICV.video_interactive_segmentation.models.segment_anything2_matting.sam2videomatting_train import MaskDecoderMatting, MLP, FUSION

__all__ = [
    'dinov3_vit_small_patch16_sam2image_matting',
    'dinov3_vit_small_plus_patch16_sam2image_matting',
    'dinov3_vit_base_patch16_sam2image_matting',
    'dinov3_vit_large_patch16_sam2image_matting',
    'dinov3_vit_large_plus_patch16_sam2image_matting',
    'dinov3_vit_huge_plus_patch16_sam2image_matting',
]


class DINOV3SAM2ImageMatting(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 image_size=1024,
                 patch_size=16,
                 prompt_encoder_embedding_planes=256,
                 prompt_encoder_mask_inter_planes=16,
                 mask_decoder_num_multimask_outputs=3,
                 mask_decoder_iou_prediction_head_block_nums=3,
                 mask_decoder_iou_prediction_head_hidden_planes=256,
                 matting_planes=[32, 256],
                 matting_cpfe_planes=32,
                 use_gradient_checkpoint=False,
                 frozen_image_encoder=False,
                 frozen_prompt_encoder=False,
                 frozen_mask_decoder=False):
        super(DINOV3SAM2ImageMatting, self).__init__()
        self.image_size = image_size

        self.image_encoder = DINOV3ViTImageEncoder(
            backbone_type=backbone_type,
            backbone_pretrained_path=backbone_pretrained_path,
            image_size=image_size,
            fpn_planes=prompt_encoder_embedding_planes,
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
            mask_decoder_iou_prediction_head_hidden_planes,
            use_high_res_features=True)

        self.num_mask_tokens = mask_decoder_num_multimask_outputs + 1
        self.fusion_pred_list = nn.ModuleList()
        for _ in range(self.num_mask_tokens):
            self.fusion_pred_list.append(
                FUSION(planes=matting_planes, cpfe_planes=matting_cpfe_planes))

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

        # features: torch.Size([B, 256, 256, 256]) torch.Size([B, 256, 128, 128]) torch.Size([B, 256, 64, 64])
        # positions: torch.Size([B, 256, 256, 256]) torch.Size([B, 256, 128, 128]) torch.Size([B, 256, 64, 64])
        batch_image_embeddings, _ = self.image_encoder(batch_images)

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

        point_inputs = None
        mask_inputs = None

        if prompt_points is not None:
            point_inputs = prompt_points

        if prompt_boxes is not None:
            # [N_o, 2, 2]
            prompt_boxes = prompt_boxes.reshape(-1, 2, 2)
            device = prompt_boxes.device
            # [N_o, 2, 1]
            prompt_boxes_label = torch.ones(
                [prompt_boxes.shape[0], prompt_boxes.shape[1], 1],
                dtype=torch.float32).to(device)
            prompt_boxes_label[:, 0] = prompt_boxes_label[:, 0] * 2
            prompt_boxes_label[:, 1] = prompt_boxes_label[:, 1] * 3
            # [N_o, 2, 3]
            prompt_boxes = torch.cat([prompt_boxes, prompt_boxes_label],
                                     dim=-1)

            if point_inputs is not None:
                point_inputs = torch.cat([
                    point_inputs,
                    prompt_boxes,
                ], dim=1)
            else:
                point_inputs = prompt_boxes

        if prompt_mask is not None:
            mask_inputs = prompt_mask

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_inputs, boxes=None, masks=mask_inputs)

        mask_preds, iou_preds, _, _, feat3, feat1 = self.mask_decoder(
            image_embeddings=batch_image_embeddings[-1],
            image_pe=self.prompt_encoder.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=batch_image_embeddings[0:2],
            mask_out_idxs=mask_out_idxs)

        global_preds, local_preds, fused_preds = [], [], []
        for idx, mask_out_idx in enumerate(mask_out_idxs):
            mask_preds_per_out_idx = mask_preds[:, idx:idx + 1, :, :]
            global_preds_per_out_idx, local_preds_per_out_idx = self.fusion_pred_list[
                mask_out_idx](mask_preds_per_out_idx, feat3, feat1)
            fused_preds_per_out_idx = self.collaborative_matting(
                global_preds_per_out_idx, local_preds_per_out_idx)

            global_preds.append(global_preds_per_out_idx)
            local_preds.append(local_preds_per_out_idx)
            fused_preds.append(fused_preds_per_out_idx)

        global_preds = torch.stack(global_preds, dim=1)
        local_preds = torch.stack(local_preds, dim=1)
        fused_preds = torch.stack(fused_preds, dim=1)

        return global_preds, local_preds, fused_preds, iou_preds

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

    def forward_image_encoder(self, batch_images):
        batch_image_embeddings, _ = self.image_encoder(batch_images)

        return batch_image_embeddings

    def forward_prompt_encoder_mask_decoder(self,
                                            batch_image_embeddings,
                                            batch_prompts,
                                            mask_out_idxs=[0, 1, 2, 3]):
        device = batch_image_embeddings[0].device

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

        point_inputs = None
        mask_inputs = None

        if prompt_points is not None:
            point_inputs = prompt_points

        if prompt_boxes is not None:
            # [N_o, 2, 2]
            prompt_boxes = prompt_boxes.reshape(-1, 2, 2)
            device = prompt_boxes.device
            # [N_o, 2, 1]
            prompt_boxes_label = torch.ones(
                [prompt_boxes.shape[0], prompt_boxes.shape[1], 1],
                dtype=torch.float32).to(device)
            prompt_boxes_label[:, 0] = prompt_boxes_label[:, 0] * 2
            prompt_boxes_label[:, 1] = prompt_boxes_label[:, 1] * 3
            # [N_o, 2, 3]
            prompt_boxes = torch.cat([prompt_boxes, prompt_boxes_label],
                                     dim=-1)

            if point_inputs is not None:
                point_inputs = torch.cat([
                    point_inputs,
                    prompt_boxes,
                ], dim=1)
            else:
                point_inputs = prompt_boxes

        if prompt_mask is not None:
            mask_inputs = prompt_mask

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_inputs, boxes=None, masks=mask_inputs)

        mask_preds, iou_preds, _, _, feat3, feat1 = self.mask_decoder(
            image_embeddings=batch_image_embeddings[-1],
            image_pe=self.prompt_encoder.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=batch_image_embeddings[0:2],
            mask_out_idxs=mask_out_idxs)

        global_preds, local_preds, fused_preds = [], [], []
        for idx, mask_out_idx in enumerate(mask_out_idxs):
            mask_preds_per_out_idx = mask_preds[:, idx:idx + 1, :, :]
            global_preds_per_out_idx, local_preds_per_out_idx = self.fusion_pred_list[
                mask_out_idx](mask_preds_per_out_idx, feat3, feat1)
            fused_preds_per_out_idx = self.collaborative_matting(
                global_preds_per_out_idx, local_preds_per_out_idx)

            global_preds.append(global_preds_per_out_idx)
            local_preds.append(local_preds_per_out_idx)
            fused_preds.append(fused_preds_per_out_idx)

        global_preds = torch.stack(global_preds, dim=1)
        local_preds = torch.stack(local_preds, dim=1)
        fused_preds = torch.stack(fused_preds, dim=1)

        return global_preds, local_preds, fused_preds, iou_preds


def _dinov3_sam2imagematting(backbone_type, backbone_pretrained_path,
                             image_size, patch_size,
                             prompt_encoder_embedding_planes, **kwargs):
    model = DINOV3SAM2ImageMatting(
        backbone_type=backbone_type,
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=prompt_encoder_embedding_planes,
        **kwargs)

    return model


def dinov3_vit_small_patch16_sam2image_matting(backbone_pretrained_path='',
                                               image_size=1024,
                                               patch_size=16,
                                               **kwargs):
    return _dinov3_sam2imagematting(
        backbone_type='dinov3_vit_small_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_small_plus_patch16_sam2image_matting(backbone_pretrained_path='',
                                                    image_size=1024,
                                                    patch_size=16,
                                                    **kwargs):
    return _dinov3_sam2imagematting(
        backbone_type='dinov3_vit_small_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_base_patch16_sam2image_matting(backbone_pretrained_path='',
                                              image_size=1024,
                                              patch_size=16,
                                              **kwargs):
    return _dinov3_sam2imagematting(
        backbone_type='dinov3_vit_base_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_large_patch16_sam2image_matting(backbone_pretrained_path='',
                                               image_size=1024,
                                               patch_size=16,
                                               **kwargs):
    return _dinov3_sam2imagematting(
        backbone_type='dinov3_vit_large_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_large_plus_patch16_sam2image_matting(backbone_pretrained_path='',
                                                    image_size=1024,
                                                    patch_size=16,
                                                    **kwargs):
    return _dinov3_sam2imagematting(
        backbone_type='dinov3_vit_large_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_huge_plus_patch16_sam2image_matting(backbone_pretrained_path='',
                                                   image_size=1024,
                                                   patch_size=16,
                                                   **kwargs):
    return _dinov3_sam2imagematting(
        backbone_type='dinov3_vit_huge_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
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

    from SimpleAICV.interactive_segmentation.datasets.sam_matting_dataset import SAMMattingDataset
    from SimpleAICV.interactive_segmentation.common_matting import SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingBatchCollater, load_state_dict

    samdataset = SAMMattingDataset(
        interactive_segmentation_dataset_path,
        set_name=[
            'DIS5K',
            'sa_000000',
        ],
        set_type='train',
        per_set_image_choose_max_num={
            'DIS5K': 1000000,
            'sa_000000': 1000000,
        },
        max_side=2048,
        kernel_size_range=[15, 15],
        per_image_mask_chosse_max_num=16,
        points_num=1,
        area_filter_ratio=0.0001,
        box_noise_wh_ratio=0.1,
        mask_noise_area_ratio=0.04,
        transform=transforms.Compose([
            SAMMattingResize(resize=1024),
            SAMMattingRandomHorizontalFlip(prob=0.5),
            SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAMMattingBatchCollater(resize=1024)
    train_loader = DataLoader(samdataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    net = dinov3_vit_base_patch16_sam2image_matting(
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

        global_preds, local_preds, fused_preds, iou_preds = net(
            input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('3333', global_preds.shape, local_preds.shape, fused_preds.shape,
              iou_preds.shape)

        batch_image_embeddings = net.forward_image_encoder(input_images)

        print('4444', batch_image_embeddings[0].shape,
              batch_image_embeddings[1].shape, batch_image_embeddings[2].shape)

        global_preds, local_preds, fused_preds, iou_preds = net.forward_prompt_encoder_mask_decoder(
            batch_image_embeddings, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('5555', global_preds.shape, local_preds.shape, fused_preds.shape,
              iou_preds.shape)

        break
