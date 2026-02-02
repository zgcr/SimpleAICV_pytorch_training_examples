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
from SimpleAICV.video_interactive_segmentation.models.segment_anything2.mask_decoder import MaskDecoder

__all__ = [
    'dinov3_vit_small_patch16_sam2image',
    'dinov3_vit_small_plus_patch16_sam2image',
    'dinov3_vit_base_patch16_sam2image',
    'dinov3_vit_large_patch16_sam2image',
    'dinov3_vit_large_plus_patch16_sam2image',
    'dinov3_vit_huge_plus_patch16_sam2image',
]


class DINOV3SAM2Image(nn.Module):

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
                 use_gradient_checkpoint=False,
                 frozen_image_encoder=False,
                 frozen_prompt_encoder=False,
                 frozen_mask_decoder=False):
        super(DINOV3SAM2Image, self).__init__()
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
        self.mask_decoder = MaskDecoder(
            inplanes=prompt_encoder_embedding_planes,
            num_multimask_outputs=mask_decoder_num_multimask_outputs,
            iou_prediction_head_block_nums=
            mask_decoder_iou_prediction_head_block_nums,
            iou_prediction_head_hidden_planes=
            mask_decoder_iou_prediction_head_hidden_planes,
            use_high_res_features=True)

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

        mask_preds, iou_preds, _, _ = self.mask_decoder(
            image_embeddings=batch_image_embeddings[-1],
            image_pe=self.prompt_encoder.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=batch_image_embeddings[0:2],
            mask_out_idxs=mask_out_idxs)

        mask_preds = F.interpolate(mask_preds,
                                   (self.image_size, self.image_size),
                                   mode="bilinear")

        return mask_preds, iou_preds

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

        mask_preds, iou_preds, _, _ = self.mask_decoder(
            image_embeddings=batch_image_embeddings[-1],
            image_pe=self.prompt_encoder.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=batch_image_embeddings[0:2],
            mask_out_idxs=mask_out_idxs)

        mask_preds = F.interpolate(mask_preds,
                                   (self.image_size, self.image_size),
                                   mode="bilinear")

        return mask_preds, iou_preds


def _dinov3_sam2image(backbone_type, backbone_pretrained_path, image_size,
                      patch_size, prompt_encoder_embedding_planes, **kwargs):
    model = DINOV3SAM2Image(
        backbone_type=backbone_type,
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=prompt_encoder_embedding_planes,
        **kwargs)

    return model


def dinov3_vit_small_patch16_sam2image(backbone_pretrained_path='',
                                       image_size=1024,
                                       patch_size=16,
                                       **kwargs):
    return _dinov3_sam2image(backbone_type='dinov3_vit_small_patch16_backbone',
                             backbone_pretrained_path=backbone_pretrained_path,
                             image_size=image_size,
                             patch_size=patch_size,
                             prompt_encoder_embedding_planes=256,
                             **kwargs)


def dinov3_vit_small_plus_patch16_sam2image(backbone_pretrained_path='',
                                            image_size=1024,
                                            patch_size=16,
                                            **kwargs):
    return _dinov3_sam2image(
        backbone_type='dinov3_vit_small_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_base_patch16_sam2image(backbone_pretrained_path='',
                                      image_size=1024,
                                      patch_size=16,
                                      **kwargs):
    return _dinov3_sam2image(backbone_type='dinov3_vit_base_patch16_backbone',
                             backbone_pretrained_path=backbone_pretrained_path,
                             image_size=image_size,
                             patch_size=patch_size,
                             prompt_encoder_embedding_planes=256,
                             **kwargs)


def dinov3_vit_large_patch16_sam2image(backbone_pretrained_path='',
                                       image_size=1024,
                                       patch_size=16,
                                       **kwargs):
    return _dinov3_sam2image(backbone_type='dinov3_vit_large_patch16_backbone',
                             backbone_pretrained_path=backbone_pretrained_path,
                             image_size=image_size,
                             patch_size=patch_size,
                             prompt_encoder_embedding_planes=256,
                             **kwargs)


def dinov3_vit_large_plus_patch16_sam2image(backbone_pretrained_path='',
                                            image_size=1024,
                                            patch_size=16,
                                            **kwargs):
    return _dinov3_sam2image(
        backbone_type='dinov3_vit_large_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_huge_plus_patch16_sam2image(backbone_pretrained_path='',
                                           image_size=1024,
                                           patch_size=16,
                                           **kwargs):
    return _dinov3_sam2image(
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

    from SimpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
    from SimpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

    samdataset = SAMSegmentationDataset(interactive_segmentation_dataset_path,
                                        set_name=[
                                            'DIS5K',
                                            'sa_000000',
                                        ],
                                        set_type='train',
                                        per_set_image_choose_max_num={
                                            'DIS5K': 1000000,
                                            'sa_000000': 1000000,
                                        },
                                        per_image_mask_chosse_max_num=16,
                                        points_num=1,
                                        area_filter_ratio=0.0001,
                                        box_noise_wh_ratio=0.1,
                                        mask_noise_area_ratio=0.04,
                                        transform=transforms.Compose([
                                            SamResize(resize=1024),
                                            SamRandomHorizontalFlip(prob=0.5),
                                            SamNormalize(
                                                mean=[123.675, 116.28, 103.53],
                                                std=[58.395, 57.12, 57.375]),
                                        ]))

    from torch.utils.data import DataLoader

    collater = SAMBatchCollater(resize=1024)
    train_loader = DataLoader(samdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    net = dinov3_vit_base_patch16_sam2image(use_gradient_checkpoint=True)
    load_state_dict('', net)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

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

        mask_preds, iou_preds = net(input_images,
                                    input_prompts,
                                    mask_out_idxs=[0, 1, 2, 3])

        print('3333', mask_preds.shape, iou_preds.shape, mask_preds.dtype,
              iou_preds.dtype)

        batch_image_embeddings = net.forward_image_encoder(input_images)

        print('4444', batch_image_embeddings[0].shape,
              batch_image_embeddings[1].shape, batch_image_embeddings[2].shape)

        mask_preds, iou_preds = net.forward_prompt_encoder_mask_decoder(
            batch_image_embeddings, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('5555', mask_preds.shape, iou_preds.shape, mask_preds.dtype,
              iou_preds.dtype)

        break
