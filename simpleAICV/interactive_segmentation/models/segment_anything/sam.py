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
from simpleAICV.interactive_segmentation.models.segment_anything.mask_decoder import MaskDecoder
from simpleAICV.interactive_segmentation.models.segment_anything.prompt_encoder import PromptEncoder

__all__ = [
    'sam_b',
    'sam_l',
    'sam_h',
]


class SAM(nn.Module):

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
                 sigmoid_out=False,
                 binary_mask_out=False,
                 mask_threshold=0.0):
        super(SAM, self).__init__()
        self.sigmoid_out = sigmoid_out
        self.binary_mask_out = binary_mask_out
        self.mask_threshold = mask_threshold

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
        self.mask_decoder = MaskDecoder(
            inplanes=prompt_encoder_embedding_planes,
            num_multimask_outputs=mask_decoder_num_multimask_outputs,
            iou_prediction_head_block_nums=
            mask_decoder_iou_prediction_head_block_nums,
            iou_prediction_head_hidden_planes=
            mask_decoder_iou_prediction_head_hidden_planes)

        if self.sigmoid_out:
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

    def forward(self, batch_images, batch_prompts, mask_out_idxs=[1, 2, 3]):
        device = batch_images.device

        # [4, 256, 64, 64]
        batch_image_embeddings = self.image_encoder(batch_images)

        batch_mask_outputs, batch_iou_outputs = [], []
        for per_image_prompt, per_image_embedding in zip(
                batch_prompts, batch_image_embeddings):
            prompt_points = None
            if per_image_prompt['prompt_point'] is not None:
                prompt_points = per_image_prompt['prompt_point']
                prompt_points = prompt_points.to(device)

            prompt_boxes = None
            if per_image_prompt['prompt_box'] is not None:
                prompt_boxes = per_image_prompt['prompt_box']
                prompt_boxes = prompt_boxes.to(device)

            prompt_mask = None
            if per_image_prompt['prompt_mask'] is not None:
                prompt_mask = per_image_prompt['prompt_mask']
                prompt_mask = prompt_mask.to(device)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=prompt_points, boxes=prompt_boxes, masks=prompt_mask)

            masks, iou_predictions = self.mask_decoder(
                image_embeddings=per_image_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe_layer(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                mask_out_idxs=mask_out_idxs)

            masks = F.interpolate(
                masks,
                (self.image_encoder.image_size, self.image_encoder.image_size),
                mode="bilinear",
                align_corners=False)

            if self.sigmoid_out:
                masks = self.sigmoid(masks)

            if self.binary_mask_out:
                masks = masks > self.mask_threshold

            batch_mask_outputs.append(masks)
            batch_iou_outputs.append(iou_predictions)

        return batch_mask_outputs, batch_iou_outputs

    def forward_per_image_encoder(self, image):
        # [4, 256, 64, 64]
        per_image_embedding = self.image_encoder(image)

        return per_image_embedding

    def forward_per_image_prompt_encoder_mask_decoder(self,
                                                      per_image_embedding,
                                                      batch_prompts,
                                                      mask_out_idxs=[1, 2, 3]):
        device = per_image_embedding.device

        batch_mask_outputs, batch_iou_outputs = [], []
        for per_prompt in batch_prompts:
            prompt_points = None
            if per_prompt['prompt_point'] is not None:
                prompt_points = per_prompt['prompt_point']
                prompt_points = prompt_points.to(device)

            prompt_boxes = None
            if per_prompt['prompt_box'] is not None:
                prompt_boxes = per_prompt['prompt_box']
                prompt_boxes = prompt_boxes.to(device)

            prompt_mask = None
            if per_prompt['prompt_mask'] is not None:
                prompt_mask = per_prompt['prompt_mask']
                prompt_mask = prompt_mask.to(device)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=prompt_points, boxes=prompt_boxes, masks=prompt_mask)

            masks, iou_predictions = self.mask_decoder(
                image_embeddings=per_image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe_layer(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                mask_out_idxs=mask_out_idxs)

            masks = F.interpolate(
                masks,
                (self.image_encoder.image_size, self.image_encoder.image_size),
                mode="bilinear",
                align_corners=False)

            if self.sigmoid_out:
                masks = self.sigmoid(masks)

            if self.binary_mask_out:
                masks = masks > self.mask_threshold

            batch_mask_outputs.append(masks)
            batch_iou_outputs.append(iou_predictions)

        return batch_mask_outputs, batch_iou_outputs


def _sam(image_size, patch_size, image_encoder_embedding_planes,
         image_encoder_block_nums, image_encoder_head_nums,
         image_encoder_global_attn_indexes, prompt_encoder_embedding_planes,
         **kwargs):
    model = SAM(
        image_size=image_size,
        patch_size=patch_size,
        image_encoder_embedding_planes=image_encoder_embedding_planes,
        image_encoder_block_nums=image_encoder_block_nums,
        image_encoder_head_nums=image_encoder_head_nums,
        image_encoder_global_attn_indexes=image_encoder_global_attn_indexes,
        prompt_encoder_embedding_planes=prompt_encoder_embedding_planes,
        **kwargs)

    return model


def sam_b(image_size=1024, patch_size=16, **kwargs):
    return _sam(image_size=image_size,
                patch_size=patch_size,
                image_encoder_embedding_planes=768,
                image_encoder_block_nums=12,
                image_encoder_head_nums=12,
                image_encoder_global_attn_indexes=[2, 5, 8, 11],
                prompt_encoder_embedding_planes=256,
                **kwargs)


def sam_l(image_size=1024, patch_size=16, **kwargs):
    return _sam(image_size=image_size,
                patch_size=patch_size,
                image_encoder_embedding_planes=1024,
                image_encoder_block_nums=24,
                image_encoder_head_nums=16,
                image_encoder_global_attn_indexes=[5, 11, 17, 23],
                prompt_encoder_embedding_planes=256,
                **kwargs)


def sam_h(image_size=1024, patch_size=16, **kwargs):
    return _sam(image_size=image_size,
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

    from tools.path import COCO2017_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.interactive_segmentation.datasets.coco2017dataset import COCO2017dataset
    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMCollater, load_state_dict

    sam1bdataset = COCO2017dataset(COCO2017_path,
                                   set_name='train2017',
                                   positive_points_num=9,
                                   negative_points_num=9,
                                   area_filter_ratio=0.0025,
                                   box_noise_pixel=50,
                                   mask_noise_pixel=100,
                                   transform=transforms.Compose([
                                       SamResize(resize=1024),
                                       SamRandomHorizontalFlip(prob=0.5),
                                       SamNormalize(
                                           mean=[123.675, 116.28, 103.53],
                                           std=[58.395, 57.12, 57.375]),
                                   ]))

    from torch.utils.data import DataLoader
    collater = SAMCollater(resize=1024,
                           positive_point_num_range=[1, 9],
                           negative_point_num_range=[1, 9],
                           batch_align_random_point_num=True,
                           positive_negative_point_num_ratio=1)
    train_loader = DataLoader(sam1bdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.interactive_segmentation.models.segment_anything import sam_b
    net = sam_b(image_size=1024,
                frozen_image_encoder=False,
                frozen_prompt_encoder=False,
                frozen_mask_decoder=False,
                use_gradient_checkpoint=True,
                sigmoid_out=False,
                binary_mask_out=False,
                mask_threshold=0.0)
    load_state_dict(
        '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/sam_official_pytorch_weights/sam_vit_b_01ec64.pth',
        net)

    for data in tqdm(train_loader):
        origin_images, origin_bboxs, origin_masks, origin_sizes = data[
            'origin_image'], data['origin_bbox'], data['origin_mask'], data[
                'origin_size']

        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_positive_prompt_points, input_negative_prompt_points, input_prompt_points = data[
            'positive_prompt_point'], data['negative_prompt_point'], data[
                'prompt_point']

        input_prompt_boxs, input_prompt_masks, batch_images, batch_masks, batch_prompts = data[
            'prompt_box'], data['prompt_mask'], data['batch_image'], data[
                'batch_mask'], data['batch_prompt']

        net = net.cuda()
        batch_images = batch_images.cuda()
        batch_masks = batch_masks.cuda()
        print('1111', batch_images.shape, batch_masks.shape)

        print('2222', len(batch_prompts), batch_prompts[0].keys())

        print('3333', batch_prompts[0]['prompt_point'].shape,
              batch_prompts[0]['prompt_box'].shape,
              batch_prompts[0]['prompt_mask'].shape)

        preds = net(batch_images, batch_prompts, mask_out_idxs=[3])

        for per_pred1, per_pred2 in zip(preds[0], preds[1]):
            print('3333', per_pred1.shape, per_pred2.shape, per_pred1.dtype,
                  per_pred2.dtype)

        break
