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

from simpleAICV.interactive_segmentation.models.light_segment_anything.light_image_encoder import LightImageEncoder
from simpleAICV.interactive_segmentation.models.segment_anything.prompt_encoder import PromptEncoder
from simpleAICV.interactive_segmentation.models.segment_anything.mask_decoder import MaskDecoder

__all__ = [
    'vanb0_light_sam',
    'vanb1_light_sam',
    'vanb2_light_sam',
    'vanb3_light_sam',
    'convformers18_light_sam',
    'convformers36_light_sam',
    'convformerm36_light_sam',
    'convformerb36_light_sam',
]


class LightSAM(nn.Module):

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
                 frozen_mask_decoder=False,
                 sigmoid_out=False,
                 binary_mask_out=False,
                 mask_threshold=0.0):
        super(LightSAM, self).__init__()
        self.image_size = image_size
        self.sigmoid_out = sigmoid_out
        self.binary_mask_out = binary_mask_out
        self.mask_threshold = mask_threshold

        self.image_encoder = LightImageEncoder(
            backbone_type=backbone_type,
            backbone_pretrained_path=backbone_pretrained_path,
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

        masks, iou_predictions = self.mask_decoder(
            image_embeddings=batch_image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            mask_out_idxs=mask_out_idxs)

        batch_mask_outputs = F.interpolate(masks,
                                           (self.image_size, self.image_size),
                                           mode="bilinear",
                                           align_corners=False)

        if self.sigmoid_out:
            batch_mask_outputs = batch_mask_outputs.float()
            batch_mask_outputs = self.sigmoid(batch_mask_outputs)

        if self.binary_mask_out:
            batch_mask_outputs = batch_mask_outputs > self.mask_threshold

        batch_iou_outputs = iou_predictions

        return batch_mask_outputs, batch_iou_outputs


def _light_sam(backbone_type, **kwargs):
    model = LightSAM(backbone_type=backbone_type, **kwargs)

    return model


def vanb0_light_sam(**kwargs):
    return _light_sam(backbone_type='vanb0backbone', **kwargs)


def vanb1_light_sam(**kwargs):
    return _light_sam(backbone_type='vanb1backbone', **kwargs)


def vanb2_light_sam(**kwargs):
    return _light_sam(backbone_type='vanb2backbone', **kwargs)


def vanb3_light_sam(**kwargs):
    return _light_sam(backbone_type='vanb3backbone', **kwargs)


def convformers18_light_sam(**kwargs):
    return _light_sam(backbone_type='convformers18backbone', **kwargs)


def convformers36_light_sam(**kwargs):
    return _light_sam(backbone_type='convformers36backbone', **kwargs)


def convformerm36_light_sam(**kwargs):
    return _light_sam(backbone_type='convformerm36backbone', **kwargs)


def convformerb36_light_sam(**kwargs):
    return _light_sam(backbone_type='convformerb36backbone', **kwargs)


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

    from simpleAICV.interactive_segmentation.datasets.sam_segmentation_dataset import SAMSegmentationDataset
    from simpleAICV.interactive_segmentation.common import SamResize, SamRandomHorizontalFlip, SamNormalize, SAMBatchCollater, load_state_dict

    samdataset = SAMSegmentationDataset(interactive_segmentation_dataset_path,
                                        set_name=[
                                            'sa_000020',
                                        ],
                                        set_type='train',
                                        per_set_image_choose_max_num={
                                            'sa_000020': 1000000,
                                        },
                                        per_image_mask_chosse_max_num=16,
                                        positive_points_num=9,
                                        negative_points_num=9,
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

    collater = SAMBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    net = vanb3_light_sam(image_size=1024,
                          frozen_image_encoder=False,
                          frozen_prompt_encoder=False,
                          frozen_mask_decoder=False,
                          use_gradient_checkpoint=True,
                          sigmoid_out=False,
                          binary_mask_out=False,
                          mask_threshold=0.0)
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

        preds = net(input_images, input_prompts, mask_out_idxs=[0, 1, 2, 3])

        print('3333', preds[0].shape, preds[1].shape, preds[0].dtype,
              preds[1].dtype)

        break
