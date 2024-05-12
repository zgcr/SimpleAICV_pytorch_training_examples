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

from simpleAICV.interactive_segmentation.models.segment_anything.transformer import TwoWayTransformer


class LayerNorm2d(nn.Module):

    def __init__(self, inplanes, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(inplanes))
        self.bias = nn.Parameter(torch.zeros(inplanes))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        return x


class MLP(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, layer_nums):
        super(MLP, self).__init__()
        self.layer_nums = layer_nums

        h = [hidden_planes] * (layer_nums - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([inplanes] + h, h + [planes]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.layer_nums - 1 else layer(x)

        return x


class MaskDecoder(nn.Module):

    def __init__(self,
                 inplanes=256,
                 num_multimask_outputs=3,
                 iou_prediction_head_block_nums=3,
                 iou_prediction_head_hidden_planes=256):
        super(MaskDecoder, self).__init__()

        self.transformer = TwoWayTransformer(block_nums=2,
                                             embedding_planes=inplanes,
                                             head_nums=8,
                                             mlp_planes=2048)

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, inplanes)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, inplanes)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(inplanes,
                               inplanes // 4,
                               kernel_size=2,
                               stride=2,
                               padding=0), LayerNorm2d(inplanes // 4),
            nn.GELU(),
            nn.ConvTranspose2d(inplanes // 4,
                               inplanes // 8,
                               kernel_size=2,
                               stride=2,
                               padding=0), nn.GELU())

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(inplanes, inplanes, inplanes // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(
            inplanes=inplanes,
            hidden_planes=iou_prediction_head_hidden_planes,
            planes=self.num_mask_tokens,
            layer_nums=iou_prediction_head_block_nums)

    def forward(self,
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                mask_out_idxs=[1, 2, 3]):
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](
                mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        masks = masks[:, mask_out_idxs, :, :]
        iou_pred = iou_pred[:, mask_out_idxs]

        # Prepare output
        return masks, iou_pred


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
                              batch_size=2,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    from simpleAICV.interactive_segmentation.models.segment_anything.image_encoder import ViTImageEncoder
    from simpleAICV.interactive_segmentation.models.segment_anything.prompt_encoder import PromptEncoder

    image_encoder_net = ViTImageEncoder(image_size=1024,
                                        patch_size=16,
                                        inplanes=3,
                                        embedding_planes=768,
                                        block_nums=12,
                                        head_nums=12,
                                        mlp_ratio=4,
                                        out_planes=256,
                                        window_size=14,
                                        global_attn_indexes=[2, 5, 8, 11],
                                        use_gradient_checkpoint=True)
    prompt_encoder_net = PromptEncoder(image_size=1024,
                                       patch_size=16,
                                       embedding_planes=256,
                                       mask_inter_planes=16)
    mask_decoder_net = MaskDecoder(inplanes=256,
                                   num_multimask_outputs=3,
                                   iou_prediction_head_block_nums=3,
                                   iou_prediction_head_hidden_planes=256)

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

        image_encoder_net = image_encoder_net.cuda()
        prompt_encoder_net = prompt_encoder_net.cuda()
        mask_decoder_net = mask_decoder_net.cuda()

        batch_images = batch_images.cuda()
        print('1111', batch_images.shape)

        device = batch_images.device

        # [4, 256, 64, 64]
        batch_image_embeddings = image_encoder_net(batch_images)

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

            sparse_embeddings, dense_embeddings = prompt_encoder_net(
                points=prompt_points, boxes=prompt_boxes, masks=prompt_mask)

            print('1111', sparse_embeddings.shape, dense_embeddings.shape)

            masks, iou_predictions = mask_decoder_net(
                image_embeddings=per_image_embedding.unsqueeze(0),
                image_pe=prompt_encoder_net.get_dense_pe_layer(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                mask_out_idxs=[0, 1, 2, 3])

            print('2222', masks.shape, iou_predictions.shape)

        break
