import numpy as np

import torch
import torch.nn as nn


class PositionEmbeddingRandom(nn.Module):

    def __init__(self, num_pos_feats=64):
        super(PositionEmbeddingRandom, self).__init__()
        self.register_buffer("positional_encoding_gaussian_matrix",
                             torch.randn((2, num_pos_feats)))

    def forward(self, size):
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self.pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        # C x H x W
        pe = pe.permute(2, 0, 1)

        return pe

    def forward_with_coords(self, coords_input, image_size):
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size
        coords[:, :, 1] = coords[:, :, 1] / image_size
        # B x N x C
        coords = self.pe_encoding(coords.to(torch.float))

        return coords

    def pe_encoding(self, coords):
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        coords = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

        return coords


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


class PromptEncoder(nn.Module):

    def __init__(self,
                 image_size=1024,
                 patch_size=16,
                 embedding_planes=256,
                 mask_inter_planes=16):
        super(PromptEncoder, self).__init__()
        self.image_size = image_size
        self.embedding_planes = embedding_planes
        self.image_embedding_size = image_size // patch_size
        self.pe_layer = PositionEmbeddingRandom(embedding_planes // 2)

        # pos/neg point + 2 box corners
        self.num_point_embeddings = 4
        point_embeddings = [
            nn.Embedding(1, embedding_planes)
            for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        self.not_a_point_embed = nn.Embedding(1, embedding_planes)
        self.no_mask_embed = nn.Embedding(1, embedding_planes)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1,
                      mask_inter_planes // 4,
                      kernel_size=2,
                      stride=2,
                      padding=0), LayerNorm2d(mask_inter_planes // 4),
            nn.GELU(),
            nn.Conv2d(mask_inter_planes // 4,
                      mask_inter_planes,
                      kernel_size=2,
                      stride=2,
                      padding=0), LayerNorm2d(mask_inter_planes), nn.GELU(),
            nn.Conv2d(mask_inter_planes,
                      embedding_planes,
                      kernel_size=1,
                      stride=1,
                      padding=0))

    def forward(self, points, boxes, masks):
        if points is not None:
            batch_size = points.shape[0]
        elif boxes is not None:
            batch_size = boxes.shape[0]
        elif masks is not None:
            batch_size = masks.shape[0]
        else:
            batch_size = 1

        device = self.point_embeddings[0].weight.device

        sparse_embeddings = torch.empty((batch_size, 0, self.embedding_planes),
                                        device=device)
        if points is not None:
            coords, labels = points[:, :, 0:2], points[:, :, 2]
            point_embeddings = self.embed_points(coords,
                                                 labels,
                                                 pad=(boxes is None))
            sparse_embeddings = torch.cat(
                [sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self.embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings],
                                          dim=1)

        if masks is not None:
            dense_embeddings = self.embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, -1, 1, 1).expand(batch_size, -1, self.image_embedding_size,
                                    self.image_embedding_size)

        return sparse_embeddings, dense_embeddings

    def get_dense_pe_layer(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def embed_points(self, points, labels, pad):
        """Embeds point prompts."""
        # Shift to center of pixel
        points = points + 0.5

        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2),
                                        device=points.device)
            padding_label = -torch.ones(
                (labels.shape[0], 1), device=labels.device)

            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(
            points, self.image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight

        return point_embedding

    def embed_boxes(self, boxes):
        """Embeds box prompts."""
        # Shift to center of pixel
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight

        return corner_embedding

    def embed_masks(self, masks):
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)

        return mask_embedding


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

        break
