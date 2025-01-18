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

from simpleAICV.interactive_segmentation.models.segment_anything.mask_decoder import MaskDecoder


class MaskDecoderMatting(MaskDecoder):

    def forward(self,
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                mask_out_idxs=[0, 1, 2, 3]):
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # one image feature for multi prompt feature
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings,
                                          tokens.shape[0],
                                          dim=0)
        else:
            # one image feature for one prompt feature/ batch image feature for batch prompt feature
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        # src:[1, 256, 64, 64]
        src = src.transpose(1, 2).view(b, c, h, w)
        feat3 = src.clone()
        # upscaled_embedding:[1, 32, 256, 256]
        upscaled_embedding = self.output_upscaling(src)
        feat1 = upscaled_embedding.clone()
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
        return masks, iou_pred, feat3, feat1


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
    from simpleAICV.interactive_segmentation.common_matting import SAMMattingResize, SAMMattingNormalize, SAMMattingRandomHorizontalFlip, SAMMattingBatchCollater

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
            # SAMMattingNormalize(mean=[123.675, 116.28, 103.53],
            #                     std=[58.395, 57.12, 57.375]),
        ]))

    from torch.utils.data import DataLoader

    collater = SAMMattingBatchCollater(resize=1024, positive_point_num_range=1)
    train_loader = DataLoader(samdataset,
                              batch_size=4,
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
    mask_decoder_net = MaskDecoderMatting(
        inplanes=256,
        num_multimask_outputs=3,
        iou_prediction_head_block_nums=3,
        iou_prediction_head_hidden_planes=256)

    for data in tqdm(train_loader):
        input_images, input_boxs, input_masks, sizes = data['image'], data[
            'box'], data['mask'], data['size']

        input_prompt_points, input_prompt_boxs, input_prompt_masks = data[
            'prompt_point'], data['prompt_box'], data['prompt_mask']

        trimaps, fg_maps, bg_maps = data['trimap'], data['fg_map'], data[
            'bg_map']

        image_encoder_net = image_encoder_net.cuda()
        prompt_encoder_net = prompt_encoder_net.cuda()
        mask_decoder_net = mask_decoder_net.cuda()

        input_images = input_images.cuda()
        print('1111', input_images.shape)

        batch_image_embeddings = image_encoder_net(input_images)

        input_prompt_points = input_prompt_points.cuda()
        input_prompt_boxs = input_prompt_boxs.cuda()
        input_prompt_masks = input_prompt_masks.cuda()

        sparse_embeddings, dense_embeddings = prompt_encoder_net(
            points=input_prompt_points,
            boxes=input_prompt_boxs,
            masks=input_prompt_masks)

        print('2222', sparse_embeddings.shape, dense_embeddings.shape)

        masks, iou_predictions, feat3, feat1 = mask_decoder_net(
            image_embeddings=batch_image_embeddings,
            image_pe=prompt_encoder_net.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            mask_out_idxs=[0, 1, 2, 3])

        print('3333', masks.shape, iou_predictions.shape, feat3.shape,
              feat1.shape)

        break
