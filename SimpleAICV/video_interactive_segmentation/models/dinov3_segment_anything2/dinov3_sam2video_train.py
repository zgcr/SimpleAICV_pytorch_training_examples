import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from SimpleAICV.video_interactive_segmentation.models.dinov3_segment_anything2.dinov3_image_encoder import DINOV3ViTImageEncoder
from SimpleAICV.video_interactive_segmentation.models.segment_anything2.prompt_encoder import PromptEncoder
from SimpleAICV.video_interactive_segmentation.models.segment_anything2.mask_decoder import MaskDecoder
from SimpleAICV.video_interactive_segmentation.models.segment_anything2.memory_attention import MemoryAttention
from SimpleAICV.video_interactive_segmentation.models.segment_anything2.memory_encoder import MemoryEncoder

__all__ = [
    'dinov3_vit_small_patch16_sam2video',
    'dinov3_vit_small_plus_patch16_sam2video',
    'dinov3_vit_base_patch16_sam2video',
    'dinov3_vit_large_patch16_sam2video',
    'dinov3_vit_large_plus_patch16_sam2video',
    'dinov3_vit_huge_plus_patch16_sam2video',
]


class MLP(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes, layer_nums):
        super(MLP, self).__init__()
        self.layer_nums = layer_nums
        h = [hidden_planes] * (layer_nums - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([inplanes] + h, h + [planes]))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.layer_nums - 1 else layer(x)

        return x


class DINOV3SAM2Video(nn.Module):

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
                 mask_out_idxs=[0, 1, 2, 3],
                 memory_inplanes=256,
                 memory_planes=64,
                 memory_mask_nums=7,
                 max_object_pointers_in_encoder=16,
                 use_gradient_checkpoint=False,
                 frozen_image_encoder=False,
                 frozen_prompt_encoder=False,
                 frozen_mask_decoder=False,
                 frozen_memory_attention=False,
                 frozen_memory_encoder=False,
                 use_single_prompt=True,
                 use_point_prompt_prob=0.25,
                 use_box_prompt_prob=0.25,
                 use_mask_prompt_prob=0.5,
                 max_condition_frame_num=2,
                 random_condition_frame_num=True,
                 max_decoder_point_iters_frame_num=2,
                 random_decoder_point_iters_frame_num=True,
                 sample_decoder_point_from_gt_mask_prob=0.1,
                 decoder_point_iters_num=7):
        super(DINOV3SAM2Video, self).__init__()
        self.image_size = image_size
        self.mask_out_idxs = mask_out_idxs
        self.memory_inplanes = memory_inplanes
        self.memory_planes = memory_planes
        self.memory_mask_nums = memory_mask_nums
        self.max_object_pointers_in_encoder = max_object_pointers_in_encoder

        assert self.memory_mask_nums > 0

        self.no_object_score = -1024.0

        self.use_single_prompt = use_single_prompt
        self.use_point_prompt_prob = use_point_prompt_prob
        self.use_box_prompt_prob = use_box_prompt_prob
        self.use_mask_prompt_prob = use_mask_prompt_prob
        self.max_condition_frame_num = max_condition_frame_num
        self.random_condition_frame_num = random_condition_frame_num
        self.max_decoder_point_iters_frame_num = max_decoder_point_iters_frame_num
        self.random_decoder_point_iters_frame_num = random_decoder_point_iters_frame_num
        self.sample_decoder_point_from_gt_mask_prob = sample_decoder_point_from_gt_mask_prob
        self.decoder_point_iters_num = decoder_point_iters_num

        assert 0.0 <= self.use_point_prompt_prob <= 1.0
        assert 0.0 <= self.use_box_prompt_prob <= 1.0
        assert 0.0 <= self.use_mask_prompt_prob <= 1.0
        if self.use_single_prompt:
            assert self.use_point_prompt_prob + self.use_box_prompt_prob + self.use_mask_prompt_prob == 1.
        else:
            assert self.use_point_prompt_prob + self.use_box_prompt_prob + self.use_mask_prompt_prob <= 3.

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

        self.memory_attention = MemoryAttention(inplanes=memory_inplanes,
                                                layer_nums=4)
        self.memory_encoder = MemoryEncoder(inplanes=memory_inplanes,
                                            planes=memory_planes)

        self.mask_input_size = (image_size // 4, image_size // 4)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        self.mask_memory_time_position_encoder = nn.Parameter(
            torch.zeros(memory_mask_nums, 1, 1, memory_planes))
        self.no_memory_embedding = nn.Parameter(
            torch.zeros(1, 1, memory_inplanes))
        self.no_memory_position_encoder = nn.Parameter(
            torch.zeros(1, 1, memory_inplanes))

        self.no_object_pointer = nn.Parameter(torch.zeros(1, memory_inplanes))
        self.no_object_embedding_spatial = nn.Parameter(
            torch.zeros(1, memory_planes))
        self.object_pointer_projection = MLP(memory_inplanes, memory_inplanes,
                                             memory_inplanes, 3)
        self.object_pointer_time_position_projection = nn.Linear(
            memory_inplanes, memory_planes)

        nn.init.trunc_normal_(self.mask_memory_time_position_encoder, std=0.02)
        nn.init.trunc_normal_(self.no_memory_embedding, std=0.02)
        nn.init.trunc_normal_(self.no_memory_position_encoder, std=0.02)
        nn.init.trunc_normal_(self.no_object_pointer, std=0.02)
        nn.init.trunc_normal_(self.no_object_embedding_spatial, std=0.02)

        if frozen_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if frozen_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
        if frozen_mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
        if frozen_memory_attention:
            for param in self.memory_attention.parameters():
                param.requires_grad = False
        if frozen_memory_encoder:
            for param in self.memory_encoder.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        # [T, B, 3, H, W] -> [B, T, 3, H, W] -> [B*T, 3, H, W]
        input_images = inputs['image'].permute(1, 0, 2, 3, 4).flatten(0, 1)
        features, positions = self.image_encoder(input_images)

        del input_images

        inputs['features'], inputs['positions'] = features, positions

        del features, positions

        inputs = self.prepare_batch_prompt_inputs(inputs)
        outs = self.forward_tracking_for_train(inputs)

        del inputs

        return outs

    def prepare_batch_prompt_inputs(self, inputs):
        """
        支持单prompt输入和多prompt输入,默认按单prompt输入,单prompt输入情况下与官方实现无区别;
        """
        # 获取视频总帧数
        frame_num = inputs['frame_num']
        assert frame_num >= 1

        # 将每一帧所有object的GT mask存在字典frame_step_gt_mask中
        # key为帧索引,value为这一帧的GT mask,维度为[N_o, 1, H, W],N_o为B个视频中object_num的总数
        # input['mask']:[T, N_o, H, W]
        # frame_step_gt_mask的每个value:[N_o, 1, H, W]
        frame_step_gt_mask = collections.OrderedDict()
        for per_frame_idx in range(frame_num):
            per_frame_gt_mask = inputs['mask'][per_frame_idx]
            per_frame_gt_mask = torch.unsqueeze(per_frame_gt_mask, dim=1)
            # per_frame_gt_mask:[N_o, 1, H, W]
            frame_step_gt_mask[per_frame_idx] = per_frame_gt_mask
        inputs['frame_step_gt_mask'] = frame_step_gt_mask

        del frame_step_gt_mask

        # 确定每个batch究竟采用哪种prompt输入
        inputs['use_prompt_point_flag'] = False
        inputs['use_prompt_box_flag'] = False
        inputs['use_prompt_mask_flag'] = False
        if self.use_single_prompt:
            # 视频只有一帧时，限制必须使用点提示
            if frame_num == 1:
                inputs['use_prompt_point_flag'] = True
            else:
                use_prompt_prob = np.random.uniform(0, 1)
                if 0. < use_prompt_prob < self.use_point_prompt_prob:
                    inputs['use_prompt_point_flag'] = True
                elif self.use_point_prompt_prob < use_prompt_prob < (
                        self.use_point_prompt_prob + self.use_box_prompt_prob):
                    inputs['use_prompt_box_flag'] = True
                elif (self.use_point_prompt_prob +
                      self.use_box_prompt_prob) < use_prompt_prob < 1.:
                    inputs['use_prompt_mask_flag'] = True
        else:
            use_point_prompt_prob = np.random.uniform(0, 1)
            use_box_prompt_prob = np.random.uniform(0, 1)
            use_mask_prompt_prob = np.random.uniform(0, 1)
            if use_point_prompt_prob < self.use_point_prompt_prob:
                inputs['use_prompt_point_flag'] = True
            if use_box_prompt_prob < self.use_box_prompt_prob:
                inputs['use_prompt_box_flag'] = True
            if use_mask_prompt_prob < self.use_mask_prompt_prob:
                inputs['use_prompt_mask_flag'] = True

        # 约束max_condition_frame_num/max_decoder_point_iters_frame_num不超过视频帧数
        max_condition_frame_num = min(frame_num, self.max_condition_frame_num)
        max_decoder_point_iters_frame_num = min(
            frame_num, self.max_decoder_point_iters_frame_num)
        random_condition_frame_num = self.random_condition_frame_num
        random_decoder_point_iters_frame_num = self.random_decoder_point_iters_frame_num

        if random_condition_frame_num and max_condition_frame_num > 1:
            condition_frame_num = np.random.randint(
                1, max_condition_frame_num + 1)
        else:
            condition_frame_num = max_condition_frame_num

        if random_decoder_point_iters_frame_num and max_decoder_point_iters_frame_num > condition_frame_num:
            decoder_point_iters_frame_num = np.random.randint(
                condition_frame_num, max_decoder_point_iters_frame_num + 1)
        else:
            decoder_point_iters_frame_num = max_decoder_point_iters_frame_num

        # 如果初始条件帧数为1,则仅选择第0帧作为初始条件帧,否则选择第0帧和其他随机帧作为初始条件帧
        if condition_frame_num == 1:
            condition_frame_idx_list = [0]
        else:
            condition_frame_idx_list = [0] + np.random.choice(
                range(1, frame_num), condition_frame_num - 1,
                replace=False).tolist()

        # 初始条件帧之外的其他帧是非初始条件帧
        not_condition_frame_idx_list = []
        for per_frame_idx in range(frame_num):
            if per_frame_idx not in condition_frame_idx_list:
                not_condition_frame_idx_list.append(per_frame_idx)

        inputs['condition_frame_idx_list'] = condition_frame_idx_list
        inputs['not_condition_frame_idx_list'] = not_condition_frame_idx_list

        # 确定点迭代帧
        if decoder_point_iters_frame_num == condition_frame_num:
            decoder_point_iters_frame_idx_list = condition_frame_idx_list
        else:
            extra_decoder_point_iters_frame_num = decoder_point_iters_frame_num - condition_frame_num
            decoder_point_iters_frame_idx_list = condition_frame_idx_list + np.random.choice(
                not_condition_frame_idx_list,
                extra_decoder_point_iters_frame_num,
                replace=False).tolist()

        inputs['decoder_point_iters_frame_idx_list'] = []
        # 如果采用点或框作为prompt输入,则采取点迭代策略,且仅有初始条件帧才采用点迭代策略
        if inputs['use_prompt_point_flag'] or inputs['use_prompt_box_flag']:
            inputs[
                'decoder_point_iters_frame_idx_list'] = decoder_point_iters_frame_idx_list

        # 获取初始条件帧用于训练的prompt
        inputs['frame_step_prompt_point'] = collections.OrderedDict()
        inputs['frame_step_prompt_box'] = collections.OrderedDict()
        inputs['frame_step_prompt_mask'] = collections.OrderedDict()
        for per_frame_idx in range(frame_num):
            inputs['frame_step_prompt_point'][per_frame_idx] = None
            inputs['frame_step_prompt_box'][per_frame_idx] = None
            inputs['frame_step_prompt_mask'][per_frame_idx] = None
            if per_frame_idx in condition_frame_idx_list:
                if inputs['use_prompt_point_flag']:
                    # [N_o, point_num, 3]
                    inputs['frame_step_prompt_point'][per_frame_idx] = inputs[
                        'prompt_point'][per_frame_idx]
                if inputs['use_prompt_box_flag']:
                    # [N_o, 2, 2]
                    per_frame_prompt_box = inputs['prompt_box'][
                        per_frame_idx].reshape(-1, 2, 2)
                    device = per_frame_prompt_box.device
                    # [N_o, 2, 1]
                    per_frame_prompt_box_label = torch.ones(
                        [
                            per_frame_prompt_box.shape[0],
                            per_frame_prompt_box.shape[1], 1
                        ],
                        dtype=torch.float32).to(device)
                    per_frame_prompt_box_label[:,
                                               0] = per_frame_prompt_box_label[:,
                                                                               0] * 2
                    per_frame_prompt_box_label[:,
                                               1] = per_frame_prompt_box_label[:,
                                                                               1] * 3
                    # [N_o, 2, 3]
                    per_frame_prompt_box = torch.cat(
                        [per_frame_prompt_box, per_frame_prompt_box_label],
                        dim=-1)
                    inputs['frame_step_prompt_box'][
                        per_frame_idx] = per_frame_prompt_box
                if inputs['use_prompt_mask_flag']:
                    # 使用GT mask做prompt mask
                    per_frame_prompt_mask = inputs['frame_step_gt_mask'][
                        per_frame_idx]
                    # [N_o, 1, H, W]
                    inputs['frame_step_prompt_mask'][
                        per_frame_idx] = per_frame_prompt_mask

        return inputs

    def forward_tracking_for_train(self, inputs):
        features, positions = inputs['features'], inputs['positions']

        # 视频帧数: frame_num
        # 初始条件帧idx list: condition_frame_idx_list
        # 非条件帧idx list: not_condition_frame_idx_list
        # 采用点迭代策略的帧idx list: decoder_point_iters_frame_idx_list
        # 处理帧顺序idx list: processing_frame_idxs,先处理初始条件帧,再处理非条件帧
        frame_num = inputs['frame_num']
        condition_frame_idx_list = inputs['condition_frame_idx_list']
        not_condition_frame_idx_list = inputs['not_condition_frame_idx_list']
        decoder_point_iters_frame_idx_list = inputs[
            'decoder_point_iters_frame_idx_list']
        processing_frame_idxs = condition_frame_idx_list + not_condition_frame_idx_list

        all_frame_outputs = {
            'condition_frame_preds': {},
            'not_condition_frame_preds': {},
        }
        for process_frame_idx in processing_frame_idxs:
            frame_idx, video_idx = inputs['object_to_frame_idx'].unbind(dim=-1)
            flatten_feature_idx = video_idx * frame_num + frame_idx
            feature_idx = flatten_feature_idx[process_frame_idx]

            process_frame_all_objects_features = [
                x[feature_idx] for x in features
            ]
            process_frame_all_objects_positions = [
                x[feature_idx] for x in positions
            ]

            process_frame_point_inputs = None
            process_frame_mask_inputs = None
            if process_frame_idx in condition_frame_idx_list:
                if inputs['use_prompt_point_flag']:
                    process_frame_prompt_points = inputs[
                        'frame_step_prompt_point'][process_frame_idx]
                    process_frame_point_inputs = process_frame_prompt_points

                if inputs['use_prompt_box_flag']:
                    process_frame_prompt_boxs = inputs[
                        'frame_step_prompt_box'][process_frame_idx]
                    if process_frame_point_inputs is not None:
                        process_frame_point_inputs = torch.cat([
                            process_frame_point_inputs,
                            process_frame_prompt_boxs,
                        ],
                                                               dim=1)
                    else:
                        process_frame_point_inputs = process_frame_prompt_boxs

                if inputs['use_prompt_mask_flag']:
                    process_frame_prompt_masks = inputs[
                        'frame_step_prompt_mask'][process_frame_idx]
                    process_frame_mask_inputs = process_frame_prompt_masks

            process_frame_gt_masks = inputs['frame_step_gt_mask'][
                process_frame_idx]

            is_condition_frame = True if process_frame_idx in condition_frame_idx_list else False
            process_frame_best_iou_mask_preds, process_frame_best_iou_preds, process_frame_mask_preds, process_frame_iou_preds, process_frame_object_score_logits, process_frame_object_pointer, process_frame_all_objects_features = self.predict_per_frame_mask(
                frame_idx=process_frame_idx,
                is_condition_frame=is_condition_frame,
                point_inputs=process_frame_point_inputs,
                mask_inputs=process_frame_mask_inputs,
                per_frame_features=process_frame_all_objects_features,
                per_frame_positions=process_frame_all_objects_positions,
                all_frame_outputs=all_frame_outputs,
                frame_nums=frame_num,
                reverse=False)

            process_frame_preds = {}
            process_frame_preds['mask_preds'] = [
                process_frame_mask_preds,
            ]
            process_frame_preds['iou_preds'] = [
                process_frame_iou_preds,
            ]
            process_frame_preds['pred_object_score_logits'] = [
                process_frame_object_score_logits,
            ]
            process_frame_preds[
                'object_pointer'] = process_frame_object_pointer

            # 如果frame_idx在采用点迭代策略的帧idx list中,则调用decoder_point_iters_sampling_and_predict_mask,添加纠正点,重新预测,更新process_frame_preds结果
            if process_frame_idx in decoder_point_iters_frame_idx_list:
                all_decoder_point_iters_best_iou_mask_preds, all_decoder_point_iters_best_iou_preds, all_decoder_point_iters_mask_preds, all_decoder_point_iters_iou_preds, all_decoder_point_iters_object_score_logits, all_decoder_point_iters_object_pointers = self.decoder_point_iters_sampling_and_predict_mask(
                    point_inputs=process_frame_point_inputs,
                    gt_masks=process_frame_gt_masks,
                    per_frame_features=process_frame_all_objects_features,
                    per_frame_mask_preds=process_frame_mask_preds,
                    per_frame_iou_preds=process_frame_iou_preds,
                    per_frame_object_score_logits=
                    process_frame_object_score_logits,
                    per_frame_object_pointer=process_frame_object_pointer,
                    per_frame_best_iou_mask_preds=
                    process_frame_best_iou_mask_preds,
                    per_frame_best_iou_preds=process_frame_best_iou_preds)

                process_frame_best_iou_mask_preds = all_decoder_point_iters_best_iou_mask_preds[
                    -1]
                process_frame_best_iou_preds = all_decoder_point_iters_best_iou_preds[
                    -1]
                process_frame_mask_preds = all_decoder_point_iters_mask_preds[
                    -1]
                process_frame_iou_preds = all_decoder_point_iters_iou_preds[-1]
                process_frame_object_score_logits = all_decoder_point_iters_object_score_logits[
                    -1]
                process_frame_object_pointer = all_decoder_point_iters_object_pointers[
                    -1]

                process_frame_preds[
                    'mask_preds'] = all_decoder_point_iters_mask_preds
                process_frame_preds[
                    'iou_preds'] = all_decoder_point_iters_iou_preds
                process_frame_preds[
                    'pred_object_score_logits'] = all_decoder_point_iters_object_score_logits
                process_frame_preds[
                    'object_pointer'] = process_frame_object_pointer

            # 更新当前frame_idx的mask_memory_features,mask_memory_positions
            mask_memory_features, mask_memory_positions = self.encode_frame_memory(
                image_embeddings=process_frame_all_objects_features[-1],
                best_iou_mask_preds=process_frame_best_iou_mask_preds,
                object_score_logits=process_frame_object_score_logits)
            process_frame_preds['mask_memory_features'] = mask_memory_features
            process_frame_preds[
                'mask_memory_positions'] = mask_memory_positions

            # 根据当前帧是否为条件帧,将输出存储在condition_frame_preds/not_condition_frame_preds
            if (process_frame_idx in condition_frame_idx_list) or (
                    process_frame_idx in decoder_point_iters_frame_idx_list):
                all_frame_outputs['condition_frame_preds'][
                    process_frame_idx] = process_frame_preds
            else:
                all_frame_outputs['not_condition_frame_preds'][
                    process_frame_idx] = process_frame_preds

        all_frame_output_dict = {}
        all_frame_output_dict.update(
            all_frame_outputs["condition_frame_preds"])
        all_frame_output_dict.update(
            all_frame_outputs["not_condition_frame_preds"])

        del all_frame_outputs

        all_frame_mask_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = [], [], []
        for frame_idx in range(frame_num):
            frame_preds = all_frame_output_dict[frame_idx]
            all_frame_mask_preds.append(frame_preds['mask_preds'])
            all_frame_iou_preds.append(frame_preds['iou_preds'])
            all_frame_pred_object_score_logits.append(
                frame_preds['pred_object_score_logits'])

        del all_frame_output_dict

        return all_frame_mask_preds, all_frame_iou_preds, all_frame_pred_object_score_logits

    def predict_per_frame_mask(self,
                               frame_idx,
                               is_condition_frame,
                               point_inputs,
                               mask_inputs,
                               per_frame_features,
                               per_frame_positions,
                               all_frame_outputs,
                               frame_nums,
                               reverse=False):
        image_embeddings, high_res_features = per_frame_features[
            -1], per_frame_features[:-1]

        if mask_inputs is not None:
            best_iou_mask_preds, best_iou_preds, mask_preds, iou_preds, object_score_logits, object_pointer = self.use_mask_as_output(
                image_embeddings, high_res_features, mask_inputs)
        else:
            image_embeddings = self.prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_condition_frame=is_condition_frame,
                per_frame_features=per_frame_features[-1:],
                per_frame_positions=per_frame_positions[-1:],
                all_frame_outputs=all_frame_outputs,
                frame_nums=frame_nums,
                track_in_reverse=reverse)

            mask_out_idxs = self.mask_out_idxs

            best_iou_mask_preds, best_iou_preds, mask_preds, iou_preds, _, object_score_logits, object_pointer = self.forward_prompt_encoder_mask_decoder(
                image_embeddings=image_embeddings,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                mask_out_idxs=mask_out_idxs)

        per_frame_features = [
            high_res_features[0],
            high_res_features[1],
            image_embeddings,
        ]

        return best_iou_mask_preds, best_iou_preds, mask_preds, iou_preds, object_score_logits, object_pointer, per_frame_features

    def use_mask_as_output(self, image_embeddings, high_res_features,
                           input_masks):
        B = input_masks.shape[0]
        device = input_masks.device
        input_masks = input_masks.float()

        out_scale, out_bias = 20.0, -10.0
        mask_preds = input_masks * out_scale + out_bias

        best_iou_mask_preds = mask_preds

        iou_preds = torch.ones(B, 1).to(device)

        best_iou_preds = iou_preds

        mask_out_idxs = self.mask_out_idxs

        input_masks = self.mask_downsample(input_masks)
        _, _, _, _, _, object_score_logits, object_pointer = self.forward_prompt_encoder_mask_decoder(
            image_embeddings=image_embeddings,
            mask_inputs=input_masks,
            high_res_features=high_res_features,
            mask_out_idxs=mask_out_idxs)

        is_obj_appearing = torch.any(input_masks.flatten(1) > 0.0,
                                     dim=1).float().unsqueeze(1)

        object_pointer = is_obj_appearing * object_pointer + (
            1 - is_obj_appearing) * self.no_object_pointer

        return best_iou_mask_preds, best_iou_preds, mask_preds, iou_preds, object_score_logits, object_pointer

    def prepare_memory_conditioned_features(self,
                                            frame_idx,
                                            is_condition_frame,
                                            per_frame_features,
                                            per_frame_positions,
                                            all_frame_outputs,
                                            frame_nums,
                                            track_in_reverse=False):
        device = per_frame_features[-1].device
        B, C, H, W = per_frame_features[-1].shape[0], per_frame_features[
            -1].shape[1], per_frame_features[-1].shape[2], per_frame_features[
                -1].shape[3]

        if not is_condition_frame:
            condition_outputs = all_frame_outputs['condition_frame_preds']
            selected_condition_outputs, _ = self.select_closest_cond_frames(
                frame_idx, condition_outputs, max_cond_frame_num=-1)
            time_position_and_prevs = [
                (0, out) for out in selected_condition_outputs.values()
            ]
            for time_position in range(1, self.memory_mask_nums):
                t_rel = self.memory_mask_nums - time_position
                if not track_in_reverse:
                    prev_frame_idx = frame_idx - t_rel
                else:
                    prev_frame_idx = frame_idx + t_rel
                out = all_frame_outputs['not_condition_frame_preds'].get(
                    prev_frame_idx, None)
                if out is None:
                    # maybe some cond frames not selected yet
                    unselected_cond = {
                        t: v
                        for t, v in condition_outputs.items()
                        if t not in selected_condition_outputs
                    }
                    out = unselected_cond.get(prev_frame_idx, None)
                time_position_and_prevs.append((time_position, out))

            max_object_pointers_in_encoder = min(
                frame_nums, self.max_object_pointers_in_encoder)
            pointer_condition_outputs = {
                t: out
                for t, out in selected_condition_outputs.items()
                if (t >= frame_idx if track_in_reverse else t <= frame_idx)
            }
            position_and_pointers = [
                ((frame_idx - t) * (-1 if track_in_reverse else 1),
                 out['object_pointer'])
                for t, out in pointer_condition_outputs.items()
            ]
            for t_diff in range(1, max_object_pointers_in_encoder):
                t = frame_idx + (t_diff if track_in_reverse else -t_diff)
                if t < 0 or (frame_nums is not None and t >= frame_nums):
                    break
                out = all_frame_outputs['not_condition_frame_preds'].get(
                    t, None)
                if out is None and t in condition_outputs:
                    out = condition_outputs[t]
                if out is not None:
                    position_and_pointers.append(
                        (t_diff, out['object_pointer']))

            to_cat_memory = []
            to_cat_memory_position_embedding = []
            for time_position, prev in time_position_and_prevs:
                if prev is None:
                    continue
                feats = prev['mask_memory_features'].to(device)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                mask_memory_positions = prev['mask_memory_positions'].to(
                    device)
                mask_memory_positions = mask_memory_positions.flatten(
                    2).permute(2, 0, 1)
                mask_memory_positions = mask_memory_positions + self.mask_memory_time_position_encoder[
                    self.memory_mask_nums - time_position - 1]
                to_cat_memory_position_embedding.append(mask_memory_positions)

            object_pointer_token_nums = 0
            if len(position_and_pointers) > 0:
                position_list, pointer_list = zip(*position_and_pointers)
                object_pointers = torch.stack(pointer_list, dim=0)
                t_diff_max = max_object_pointers_in_encoder - 1
                time_position_dim = C
                object_position = torch.tensor(position_list).to(
                    device).float()
                object_position = self.get_1d_sine_pe(object_position /
                                                      t_diff_max,
                                                      dim=time_position_dim)
                object_position = self.object_pointer_time_position_projection(
                    object_position)
                object_position = object_position.unsqueeze(1).expand(
                    -1, B, self.memory_planes)

                if self.memory_planes < C:
                    object_pointers = object_pointers.reshape(
                        -1, B, C // self.memory_planes, self.memory_planes)
                    object_pointers = object_pointers.permute(0, 2, 1,
                                                              3).flatten(0, 1)
                    object_position = object_position.repeat_interleave(
                        C // self.memory_planes, dim=0)

                to_cat_memory.append(object_pointers)
                to_cat_memory_position_embedding.append(object_position)
                object_pointer_token_nums = object_pointers.shape[0]

            if len(to_cat_memory) == 0:
                image_embeddings = per_frame_features[-1].flatten(2).permute(
                    2, 0, 1) + self.no_memory_embedding
                image_embeddings = image_embeddings.permute(1, 2, 0).view(
                    B, C, H, W)
            else:
                memory = torch.cat(to_cat_memory, dim=0)
                memory_position_embedding = torch.cat(
                    to_cat_memory_position_embedding, dim=0)
                features = [
                    x.flatten(2).permute(2, 0, 1) for x in per_frame_features
                ]
                positions = [
                    x.flatten(2).permute(2, 0, 1) for x in per_frame_positions
                ]
                image_embeddings = self.memory_attention(
                    curr=features,
                    curr_pos=positions,
                    memory=memory,
                    memory_pos=memory_position_embedding,
                    num_obj_ptr_tokens=object_pointer_token_nums)

                image_embeddings = image_embeddings.permute(1, 2, 0).view(
                    B, C, H, W)

            return image_embeddings
        else:
            image_embeddings = per_frame_features[-1].flatten(2).permute(
                2, 0, 1) + self.no_memory_embedding
            image_embeddings = image_embeddings.permute(1, 2,
                                                        0).view(B, C, H, W)

            return image_embeddings

    def get_1d_sine_pe(self, pos_inds, dim, temperature=10000):
        pe_dim = dim // 2
        dim_t = torch.arange(pe_dim,
                             dtype=torch.float32,
                             device=pos_inds.device)
        dim_t = temperature**(2 * (dim_t // 2) / pe_dim)
        pos_embed = pos_inds.unsqueeze(-1) / dim_t
        pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)

        return pos_embed

    def select_closest_cond_frames(self,
                                   frame_idx,
                                   cond_frame_outputs,
                                   max_cond_frame_num=-1):
        if max_cond_frame_num == -1 or len(
                cond_frame_outputs) <= max_cond_frame_num:
            selected_outputs = cond_frame_outputs
            unselected_outputs = {}
        else:
            assert max_cond_frame_num >= 2
            selected_outputs = {}
            idx_before = max((t for t in cond_frame_outputs if t < frame_idx),
                             default=None)
            if idx_before is not None:
                selected_outputs[idx_before] = cond_frame_outputs[idx_before]
            idx_after = min((t for t in cond_frame_outputs if t >= frame_idx),
                            default=None)
            if idx_after is not None:
                selected_outputs[idx_after] = cond_frame_outputs[idx_after]
            num_remain = max_cond_frame_num - len(selected_outputs)
            inds_remain = sorted(
                (t for t in cond_frame_outputs if t not in selected_outputs),
                key=lambda x: abs(x - frame_idx))[:num_remain]
            selected_outputs.update(
                (t, cond_frame_outputs[t]) for t in inds_remain)
            unselected_outputs = {
                t: v
                for t, v in cond_frame_outputs.items()
                if t not in selected_outputs
            }

        return selected_outputs, unselected_outputs

    def forward_prompt_encoder_mask_decoder(self,
                                            image_embeddings,
                                            point_inputs=None,
                                            mask_inputs=None,
                                            high_res_features=None,
                                            mask_out_idxs=[0, 1, 2, 3]):
        B = image_embeddings.shape[0]
        device = image_embeddings.device

        if point_inputs is None:
            point_input_coords = torch.zeros(B, 1, 2, device=device)
            point_input_labels = -torch.ones(
                B, 1, 1, dtype=torch.int32, device=device)
            point_inputs = torch.cat([point_input_coords, point_input_labels],
                                     axis=-1)

        if mask_inputs is not None:
            if mask_inputs.shape[-2:] != self.mask_input_size:
                mask_inputs = F.interpolate(mask_inputs.float(),
                                            size=self.mask_input_size,
                                            mode='bilinear',
                                            antialias=True)
            else:
                mask_inputs = mask_inputs
        else:
            mask_inputs = None

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_inputs, boxes=None, masks=mask_inputs)

        mask_preds, iou_preds, mask_tokens_out, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe_layer(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=high_res_features,
            mask_out_idxs=mask_out_idxs)

        is_obj_appearing = object_score_logits > 0
        mask_preds = torch.where(is_obj_appearing[:, None, None], mask_preds,
                                 self.no_object_score).float()
        mask_preds = F.interpolate(mask_preds,
                                   size=(self.image_size, self.image_size),
                                   mode='bilinear')

        output_tokens = mask_tokens_out[:, 0]
        best_iou_mask_preds = mask_preds
        best_iou_preds = iou_preds
        if len(mask_out_idxs) > 1:
            best_iou_idxs = torch.argmax(iou_preds, dim=-1)
            batch_idxs = torch.arange(B, device=device)
            best_iou_mask_preds = mask_preds[batch_idxs,
                                             best_iou_idxs].unsqueeze(1)
            best_iou_preds = iou_preds[batch_idxs, best_iou_idxs].unsqueeze(1)
            if mask_tokens_out.shape[1] > 1:
                output_tokens = mask_tokens_out[batch_idxs, best_iou_idxs]

        object_pointer = self.object_pointer_projection(output_tokens)

        is_obj_appearing = is_obj_appearing.float()

        object_pointer = is_obj_appearing * object_pointer + (
            1 - is_obj_appearing) * self.no_object_pointer

        return best_iou_mask_preds, best_iou_preds, mask_preds, iou_preds, output_tokens, object_score_logits, object_pointer

    def decoder_point_iters_sampling_and_predict_mask(
        self,
        point_inputs,
        gt_masks,
        per_frame_features,
        per_frame_mask_preds,
        per_frame_iou_preds,
        per_frame_object_score_logits,
        per_frame_object_pointer,
        per_frame_best_iou_mask_preds,
        per_frame_best_iou_preds,
    ):
        image_embeddings = per_frame_features[-1]
        high_res_features = per_frame_features[:-1]

        all_decoder_point_iters_best_iou_mask_preds = [
            per_frame_best_iou_mask_preds
        ]
        all_decoder_point_iters_best_iou_preds = [per_frame_best_iou_preds]

        all_decoder_point_iters_mask_preds = [per_frame_mask_preds]
        all_decoder_point_iters_iou_preds = [per_frame_iou_preds]

        all_decoder_point_iters_object_score_logits = [
            per_frame_object_score_logits
        ]
        all_decoder_point_iters_object_pointers = [per_frame_object_pointer]

        mask_inputs = per_frame_best_iou_mask_preds
        current_best_iou_mask_preds = per_frame_best_iou_mask_preds
        for _ in range(self.decoder_point_iters_num):
            if np.random.uniform(
                    0, 1) < self.sample_decoder_point_from_gt_mask_prob:
                new_point_inputs = self.sample_random_point(gt_masks,
                                                            None,
                                                            num_pt=1)
            else:
                new_point_inputs = self.sample_random_point(
                    gt_masks, (current_best_iou_mask_preds > 0), num_pt=1)

            if point_inputs is not None:
                point_inputs = torch.cat([point_inputs, new_point_inputs],
                                         dim=1)
            else:
                point_inputs = new_point_inputs

            mask_out_idxs = self.mask_out_idxs

            per_decoder_point_iter_best_iou_mask_preds, per_decoder_point_iter_best_iou_preds, per_decoder_point_iter_mask_preds, per_decoder_point_iter_iou_preds, _, per_decoder_point_iter_object_score_logits, per_decoder_point_iter_object_pointer = self.forward_prompt_encoder_mask_decoder(
                image_embeddings=image_embeddings,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                mask_out_idxs=mask_out_idxs)
            mask_inputs = per_decoder_point_iter_best_iou_mask_preds
            current_best_iou_mask_preds = per_decoder_point_iter_best_iou_mask_preds

            all_decoder_point_iters_best_iou_mask_preds.append(
                per_decoder_point_iter_best_iou_mask_preds)
            all_decoder_point_iters_best_iou_preds.append(
                per_decoder_point_iter_best_iou_preds)
            all_decoder_point_iters_mask_preds.append(
                per_decoder_point_iter_mask_preds)
            all_decoder_point_iters_iou_preds.append(
                per_decoder_point_iter_iou_preds)
            all_decoder_point_iters_object_score_logits.append(
                per_decoder_point_iter_object_score_logits)
            all_decoder_point_iters_object_pointers.append(
                per_decoder_point_iter_object_pointer)

        return all_decoder_point_iters_best_iou_mask_preds, all_decoder_point_iters_best_iou_preds, all_decoder_point_iters_mask_preds, all_decoder_point_iters_iou_preds, all_decoder_point_iters_object_score_logits, all_decoder_point_iters_object_pointers

    def sample_random_point(self, gt_masks, pred_masks, num_pt=1):
        gt_masks = gt_masks.bool()

        if pred_masks is None:
            pred_masks = torch.zeros_like(gt_masks)

        pred_masks = pred_masks.bool()

        B, _, H_im, W_im = gt_masks.shape
        device = gt_masks.device
        fp_masks = ~gt_masks & pred_masks
        fn_masks = gt_masks & ~pred_masks
        all_correct = torch.all((gt_masks == pred_masks).flatten(2),
                                dim=2)[..., None, None]
        pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
        pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
        pts_noise[..., 1] *= fn_masks
        pts_idx = pts_noise.flatten(2).argmax(dim=2)
        labels = (pts_idx % 2).to(torch.int32)
        pts_idx = pts_idx // 2
        pts_x = pts_idx % W_im
        pts_y = pts_idx // W_im
        points = torch.stack([pts_x, pts_y], dim=2).float()

        labels = labels.unsqueeze(dim=-1)
        new_points = torch.cat([points, labels], dim=-1)

        return new_points

    def encode_frame_memory(self, image_embeddings, best_iou_mask_preds,
                            object_score_logits):
        B = image_embeddings.shape[0]
        device = image_embeddings.device

        mask_for_memory = torch.sigmoid(best_iou_mask_preds)
        mask_for_memory = mask_for_memory * 20.0 - 10.0
        mask_memory_features, mask_memory_positions = self.memory_encoder(
            image_embeddings, mask_for_memory, skip_mask_sigmoid=True)

        if object_score_logits is not None:
            is_obj_appearing = (object_score_logits > 0).float()
        else:
            is_obj_appearing = torch.ones(B, 1).to(device)
        is_obj_appearing = is_obj_appearing.unsqueeze(dim=-1).unsqueeze(dim=-1)

        mask_memory_features = mask_memory_features + (
            1 - is_obj_appearing) * self.no_object_embedding_spatial.unsqueeze(
                dim=-1).unsqueeze(dim=-1)

        return mask_memory_features, mask_memory_positions


def _dinov3_sam2video(backbone_type, backbone_pretrained_path, image_size,
                      patch_size, prompt_encoder_embedding_planes, **kwargs):
    model = DINOV3SAM2Video(
        backbone_type=backbone_type,
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=prompt_encoder_embedding_planes,
        **kwargs)

    return model


def dinov3_vit_small_patch16_sam2video(backbone_pretrained_path='',
                                       image_size=1024,
                                       patch_size=16,
                                       **kwargs):
    return _dinov3_sam2video(backbone_type='dinov3_vit_small_patch16_backbone',
                             backbone_pretrained_path=backbone_pretrained_path,
                             image_size=image_size,
                             patch_size=patch_size,
                             prompt_encoder_embedding_planes=256,
                             **kwargs)


def dinov3_vit_small_plus_patch16_sam2video(backbone_pretrained_path='',
                                            image_size=1024,
                                            patch_size=16,
                                            **kwargs):
    return _dinov3_sam2video(
        backbone_type='dinov3_vit_small_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_base_patch16_sam2video(backbone_pretrained_path='',
                                      image_size=1024,
                                      patch_size=16,
                                      **kwargs):
    return _dinov3_sam2video(backbone_type='dinov3_vit_base_patch16_backbone',
                             backbone_pretrained_path=backbone_pretrained_path,
                             image_size=image_size,
                             patch_size=patch_size,
                             prompt_encoder_embedding_planes=256,
                             **kwargs)


def dinov3_vit_large_patch16_sam2video(backbone_pretrained_path='',
                                       image_size=1024,
                                       patch_size=16,
                                       **kwargs):
    return _dinov3_sam2video(backbone_type='dinov3_vit_large_patch16_backbone',
                             backbone_pretrained_path=backbone_pretrained_path,
                             image_size=image_size,
                             patch_size=patch_size,
                             prompt_encoder_embedding_planes=256,
                             **kwargs)


def dinov3_vit_large_plus_patch16_sam2video(backbone_pretrained_path='',
                                            image_size=1024,
                                            patch_size=16,
                                            **kwargs):
    return _dinov3_sam2video(
        backbone_type='dinov3_vit_large_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_huge_plus_patch16_sam2video(backbone_pretrained_path='',
                                           image_size=1024,
                                           patch_size=16,
                                           **kwargs):
    return _dinov3_sam2video(
        backbone_type='dinov3_vit_huge_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)
