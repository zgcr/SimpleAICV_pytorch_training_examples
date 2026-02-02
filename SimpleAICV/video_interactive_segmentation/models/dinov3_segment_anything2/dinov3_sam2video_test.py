import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

import collections
import cv2
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
    'dinov3_vit_small_patch16_sam2video_test',
    'dinov3_vit_small_plus_patch16_sam2video_test',
    'dinov3_vit_base_patch16_sam2video_test',
    'dinov3_vit_large_patch16_sam2video_test',
    'dinov3_vit_large_plus_patch16_sam2video_test',
    'dinov3_vit_huge_plus_patch16_sam2video_test',
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


class DINOV3SAM2VideoTest(nn.Module):

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
                 max_object_pointers_in_encoder=16):
        super(DINOV3SAM2VideoTest, self).__init__()
        self.image_size = image_size
        self.mask_out_idxs = mask_out_idxs
        self.memory_inplanes = memory_inplanes
        self.memory_planes = memory_planes
        self.memory_mask_nums = memory_mask_nums
        self.max_object_pointers_in_encoder = max_object_pointers_in_encoder

        assert self.memory_mask_nums > 0

        self.no_object_score = -1024.0

        self.image_encoder = DINOV3ViTImageEncoder(
            backbone_type=backbone_type,
            backbone_pretrained_path=backbone_pretrained_path,
            image_size=image_size,
            fpn_planes=prompt_encoder_embedding_planes,
            use_gradient_checkpoint=False)
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

    @staticmethod
    def timer(func):
        """计时装饰器，用于统计函数执行耗时"""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            # 将秒转换为毫秒
            elapsed_ms = (end_time - start_time) * 1000
            print(f"{func.__name__} cost time: {elapsed_ms:.6f} ms")

            return result

        return wrapper

    @timer.__func__
    def load_all_video_frames(self, video_dir_path):
        with torch.no_grad():
            with ThreadPoolExecutor(max_workers=8) as executor:
                image_names = list(
                    executor.map(
                        lambda per_frame_name: per_frame_name
                        if '.jpg' in per_frame_name else None,
                        os.listdir(video_dir_path)))
            frames_name_list = [
                name for name in image_names if name is not None
            ]
            frames_name_list = sorted(frames_name_list)
            frames_path_list = [
                os.path.join(video_dir_path, n) for n in frames_name_list
            ]

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    tqdm(executor.map(
                        lambda per_frame_path:
                        (lambda result: (torch.from_numpy(result[0]).float(
                        ).permute(2, 0, 1), result[1], result[2]))
                        (self.preprocess_single_image(per_frame_path)),
                        frames_path_list),
                         total=len(frames_path_list)))

            all_video_frames_image = torch.stack(
                [result[0] for result in results], dim=0).cpu()

            origin_video_h, origin_video_w = results[-1][1], results[-1][2]

            return all_video_frames_image, origin_video_h, origin_video_w

    def preprocess_single_image(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        origin_image_h, origin_image_w = image.shape[0], image.shape[1]
        image = cv2.resize(image, (self.image_size, self.image_size))

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        image = (image - mean) / std
        image = image.astype(np.float32)

        return image, origin_image_h, origin_image_w

    def get_per_frame_feature_position(self, video_state_dict, frame_idx):
        device = video_state_dict['video_device']

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                per_frame_input_image = video_state_dict['video_image'][
                    frame_idx:frame_idx + 1].to(device)
                per_frame_feature, per_frame_position = self.image_encoder(
                    per_frame_input_image)

                return per_frame_feature, per_frame_position

    @timer.__func__
    def get_video_features_positions(self, video_state_dict):
        device = video_state_dict['video_device']
        video_frame_num = video_state_dict['video_frame_num']

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                for frame_idx in tqdm(range(video_frame_num)):
                    per_frame_feature, per_frame_position = self.get_per_frame_feature_position(
                        video_state_dict, frame_idx)
                    video_state_dict['video_feature'][
                        frame_idx] = per_frame_feature
                    video_state_dict['video_position'][
                        frame_idx] = per_frame_position

                return video_state_dict

    def init_video_state_dict(self, video_dir_path):
        all_video_frames_image, video_height, video_width = self.load_all_video_frames(
            video_dir_path)

        video_state_dict = {}
        video_state_dict['video_device'] = next(self.parameters()).device
        video_state_dict['video_image'] = all_video_frames_image.to(
            video_state_dict['video_device'])
        video_state_dict['video_height'] = video_height
        video_state_dict['video_width'] = video_width
        video_state_dict['video_frame_num'] = len(all_video_frames_image)
        video_state_dict['video_frame_idx_list'] = list(
            range(len(all_video_frames_image)))

        video_state_dict['video_feature'] = collections.OrderedDict()
        video_state_dict['video_position'] = collections.OrderedDict()
        video_state_dict = self.get_video_features_positions(video_state_dict)

        video_state_dict['object_prompt_input'] = collections.OrderedDict()
        video_state_dict['object_track_state'] = collections.OrderedDict()
        video_state_dict['object_track_result'] = collections.OrderedDict()

        return video_state_dict

    def clear_video_state_dict_all_info(self, video_state_dict):
        # 用于清空video_state_dict所有信息,便于进行下一个视频的预测
        video_state_dict = {}

        return video_state_dict

    def clear_video_state_dict_all_object_info(self, video_state_dict):
        # 用于清空video_state_dict所有object信息,便于重新开始添加object并重新预测
        video_state_dict['object_prompt_input'] = collections.OrderedDict()
        video_state_dict['object_track_state'] = collections.OrderedDict()
        video_state_dict['object_track_result'] = collections.OrderedDict()

        return video_state_dict

    def clear_video_state_dict_one_object_info(self, video_state_dict,
                                               del_object_id):
        # 用于清空video_state_dict指定object id的object信息
        video_state_dict['object_prompt_input'].pop(del_object_id, None)
        video_state_dict['object_track_state'].pop(del_object_id, None)
        video_state_dict['object_track_result'].pop(del_object_id, None)

        return video_state_dict

    @timer.__func__
    def add_new_object_prompt_input(self,
                                    video_state_dict,
                                    frame_idx,
                                    prompt_point=None,
                                    prompt_box=None,
                                    prompt_mask=None):
        with torch.no_grad():
            assert frame_idx >= 0
            assert prompt_point is not None or prompt_box is not None or prompt_mask is not None

            video_width = video_state_dict['video_width']
            video_height = video_state_dict['video_height']
            w_factor = self.image_size / video_width
            h_factor = self.image_size / video_height

            if prompt_point is not None:
                # 如 [1,1,3],第一个1代表只有一个目标,坐标应当为原图分辨率上坐标,3中前两个值为坐标,第三个值为点类型,必须为1(前景点)或0(背景点)
                assert isinstance(prompt_point, np.ndarray)
                assert len(prompt_point.shape) == 3
                assert prompt_point.shape[0] == 1 and prompt_point.shape[
                    -1] == 3

                # 转换到模型输入分辨率尺度
                # [1,1,3]
                prompt_point = torch.tensor(prompt_point).float().cpu()
                prompt_point[:, :, 0] = prompt_point[:, :, 0] * w_factor
                prompt_point[:, :, 1] = prompt_point[:, :, 1] * h_factor

            if prompt_box is not None:
                # 如 [1,4],第一个1代表只有一个目标,坐标应当为原图分辨率上坐标
                assert isinstance(prompt_box, np.ndarray)
                assert len(prompt_box.shape) == 2
                assert prompt_box.shape[0] == 1 and prompt_box.shape[1] == 4

                # 转换到模型输入分辨率尺度
                # [1,2,2]
                prompt_box = torch.tensor(prompt_box).float().cpu()
                # [1,4]->[1,2,2]
                prompt_box = prompt_box.reshape(-1, 2, 2)
                prompt_box[:, :, 0] = prompt_box[:, :, 0] * w_factor
                prompt_box[:, :, 1] = prompt_box[:, :, 1] * h_factor
                # [1,2,1]
                prompt_box_label = torch.ones(
                    [prompt_box.shape[0], prompt_box.shape[1], 1],
                    dtype=torch.float32)
                # if box left top point, input box label = 2 ,elif box bottom right point, input box label = 3
                prompt_box_label[:, 0, :] = prompt_box_label[:, 0, :] * 2
                prompt_box_label[:, 1, :] = prompt_box_label[:, 1, :] * 3
                # [1,2,3]
                prompt_box = torch.cat([prompt_box, prompt_box_label], axis=-1)

            if prompt_mask is not None:
                # 如 [h,w],prompt_mask应当为原图分辨率,必须为0/1二值化mask
                assert isinstance(prompt_mask, np.ndarray)
                assert len(prompt_mask.shape) == 2
                assert prompt_mask.shape[
                    0] == video_height and prompt_mask.shape[1] == video_width

                # [h,w]
                prompt_mask = torch.tensor(prompt_mask).float().cpu()
                # [1,1,h,w]
                prompt_mask = prompt_mask.unsqueeze(0).unsqueeze(0)
                # [1,1,1024,1024]
                prompt_mask = F.interpolate(prompt_mask,
                                            (self.image_size, self.image_size),
                                            mode='nearest')

            exist_object_ids = sorted(
                list(video_state_dict['object_prompt_input'].keys()))
            if len(exist_object_ids) > 0:
                assert all(
                    isinstance(per_object_id, int)
                    for per_object_id in exist_object_ids)

                max_exist_object_id = max(exist_object_ids)
                new_object_id = max_exist_object_id + 1
            else:
                new_object_id = 0

            has_prompt_point = True if prompt_point is not None else False
            has_prompt_box = True if prompt_box is not None else False
            has_prompt_mask = True if prompt_mask is not None else False

            video_state_dict['object_prompt_input'][
                new_object_id] = collections.OrderedDict()
            video_state_dict['object_prompt_input'][new_object_id][
                frame_idx] = {
                    'prompt_point': prompt_point,
                    'prompt_box': prompt_box,
                    'prompt_mask': prompt_mask,
                }

            video_state_dict['object_track_state'][new_object_id] = False
            video_state_dict['object_track_result'][
                new_object_id] = collections.OrderedDict()

            return exist_object_ids, frame_idx, new_object_id, has_prompt_point, has_prompt_box, has_prompt_mask

    @timer.__func__
    def forward_tracking_for_test(self,
                                  video_state_dict,
                                  start_tracking_frame_idx,
                                  tracking_object_ids,
                                  use_point_prompt_input=True,
                                  use_box_prompt_input=False,
                                  use_mask_prompt_input=False):
        device = video_state_dict['video_device']

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                video_height, video_width = video_state_dict[
                    'video_height'], video_state_dict['video_width']
                video_frame_num = video_state_dict['video_frame_num']
                condition_frame_idx_list = [start_tracking_frame_idx]
                # 若起始帧是第0帧或最后一帧
                if start_tracking_frame_idx == 0 or start_tracking_frame_idx == video_frame_num - 1:
                    if start_tracking_frame_idx == 0:
                        processing_frame_idxs = list(
                            range(0, video_frame_num, 1))
                        processing_frame_reverse_flag = False

                    elif start_tracking_frame_idx == video_frame_num - 1:
                        processing_frame_idxs = list(
                            range(video_frame_num - 1, -1, -1))
                        processing_frame_reverse_flag = True

                    all_frame_mask_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = self.process_one_video_frame_sequence(
                        video_state_dict,
                        processing_frame_idxs=processing_frame_idxs,
                        processing_frame_reverse_flag=
                        processing_frame_reverse_flag,
                        condition_frame_idx_list=condition_frame_idx_list,
                        tracking_object_ids=tracking_object_ids,
                        use_point_prompt_input=use_point_prompt_input,
                        use_box_prompt_input=use_box_prompt_input,
                        use_mask_prompt_input=use_mask_prompt_input)

                    # 记录object的track状态
                    for per_object_id in tracking_object_ids:
                        video_state_dict['object_track_state'][
                            per_object_id] = True

                    for per_frame_idx, per_frame_mask_preds, per_frame_iou_preds, per_frame_pred_object_score_logits in zip(
                            processing_frame_idxs, all_frame_mask_preds,
                            all_frame_iou_preds,
                            all_frame_pred_object_score_logits):
                        # 恢复原始尺寸
                        per_frame_mask_preds = F.interpolate(
                            per_frame_mask_preds,
                            size=(video_height, video_width),
                            mode='bilinear')
                        for per_object_id in tracking_object_ids:
                            per_object_id_idx = tracking_object_ids.index(
                                per_object_id)
                            per_frame_per_object_mask = per_frame_mask_preds[
                                per_object_id_idx][0].cpu().float().numpy()
                            per_frame_per_object_iou = per_frame_iou_preds[
                                per_object_id_idx][0].cpu().float().numpy()
                            per_frame_per_object_score = per_frame_pred_object_score_logits[
                                per_object_id_idx][0].cpu().float().numpy()

                            video_state_dict['object_track_result'][
                                per_object_id][per_frame_idx] = {
                                    'pred_mask': per_frame_per_object_mask,
                                    'pred_iou': per_frame_per_object_iou,
                                    'pred_object_score':
                                    per_frame_per_object_score,
                                }

                # 若起始帧为视频中间的一帧
                else:
                    processing_frame_idxs1 = list(
                        range(start_tracking_frame_idx, video_frame_num, 1))
                    processing_frame_idxs1_reverse_flag = False

                    all_frame_mask_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = self.process_one_video_frame_sequence(
                        video_state_dict,
                        processing_frame_idxs=processing_frame_idxs1,
                        processing_frame_reverse_flag=
                        processing_frame_idxs1_reverse_flag,
                        condition_frame_idx_list=condition_frame_idx_list,
                        tracking_object_ids=tracking_object_ids,
                        use_point_prompt_input=use_point_prompt_input,
                        use_box_prompt_input=use_box_prompt_input,
                        use_mask_prompt_input=use_mask_prompt_input)

                    # 记录object的track状态
                    for per_object_id in tracking_object_ids:
                        video_state_dict['object_track_state'][
                            per_object_id] = True

                    for per_frame_idx, per_frame_mask_preds, per_frame_iou_preds, per_frame_pred_object_score_logits in zip(
                            processing_frame_idxs1, all_frame_mask_preds,
                            all_frame_iou_preds,
                            all_frame_pred_object_score_logits):
                        # 恢复原始尺寸
                        per_frame_mask_preds = F.interpolate(
                            per_frame_mask_preds,
                            size=(video_height, video_width),
                            mode='bilinear')
                        for per_object_id in tracking_object_ids:
                            per_object_id_idx = tracking_object_ids.index(
                                per_object_id)
                            per_frame_per_object_mask = per_frame_mask_preds[
                                per_object_id_idx][0].cpu().float().numpy()
                            per_frame_per_object_iou = per_frame_iou_preds[
                                per_object_id_idx][0].cpu().float().numpy()
                            per_frame_per_object_score = per_frame_pred_object_score_logits[
                                per_object_id_idx][0].cpu().float().numpy()

                            video_state_dict['object_track_result'][
                                per_object_id][per_frame_idx] = {
                                    'pred_mask': per_frame_per_object_mask,
                                    'pred_iou': per_frame_per_object_iou,
                                    'pred_object_score':
                                    per_frame_per_object_score,
                                }

                    processing_frame_idxs2 = list(
                        range(start_tracking_frame_idx, -1, -1))
                    processing_frame_idxs2_reverse_flag = True

                    all_frame_mask_preds, all_frame_iou_preds, all_frame_pred_object_score_logits = self.process_one_video_frame_sequence(
                        video_state_dict,
                        processing_frame_idxs=processing_frame_idxs2,
                        processing_frame_reverse_flag=
                        processing_frame_idxs2_reverse_flag,
                        condition_frame_idx_list=condition_frame_idx_list,
                        tracking_object_ids=tracking_object_ids,
                        use_point_prompt_input=use_point_prompt_input,
                        use_box_prompt_input=use_box_prompt_input,
                        use_mask_prompt_input=use_mask_prompt_input)

                    # 记录object的track状态
                    for per_object_id in tracking_object_ids:
                        video_state_dict['object_track_state'][
                            per_object_id] = True

                    for per_frame_idx, per_frame_mask_preds, per_frame_iou_preds, per_frame_pred_object_score_logits in zip(
                            processing_frame_idxs2, all_frame_mask_preds,
                            all_frame_iou_preds,
                            all_frame_pred_object_score_logits):
                        # 恢复原始尺寸
                        per_frame_mask_preds = F.interpolate(
                            per_frame_mask_preds,
                            size=(video_height, video_width),
                            mode='bilinear')
                        for per_object_id in tracking_object_ids:
                            per_object_id_idx = tracking_object_ids.index(
                                per_object_id)
                            per_frame_per_object_mask = per_frame_mask_preds[
                                per_object_id_idx][0].cpu().float().numpy()
                            per_frame_per_object_iou = per_frame_iou_preds[
                                per_object_id_idx][0].cpu().float().numpy()
                            per_frame_per_object_score = per_frame_pred_object_score_logits[
                                per_object_id_idx][0].cpu().float().numpy()

                            video_state_dict['object_track_result'][
                                per_object_id][per_frame_idx] = {
                                    'pred_mask': per_frame_per_object_mask,
                                    'pred_iou': per_frame_per_object_iou,
                                    'pred_object_score':
                                    per_frame_per_object_score,
                                }

                return video_state_dict

    def process_one_video_frame_sequence(self,
                                         video_state_dict,
                                         processing_frame_idxs,
                                         processing_frame_reverse_flag,
                                         condition_frame_idx_list,
                                         tracking_object_ids,
                                         use_point_prompt_input=True,
                                         use_box_prompt_input=False,
                                         use_mask_prompt_input=False):
        for per_frame_idx in condition_frame_idx_list:
            assert per_frame_idx in processing_frame_idxs
        assert int(use_point_prompt_input) + int(use_box_prompt_input) + int(
            use_mask_prompt_input) >= 1

        assert len(tracking_object_ids) > 0 and isinstance(
            tracking_object_ids, list)
        tracking_object_id_nums = len(tracking_object_ids)
        processing_frame_num = len(processing_frame_idxs)

        device = video_state_dict['video_device']
        video_features = video_state_dict['video_feature']
        video_positions = video_state_dict['video_position']

        all_frame_outputs = {
            'condition_frame_preds': {},
            'not_condition_frame_preds': {},
        }
        for process_frame_idx in processing_frame_idxs:
            process_frame_all_objects_features = [
                torch.repeat_interleave(x, tracking_object_id_nums,
                                        dim=0).to(device)
                for x in video_features[process_frame_idx]
            ]
            process_frame_all_objects_positions = [
                torch.repeat_interleave(x, tracking_object_id_nums,
                                        dim=0).to(device)
                for x in video_positions[process_frame_idx]
            ]

            process_frame_point_inputs = None
            process_frame_mask_inputs = None
            if process_frame_idx in condition_frame_idx_list:
                if use_point_prompt_input:
                    # 请手动确认不同目标的point数量保持一致
                    all_object_prompt_point = [
                        video_state_dict['object_prompt_input']
                        [per_tracking_object_id][process_frame_idx]
                        ['prompt_point']
                        for per_tracking_object_id in tracking_object_ids
                    ]
                    all_object_prompt_point = torch.cat(
                        all_object_prompt_point, dim=0).to(device)
                    assert all_object_prompt_point.shape[
                        0] == tracking_object_id_nums
                    process_frame_point_inputs = all_object_prompt_point

                if use_box_prompt_input:
                    all_object_prompt_box = [
                        video_state_dict['object_prompt_input']
                        [per_tracking_object_id][process_frame_idx]
                        ['prompt_box']
                        for per_tracking_object_id in tracking_object_ids
                    ]
                    all_object_prompt_box = torch.cat(all_object_prompt_box,
                                                      dim=0).to(device)
                    assert all_object_prompt_box.shape[
                        0] == tracking_object_id_nums

                    if process_frame_point_inputs is not None:
                        process_frame_point_inputs = torch.cat([
                            process_frame_point_inputs,
                            all_object_prompt_box,
                        ],
                                                               dim=1)
                    else:
                        process_frame_point_inputs = all_object_prompt_box

                if use_mask_prompt_input:
                    all_object_prompt_mask = [
                        video_state_dict['object_prompt_input']
                        [per_tracking_object_id][process_frame_idx]
                        ['prompt_mask']
                        for per_tracking_object_id in tracking_object_ids
                    ]
                    all_object_prompt_mask = torch.cat(all_object_prompt_mask,
                                                       dim=0).to(device)
                    assert all_object_prompt_mask.shape[
                        0] == tracking_object_id_nums

                    process_frame_mask_inputs = all_object_prompt_mask

            is_condition_frame = True if process_frame_idx in condition_frame_idx_list else False
            process_frame_best_iou_mask_preds, process_frame_best_iou_preds, process_frame_mask_preds, process_frame_iou_preds, process_frame_object_score_logits, process_frame_object_pointer, process_frame_all_objects_features = self.predict_per_frame_mask(
                frame_idx=process_frame_idx,
                is_condition_frame=is_condition_frame,
                point_inputs=process_frame_point_inputs,
                mask_inputs=process_frame_mask_inputs,
                per_frame_features=process_frame_all_objects_features,
                per_frame_positions=process_frame_all_objects_positions,
                all_frame_outputs=all_frame_outputs,
                frame_nums=processing_frame_num,
                reverse=processing_frame_reverse_flag)

            process_frame_preds = {}
            process_frame_preds[
                'best_mask_preds'] = process_frame_best_iou_mask_preds
            process_frame_preds[
                'best_iou_preds'] = process_frame_best_iou_preds
            process_frame_preds[
                'pred_object_score_logits'] = process_frame_object_score_logits
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
            if process_frame_idx in condition_frame_idx_list:
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
        for frame_idx in processing_frame_idxs:
            frame_preds = all_frame_output_dict[frame_idx]
            all_frame_mask_preds.append(frame_preds['best_mask_preds'])
            all_frame_iou_preds.append(frame_preds['best_iou_preds'])
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
                if t < 0 or (frame_nums is not None and t_diff >= frame_nums):
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

    def forward_one_image_test(self,
                               video_state_dict,
                               object_id,
                               frame_idx,
                               mask_out_idxs=[0, 1, 2, 3]):
        device = video_state_dict['video_device']

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                video_height, video_width = video_state_dict[
                    'video_height'], video_state_dict['video_width']

                one_frame_image = video_state_dict['video_image'][
                    frame_idx:frame_idx + 1].to(device)

                # features: torch.Size([1, 256, 256, 256]) torch.Size([1, 256, 128, 128]) torch.Size([1, 256, 64, 64])
                # positions: torch.Size([1, 256, 256, 256]) torch.Size([1, 256, 128, 128]) torch.Size([1, 256, 64, 64])
                one_frame_feature, _ = self.image_encoder(one_frame_image)

                prompt_points = None
                if video_state_dict['object_prompt_input'][object_id][
                        frame_idx]['prompt_point'] is not None:
                    prompt_points = video_state_dict['object_prompt_input'][
                        object_id][frame_idx]['prompt_point']
                    prompt_points = prompt_points.to(device)

                prompt_boxes = None
                if video_state_dict['object_prompt_input'][object_id][
                        frame_idx]['prompt_box'] is not None:
                    prompt_boxes = video_state_dict['object_prompt_input'][
                        object_id][frame_idx]['prompt_box']
                    prompt_boxes = prompt_boxes.to(device)

                prompt_mask = None
                if video_state_dict['object_prompt_input'][object_id][
                        frame_idx]['prompt_mask'] is not None:
                    prompt_mask = video_state_dict['object_prompt_input'][
                        object_id][frame_idx]['prompt_mask']
                    prompt_mask = prompt_mask.to(device)

                    if prompt_mask.shape[-2:] != self.mask_input_size:
                        prompt_mask = F.interpolate(prompt_mask.float(),
                                                    size=self.mask_input_size,
                                                    mode='bilinear',
                                                    antialias=True)

                point_inputs = None
                mask_inputs = None

                if prompt_points is not None:
                    point_inputs = prompt_points

                if prompt_boxes is not None:
                    if point_inputs is not None:
                        point_inputs = torch.cat([
                            point_inputs,
                            prompt_boxes,
                        ],
                                                 dim=1)
                    else:
                        point_inputs = prompt_boxes

                if prompt_mask is not None:
                    mask_inputs = prompt_mask

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=point_inputs, boxes=None, masks=mask_inputs)

                mask_preds, iou_preds, _, _ = self.mask_decoder(
                    image_embeddings=one_frame_feature[-1],
                    image_pe=self.prompt_encoder.get_dense_pe_layer(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    high_res_features=one_frame_feature[0:2],
                    mask_out_idxs=mask_out_idxs)

                mask_preds = F.interpolate(mask_preds,
                                           (self.image_size, self.image_size),
                                           mode="bilinear")

                # 恢复原始尺寸
                mask_preds = F.interpolate(mask_preds,
                                           size=(video_height, video_width),
                                           mode='bilinear')

                return mask_preds, iou_preds

    def forward_one_image_encoder(self, video_state_dict, frame_idx):
        device = video_state_dict['video_device']

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                one_frame_image = video_state_dict['video_image'][
                    frame_idx:frame_idx + 1].to(device)

                # features: torch.Size([1, 256, 256, 256]) torch.Size([1, 256, 128, 128]) torch.Size([1, 256, 64, 64])
                # positions: torch.Size([1, 256, 256, 256]) torch.Size([1, 256, 128, 128]) torch.Size([1, 256, 64, 64])
                one_frame_feature, one_frame_position = self.image_encoder(
                    one_frame_image)

                assert one_frame_feature[0].shape[0] == 1

                return one_frame_feature, one_frame_position

    def forward_one_image_prompt_encoder_mask_decoder(
            self,
            video_state_dict,
            object_id,
            frame_idx,
            mask_out_idxs=[0, 1, 2, 3]):
        device = video_state_dict['video_device']

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                video_height, video_width = video_state_dict[
                    'video_height'], video_state_dict['video_width']

                one_frame_feature, _ = video_state_dict['video_feature'][
                    frame_idx], video_state_dict['video_position'][frame_idx]

                prompt_points = None
                if video_state_dict['object_prompt_input'][object_id][
                        frame_idx]['prompt_point'] is not None:
                    prompt_points = video_state_dict['object_prompt_input'][
                        object_id][frame_idx]['prompt_point']
                    prompt_points = prompt_points.to(device)

                prompt_boxes = None
                if video_state_dict['object_prompt_input'][object_id][
                        frame_idx]['prompt_box'] is not None:
                    prompt_boxes = video_state_dict['object_prompt_input'][
                        object_id][frame_idx]['prompt_box']
                    prompt_boxes = prompt_boxes.to(device)

                prompt_mask = None
                if video_state_dict['object_prompt_input'][object_id][
                        frame_idx]['prompt_mask'] is not None:
                    prompt_mask = video_state_dict['object_prompt_input'][
                        object_id][frame_idx]['prompt_mask']
                    prompt_mask = prompt_mask.to(device)

                    if prompt_mask.shape[-2:] != self.mask_input_size:
                        prompt_mask = F.interpolate(prompt_mask.float(),
                                                    size=self.mask_input_size,
                                                    mode='bilinear',
                                                    antialias=True)

                point_inputs = None
                mask_inputs = None

                if prompt_points is not None:
                    point_inputs = prompt_points

                if prompt_boxes is not None:
                    if point_inputs is not None:
                        point_inputs = torch.cat([
                            point_inputs,
                            prompt_boxes,
                        ],
                                                 dim=1)
                    else:
                        point_inputs = prompt_boxes

                if prompt_mask is not None:
                    mask_inputs = prompt_mask

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=point_inputs, boxes=None, masks=mask_inputs)

                mask_preds, iou_preds, _, _ = self.mask_decoder(
                    image_embeddings=one_frame_feature[-1],
                    image_pe=self.prompt_encoder.get_dense_pe_layer(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    high_res_features=one_frame_feature[0:2],
                    mask_out_idxs=mask_out_idxs)

                mask_preds = F.interpolate(mask_preds,
                                           (self.image_size, self.image_size),
                                           mode="bilinear")

                # 恢复原始尺寸
                mask_preds = F.interpolate(mask_preds,
                                           size=(video_height, video_width),
                                           mode='bilinear')

                return mask_preds, iou_preds


def _dinov3_sam2videotest(backbone_type, backbone_pretrained_path, image_size,
                          patch_size, prompt_encoder_embedding_planes,
                          **kwargs):
    model = DINOV3SAM2VideoTest(
        backbone_type=backbone_type,
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=prompt_encoder_embedding_planes,
        **kwargs)

    return model


def dinov3_vit_small_patch16_sam2video_test(backbone_pretrained_path='',
                                            image_size=1024,
                                            patch_size=16,
                                            **kwargs):
    return _dinov3_sam2videotest(
        backbone_type='dinov3_vit_small_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_small_plus_patch16_sam2video_test(backbone_pretrained_path='',
                                                 image_size=1024,
                                                 patch_size=16,
                                                 **kwargs):
    return _dinov3_sam2videotest(
        backbone_type='dinov3_vit_small_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_base_patch16_sam2video_test(backbone_pretrained_path='',
                                           image_size=1024,
                                           patch_size=16,
                                           **kwargs):
    return _dinov3_sam2videotest(
        backbone_type='dinov3_vit_base_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_large_patch16_sam2video_test(backbone_pretrained_path='',
                                            image_size=1024,
                                            patch_size=16,
                                            **kwargs):
    return _dinov3_sam2videotest(
        backbone_type='dinov3_vit_large_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_large_plus_patch16_sam2video_test(backbone_pretrained_path='',
                                                 image_size=1024,
                                                 patch_size=16,
                                                 **kwargs):
    return _dinov3_sam2videotest(
        backbone_type='dinov3_vit_large_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)


def dinov3_vit_huge_plus_patch16_sam2video_test(backbone_pretrained_path='',
                                                image_size=1024,
                                                patch_size=16,
                                                **kwargs):
    return _dinov3_sam2videotest(
        backbone_type='dinov3_vit_huge_plus_patch16_backbone',
        backbone_pretrained_path=backbone_pretrained_path,
        image_size=image_size,
        patch_size=patch_size,
        prompt_encoder_embedding_planes=256,
        **kwargs)
