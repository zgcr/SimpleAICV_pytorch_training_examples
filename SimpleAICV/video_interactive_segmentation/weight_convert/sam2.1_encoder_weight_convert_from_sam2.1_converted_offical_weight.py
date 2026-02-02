import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleAICV.video_interactive_segmentation.models.segment_anything2 import sam2video_train

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    network = 'hiera_l_sam2video'

    model = sam2video_train.__dict__[network](**{})
    model.eval()

    model_name_list = []
    for name, weight in model.state_dict().items():
        if 'image_encoder' in name:
            model_name_list.append([name, weight.shape])

    print('1111', len(model_name_list))
    # for name, shape in model_name_list:
    #     print('1111', name, shape)

    saved_model_path = '/root/autodl-tmp/pretrained_models/sam2.1_convert_from_pytorch_official_weights/sam2.1_hiera_large_convert_from_pytorch_official_weight.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    save_name_list = []
    for name, weight in saved_state_dict.items():
        if 'image_encoder' in name:
            save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, shape in save_name_list:
    #     print('2222', name, shape)

    convert_dict = {}
    for key, value in saved_state_dict.items():
        if 'image_encoder' in key:
            if key in model.state_dict().keys():
                if value.shape == model.state_dict()[key].shape:
                    key = key.replace('image_encoder.', '')
                    convert_dict[key] = value
            else:
                print('2323', key)

    convert_name_list = []
    for name, weight in convert_dict.items():
        convert_name_list.append([name, weight.shape])

    print('3333', len(convert_name_list))
    # for name, weight_shape in convert_name_list:
    #     print('3333', name, weight_shape)

    save_model_name = saved_model_path.split('/')[-1][:-4]
    torch.save(
        convert_dict,
        f'/root/autodl-tmp/pretrained_models/sam2.1_encoder_weights_convert_from_pytorch_official_weights/{save_model_name}_encoder_convert_from_pytorch_official_weight.pth'
    )
