import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from simpleAICV.interactive_segmentation.models.segment_anything2 import sam2image

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    network = 'sam2image_hiera_l'
    model = sam2image.__dict__[network](**{})

    model_name_list = []
    for name, weight in model.state_dict().items():
        model_name_list.append([name, weight.shape])

    print('1111', len(model_name_list))
    # for name, weight_shape in model_name_list:
    #     print('1111', name, weight_shape)

    saved_model_path = '/root/autodl-tmp/pretrained_models/sam2_official_pytorch_weights/sam2_hiera_large.pt'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

    save_name_list = []
    for name, weight in saved_state_dict['model'].items():
        if 'image_encoder' in name or 'sam_prompt_encoder' in name or 'sam_mask_decoder' in name:
            save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, weight_shape in save_name_list:
    #     print('2222', name, weight_shape)

    convert_dict = {}
    for key, value in saved_state_dict['model'].items():
        if 'image_encoder' in key:
            key = key.replace('image_encoder.', 'image_encoder.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2323', key)
        elif 'sam_prompt_encoder' in key:
            key = key.replace('sam_prompt_encoder.', 'prompt_encoder.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2424', key)
        elif 'sam_mask_decoder' in key:
            key = key.replace('sam_mask_decoder.', 'mask_decoder.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2525', key)

    convert_name_list = []
    for name, weight in convert_dict.items():
        convert_name_list.append([name, weight.shape])

    print('3333', len(convert_name_list))
    # for name, weight_shape in convert_name_list:
    #     print('3333', name, weight_shape)

    in_count = 0
    for key, value in convert_dict.items():
        if key in model.state_dict().keys():
            if value.shape == model.state_dict()[key].shape:
                in_count += 1
        else:
            print(key)
    print('4444', in_count)

    save_model_name = saved_model_path.split('/')[-1][:-3]
    torch.save(
        convert_dict,
        f'/root/autodl-tmp/pretrained_models/sam2image_weights_from_official_pytorch_weights/{save_model_name}_image_convert_from_pytorch_official_weight.pth'
    )
