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

    model_name_list = []
    for name, weight in model.state_dict().items():
        model_name_list.append([name, weight.shape])

    print('1111', len(model_name_list))
    # for name, weight_shape in model_name_list:
    #     print('1111', name, weight_shape)

    saved_model_path = '/root/autodl-tmp/pretrained_models/sam2.1_pytorch_official_weights/sam2.1_hiera_large.pt'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    save_name_list = []
    for name, weight in saved_state_dict['model'].items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, weight_shape in save_name_list:
    #     print('2222', name, weight_shape)

    convert_dict = {}
    for key, value in saved_state_dict['model'].items():
        if 'image_encoder.' in key:
            key = key.replace('image_encoder.', 'image_encoder.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2323', key)
        elif 'sam_prompt_encoder.' in key:
            key = key.replace('sam_prompt_encoder.', 'prompt_encoder.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2424', key)
        elif 'sam_mask_decoder.' in key:
            key = key.replace('sam_mask_decoder.', 'mask_decoder.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2525', key)
        elif 'maskmem_tpos_enc' in key:
            key = key.replace('maskmem_tpos_enc',
                              'mask_memory_time_position_encoder')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2626', key)
        elif 'no_mem_embed' in key:
            key = key.replace('no_mem_embed', 'no_memory_embedding')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2727', key)
        elif 'no_mem_pos_enc' in key:
            key = key.replace('no_mem_pos_enc', 'no_memory_position_encoder')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2828', key)
        elif 'no_obj_ptr' in key:
            key = key.replace('no_obj_ptr', 'no_object_pointer')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('2929', key)
        elif 'no_obj_embed_spatial' in key:
            key = key.replace('no_obj_embed_spatial',
                              'no_object_embedding_spatial')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('3131', key)
        elif 'obj_ptr_proj.' in key:
            key = key.replace('obj_ptr_proj.', 'object_pointer_projection.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('3232', key)
        elif 'obj_ptr_tpos_proj.' in key:
            key = key.replace('obj_ptr_tpos_proj.',
                              'object_pointer_time_position_projection.')
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('3333', key)
        else:
            if key in model.state_dict().keys():
                convert_dict[key] = value
            else:
                print('9191', key)

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
        f'/root/autodl-tmp/pretrained_models/sam2.1_convert_from_pytorch_official_weights/{save_model_name}_convert_from_pytorch_official_weight.pth'
    )
