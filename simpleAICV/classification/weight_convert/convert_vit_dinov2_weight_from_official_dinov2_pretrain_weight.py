import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from simpleAICV.classification import backbones

import torch
import torch.nn.functional as F

if __name__ == '__main__':
    network = 'vit_large_patch14'
    num_classes = 1000
    input_image_size = 518
    scale = 256 / 224
    model = backbones.__dict__[network](**{
        'image_size': input_image_size,
        'global_pool': True,
        'num_classes': num_classes,
    })

    model_name_list = []
    for name, weight in model.state_dict().items():
        model_name_list.append([name, weight.shape])

    print('1111', len(model_name_list))
    for name, weight_shape in model_name_list:
        print('1212', name, weight_shape)

    saved_model_path = '/root/autodl-tmp/pretrained_models/dinov2_pretrain_official_pytorch_weights/dinov2_vitl14_reg4_pretrain.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

    save_name_list = []
    for name, weight in saved_state_dict.items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    for name, weight_shape in save_name_list:
        print('2323', name, weight_shape)

    convert_dict = {}
    for key, value in saved_state_dict.items():
        if key in model.state_dict().keys():
            convert_dict[key] = value
        else:
            print('3434', key)

    print('3333', len(convert_dict))

    in_count = 0
    for key, value in convert_dict.items():
        if key in model.state_dict().keys():
            if value.shape == model.state_dict()[key].shape:
                in_count += 1
            else:
                print('4545', key, value.shape, model.state_dict()[key].shape)
        else:
            print('4646', key)

    print('4444', in_count)

    save_model_name = saved_model_path.split('/')[-1][:-4]
    torch.save(
        convert_dict,
        f'/root/autodl-tmp/pretrained_models/dinov2_convert_from_official_dinov2_pretrain/{save_model_name}_pytorch_official_weight_convert.pth'
    )
