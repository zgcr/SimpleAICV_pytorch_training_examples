import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleAICV.classification import backbones

import torch
import torch.nn.functional as F

if __name__ == '__main__':
    network = 'vit_huge_patch14'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224
    model = backbones.__dict__[network](**{
        'image_size': 224,
        'global_pool': True,
        'num_classes': num_classes,
    })

    model_name_list = []
    for name, weight in model.state_dict().items():
        model_name_list.append([name, weight.shape])

    print('1111', len(model_name_list))
    # for name, _ in model_name_list:
    #     print(name)

    saved_model_path = '/root/autodl-tmp/pretrained_models/vit_mae_pretrain_pytorch_official_weights/mae_pretrain_vit_huge.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    save_name_list = []
    for name, weight in saved_state_dict['model'].items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, _ in save_name_list:
    #     print(name)

    convert_dict = {}
    for key, value in saved_state_dict['model'].items():
        if key in model.state_dict().keys():
            convert_dict[key] = value
        else:
            print('2323', key)

    print('3333', len(convert_dict))

    in_count = 0
    for key, value in convert_dict.items():
        if key in model.state_dict().keys():
            if value.shape == model.state_dict()[key].shape:
                in_count += 1
        else:
            print(key)
    print('4444', in_count)

    save_model_name = saved_model_path.split('/')[-1][:-4]
    torch.save(
        convert_dict,
        f'/root/autodl-tmp/pretrained_models/vit_mae_pretrain_convert_from_pytorch_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    )
