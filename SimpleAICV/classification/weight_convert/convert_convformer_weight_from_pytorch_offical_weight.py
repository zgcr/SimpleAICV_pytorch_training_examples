import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleAICV.classification import backbones

import torch
import torch.nn.functional as F

convert_common_dict = {
    'norm_head.weight': 'norm.weight',
    'norm_head.bias': 'norm.bias',
    'head.weight': 'head.weight',
    'head.bias': 'head.bias',
}

if __name__ == '__main__':
    network = 'convformer_b36'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    model_name_list = []
    for name, weight in model.state_dict().items():
        model_name_list.append([name, weight.shape])

    print('1111', len(model_name_list))
    # for name, shape in model_name_list:
    #     print('1111', name, shape)

    saved_model_path = '/root/autodl-tmp/pretrained_models/convformer_pytorch_official_weights/convformer_b36_in21k.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    save_name_list = []
    for name, weight in saved_state_dict.items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, shape in save_name_list:
    #     print('2222', name, shape)

    convert_dict = {}
    not_include_key = []
    for key, value in saved_state_dict.items():
        if key in model.state_dict().keys():
            if value.shape == model.state_dict()[key].shape:
                convert_dict[key] = value
        else:
            # print('2323', key)
            not_include_key.append(key)

    print('3333', len(convert_dict), len(not_include_key))

    save_model_name = saved_model_path.split('/')[-1][:-4]
    save_pth_path = f'/root/autodl-tmp/pretrained_models/convformer_convert_from_pytorch_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    print('4444', save_pth_path)
    torch.save(convert_dict, save_pth_path)
