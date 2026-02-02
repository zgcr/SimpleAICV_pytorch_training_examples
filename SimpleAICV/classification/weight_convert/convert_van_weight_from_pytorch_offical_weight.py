import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from SimpleAICV.classification import backbones

import torch
import torch.nn.functional as F

filter_list = {
    'norm1.weight',
    'norm1.bias',
    'norm1.running_mean',
    'norm1.running_var',
    'norm1.num_batches_tracked',
    'norm2.weight',
    'norm2.bias',
    'norm2.running_mean',
    'norm2.running_var',
    'norm2.num_batches_tracked',
    'norm3.weight',
    'norm3.bias',
    'norm3.running_mean',
    'norm3.running_var',
    'norm3.num_batches_tracked',
    'norm4.weight',
    'norm4.bias',
    'norm4.running_mean',
    'norm4.running_var',
    'norm4.num_batches_tracked',
    'head.weight',
    'head.bias',
}

if __name__ == '__main__':
    # network = 'van_b4'
    # num_classes = 1000
    # input_image_size = 224
    # scale = 256 / 224
    # model = backbones.__dict__[network](**{
    #     'num_classes': num_classes,
    # })

    # model_name_list = []
    # num_batches_tracked_num = 0
    # for name, weight in model.state_dict().items():
    #     model_name_list.append([name, weight.shape])
    #     if 'num_batches_tracked' in name:
    #         num_batches_tracked_num += 1

    # print('1111', len(model_name_list), num_batches_tracked_num)
    # # for name, weight_shape in model_name_list:
    # #     print('1111', name, weight_shape)

    # saved_model_path = '/root/autodl-tmp/pretrained_models/van_pytorch_official_weights/van_b4.pth'
    # saved_state_dict = torch.load(saved_model_path,
    #                               map_location=torch.device('cpu'),
    #                               weights_only=True)

    # save_name_list = []
    # for name, weight in saved_state_dict['state_dict'].items():
    #     save_name_list.append([name, weight.shape])

    # print('2222', len(save_name_list))
    # # for name, weight_shape in save_name_list:
    # #     print('2222', name, weight_shape)

    # convert_dict = {}
    # for key, value in saved_state_dict['state_dict'].items():
    #     if key not in model.state_dict().keys():
    #         print('3333', key)
    #     elif key in filter_list:
    #         print('4444', key)
    #     elif 'layer_scale' in key:
    #         print('5555', key)
    #     else:
    #         key = key
    #         value = value
    #         convert_dict[key] = value

    # convert_name_list = []
    # for name, weight in convert_dict.items():
    #     convert_name_list.append([name, weight.shape])

    # print('3333', len(convert_name_list))
    # # for name, weight_shape in convert_name_list:
    # #     print('3333', name, weight_shape)

    # in_count = 0
    # for key, value in convert_dict.items():
    #     if key in model.state_dict().keys():
    #         if value.shape == model.state_dict()[key].shape:
    #             in_count += 1
    #         else:
    #             print('4444', key)
    #     else:
    #         print('5555', key)
    # print('4444', in_count)

    # save_model_name = saved_model_path.split('/')[-1][:-4]
    # torch.save(
    #     convert_dict,
    #     f'/root/autodl-tmp/pretrained_models/van_convert_from_pytorch_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    # )

    ###################################################################
    network = 'van_b6'
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224
    model = backbones.__dict__[network](**{
        'num_classes': num_classes,
    })

    model_name_list = []
    num_batches_tracked_num = 0
    for name, weight in model.state_dict().items():
        model_name_list.append([name, weight.shape])
        if 'num_batches_tracked' in name:
            num_batches_tracked_num += 1

    print('1111', len(model_name_list), num_batches_tracked_num)
    # for name, weight_shape in model_name_list:
    #     print('1111', name, weight_shape)

    saved_model_path = '/root/autodl-tmp/pretrained_models/van_pytorch_official_weights/van_b6_22k.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)
    print(saved_state_dict.keys())
    save_name_list = []
    for name, weight in saved_state_dict['state_dict'].items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, weight_shape in save_name_list:
    #     print('2222', name, weight_shape)

    convert_dict = {}
    for key, value in saved_state_dict['state_dict'].items():
        if key not in model.state_dict().keys():
            print('3333', key)
        elif key in filter_list:
            print('4444', key)
        elif 'layer_scale' in key:
            print('5555', key)
        else:
            key = key
            value = value
            convert_dict[key] = value

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
                print('4444', key)
        else:
            print('5555', key)
    print('4444', in_count)

    save_model_name = saved_model_path.split('/')[-1][:-4]
    torch.save(
        convert_dict,
        f'/root/autodl-tmp/pretrained_models/van_convert_from_pytorch_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    )
