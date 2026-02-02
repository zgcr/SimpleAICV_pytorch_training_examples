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
    'conv1.weight': 'conv1.layer.0.weight',
    'bn1.weight': 'conv1.layer.1.weight',
    'bn1.bias': 'conv1.layer.1.bias',
    'bn1.running_mean': 'conv1.layer.1.running_mean',
    'bn1.running_var': 'conv1.layer.1.running_var',
    'bn1.num_batches_tracked': 'conv1.layer.1.num_batches_tracked',
    'fc.weight': 'fc.weight',
    'fc.bias': 'fc.bias',
}

convert_other_dict = {
    'conv1.weight':
    'conv1.layer.0.weight',
    'bn1.weight':
    'conv1.layer.1.weight',
    'bn1.bias':
    'conv1.layer.1.bias',
    'bn1.running_mean':
    'conv1.layer.1.running_mean',
    'bn1.running_var':
    'conv1.layer.1.running_var',
    'bn1.num_batches_tracked':
    'conv1.layer.1.num_batches_tracked',
    'conv2.weight':
    'conv2.layer.0.weight',
    'bn2.weight':
    'conv2.layer.1.weight',
    'bn2.bias':
    'conv2.layer.1.bias',
    'bn2.running_mean':
    'conv2.layer.1.running_mean',
    'bn2.running_var':
    'conv2.layer.1.running_var',
    'bn2.num_batches_tracked':
    'conv2.layer.1.num_batches_tracked',
    'conv3.weight':
    'conv3.layer.0.weight',
    'bn3.weight':
    'conv3.layer.1.weight',
    'bn3.bias':
    'conv3.layer.1.bias',
    'bn3.running_mean':
    'conv3.layer.1.running_mean',
    'bn3.running_var':
    'conv3.layer.1.running_var',
    'bn3.num_batches_tracked':
    'conv3.layer.1.num_batches_tracked',
    'downsample.0.weight':
    'downsample_conv.layer.0.weight',
    'downsample.1.weight':
    'downsample_conv.layer.1.weight',
    'downsample.1.bias':
    'downsample_conv.layer.1.bias',
    'downsample.1.running_mean':
    'downsample_conv.layer.1.running_mean',
    'downsample.1.running_var':
    'downsample_conv.layer.1.running_var',
    'downsample.1.num_batches_tracked':
    'downsample_conv.layer.1.num_batches_tracked',
}

convert_resnet50_dict = {}

if __name__ == '__main__':
    network = 'resnet152'
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
    # for name, _ in model_name_list:
    #     print(name)

    saved_model_path = '/root/autodl-tmp/pretrained_models/resnet_pytorch_official_weights/resnet152-f82ba261-acc1-82.284.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    save_name_list = []
    for name, weight in saved_state_dict.items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, _ in save_name_list:
    #     print(name)

    convert_dict = {}
    for key, value in saved_state_dict.items():
        if key in convert_common_dict.keys():
            key = convert_common_dict[key]
            convert_dict[key] = value
        else:
            for sub_key in convert_other_dict.keys():
                if sub_key in key:
                    key = key.replace(sub_key, convert_other_dict[sub_key])
                    convert_dict[key] = value
                    break

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
        f'/root/autodl-tmp/pretrained_models/resnet_convert_from_pytorch_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    )
