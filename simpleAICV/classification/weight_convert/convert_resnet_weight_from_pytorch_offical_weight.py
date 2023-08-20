import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from simpleAICV.classification import backbones

import torch
import torch.nn.functional as F


def load_state_dict(saved_model_path,
                    model,
                    excluded_layer_name=(),
                    loading_new_input_size_position_encoding_weight=False):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    loading_new_input_size_position_encoding_weight: default False, for vit net, loading a position encoding layer with new input size, set True
    only load layer parameters which has same layer name and same layer weight shape
    '''
    if not saved_model_path:
        print('No pretrained model file!')
        return

    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))
    not_loaded_save_state_dict = []
    filtered_state_dict = {}
    for name, weight in saved_state_dict.items():
        if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name
        ) and weight.shape == model.state_dict()[name].shape:
            filtered_state_dict[name] = weight
        else:
            not_loaded_save_state_dict.append(name)

    position_encoding_already_loaded = False
    if 'position_encoding' in filtered_state_dict.keys():
        position_encoding_already_loaded = True

    # for vit net, loading a position encoding layer with new input size
    if loading_new_input_size_position_encoding_weight and not position_encoding_already_loaded:
        # assert position_encoding_layer name are unchanged for model and saved_model
        # assert class_token num are unchanged for model and saved_model
        # assert embedding_planes are unchanged for model and saved_model
        model_num_cls_token = model.cls_token.shape[1]
        model_embedding_planes = model.position_encoding.shape[2]
        model_encoding_shape = int(
            (model.position_encoding.shape[1] - model_num_cls_token)**0.5)
        encoding_layer_name, encoding_layer_weight = None, None
        for name, weight in saved_state_dict.items():
            if 'position_encoding' in name:
                encoding_layer_name = name
                encoding_layer_weight = weight
                break
        save_model_encoding_shape = int(
            (encoding_layer_weight.shape[1] - model_num_cls_token)**0.5)

        save_model_cls_token_weight = encoding_layer_weight[:, 0:
                                                            model_num_cls_token, :]
        save_model_position_weight = encoding_layer_weight[:,
                                                           model_num_cls_token:, :]
        save_model_position_weight = save_model_position_weight.reshape(
            -1, save_model_encoding_shape, save_model_encoding_shape,
            model_embedding_planes).permute(0, 3, 1, 2)
        save_model_position_weight = F.interpolate(save_model_position_weight,
                                                   size=(model_encoding_shape,
                                                         model_encoding_shape),
                                                   mode='bicubic',
                                                   align_corners=False)
        save_model_position_weight = save_model_position_weight.permute(
            0, 2, 3, 1).flatten(1, 2)
        model_encoding_layer_weight = torch.cat(
            (save_model_cls_token_weight, save_model_position_weight), dim=1)

        filtered_state_dict[encoding_layer_name] = model_encoding_layer_weight
        not_loaded_save_state_dict.remove('position_encoding')

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return


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

    saved_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/resnet_pytorch_official_weights/resnet152-f82ba261-acc1-82.284.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

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
        f'/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/resnet_convert_from_pytorch_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    )
