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


# filter_list = {
#     'norm1.weight',
#     'norm1.bias',
#     'norm1.running_mean',
#     'norm1.running_var',
#     'norm1.num_batches_tracked',
#     'norm2.weight',
#     'norm2.bias',
#     'norm2.running_mean',
#     'norm2.running_var',
#     'norm2.num_batches_tracked',
#     'norm3.weight',
#     'norm3.bias',
#     'norm3.running_mean',
#     'norm3.running_var',
#     'norm3.num_batches_tracked',
#     'norm4.weight',
#     'norm4.bias',
#     'norm4.running_mean',
#     'norm4.running_var',
#     'norm4.num_batches_tracked',
# }

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
    # for name, weight_shape in model_name_list:
    #     print('1111', name, weight_shape)

    # saved_model_path = '/root/autodl-tmp/van_official_model/van_b4.pth'
    # saved_state_dict = torch.load(saved_model_path,
    #                               map_location=torch.device('cpu'))

    # save_name_list = []
    # for name, weight in saved_state_dict['state_dict'].items():
    #     save_name_list.append([name, weight.shape])

    # print('2222', len(save_name_list))
    # for name, weight_shape in save_name_list:
    #     print('2222', name, weight_shape)

    # convert_dict = {}
    # for key, value in saved_state_dict['state_dict'].items():
    #     if key not in model.state_dict().keys():
    #         print('3333', key)
    #     elif key in filter_list:
    #         print('4444', key)
    #     elif 'layer_scale' in key:
    #         key_planes = value.shape[0]
    #         value = value.reshape(1, key_planes, 1, 1)
    #         convert_dict[key] = value
    #     else:
    #         key = key
    #         value = value
    #         convert_dict[key] = value

    # convert_name_list = []
    # for name, weight in convert_dict.items():
    #     convert_name_list.append([name, weight.shape])

    # print('3333', len(convert_name_list))
    # for name, weight_shape in convert_name_list:
    #     print('3333', name, weight_shape)

    # in_count = 0
    # for key, value in convert_dict.items():
    #     if key in model.state_dict().keys():
    #         if value.shape == model.state_dict()[key].shape:
    #             in_count += 1
    #     else:
    #         print(key)
    # print('4444', in_count)

    # save_model_name = saved_model_path.split('/')[-1][:-4]
    # torch.save(
    #     convert_dict,
    #     f'/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/van_weight_convert_from_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    # )

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
    for name, weight_shape in model_name_list:
        print('1111', name, weight_shape)

    saved_model_path = '/root/autodl-tmp/van_official_model/van_b6_22k.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))
    print(saved_state_dict.keys())
    save_name_list = []
    for name, weight in saved_state_dict['state_dict'].items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    for name, weight_shape in save_name_list:
        print('2222', name, weight_shape)

    convert_dict = {}
    for key, value in saved_state_dict['state_dict'].items():
        if key not in model.state_dict().keys():
            print('3333', key)
        elif key in filter_list:
            print('4444', key)
        elif 'layer_scale' in key:
            key_planes = value.shape[0]
            value = value.reshape(1, key_planes, 1, 1)
            convert_dict[key] = value
        else:
            key = key
            value = value
            convert_dict[key] = value

    convert_name_list = []
    for name, weight in convert_dict.items():
        convert_name_list.append([name, weight.shape])

    print('3333', len(convert_name_list))
    for name, weight_shape in convert_name_list:
        print('3333', name, weight_shape)

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
        f'/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/van_weight_convert_from_official_weights/{save_model_name}_pytorch_official_weight_convert.pth'
    )
