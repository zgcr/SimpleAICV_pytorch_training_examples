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


convert_origin_dict = {
    'blocks.0.norm1.weight': 'blocks.0.norm1.weight',
    'blocks.0.norm1.bias': 'blocks.0.norm1.bias',
    'blocks.0.attn.qkv.weight': 'blocks.0.attention.qkv_linear.weight',
    'blocks.0.attn.qkv.bias': 'blocks.0.attention.qkv_linear.bias',
    'blocks.0.attn.proj.weight': 'blocks.0.attention.out_linear.weight',
    'blocks.0.attn.proj.bias': 'blocks.0.attention.out_linear.bias',
    'blocks.0.norm2.weight': 'blocks.0.norm2.weight',
    'blocks.0.norm2.bias': 'blocks.0.norm2.bias',
    'blocks.0.mlp.fc1.weight': 'blocks.0.feed_forward.fc1.weight',
    'blocks.0.mlp.fc1.bias': 'blocks.0.feed_forward.fc1.bias',
    'blocks.0.mlp.fc2.weight': 'blocks.0.feed_forward.fc2.weight',
    'blocks.0.mlp.fc2.bias': 'blocks.0.feed_forward.fc2.bias',
}

convert_dict = {
    'cls_token': 'cls_token',
    'pos_embed': 'position_encoding',
    'patch_embed.proj.weight': 'patch_embedding.conv.weight',
    'patch_embed.proj.bias': 'patch_embedding.conv.bias',
    'norm.weight': 'norm.weight',
    'norm.bias': 'norm.bias',
}

convert_vit_base_dict = {
    'cls_token': 'cls_token',
    'pos_embed': 'position_encoding',
    'patch_embed.proj.weight': 'patch_embedding.conv.weight',
    'patch_embed.proj.bias': 'patch_embedding.conv.bias',
    'blocks.0.norm1.weight': 'blocks.0.norm1.weight',
    'blocks.0.norm1.bias': 'blocks.0.norm1.bias',
    'blocks.0.attn.qkv.weight': 'blocks.0.attention.qkv_linear.weight',
    'blocks.0.attn.qkv.bias': 'blocks.0.attention.qkv_linear.bias',
    'blocks.0.attn.proj.weight': 'blocks.0.attention.out_linear.weight',
    'blocks.0.attn.proj.bias': 'blocks.0.attention.out_linear.bias',
    'blocks.0.norm2.weight': 'blocks.0.norm2.weight',
    'blocks.0.norm2.bias': 'blocks.0.norm2.bias',
    'blocks.0.mlp.fc1.weight': 'blocks.0.feed_forward.fc1.weight',
    'blocks.0.mlp.fc1.bias': 'blocks.0.feed_forward.fc1.bias',
    'blocks.0.mlp.fc2.weight': 'blocks.0.feed_forward.fc2.weight',
    'blocks.0.mlp.fc2.bias': 'blocks.0.feed_forward.fc2.bias',
    'blocks.1.norm1.weight': 'blocks.1.norm1.weight',
    'blocks.1.norm1.bias': 'blocks.1.norm1.bias',
    'blocks.1.attn.qkv.weight': 'blocks.1.attention.qkv_linear.weight',
    'blocks.1.attn.qkv.bias': 'blocks.1.attention.qkv_linear.bias',
    'blocks.1.attn.proj.weight': 'blocks.1.attention.out_linear.weight',
    'blocks.1.attn.proj.bias': 'blocks.1.attention.out_linear.bias',
    'blocks.1.norm2.weight': 'blocks.1.norm2.weight',
    'blocks.1.norm2.bias': 'blocks.1.norm2.bias',
    'blocks.1.mlp.fc1.weight': 'blocks.1.feed_forward.fc1.weight',
    'blocks.1.mlp.fc1.bias': 'blocks.1.feed_forward.fc1.bias',
    'blocks.1.mlp.fc2.weight': 'blocks.1.feed_forward.fc2.weight',
    'blocks.1.mlp.fc2.bias': 'blocks.1.feed_forward.fc2.bias',
    'blocks.2.norm1.weight': 'blocks.2.norm1.weight',
    'blocks.2.norm1.bias': 'blocks.2.norm1.bias',
    'blocks.2.attn.qkv.weight': 'blocks.2.attention.qkv_linear.weight',
    'blocks.2.attn.qkv.bias': 'blocks.2.attention.qkv_linear.bias',
    'blocks.2.attn.proj.weight': 'blocks.2.attention.out_linear.weight',
    'blocks.2.attn.proj.bias': 'blocks.2.attention.out_linear.bias',
    'blocks.2.norm2.weight': 'blocks.2.norm2.weight',
    'blocks.2.norm2.bias': 'blocks.2.norm2.bias',
    'blocks.2.mlp.fc1.weight': 'blocks.2.feed_forward.fc1.weight',
    'blocks.2.mlp.fc1.bias': 'blocks.2.feed_forward.fc1.bias',
    'blocks.2.mlp.fc2.weight': 'blocks.2.feed_forward.fc2.weight',
    'blocks.2.mlp.fc2.bias': 'blocks.2.feed_forward.fc2.bias',
    'blocks.3.norm1.weight': 'blocks.3.norm1.weight',
    'blocks.3.norm1.bias': 'blocks.3.norm1.bias',
    'blocks.3.attn.qkv.weight': 'blocks.3.attention.qkv_linear.weight',
    'blocks.3.attn.qkv.bias': 'blocks.3.attention.qkv_linear.bias',
    'blocks.3.attn.proj.weight': 'blocks.3.attention.out_linear.weight',
    'blocks.3.attn.proj.bias': 'blocks.3.attention.out_linear.bias',
    'blocks.3.norm2.weight': 'blocks.3.norm2.weight',
    'blocks.3.norm2.bias': 'blocks.3.norm2.bias',
    'blocks.3.mlp.fc1.weight': 'blocks.3.feed_forward.fc1.weight',
    'blocks.3.mlp.fc1.bias': 'blocks.3.feed_forward.fc1.bias',
    'blocks.3.mlp.fc2.weight': 'blocks.3.feed_forward.fc2.weight',
    'blocks.3.mlp.fc2.bias': 'blocks.3.feed_forward.fc2.bias',
    'blocks.4.norm1.weight': 'blocks.4.norm1.weight',
    'blocks.4.norm1.bias': 'blocks.4.norm1.bias',
    'blocks.4.attn.qkv.weight': 'blocks.4.attention.qkv_linear.weight',
    'blocks.4.attn.qkv.bias': 'blocks.4.attention.qkv_linear.bias',
    'blocks.4.attn.proj.weight': 'blocks.4.attention.out_linear.weight',
    'blocks.4.attn.proj.bias': 'blocks.4.attention.out_linear.bias',
    'blocks.4.norm2.weight': 'blocks.4.norm2.weight',
    'blocks.4.norm2.bias': 'blocks.4.norm2.bias',
    'blocks.4.mlp.fc1.weight': 'blocks.4.feed_forward.fc1.weight',
    'blocks.4.mlp.fc1.bias': 'blocks.4.feed_forward.fc1.bias',
    'blocks.4.mlp.fc2.weight': 'blocks.4.feed_forward.fc2.weight',
    'blocks.4.mlp.fc2.bias': 'blocks.4.feed_forward.fc2.bias',
    'blocks.5.norm1.weight': 'blocks.5.norm1.weight',
    'blocks.5.norm1.bias': 'blocks.5.norm1.bias',
    'blocks.5.attn.qkv.weight': 'blocks.5.attention.qkv_linear.weight',
    'blocks.5.attn.qkv.bias': 'blocks.5.attention.qkv_linear.bias',
    'blocks.5.attn.proj.weight': 'blocks.5.attention.out_linear.weight',
    'blocks.5.attn.proj.bias': 'blocks.5.attention.out_linear.bias',
    'blocks.5.norm2.weight': 'blocks.5.norm2.weight',
    'blocks.5.norm2.bias': 'blocks.5.norm2.bias',
    'blocks.5.mlp.fc1.weight': 'blocks.5.feed_forward.fc1.weight',
    'blocks.5.mlp.fc1.bias': 'blocks.5.feed_forward.fc1.bias',
    'blocks.5.mlp.fc2.weight': 'blocks.5.feed_forward.fc2.weight',
    'blocks.5.mlp.fc2.bias': 'blocks.5.feed_forward.fc2.bias',
    'blocks.6.norm1.weight': 'blocks.6.norm1.weight',
    'blocks.6.norm1.bias': 'blocks.6.norm1.bias',
    'blocks.6.attn.qkv.weight': 'blocks.6.attention.qkv_linear.weight',
    'blocks.6.attn.qkv.bias': 'blocks.6.attention.qkv_linear.bias',
    'blocks.6.attn.proj.weight': 'blocks.6.attention.out_linear.weight',
    'blocks.6.attn.proj.bias': 'blocks.6.attention.out_linear.bias',
    'blocks.6.norm2.weight': 'blocks.6.norm2.weight',
    'blocks.6.norm2.bias': 'blocks.6.norm2.bias',
    'blocks.6.mlp.fc1.weight': 'blocks.6.feed_forward.fc1.weight',
    'blocks.6.mlp.fc1.bias': 'blocks.6.feed_forward.fc1.bias',
    'blocks.6.mlp.fc2.weight': 'blocks.6.feed_forward.fc2.weight',
    'blocks.6.mlp.fc2.bias': 'blocks.6.feed_forward.fc2.bias',
    'blocks.7.norm1.weight': 'blocks.7.norm1.weight',
    'blocks.7.norm1.bias': 'blocks.7.norm1.bias',
    'blocks.7.attn.qkv.weight': 'blocks.7.attention.qkv_linear.weight',
    'blocks.7.attn.qkv.bias': 'blocks.7.attention.qkv_linear.bias',
    'blocks.7.attn.proj.weight': 'blocks.7.attention.out_linear.weight',
    'blocks.7.attn.proj.bias': 'blocks.7.attention.out_linear.bias',
    'blocks.7.norm2.weight': 'blocks.7.norm2.weight',
    'blocks.7.norm2.bias': 'blocks.7.norm2.bias',
    'blocks.7.mlp.fc1.weight': 'blocks.7.feed_forward.fc1.weight',
    'blocks.7.mlp.fc1.bias': 'blocks.7.feed_forward.fc1.bias',
    'blocks.7.mlp.fc2.weight': 'blocks.7.feed_forward.fc2.weight',
    'blocks.7.mlp.fc2.bias': 'blocks.7.feed_forward.fc2.bias',
    'blocks.8.norm1.weight': 'blocks.8.norm1.weight',
    'blocks.8.norm1.bias': 'blocks.8.norm1.bias',
    'blocks.8.attn.qkv.weight': 'blocks.8.attention.qkv_linear.weight',
    'blocks.8.attn.qkv.bias': 'blocks.8.attention.qkv_linear.bias',
    'blocks.8.attn.proj.weight': 'blocks.8.attention.out_linear.weight',
    'blocks.8.attn.proj.bias': 'blocks.8.attention.out_linear.bias',
    'blocks.8.norm2.weight': 'blocks.8.norm2.weight',
    'blocks.8.norm2.bias': 'blocks.8.norm2.bias',
    'blocks.8.mlp.fc1.weight': 'blocks.8.feed_forward.fc1.weight',
    'blocks.8.mlp.fc1.bias': 'blocks.8.feed_forward.fc1.bias',
    'blocks.8.mlp.fc2.weight': 'blocks.8.feed_forward.fc2.weight',
    'blocks.8.mlp.fc2.bias': 'blocks.8.feed_forward.fc2.bias',
    'blocks.9.norm1.weight': 'blocks.9.norm1.weight',
    'blocks.9.norm1.bias': 'blocks.9.norm1.bias',
    'blocks.9.attn.qkv.weight': 'blocks.9.attention.qkv_linear.weight',
    'blocks.9.attn.qkv.bias': 'blocks.9.attention.qkv_linear.bias',
    'blocks.9.attn.proj.weight': 'blocks.9.attention.out_linear.weight',
    'blocks.9.attn.proj.bias': 'blocks.9.attention.out_linear.bias',
    'blocks.9.norm2.weight': 'blocks.9.norm2.weight',
    'blocks.9.norm2.bias': 'blocks.9.norm2.bias',
    'blocks.9.mlp.fc1.weight': 'blocks.9.feed_forward.fc1.weight',
    'blocks.9.mlp.fc1.bias': 'blocks.9.feed_forward.fc1.bias',
    'blocks.9.mlp.fc2.weight': 'blocks.9.feed_forward.fc2.weight',
    'blocks.9.mlp.fc2.bias': 'blocks.9.feed_forward.fc2.bias',
    'blocks.10.norm1.weight': 'blocks.10.norm1.weight',
    'blocks.10.norm1.bias': 'blocks.10.norm1.bias',
    'blocks.10.attn.qkv.weight': 'blocks.10.attention.qkv_linear.weight',
    'blocks.10.attn.qkv.bias': 'blocks.10.attention.qkv_linear.bias',
    'blocks.10.attn.proj.weight': 'blocks.10.attention.out_linear.weight',
    'blocks.10.attn.proj.bias': 'blocks.10.attention.out_linear.bias',
    'blocks.10.norm2.weight': 'blocks.10.norm2.weight',
    'blocks.10.norm2.bias': 'blocks.10.norm2.bias',
    'blocks.10.mlp.fc1.weight': 'blocks.10.feed_forward.fc1.weight',
    'blocks.10.mlp.fc1.bias': 'blocks.10.feed_forward.fc1.bias',
    'blocks.10.mlp.fc2.weight': 'blocks.10.feed_forward.fc2.weight',
    'blocks.10.mlp.fc2.bias': 'blocks.10.feed_forward.fc2.bias',
    'blocks.11.norm1.weight': 'blocks.11.norm1.weight',
    'blocks.11.norm1.bias': 'blocks.11.norm1.bias',
    'blocks.11.attn.qkv.weight': 'blocks.11.attention.qkv_linear.weight',
    'blocks.11.attn.qkv.bias': 'blocks.11.attention.qkv_linear.bias',
    'blocks.11.attn.proj.weight': 'blocks.11.attention.out_linear.weight',
    'blocks.11.attn.proj.bias': 'blocks.11.attention.out_linear.bias',
    'blocks.11.norm2.weight': 'blocks.11.norm2.weight',
    'blocks.11.norm2.bias': 'blocks.11.norm2.bias',
    'blocks.11.mlp.fc1.weight': 'blocks.11.feed_forward.fc1.weight',
    'blocks.11.mlp.fc1.bias': 'blocks.11.feed_forward.fc1.bias',
    'blocks.11.mlp.fc2.weight': 'blocks.11.feed_forward.fc2.weight',
    'blocks.11.mlp.fc2.bias': 'blocks.11.feed_forward.fc2.bias',
    'norm.weight': 'norm.weight',
    'norm.bias': 'norm.bias',
}

convert_vit_large_dict = {
    'cls_token': 'cls_token',
    'pos_embed': 'position_encoding',
    'patch_embed.proj.weight': 'patch_embedding.conv.weight',
    'patch_embed.proj.bias': 'patch_embedding.conv.bias',
    'norm.weight': 'norm.weight',
    'norm.bias': 'norm.bias',
    'blocks.0.norm1.weight': 'blocks.0.norm1.weight',
    'blocks.0.norm1.bias': 'blocks.0.norm1.bias',
    'blocks.0.attn.qkv.weight': 'blocks.0.attention.qkv_linear.weight',
    'blocks.0.attn.qkv.bias': 'blocks.0.attention.qkv_linear.bias',
    'blocks.0.attn.proj.weight': 'blocks.0.attention.out_linear.weight',
    'blocks.0.attn.proj.bias': 'blocks.0.attention.out_linear.bias',
    'blocks.0.norm2.weight': 'blocks.0.norm2.weight',
    'blocks.0.norm2.bias': 'blocks.0.norm2.bias',
    'blocks.0.mlp.fc1.weight': 'blocks.0.feed_forward.fc1.weight',
    'blocks.0.mlp.fc1.bias': 'blocks.0.feed_forward.fc1.bias',
    'blocks.0.mlp.fc2.weight': 'blocks.0.feed_forward.fc2.weight',
    'blocks.0.mlp.fc2.bias': 'blocks.0.feed_forward.fc2.bias',
    'blocks.1.norm1.weight': 'blocks.1.norm1.weight',
    'blocks.1.norm1.bias': 'blocks.1.norm1.bias',
    'blocks.1.attn.qkv.weight': 'blocks.1.attention.qkv_linear.weight',
    'blocks.1.attn.qkv.bias': 'blocks.1.attention.qkv_linear.bias',
    'blocks.1.attn.proj.weight': 'blocks.1.attention.out_linear.weight',
    'blocks.1.attn.proj.bias': 'blocks.1.attention.out_linear.bias',
    'blocks.1.norm2.weight': 'blocks.1.norm2.weight',
    'blocks.1.norm2.bias': 'blocks.1.norm2.bias',
    'blocks.1.mlp.fc1.weight': 'blocks.1.feed_forward.fc1.weight',
    'blocks.1.mlp.fc1.bias': 'blocks.1.feed_forward.fc1.bias',
    'blocks.1.mlp.fc2.weight': 'blocks.1.feed_forward.fc2.weight',
    'blocks.1.mlp.fc2.bias': 'blocks.1.feed_forward.fc2.bias',
    'blocks.2.norm1.weight': 'blocks.2.norm1.weight',
    'blocks.2.norm1.bias': 'blocks.2.norm1.bias',
    'blocks.2.attn.qkv.weight': 'blocks.2.attention.qkv_linear.weight',
    'blocks.2.attn.qkv.bias': 'blocks.2.attention.qkv_linear.bias',
    'blocks.2.attn.proj.weight': 'blocks.2.attention.out_linear.weight',
    'blocks.2.attn.proj.bias': 'blocks.2.attention.out_linear.bias',
    'blocks.2.norm2.weight': 'blocks.2.norm2.weight',
    'blocks.2.norm2.bias': 'blocks.2.norm2.bias',
    'blocks.2.mlp.fc1.weight': 'blocks.2.feed_forward.fc1.weight',
    'blocks.2.mlp.fc1.bias': 'blocks.2.feed_forward.fc1.bias',
    'blocks.2.mlp.fc2.weight': 'blocks.2.feed_forward.fc2.weight',
    'blocks.2.mlp.fc2.bias': 'blocks.2.feed_forward.fc2.bias',
    'blocks.3.norm1.weight': 'blocks.3.norm1.weight',
    'blocks.3.norm1.bias': 'blocks.3.norm1.bias',
    'blocks.3.attn.qkv.weight': 'blocks.3.attention.qkv_linear.weight',
    'blocks.3.attn.qkv.bias': 'blocks.3.attention.qkv_linear.bias',
    'blocks.3.attn.proj.weight': 'blocks.3.attention.out_linear.weight',
    'blocks.3.attn.proj.bias': 'blocks.3.attention.out_linear.bias',
    'blocks.3.norm2.weight': 'blocks.3.norm2.weight',
    'blocks.3.norm2.bias': 'blocks.3.norm2.bias',
    'blocks.3.mlp.fc1.weight': 'blocks.3.feed_forward.fc1.weight',
    'blocks.3.mlp.fc1.bias': 'blocks.3.feed_forward.fc1.bias',
    'blocks.3.mlp.fc2.weight': 'blocks.3.feed_forward.fc2.weight',
    'blocks.3.mlp.fc2.bias': 'blocks.3.feed_forward.fc2.bias',
    'blocks.4.norm1.weight': 'blocks.4.norm1.weight',
    'blocks.4.norm1.bias': 'blocks.4.norm1.bias',
    'blocks.4.attn.qkv.weight': 'blocks.4.attention.qkv_linear.weight',
    'blocks.4.attn.qkv.bias': 'blocks.4.attention.qkv_linear.bias',
    'blocks.4.attn.proj.weight': 'blocks.4.attention.out_linear.weight',
    'blocks.4.attn.proj.bias': 'blocks.4.attention.out_linear.bias',
    'blocks.4.norm2.weight': 'blocks.4.norm2.weight',
    'blocks.4.norm2.bias': 'blocks.4.norm2.bias',
    'blocks.4.mlp.fc1.weight': 'blocks.4.feed_forward.fc1.weight',
    'blocks.4.mlp.fc1.bias': 'blocks.4.feed_forward.fc1.bias',
    'blocks.4.mlp.fc2.weight': 'blocks.4.feed_forward.fc2.weight',
    'blocks.4.mlp.fc2.bias': 'blocks.4.feed_forward.fc2.bias',
    'blocks.5.norm1.weight': 'blocks.5.norm1.weight',
    'blocks.5.norm1.bias': 'blocks.5.norm1.bias',
    'blocks.5.attn.qkv.weight': 'blocks.5.attention.qkv_linear.weight',
    'blocks.5.attn.qkv.bias': 'blocks.5.attention.qkv_linear.bias',
    'blocks.5.attn.proj.weight': 'blocks.5.attention.out_linear.weight',
    'blocks.5.attn.proj.bias': 'blocks.5.attention.out_linear.bias',
    'blocks.5.norm2.weight': 'blocks.5.norm2.weight',
    'blocks.5.norm2.bias': 'blocks.5.norm2.bias',
    'blocks.5.mlp.fc1.weight': 'blocks.5.feed_forward.fc1.weight',
    'blocks.5.mlp.fc1.bias': 'blocks.5.feed_forward.fc1.bias',
    'blocks.5.mlp.fc2.weight': 'blocks.5.feed_forward.fc2.weight',
    'blocks.5.mlp.fc2.bias': 'blocks.5.feed_forward.fc2.bias',
    'blocks.6.norm1.weight': 'blocks.6.norm1.weight',
    'blocks.6.norm1.bias': 'blocks.6.norm1.bias',
    'blocks.6.attn.qkv.weight': 'blocks.6.attention.qkv_linear.weight',
    'blocks.6.attn.qkv.bias': 'blocks.6.attention.qkv_linear.bias',
    'blocks.6.attn.proj.weight': 'blocks.6.attention.out_linear.weight',
    'blocks.6.attn.proj.bias': 'blocks.6.attention.out_linear.bias',
    'blocks.6.norm2.weight': 'blocks.6.norm2.weight',
    'blocks.6.norm2.bias': 'blocks.6.norm2.bias',
    'blocks.6.mlp.fc1.weight': 'blocks.6.feed_forward.fc1.weight',
    'blocks.6.mlp.fc1.bias': 'blocks.6.feed_forward.fc1.bias',
    'blocks.6.mlp.fc2.weight': 'blocks.6.feed_forward.fc2.weight',
    'blocks.6.mlp.fc2.bias': 'blocks.6.feed_forward.fc2.bias',
    'blocks.7.norm1.weight': 'blocks.7.norm1.weight',
    'blocks.7.norm1.bias': 'blocks.7.norm1.bias',
    'blocks.7.attn.qkv.weight': 'blocks.7.attention.qkv_linear.weight',
    'blocks.7.attn.qkv.bias': 'blocks.7.attention.qkv_linear.bias',
    'blocks.7.attn.proj.weight': 'blocks.7.attention.out_linear.weight',
    'blocks.7.attn.proj.bias': 'blocks.7.attention.out_linear.bias',
    'blocks.7.norm2.weight': 'blocks.7.norm2.weight',
    'blocks.7.norm2.bias': 'blocks.7.norm2.bias',
    'blocks.7.mlp.fc1.weight': 'blocks.7.feed_forward.fc1.weight',
    'blocks.7.mlp.fc1.bias': 'blocks.7.feed_forward.fc1.bias',
    'blocks.7.mlp.fc2.weight': 'blocks.7.feed_forward.fc2.weight',
    'blocks.7.mlp.fc2.bias': 'blocks.7.feed_forward.fc2.bias',
    'blocks.8.norm1.weight': 'blocks.8.norm1.weight',
    'blocks.8.norm1.bias': 'blocks.8.norm1.bias',
    'blocks.8.attn.qkv.weight': 'blocks.8.attention.qkv_linear.weight',
    'blocks.8.attn.qkv.bias': 'blocks.8.attention.qkv_linear.bias',
    'blocks.8.attn.proj.weight': 'blocks.8.attention.out_linear.weight',
    'blocks.8.attn.proj.bias': 'blocks.8.attention.out_linear.bias',
    'blocks.8.norm2.weight': 'blocks.8.norm2.weight',
    'blocks.8.norm2.bias': 'blocks.8.norm2.bias',
    'blocks.8.mlp.fc1.weight': 'blocks.8.feed_forward.fc1.weight',
    'blocks.8.mlp.fc1.bias': 'blocks.8.feed_forward.fc1.bias',
    'blocks.8.mlp.fc2.weight': 'blocks.8.feed_forward.fc2.weight',
    'blocks.8.mlp.fc2.bias': 'blocks.8.feed_forward.fc2.bias',
    'blocks.9.norm1.weight': 'blocks.9.norm1.weight',
    'blocks.9.norm1.bias': 'blocks.9.norm1.bias',
    'blocks.9.attn.qkv.weight': 'blocks.9.attention.qkv_linear.weight',
    'blocks.9.attn.qkv.bias': 'blocks.9.attention.qkv_linear.bias',
    'blocks.9.attn.proj.weight': 'blocks.9.attention.out_linear.weight',
    'blocks.9.attn.proj.bias': 'blocks.9.attention.out_linear.bias',
    'blocks.9.norm2.weight': 'blocks.9.norm2.weight',
    'blocks.9.norm2.bias': 'blocks.9.norm2.bias',
    'blocks.9.mlp.fc1.weight': 'blocks.9.feed_forward.fc1.weight',
    'blocks.9.mlp.fc1.bias': 'blocks.9.feed_forward.fc1.bias',
    'blocks.9.mlp.fc2.weight': 'blocks.9.feed_forward.fc2.weight',
    'blocks.9.mlp.fc2.bias': 'blocks.9.feed_forward.fc2.bias',
    'blocks.10.norm1.weight': 'blocks.10.norm1.weight',
    'blocks.10.norm1.bias': 'blocks.10.norm1.bias',
    'blocks.10.attn.qkv.weight': 'blocks.10.attention.qkv_linear.weight',
    'blocks.10.attn.qkv.bias': 'blocks.10.attention.qkv_linear.bias',
    'blocks.10.attn.proj.weight': 'blocks.10.attention.out_linear.weight',
    'blocks.10.attn.proj.bias': 'blocks.10.attention.out_linear.bias',
    'blocks.10.norm2.weight': 'blocks.10.norm2.weight',
    'blocks.10.norm2.bias': 'blocks.10.norm2.bias',
    'blocks.10.mlp.fc1.weight': 'blocks.10.feed_forward.fc1.weight',
    'blocks.10.mlp.fc1.bias': 'blocks.10.feed_forward.fc1.bias',
    'blocks.10.mlp.fc2.weight': 'blocks.10.feed_forward.fc2.weight',
    'blocks.10.mlp.fc2.bias': 'blocks.10.feed_forward.fc2.bias',
    'blocks.11.norm1.weight': 'blocks.11.norm1.weight',
    'blocks.11.norm1.bias': 'blocks.11.norm1.bias',
    'blocks.11.attn.qkv.weight': 'blocks.11.attention.qkv_linear.weight',
    'blocks.11.attn.qkv.bias': 'blocks.11.attention.qkv_linear.bias',
    'blocks.11.attn.proj.weight': 'blocks.11.attention.out_linear.weight',
    'blocks.11.attn.proj.bias': 'blocks.11.attention.out_linear.bias',
    'blocks.11.norm2.weight': 'blocks.11.norm2.weight',
    'blocks.11.norm2.bias': 'blocks.11.norm2.bias',
    'blocks.11.mlp.fc1.weight': 'blocks.11.feed_forward.fc1.weight',
    'blocks.11.mlp.fc1.bias': 'blocks.11.feed_forward.fc1.bias',
    'blocks.11.mlp.fc2.weight': 'blocks.11.feed_forward.fc2.weight',
    'blocks.11.mlp.fc2.bias': 'blocks.11.feed_forward.fc2.bias',
    'blocks.12.norm1.weight': 'blocks.12.norm1.weight',
    'blocks.12.norm1.bias': 'blocks.12.norm1.bias',
    'blocks.12.attn.qkv.weight': 'blocks.12.attention.qkv_linear.weight',
    'blocks.12.attn.qkv.bias': 'blocks.12.attention.qkv_linear.bias',
    'blocks.12.attn.proj.weight': 'blocks.12.attention.out_linear.weight',
    'blocks.12.attn.proj.bias': 'blocks.12.attention.out_linear.bias',
    'blocks.12.norm2.weight': 'blocks.12.norm2.weight',
    'blocks.12.norm2.bias': 'blocks.12.norm2.bias',
    'blocks.12.mlp.fc1.weight': 'blocks.12.feed_forward.fc1.weight',
    'blocks.12.mlp.fc1.bias': 'blocks.12.feed_forward.fc1.bias',
    'blocks.12.mlp.fc2.weight': 'blocks.12.feed_forward.fc2.weight',
    'blocks.12.mlp.fc2.bias': 'blocks.12.feed_forward.fc2.bias',
    'blocks.13.norm1.weight': 'blocks.13.norm1.weight',
    'blocks.13.norm1.bias': 'blocks.13.norm1.bias',
    'blocks.13.attn.qkv.weight': 'blocks.13.attention.qkv_linear.weight',
    'blocks.13.attn.qkv.bias': 'blocks.13.attention.qkv_linear.bias',
    'blocks.13.attn.proj.weight': 'blocks.13.attention.out_linear.weight',
    'blocks.13.attn.proj.bias': 'blocks.13.attention.out_linear.bias',
    'blocks.13.norm2.weight': 'blocks.13.norm2.weight',
    'blocks.13.norm2.bias': 'blocks.13.norm2.bias',
    'blocks.13.mlp.fc1.weight': 'blocks.13.feed_forward.fc1.weight',
    'blocks.13.mlp.fc1.bias': 'blocks.13.feed_forward.fc1.bias',
    'blocks.13.mlp.fc2.weight': 'blocks.13.feed_forward.fc2.weight',
    'blocks.13.mlp.fc2.bias': 'blocks.13.feed_forward.fc2.bias',
    'blocks.14.norm1.weight': 'blocks.14.norm1.weight',
    'blocks.14.norm1.bias': 'blocks.14.norm1.bias',
    'blocks.14.attn.qkv.weight': 'blocks.14.attention.qkv_linear.weight',
    'blocks.14.attn.qkv.bias': 'blocks.14.attention.qkv_linear.bias',
    'blocks.14.attn.proj.weight': 'blocks.14.attention.out_linear.weight',
    'blocks.14.attn.proj.bias': 'blocks.14.attention.out_linear.bias',
    'blocks.14.norm2.weight': 'blocks.14.norm2.weight',
    'blocks.14.norm2.bias': 'blocks.14.norm2.bias',
    'blocks.14.mlp.fc1.weight': 'blocks.14.feed_forward.fc1.weight',
    'blocks.14.mlp.fc1.bias': 'blocks.14.feed_forward.fc1.bias',
    'blocks.14.mlp.fc2.weight': 'blocks.14.feed_forward.fc2.weight',
    'blocks.14.mlp.fc2.bias': 'blocks.14.feed_forward.fc2.bias',
    'blocks.15.norm1.weight': 'blocks.15.norm1.weight',
    'blocks.15.norm1.bias': 'blocks.15.norm1.bias',
    'blocks.15.attn.qkv.weight': 'blocks.15.attention.qkv_linear.weight',
    'blocks.15.attn.qkv.bias': 'blocks.15.attention.qkv_linear.bias',
    'blocks.15.attn.proj.weight': 'blocks.15.attention.out_linear.weight',
    'blocks.15.attn.proj.bias': 'blocks.15.attention.out_linear.bias',
    'blocks.15.norm2.weight': 'blocks.15.norm2.weight',
    'blocks.15.norm2.bias': 'blocks.15.norm2.bias',
    'blocks.15.mlp.fc1.weight': 'blocks.15.feed_forward.fc1.weight',
    'blocks.15.mlp.fc1.bias': 'blocks.15.feed_forward.fc1.bias',
    'blocks.15.mlp.fc2.weight': 'blocks.15.feed_forward.fc2.weight',
    'blocks.15.mlp.fc2.bias': 'blocks.15.feed_forward.fc2.bias',
    'blocks.16.norm1.weight': 'blocks.16.norm1.weight',
    'blocks.16.norm1.bias': 'blocks.16.norm1.bias',
    'blocks.16.attn.qkv.weight': 'blocks.16.attention.qkv_linear.weight',
    'blocks.16.attn.qkv.bias': 'blocks.16.attention.qkv_linear.bias',
    'blocks.16.attn.proj.weight': 'blocks.16.attention.out_linear.weight',
    'blocks.16.attn.proj.bias': 'blocks.16.attention.out_linear.bias',
    'blocks.16.norm2.weight': 'blocks.16.norm2.weight',
    'blocks.16.norm2.bias': 'blocks.16.norm2.bias',
    'blocks.16.mlp.fc1.weight': 'blocks.16.feed_forward.fc1.weight',
    'blocks.16.mlp.fc1.bias': 'blocks.16.feed_forward.fc1.bias',
    'blocks.16.mlp.fc2.weight': 'blocks.16.feed_forward.fc2.weight',
    'blocks.16.mlp.fc2.bias': 'blocks.16.feed_forward.fc2.bias',
    'blocks.17.norm1.weight': 'blocks.17.norm1.weight',
    'blocks.17.norm1.bias': 'blocks.17.norm1.bias',
    'blocks.17.attn.qkv.weight': 'blocks.17.attention.qkv_linear.weight',
    'blocks.17.attn.qkv.bias': 'blocks.17.attention.qkv_linear.bias',
    'blocks.17.attn.proj.weight': 'blocks.17.attention.out_linear.weight',
    'blocks.17.attn.proj.bias': 'blocks.17.attention.out_linear.bias',
    'blocks.17.norm2.weight': 'blocks.17.norm2.weight',
    'blocks.17.norm2.bias': 'blocks.17.norm2.bias',
    'blocks.17.mlp.fc1.weight': 'blocks.17.feed_forward.fc1.weight',
    'blocks.17.mlp.fc1.bias': 'blocks.17.feed_forward.fc1.bias',
    'blocks.17.mlp.fc2.weight': 'blocks.17.feed_forward.fc2.weight',
    'blocks.17.mlp.fc2.bias': 'blocks.17.feed_forward.fc2.bias',
    'blocks.18.norm1.weight': 'blocks.18.norm1.weight',
    'blocks.18.norm1.bias': 'blocks.18.norm1.bias',
    'blocks.18.attn.qkv.weight': 'blocks.18.attention.qkv_linear.weight',
    'blocks.18.attn.qkv.bias': 'blocks.18.attention.qkv_linear.bias',
    'blocks.18.attn.proj.weight': 'blocks.18.attention.out_linear.weight',
    'blocks.18.attn.proj.bias': 'blocks.18.attention.out_linear.bias',
    'blocks.18.norm2.weight': 'blocks.18.norm2.weight',
    'blocks.18.norm2.bias': 'blocks.18.norm2.bias',
    'blocks.18.mlp.fc1.weight': 'blocks.18.feed_forward.fc1.weight',
    'blocks.18.mlp.fc1.bias': 'blocks.18.feed_forward.fc1.bias',
    'blocks.18.mlp.fc2.weight': 'blocks.18.feed_forward.fc2.weight',
    'blocks.18.mlp.fc2.bias': 'blocks.18.feed_forward.fc2.bias',
    'blocks.19.norm1.weight': 'blocks.19.norm1.weight',
    'blocks.19.norm1.bias': 'blocks.19.norm1.bias',
    'blocks.19.attn.qkv.weight': 'blocks.19.attention.qkv_linear.weight',
    'blocks.19.attn.qkv.bias': 'blocks.19.attention.qkv_linear.bias',
    'blocks.19.attn.proj.weight': 'blocks.19.attention.out_linear.weight',
    'blocks.19.attn.proj.bias': 'blocks.19.attention.out_linear.bias',
    'blocks.19.norm2.weight': 'blocks.19.norm2.weight',
    'blocks.19.norm2.bias': 'blocks.19.norm2.bias',
    'blocks.19.mlp.fc1.weight': 'blocks.19.feed_forward.fc1.weight',
    'blocks.19.mlp.fc1.bias': 'blocks.19.feed_forward.fc1.bias',
    'blocks.19.mlp.fc2.weight': 'blocks.19.feed_forward.fc2.weight',
    'blocks.19.mlp.fc2.bias': 'blocks.19.feed_forward.fc2.bias',
    'blocks.20.norm1.weight': 'blocks.20.norm1.weight',
    'blocks.20.norm1.bias': 'blocks.20.norm1.bias',
    'blocks.20.attn.qkv.weight': 'blocks.20.attention.qkv_linear.weight',
    'blocks.20.attn.qkv.bias': 'blocks.20.attention.qkv_linear.bias',
    'blocks.20.attn.proj.weight': 'blocks.20.attention.out_linear.weight',
    'blocks.20.attn.proj.bias': 'blocks.20.attention.out_linear.bias',
    'blocks.20.norm2.weight': 'blocks.20.norm2.weight',
    'blocks.20.norm2.bias': 'blocks.20.norm2.bias',
    'blocks.20.mlp.fc1.weight': 'blocks.20.feed_forward.fc1.weight',
    'blocks.20.mlp.fc1.bias': 'blocks.20.feed_forward.fc1.bias',
    'blocks.20.mlp.fc2.weight': 'blocks.20.feed_forward.fc2.weight',
    'blocks.20.mlp.fc2.bias': 'blocks.20.feed_forward.fc2.bias',
    'blocks.21.norm1.weight': 'blocks.21.norm1.weight',
    'blocks.21.norm1.bias': 'blocks.21.norm1.bias',
    'blocks.21.attn.qkv.weight': 'blocks.21.attention.qkv_linear.weight',
    'blocks.21.attn.qkv.bias': 'blocks.21.attention.qkv_linear.bias',
    'blocks.21.attn.proj.weight': 'blocks.21.attention.out_linear.weight',
    'blocks.21.attn.proj.bias': 'blocks.21.attention.out_linear.bias',
    'blocks.21.norm2.weight': 'blocks.21.norm2.weight',
    'blocks.21.norm2.bias': 'blocks.21.norm2.bias',
    'blocks.21.mlp.fc1.weight': 'blocks.21.feed_forward.fc1.weight',
    'blocks.21.mlp.fc1.bias': 'blocks.21.feed_forward.fc1.bias',
    'blocks.21.mlp.fc2.weight': 'blocks.21.feed_forward.fc2.weight',
    'blocks.21.mlp.fc2.bias': 'blocks.21.feed_forward.fc2.bias',
    'blocks.22.norm1.weight': 'blocks.22.norm1.weight',
    'blocks.22.norm1.bias': 'blocks.22.norm1.bias',
    'blocks.22.attn.qkv.weight': 'blocks.22.attention.qkv_linear.weight',
    'blocks.22.attn.qkv.bias': 'blocks.22.attention.qkv_linear.bias',
    'blocks.22.attn.proj.weight': 'blocks.22.attention.out_linear.weight',
    'blocks.22.attn.proj.bias': 'blocks.22.attention.out_linear.bias',
    'blocks.22.norm2.weight': 'blocks.22.norm2.weight',
    'blocks.22.norm2.bias': 'blocks.22.norm2.bias',
    'blocks.22.mlp.fc1.weight': 'blocks.22.feed_forward.fc1.weight',
    'blocks.22.mlp.fc1.bias': 'blocks.22.feed_forward.fc1.bias',
    'blocks.22.mlp.fc2.weight': 'blocks.22.feed_forward.fc2.weight',
    'blocks.22.mlp.fc2.bias': 'blocks.22.feed_forward.fc2.bias',
    'blocks.23.norm1.weight': 'blocks.23.norm1.weight',
    'blocks.23.norm1.bias': 'blocks.23.norm1.bias',
    'blocks.23.attn.qkv.weight': 'blocks.23.attention.qkv_linear.weight',
    'blocks.23.attn.qkv.bias': 'blocks.23.attention.qkv_linear.bias',
    'blocks.23.attn.proj.weight': 'blocks.23.attention.out_linear.weight',
    'blocks.23.attn.proj.bias': 'blocks.23.attention.out_linear.bias',
    'blocks.23.norm2.weight': 'blocks.23.norm2.weight',
    'blocks.23.norm2.bias': 'blocks.23.norm2.bias',
    'blocks.23.mlp.fc1.weight': 'blocks.23.feed_forward.fc1.weight',
    'blocks.23.mlp.fc1.bias': 'blocks.23.feed_forward.fc1.bias',
    'blocks.23.mlp.fc2.weight': 'blocks.23.feed_forward.fc2.weight',
    'blocks.23.mlp.fc2.bias': 'blocks.23.feed_forward.fc2.bias'
}

convert_vit_huge_dict = {
    'cls_token': 'cls_token',
    'pos_embed': 'position_encoding',
    'patch_embed.proj.weight': 'patch_embedding.conv.weight',
    'patch_embed.proj.bias': 'patch_embedding.conv.bias',
    'norm.weight': 'norm.weight',
    'norm.bias': 'norm.bias',
    'blocks.0.norm1.weight': 'blocks.0.norm1.weight',
    'blocks.0.norm1.bias': 'blocks.0.norm1.bias',
    'blocks.0.attn.qkv.weight': 'blocks.0.attention.qkv_linear.weight',
    'blocks.0.attn.qkv.bias': 'blocks.0.attention.qkv_linear.bias',
    'blocks.0.attn.proj.weight': 'blocks.0.attention.out_linear.weight',
    'blocks.0.attn.proj.bias': 'blocks.0.attention.out_linear.bias',
    'blocks.0.norm2.weight': 'blocks.0.norm2.weight',
    'blocks.0.norm2.bias': 'blocks.0.norm2.bias',
    'blocks.0.mlp.fc1.weight': 'blocks.0.feed_forward.fc1.weight',
    'blocks.0.mlp.fc1.bias': 'blocks.0.feed_forward.fc1.bias',
    'blocks.0.mlp.fc2.weight': 'blocks.0.feed_forward.fc2.weight',
    'blocks.0.mlp.fc2.bias': 'blocks.0.feed_forward.fc2.bias',
    'blocks.1.norm1.weight': 'blocks.1.norm1.weight',
    'blocks.1.norm1.bias': 'blocks.1.norm1.bias',
    'blocks.1.attn.qkv.weight': 'blocks.1.attention.qkv_linear.weight',
    'blocks.1.attn.qkv.bias': 'blocks.1.attention.qkv_linear.bias',
    'blocks.1.attn.proj.weight': 'blocks.1.attention.out_linear.weight',
    'blocks.1.attn.proj.bias': 'blocks.1.attention.out_linear.bias',
    'blocks.1.norm2.weight': 'blocks.1.norm2.weight',
    'blocks.1.norm2.bias': 'blocks.1.norm2.bias',
    'blocks.1.mlp.fc1.weight': 'blocks.1.feed_forward.fc1.weight',
    'blocks.1.mlp.fc1.bias': 'blocks.1.feed_forward.fc1.bias',
    'blocks.1.mlp.fc2.weight': 'blocks.1.feed_forward.fc2.weight',
    'blocks.1.mlp.fc2.bias': 'blocks.1.feed_forward.fc2.bias',
    'blocks.2.norm1.weight': 'blocks.2.norm1.weight',
    'blocks.2.norm1.bias': 'blocks.2.norm1.bias',
    'blocks.2.attn.qkv.weight': 'blocks.2.attention.qkv_linear.weight',
    'blocks.2.attn.qkv.bias': 'blocks.2.attention.qkv_linear.bias',
    'blocks.2.attn.proj.weight': 'blocks.2.attention.out_linear.weight',
    'blocks.2.attn.proj.bias': 'blocks.2.attention.out_linear.bias',
    'blocks.2.norm2.weight': 'blocks.2.norm2.weight',
    'blocks.2.norm2.bias': 'blocks.2.norm2.bias',
    'blocks.2.mlp.fc1.weight': 'blocks.2.feed_forward.fc1.weight',
    'blocks.2.mlp.fc1.bias': 'blocks.2.feed_forward.fc1.bias',
    'blocks.2.mlp.fc2.weight': 'blocks.2.feed_forward.fc2.weight',
    'blocks.2.mlp.fc2.bias': 'blocks.2.feed_forward.fc2.bias',
    'blocks.3.norm1.weight': 'blocks.3.norm1.weight',
    'blocks.3.norm1.bias': 'blocks.3.norm1.bias',
    'blocks.3.attn.qkv.weight': 'blocks.3.attention.qkv_linear.weight',
    'blocks.3.attn.qkv.bias': 'blocks.3.attention.qkv_linear.bias',
    'blocks.3.attn.proj.weight': 'blocks.3.attention.out_linear.weight',
    'blocks.3.attn.proj.bias': 'blocks.3.attention.out_linear.bias',
    'blocks.3.norm2.weight': 'blocks.3.norm2.weight',
    'blocks.3.norm2.bias': 'blocks.3.norm2.bias',
    'blocks.3.mlp.fc1.weight': 'blocks.3.feed_forward.fc1.weight',
    'blocks.3.mlp.fc1.bias': 'blocks.3.feed_forward.fc1.bias',
    'blocks.3.mlp.fc2.weight': 'blocks.3.feed_forward.fc2.weight',
    'blocks.3.mlp.fc2.bias': 'blocks.3.feed_forward.fc2.bias',
    'blocks.4.norm1.weight': 'blocks.4.norm1.weight',
    'blocks.4.norm1.bias': 'blocks.4.norm1.bias',
    'blocks.4.attn.qkv.weight': 'blocks.4.attention.qkv_linear.weight',
    'blocks.4.attn.qkv.bias': 'blocks.4.attention.qkv_linear.bias',
    'blocks.4.attn.proj.weight': 'blocks.4.attention.out_linear.weight',
    'blocks.4.attn.proj.bias': 'blocks.4.attention.out_linear.bias',
    'blocks.4.norm2.weight': 'blocks.4.norm2.weight',
    'blocks.4.norm2.bias': 'blocks.4.norm2.bias',
    'blocks.4.mlp.fc1.weight': 'blocks.4.feed_forward.fc1.weight',
    'blocks.4.mlp.fc1.bias': 'blocks.4.feed_forward.fc1.bias',
    'blocks.4.mlp.fc2.weight': 'blocks.4.feed_forward.fc2.weight',
    'blocks.4.mlp.fc2.bias': 'blocks.4.feed_forward.fc2.bias',
    'blocks.5.norm1.weight': 'blocks.5.norm1.weight',
    'blocks.5.norm1.bias': 'blocks.5.norm1.bias',
    'blocks.5.attn.qkv.weight': 'blocks.5.attention.qkv_linear.weight',
    'blocks.5.attn.qkv.bias': 'blocks.5.attention.qkv_linear.bias',
    'blocks.5.attn.proj.weight': 'blocks.5.attention.out_linear.weight',
    'blocks.5.attn.proj.bias': 'blocks.5.attention.out_linear.bias',
    'blocks.5.norm2.weight': 'blocks.5.norm2.weight',
    'blocks.5.norm2.bias': 'blocks.5.norm2.bias',
    'blocks.5.mlp.fc1.weight': 'blocks.5.feed_forward.fc1.weight',
    'blocks.5.mlp.fc1.bias': 'blocks.5.feed_forward.fc1.bias',
    'blocks.5.mlp.fc2.weight': 'blocks.5.feed_forward.fc2.weight',
    'blocks.5.mlp.fc2.bias': 'blocks.5.feed_forward.fc2.bias',
    'blocks.6.norm1.weight': 'blocks.6.norm1.weight',
    'blocks.6.norm1.bias': 'blocks.6.norm1.bias',
    'blocks.6.attn.qkv.weight': 'blocks.6.attention.qkv_linear.weight',
    'blocks.6.attn.qkv.bias': 'blocks.6.attention.qkv_linear.bias',
    'blocks.6.attn.proj.weight': 'blocks.6.attention.out_linear.weight',
    'blocks.6.attn.proj.bias': 'blocks.6.attention.out_linear.bias',
    'blocks.6.norm2.weight': 'blocks.6.norm2.weight',
    'blocks.6.norm2.bias': 'blocks.6.norm2.bias',
    'blocks.6.mlp.fc1.weight': 'blocks.6.feed_forward.fc1.weight',
    'blocks.6.mlp.fc1.bias': 'blocks.6.feed_forward.fc1.bias',
    'blocks.6.mlp.fc2.weight': 'blocks.6.feed_forward.fc2.weight',
    'blocks.6.mlp.fc2.bias': 'blocks.6.feed_forward.fc2.bias',
    'blocks.7.norm1.weight': 'blocks.7.norm1.weight',
    'blocks.7.norm1.bias': 'blocks.7.norm1.bias',
    'blocks.7.attn.qkv.weight': 'blocks.7.attention.qkv_linear.weight',
    'blocks.7.attn.qkv.bias': 'blocks.7.attention.qkv_linear.bias',
    'blocks.7.attn.proj.weight': 'blocks.7.attention.out_linear.weight',
    'blocks.7.attn.proj.bias': 'blocks.7.attention.out_linear.bias',
    'blocks.7.norm2.weight': 'blocks.7.norm2.weight',
    'blocks.7.norm2.bias': 'blocks.7.norm2.bias',
    'blocks.7.mlp.fc1.weight': 'blocks.7.feed_forward.fc1.weight',
    'blocks.7.mlp.fc1.bias': 'blocks.7.feed_forward.fc1.bias',
    'blocks.7.mlp.fc2.weight': 'blocks.7.feed_forward.fc2.weight',
    'blocks.7.mlp.fc2.bias': 'blocks.7.feed_forward.fc2.bias',
    'blocks.8.norm1.weight': 'blocks.8.norm1.weight',
    'blocks.8.norm1.bias': 'blocks.8.norm1.bias',
    'blocks.8.attn.qkv.weight': 'blocks.8.attention.qkv_linear.weight',
    'blocks.8.attn.qkv.bias': 'blocks.8.attention.qkv_linear.bias',
    'blocks.8.attn.proj.weight': 'blocks.8.attention.out_linear.weight',
    'blocks.8.attn.proj.bias': 'blocks.8.attention.out_linear.bias',
    'blocks.8.norm2.weight': 'blocks.8.norm2.weight',
    'blocks.8.norm2.bias': 'blocks.8.norm2.bias',
    'blocks.8.mlp.fc1.weight': 'blocks.8.feed_forward.fc1.weight',
    'blocks.8.mlp.fc1.bias': 'blocks.8.feed_forward.fc1.bias',
    'blocks.8.mlp.fc2.weight': 'blocks.8.feed_forward.fc2.weight',
    'blocks.8.mlp.fc2.bias': 'blocks.8.feed_forward.fc2.bias',
    'blocks.9.norm1.weight': 'blocks.9.norm1.weight',
    'blocks.9.norm1.bias': 'blocks.9.norm1.bias',
    'blocks.9.attn.qkv.weight': 'blocks.9.attention.qkv_linear.weight',
    'blocks.9.attn.qkv.bias': 'blocks.9.attention.qkv_linear.bias',
    'blocks.9.attn.proj.weight': 'blocks.9.attention.out_linear.weight',
    'blocks.9.attn.proj.bias': 'blocks.9.attention.out_linear.bias',
    'blocks.9.norm2.weight': 'blocks.9.norm2.weight',
    'blocks.9.norm2.bias': 'blocks.9.norm2.bias',
    'blocks.9.mlp.fc1.weight': 'blocks.9.feed_forward.fc1.weight',
    'blocks.9.mlp.fc1.bias': 'blocks.9.feed_forward.fc1.bias',
    'blocks.9.mlp.fc2.weight': 'blocks.9.feed_forward.fc2.weight',
    'blocks.9.mlp.fc2.bias': 'blocks.9.feed_forward.fc2.bias',
    'blocks.10.norm1.weight': 'blocks.10.norm1.weight',
    'blocks.10.norm1.bias': 'blocks.10.norm1.bias',
    'blocks.10.attn.qkv.weight': 'blocks.10.attention.qkv_linear.weight',
    'blocks.10.attn.qkv.bias': 'blocks.10.attention.qkv_linear.bias',
    'blocks.10.attn.proj.weight': 'blocks.10.attention.out_linear.weight',
    'blocks.10.attn.proj.bias': 'blocks.10.attention.out_linear.bias',
    'blocks.10.norm2.weight': 'blocks.10.norm2.weight',
    'blocks.10.norm2.bias': 'blocks.10.norm2.bias',
    'blocks.10.mlp.fc1.weight': 'blocks.10.feed_forward.fc1.weight',
    'blocks.10.mlp.fc1.bias': 'blocks.10.feed_forward.fc1.bias',
    'blocks.10.mlp.fc2.weight': 'blocks.10.feed_forward.fc2.weight',
    'blocks.10.mlp.fc2.bias': 'blocks.10.feed_forward.fc2.bias',
    'blocks.11.norm1.weight': 'blocks.11.norm1.weight',
    'blocks.11.norm1.bias': 'blocks.11.norm1.bias',
    'blocks.11.attn.qkv.weight': 'blocks.11.attention.qkv_linear.weight',
    'blocks.11.attn.qkv.bias': 'blocks.11.attention.qkv_linear.bias',
    'blocks.11.attn.proj.weight': 'blocks.11.attention.out_linear.weight',
    'blocks.11.attn.proj.bias': 'blocks.11.attention.out_linear.bias',
    'blocks.11.norm2.weight': 'blocks.11.norm2.weight',
    'blocks.11.norm2.bias': 'blocks.11.norm2.bias',
    'blocks.11.mlp.fc1.weight': 'blocks.11.feed_forward.fc1.weight',
    'blocks.11.mlp.fc1.bias': 'blocks.11.feed_forward.fc1.bias',
    'blocks.11.mlp.fc2.weight': 'blocks.11.feed_forward.fc2.weight',
    'blocks.11.mlp.fc2.bias': 'blocks.11.feed_forward.fc2.bias',
    'blocks.12.norm1.weight': 'blocks.12.norm1.weight',
    'blocks.12.norm1.bias': 'blocks.12.norm1.bias',
    'blocks.12.attn.qkv.weight': 'blocks.12.attention.qkv_linear.weight',
    'blocks.12.attn.qkv.bias': 'blocks.12.attention.qkv_linear.bias',
    'blocks.12.attn.proj.weight': 'blocks.12.attention.out_linear.weight',
    'blocks.12.attn.proj.bias': 'blocks.12.attention.out_linear.bias',
    'blocks.12.norm2.weight': 'blocks.12.norm2.weight',
    'blocks.12.norm2.bias': 'blocks.12.norm2.bias',
    'blocks.12.mlp.fc1.weight': 'blocks.12.feed_forward.fc1.weight',
    'blocks.12.mlp.fc1.bias': 'blocks.12.feed_forward.fc1.bias',
    'blocks.12.mlp.fc2.weight': 'blocks.12.feed_forward.fc2.weight',
    'blocks.12.mlp.fc2.bias': 'blocks.12.feed_forward.fc2.bias',
    'blocks.13.norm1.weight': 'blocks.13.norm1.weight',
    'blocks.13.norm1.bias': 'blocks.13.norm1.bias',
    'blocks.13.attn.qkv.weight': 'blocks.13.attention.qkv_linear.weight',
    'blocks.13.attn.qkv.bias': 'blocks.13.attention.qkv_linear.bias',
    'blocks.13.attn.proj.weight': 'blocks.13.attention.out_linear.weight',
    'blocks.13.attn.proj.bias': 'blocks.13.attention.out_linear.bias',
    'blocks.13.norm2.weight': 'blocks.13.norm2.weight',
    'blocks.13.norm2.bias': 'blocks.13.norm2.bias',
    'blocks.13.mlp.fc1.weight': 'blocks.13.feed_forward.fc1.weight',
    'blocks.13.mlp.fc1.bias': 'blocks.13.feed_forward.fc1.bias',
    'blocks.13.mlp.fc2.weight': 'blocks.13.feed_forward.fc2.weight',
    'blocks.13.mlp.fc2.bias': 'blocks.13.feed_forward.fc2.bias',
    'blocks.14.norm1.weight': 'blocks.14.norm1.weight',
    'blocks.14.norm1.bias': 'blocks.14.norm1.bias',
    'blocks.14.attn.qkv.weight': 'blocks.14.attention.qkv_linear.weight',
    'blocks.14.attn.qkv.bias': 'blocks.14.attention.qkv_linear.bias',
    'blocks.14.attn.proj.weight': 'blocks.14.attention.out_linear.weight',
    'blocks.14.attn.proj.bias': 'blocks.14.attention.out_linear.bias',
    'blocks.14.norm2.weight': 'blocks.14.norm2.weight',
    'blocks.14.norm2.bias': 'blocks.14.norm2.bias',
    'blocks.14.mlp.fc1.weight': 'blocks.14.feed_forward.fc1.weight',
    'blocks.14.mlp.fc1.bias': 'blocks.14.feed_forward.fc1.bias',
    'blocks.14.mlp.fc2.weight': 'blocks.14.feed_forward.fc2.weight',
    'blocks.14.mlp.fc2.bias': 'blocks.14.feed_forward.fc2.bias',
    'blocks.15.norm1.weight': 'blocks.15.norm1.weight',
    'blocks.15.norm1.bias': 'blocks.15.norm1.bias',
    'blocks.15.attn.qkv.weight': 'blocks.15.attention.qkv_linear.weight',
    'blocks.15.attn.qkv.bias': 'blocks.15.attention.qkv_linear.bias',
    'blocks.15.attn.proj.weight': 'blocks.15.attention.out_linear.weight',
    'blocks.15.attn.proj.bias': 'blocks.15.attention.out_linear.bias',
    'blocks.15.norm2.weight': 'blocks.15.norm2.weight',
    'blocks.15.norm2.bias': 'blocks.15.norm2.bias',
    'blocks.15.mlp.fc1.weight': 'blocks.15.feed_forward.fc1.weight',
    'blocks.15.mlp.fc1.bias': 'blocks.15.feed_forward.fc1.bias',
    'blocks.15.mlp.fc2.weight': 'blocks.15.feed_forward.fc2.weight',
    'blocks.15.mlp.fc2.bias': 'blocks.15.feed_forward.fc2.bias',
    'blocks.16.norm1.weight': 'blocks.16.norm1.weight',
    'blocks.16.norm1.bias': 'blocks.16.norm1.bias',
    'blocks.16.attn.qkv.weight': 'blocks.16.attention.qkv_linear.weight',
    'blocks.16.attn.qkv.bias': 'blocks.16.attention.qkv_linear.bias',
    'blocks.16.attn.proj.weight': 'blocks.16.attention.out_linear.weight',
    'blocks.16.attn.proj.bias': 'blocks.16.attention.out_linear.bias',
    'blocks.16.norm2.weight': 'blocks.16.norm2.weight',
    'blocks.16.norm2.bias': 'blocks.16.norm2.bias',
    'blocks.16.mlp.fc1.weight': 'blocks.16.feed_forward.fc1.weight',
    'blocks.16.mlp.fc1.bias': 'blocks.16.feed_forward.fc1.bias',
    'blocks.16.mlp.fc2.weight': 'blocks.16.feed_forward.fc2.weight',
    'blocks.16.mlp.fc2.bias': 'blocks.16.feed_forward.fc2.bias',
    'blocks.17.norm1.weight': 'blocks.17.norm1.weight',
    'blocks.17.norm1.bias': 'blocks.17.norm1.bias',
    'blocks.17.attn.qkv.weight': 'blocks.17.attention.qkv_linear.weight',
    'blocks.17.attn.qkv.bias': 'blocks.17.attention.qkv_linear.bias',
    'blocks.17.attn.proj.weight': 'blocks.17.attention.out_linear.weight',
    'blocks.17.attn.proj.bias': 'blocks.17.attention.out_linear.bias',
    'blocks.17.norm2.weight': 'blocks.17.norm2.weight',
    'blocks.17.norm2.bias': 'blocks.17.norm2.bias',
    'blocks.17.mlp.fc1.weight': 'blocks.17.feed_forward.fc1.weight',
    'blocks.17.mlp.fc1.bias': 'blocks.17.feed_forward.fc1.bias',
    'blocks.17.mlp.fc2.weight': 'blocks.17.feed_forward.fc2.weight',
    'blocks.17.mlp.fc2.bias': 'blocks.17.feed_forward.fc2.bias',
    'blocks.18.norm1.weight': 'blocks.18.norm1.weight',
    'blocks.18.norm1.bias': 'blocks.18.norm1.bias',
    'blocks.18.attn.qkv.weight': 'blocks.18.attention.qkv_linear.weight',
    'blocks.18.attn.qkv.bias': 'blocks.18.attention.qkv_linear.bias',
    'blocks.18.attn.proj.weight': 'blocks.18.attention.out_linear.weight',
    'blocks.18.attn.proj.bias': 'blocks.18.attention.out_linear.bias',
    'blocks.18.norm2.weight': 'blocks.18.norm2.weight',
    'blocks.18.norm2.bias': 'blocks.18.norm2.bias',
    'blocks.18.mlp.fc1.weight': 'blocks.18.feed_forward.fc1.weight',
    'blocks.18.mlp.fc1.bias': 'blocks.18.feed_forward.fc1.bias',
    'blocks.18.mlp.fc2.weight': 'blocks.18.feed_forward.fc2.weight',
    'blocks.18.mlp.fc2.bias': 'blocks.18.feed_forward.fc2.bias',
    'blocks.19.norm1.weight': 'blocks.19.norm1.weight',
    'blocks.19.norm1.bias': 'blocks.19.norm1.bias',
    'blocks.19.attn.qkv.weight': 'blocks.19.attention.qkv_linear.weight',
    'blocks.19.attn.qkv.bias': 'blocks.19.attention.qkv_linear.bias',
    'blocks.19.attn.proj.weight': 'blocks.19.attention.out_linear.weight',
    'blocks.19.attn.proj.bias': 'blocks.19.attention.out_linear.bias',
    'blocks.19.norm2.weight': 'blocks.19.norm2.weight',
    'blocks.19.norm2.bias': 'blocks.19.norm2.bias',
    'blocks.19.mlp.fc1.weight': 'blocks.19.feed_forward.fc1.weight',
    'blocks.19.mlp.fc1.bias': 'blocks.19.feed_forward.fc1.bias',
    'blocks.19.mlp.fc2.weight': 'blocks.19.feed_forward.fc2.weight',
    'blocks.19.mlp.fc2.bias': 'blocks.19.feed_forward.fc2.bias',
    'blocks.20.norm1.weight': 'blocks.20.norm1.weight',
    'blocks.20.norm1.bias': 'blocks.20.norm1.bias',
    'blocks.20.attn.qkv.weight': 'blocks.20.attention.qkv_linear.weight',
    'blocks.20.attn.qkv.bias': 'blocks.20.attention.qkv_linear.bias',
    'blocks.20.attn.proj.weight': 'blocks.20.attention.out_linear.weight',
    'blocks.20.attn.proj.bias': 'blocks.20.attention.out_linear.bias',
    'blocks.20.norm2.weight': 'blocks.20.norm2.weight',
    'blocks.20.norm2.bias': 'blocks.20.norm2.bias',
    'blocks.20.mlp.fc1.weight': 'blocks.20.feed_forward.fc1.weight',
    'blocks.20.mlp.fc1.bias': 'blocks.20.feed_forward.fc1.bias',
    'blocks.20.mlp.fc2.weight': 'blocks.20.feed_forward.fc2.weight',
    'blocks.20.mlp.fc2.bias': 'blocks.20.feed_forward.fc2.bias',
    'blocks.21.norm1.weight': 'blocks.21.norm1.weight',
    'blocks.21.norm1.bias': 'blocks.21.norm1.bias',
    'blocks.21.attn.qkv.weight': 'blocks.21.attention.qkv_linear.weight',
    'blocks.21.attn.qkv.bias': 'blocks.21.attention.qkv_linear.bias',
    'blocks.21.attn.proj.weight': 'blocks.21.attention.out_linear.weight',
    'blocks.21.attn.proj.bias': 'blocks.21.attention.out_linear.bias',
    'blocks.21.norm2.weight': 'blocks.21.norm2.weight',
    'blocks.21.norm2.bias': 'blocks.21.norm2.bias',
    'blocks.21.mlp.fc1.weight': 'blocks.21.feed_forward.fc1.weight',
    'blocks.21.mlp.fc1.bias': 'blocks.21.feed_forward.fc1.bias',
    'blocks.21.mlp.fc2.weight': 'blocks.21.feed_forward.fc2.weight',
    'blocks.21.mlp.fc2.bias': 'blocks.21.feed_forward.fc2.bias',
    'blocks.22.norm1.weight': 'blocks.22.norm1.weight',
    'blocks.22.norm1.bias': 'blocks.22.norm1.bias',
    'blocks.22.attn.qkv.weight': 'blocks.22.attention.qkv_linear.weight',
    'blocks.22.attn.qkv.bias': 'blocks.22.attention.qkv_linear.bias',
    'blocks.22.attn.proj.weight': 'blocks.22.attention.out_linear.weight',
    'blocks.22.attn.proj.bias': 'blocks.22.attention.out_linear.bias',
    'blocks.22.norm2.weight': 'blocks.22.norm2.weight',
    'blocks.22.norm2.bias': 'blocks.22.norm2.bias',
    'blocks.22.mlp.fc1.weight': 'blocks.22.feed_forward.fc1.weight',
    'blocks.22.mlp.fc1.bias': 'blocks.22.feed_forward.fc1.bias',
    'blocks.22.mlp.fc2.weight': 'blocks.22.feed_forward.fc2.weight',
    'blocks.22.mlp.fc2.bias': 'blocks.22.feed_forward.fc2.bias',
    'blocks.23.norm1.weight': 'blocks.23.norm1.weight',
    'blocks.23.norm1.bias': 'blocks.23.norm1.bias',
    'blocks.23.attn.qkv.weight': 'blocks.23.attention.qkv_linear.weight',
    'blocks.23.attn.qkv.bias': 'blocks.23.attention.qkv_linear.bias',
    'blocks.23.attn.proj.weight': 'blocks.23.attention.out_linear.weight',
    'blocks.23.attn.proj.bias': 'blocks.23.attention.out_linear.bias',
    'blocks.23.norm2.weight': 'blocks.23.norm2.weight',
    'blocks.23.norm2.bias': 'blocks.23.norm2.bias',
    'blocks.23.mlp.fc1.weight': 'blocks.23.feed_forward.fc1.weight',
    'blocks.23.mlp.fc1.bias': 'blocks.23.feed_forward.fc1.bias',
    'blocks.23.mlp.fc2.weight': 'blocks.23.feed_forward.fc2.weight',
    'blocks.23.mlp.fc2.bias': 'blocks.23.feed_forward.fc2.bias',
    'blocks.24.norm1.weight': 'blocks.24.norm1.weight',
    'blocks.24.norm1.bias': 'blocks.24.norm1.bias',
    'blocks.24.attn.qkv.weight': 'blocks.24.attention.qkv_linear.weight',
    'blocks.24.attn.qkv.bias': 'blocks.24.attention.qkv_linear.bias',
    'blocks.24.attn.proj.weight': 'blocks.24.attention.out_linear.weight',
    'blocks.24.attn.proj.bias': 'blocks.24.attention.out_linear.bias',
    'blocks.24.norm2.weight': 'blocks.24.norm2.weight',
    'blocks.24.norm2.bias': 'blocks.24.norm2.bias',
    'blocks.24.mlp.fc1.weight': 'blocks.24.feed_forward.fc1.weight',
    'blocks.24.mlp.fc1.bias': 'blocks.24.feed_forward.fc1.bias',
    'blocks.24.mlp.fc2.weight': 'blocks.24.feed_forward.fc2.weight',
    'blocks.24.mlp.fc2.bias': 'blocks.24.feed_forward.fc2.bias',
    'blocks.25.norm1.weight': 'blocks.25.norm1.weight',
    'blocks.25.norm1.bias': 'blocks.25.norm1.bias',
    'blocks.25.attn.qkv.weight': 'blocks.25.attention.qkv_linear.weight',
    'blocks.25.attn.qkv.bias': 'blocks.25.attention.qkv_linear.bias',
    'blocks.25.attn.proj.weight': 'blocks.25.attention.out_linear.weight',
    'blocks.25.attn.proj.bias': 'blocks.25.attention.out_linear.bias',
    'blocks.25.norm2.weight': 'blocks.25.norm2.weight',
    'blocks.25.norm2.bias': 'blocks.25.norm2.bias',
    'blocks.25.mlp.fc1.weight': 'blocks.25.feed_forward.fc1.weight',
    'blocks.25.mlp.fc1.bias': 'blocks.25.feed_forward.fc1.bias',
    'blocks.25.mlp.fc2.weight': 'blocks.25.feed_forward.fc2.weight',
    'blocks.25.mlp.fc2.bias': 'blocks.25.feed_forward.fc2.bias',
    'blocks.26.norm1.weight': 'blocks.26.norm1.weight',
    'blocks.26.norm1.bias': 'blocks.26.norm1.bias',
    'blocks.26.attn.qkv.weight': 'blocks.26.attention.qkv_linear.weight',
    'blocks.26.attn.qkv.bias': 'blocks.26.attention.qkv_linear.bias',
    'blocks.26.attn.proj.weight': 'blocks.26.attention.out_linear.weight',
    'blocks.26.attn.proj.bias': 'blocks.26.attention.out_linear.bias',
    'blocks.26.norm2.weight': 'blocks.26.norm2.weight',
    'blocks.26.norm2.bias': 'blocks.26.norm2.bias',
    'blocks.26.mlp.fc1.weight': 'blocks.26.feed_forward.fc1.weight',
    'blocks.26.mlp.fc1.bias': 'blocks.26.feed_forward.fc1.bias',
    'blocks.26.mlp.fc2.weight': 'blocks.26.feed_forward.fc2.weight',
    'blocks.26.mlp.fc2.bias': 'blocks.26.feed_forward.fc2.bias',
    'blocks.27.norm1.weight': 'blocks.27.norm1.weight',
    'blocks.27.norm1.bias': 'blocks.27.norm1.bias',
    'blocks.27.attn.qkv.weight': 'blocks.27.attention.qkv_linear.weight',
    'blocks.27.attn.qkv.bias': 'blocks.27.attention.qkv_linear.bias',
    'blocks.27.attn.proj.weight': 'blocks.27.attention.out_linear.weight',
    'blocks.27.attn.proj.bias': 'blocks.27.attention.out_linear.bias',
    'blocks.27.norm2.weight': 'blocks.27.norm2.weight',
    'blocks.27.norm2.bias': 'blocks.27.norm2.bias',
    'blocks.27.mlp.fc1.weight': 'blocks.27.feed_forward.fc1.weight',
    'blocks.27.mlp.fc1.bias': 'blocks.27.feed_forward.fc1.bias',
    'blocks.27.mlp.fc2.weight': 'blocks.27.feed_forward.fc2.weight',
    'blocks.27.mlp.fc2.bias': 'blocks.27.feed_forward.fc2.bias',
    'blocks.28.norm1.weight': 'blocks.28.norm1.weight',
    'blocks.28.norm1.bias': 'blocks.28.norm1.bias',
    'blocks.28.attn.qkv.weight': 'blocks.28.attention.qkv_linear.weight',
    'blocks.28.attn.qkv.bias': 'blocks.28.attention.qkv_linear.bias',
    'blocks.28.attn.proj.weight': 'blocks.28.attention.out_linear.weight',
    'blocks.28.attn.proj.bias': 'blocks.28.attention.out_linear.bias',
    'blocks.28.norm2.weight': 'blocks.28.norm2.weight',
    'blocks.28.norm2.bias': 'blocks.28.norm2.bias',
    'blocks.28.mlp.fc1.weight': 'blocks.28.feed_forward.fc1.weight',
    'blocks.28.mlp.fc1.bias': 'blocks.28.feed_forward.fc1.bias',
    'blocks.28.mlp.fc2.weight': 'blocks.28.feed_forward.fc2.weight',
    'blocks.28.mlp.fc2.bias': 'blocks.28.feed_forward.fc2.bias',
    'blocks.29.norm1.weight': 'blocks.29.norm1.weight',
    'blocks.29.norm1.bias': 'blocks.29.norm1.bias',
    'blocks.29.attn.qkv.weight': 'blocks.29.attention.qkv_linear.weight',
    'blocks.29.attn.qkv.bias': 'blocks.29.attention.qkv_linear.bias',
    'blocks.29.attn.proj.weight': 'blocks.29.attention.out_linear.weight',
    'blocks.29.attn.proj.bias': 'blocks.29.attention.out_linear.bias',
    'blocks.29.norm2.weight': 'blocks.29.norm2.weight',
    'blocks.29.norm2.bias': 'blocks.29.norm2.bias',
    'blocks.29.mlp.fc1.weight': 'blocks.29.feed_forward.fc1.weight',
    'blocks.29.mlp.fc1.bias': 'blocks.29.feed_forward.fc1.bias',
    'blocks.29.mlp.fc2.weight': 'blocks.29.feed_forward.fc2.weight',
    'blocks.29.mlp.fc2.bias': 'blocks.29.feed_forward.fc2.bias',
    'blocks.30.norm1.weight': 'blocks.30.norm1.weight',
    'blocks.30.norm1.bias': 'blocks.30.norm1.bias',
    'blocks.30.attn.qkv.weight': 'blocks.30.attention.qkv_linear.weight',
    'blocks.30.attn.qkv.bias': 'blocks.30.attention.qkv_linear.bias',
    'blocks.30.attn.proj.weight': 'blocks.30.attention.out_linear.weight',
    'blocks.30.attn.proj.bias': 'blocks.30.attention.out_linear.bias',
    'blocks.30.norm2.weight': 'blocks.30.norm2.weight',
    'blocks.30.norm2.bias': 'blocks.30.norm2.bias',
    'blocks.30.mlp.fc1.weight': 'blocks.30.feed_forward.fc1.weight',
    'blocks.30.mlp.fc1.bias': 'blocks.30.feed_forward.fc1.bias',
    'blocks.30.mlp.fc2.weight': 'blocks.30.feed_forward.fc2.weight',
    'blocks.30.mlp.fc2.bias': 'blocks.30.feed_forward.fc2.bias',
    'blocks.31.norm1.weight': 'blocks.31.norm1.weight',
    'blocks.31.norm1.bias': 'blocks.31.norm1.bias',
    'blocks.31.attn.qkv.weight': 'blocks.31.attention.qkv_linear.weight',
    'blocks.31.attn.qkv.bias': 'blocks.31.attention.qkv_linear.bias',
    'blocks.31.attn.proj.weight': 'blocks.31.attention.out_linear.weight',
    'blocks.31.attn.proj.bias': 'blocks.31.attention.out_linear.bias',
    'blocks.31.norm2.weight': 'blocks.31.norm2.weight',
    'blocks.31.norm2.bias': 'blocks.31.norm2.bias',
    'blocks.31.mlp.fc1.weight': 'blocks.31.feed_forward.fc1.weight',
    'blocks.31.mlp.fc1.bias': 'blocks.31.feed_forward.fc1.bias',
    'blocks.31.mlp.fc2.weight': 'blocks.31.feed_forward.fc2.weight',
    'blocks.31.mlp.fc2.bias': 'blocks.31.feed_forward.fc2.bias'
}

if __name__ == '__main__':
    # network = 'vit_base_patch16'
    # num_classes = 1000
    # input_image_size = 224
    # scale = 256 / 224
    # model = backbones.__dict__[network](**{
    #     'image_size': 224,
    #     'global_pool': True,
    #     'num_classes': num_classes,
    # })

    # model_name_list = []
    # for name, weight in model.state_dict().items():
    #     model_name_list.append([name, weight.shape])

    # print('1111', len(model_name_list))
    # # for name, _ in model_name_list:
    # #     print(name)

    # saved_model_path = '/root/autodl-tmp/weights/mae_pretrain_vit_base.pth'
    # saved_state_dict = torch.load(saved_model_path,
    #                               map_location=torch.device('cpu'))

    # save_name_list = []
    # for name, weight in saved_state_dict['model'].items():
    #     save_name_list.append([name, weight.shape])

    # print('2222', len(save_name_list))
    # # for name, _ in save_name_list:
    # #     print(name)

    # # for i in range(32):
    # #     for key, value in convert_origin_dict.items():
    # #         if i != 0:
    # #             key = key.replace('blocks.0.', f'blocks.{str(i)}.')
    # #             value = value.replace('blocks.0.', f'blocks.{str(i)}.')

    # #         convert_dict[key] = value

    # # print(convert_dict)

    # new_save_dict = {}
    # for name, weight in saved_state_dict['model'].items():
    #     model_name = convert_vit_huge_dict[name]
    #     model_weight = model.state_dict()[model_name]
    #     if weight.shape != model_weight.shape:
    #         print('2222', name, model_name, weight.shape, model_weight.shape)
    #     new_save_dict[model_name] = weight

    # print('3333', len(new_save_dict))
    # torch.save(
    #     new_save_dict,
    #     f'/root/autodl-tmp/weights/vit_base_patch16_official_mae_pretrain_convert.pth'
    # )

    # network = 'vit_large_patch16'
    # num_classes = 1000
    # input_image_size = 224
    # scale = 256 / 224
    # model = backbones.__dict__[network](**{
    #     'image_size': 224,
    #     'global_pool': True,
    #     'num_classes': num_classes,
    # })

    # model_name_list = []
    # for name, weight in model.state_dict().items():
    #     model_name_list.append([name, weight.shape])

    # print('1111', len(model_name_list))
    # # for name, _ in model_name_list:
    # #     print(name)

    # saved_model_path = '/root/autodl-tmp/weights/mae_pretrain_vit_large.pth'
    # saved_state_dict = torch.load(saved_model_path,
    #                               map_location=torch.device('cpu'))

    # save_name_list = []
    # for name, weight in saved_state_dict['model'].items():
    #     save_name_list.append([name, weight.shape])

    # print('2222', len(save_name_list))
    # # for name, _ in save_name_list:
    # #     print(name)

    # # for i in range(32):
    # #     for key, value in convert_origin_dict.items():
    # #         if i != 0:
    # #             key = key.replace('blocks.0.', f'blocks.{str(i)}.')
    # #             value = value.replace('blocks.0.', f'blocks.{str(i)}.')

    # #         convert_dict[key] = value

    # # print(convert_dict)

    # new_save_dict = {}
    # for name, weight in saved_state_dict['model'].items():
    #     model_name = convert_vit_huge_dict[name]
    #     model_weight = model.state_dict()[model_name]
    #     if weight.shape != model_weight.shape:
    #         print('2222', name, model_name, weight.shape, model_weight.shape)
    #     new_save_dict[model_name] = weight

    # print('3333', len(new_save_dict))
    # torch.save(
    #     new_save_dict,
    #     f'/root/autodl-tmp/weights/vit_large_patch16_official_mae_pretrain_convert.pth'
    # )

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

    saved_model_path = '/root/autodl-tmp/weights/mae_pretrain_vit_huge.pth'
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

    save_name_list = []
    for name, weight in saved_state_dict['model'].items():
        save_name_list.append([name, weight.shape])

    print('2222', len(save_name_list))
    # for name, _ in save_name_list:
    #     print(name)

    # for i in range(32):
    #     for key, value in convert_origin_dict.items():
    #         if i != 0:
    #             key = key.replace('blocks.0.', f'blocks.{str(i)}.')
    #             value = value.replace('blocks.0.', f'blocks.{str(i)}.')

    #         convert_dict[key] = value

    # print(convert_dict)

    new_save_dict = {}
    for name, weight in saved_state_dict['model'].items():
        model_name = convert_vit_huge_dict[name]
        model_weight = model.state_dict()[model_name]
        if weight.shape != model_weight.shape:
            print('2222', name, model_name, weight.shape, model_weight.shape)
        new_save_dict[model_name] = weight

    print('3333', len(new_save_dict))
    torch.save(
        new_save_dict,
        f'/root/autodl-tmp/weights/vit_huge_patch14_official_mae_pretrain_convert.pth'
    )