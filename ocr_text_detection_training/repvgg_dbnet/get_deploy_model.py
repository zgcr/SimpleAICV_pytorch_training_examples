import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn


def deploy_model(trained_model, deployed_model):
    deploy_model_weights = {}
    for name, module in trained_model.named_modules():
        if hasattr(module, 'get_equivalent_conv_kernel_bias'):
            kernel, bias = module.get_equivalent_conv_kernel_bias()
            deploy_model_weights[name +
                                 '.fuse_equivalent_conv.weight'] = kernel
            deploy_model_weights[name + '.fuse_equivalent_conv.bias'] = bias
        elif isinstance(module, nn.Linear):
            deploy_model_weights[name +
                                 '.weight'] = module.weight.detach().cpu()
            deploy_model_weights[name + '.bias'] = module.bias.detach().cpu()
        else:
            # named_parameters return all layers that need to be backpropagated,such as conv layer or linear layer
            for layer_name, layer_weights in module.named_parameters():
                full_name = name + '.' + layer_name
                if full_name not in deploy_model_weights.keys():
                    deploy_model_weights[full_name] = layer_weights.detach(
                    ).cpu()
            # named_buffers return all layers that don't need to be backpropagated,such as bn layer
            for layer_name, layer_weights in module.named_buffers():
                full_name = name + '.' + layer_name
                if full_name not in deploy_model_weights.keys():
                    deploy_model_weights[full_name] = layer_weights.cpu()

    # load all equivalent weights,and the other weights will be abandoned(self.conv3x3,self.conv1x1,self.identity in RepVGGBlock).
    deployed_model.load_state_dict(deploy_model_weights, strict=False)

    return deployed_model


if __name__ == '__main__':
    trained_model_path = ''
    # save deployed model parameters
    save_deploy_model_base_path = './checkpoints'

    from simpleAICV.text_detection import models
    from simpleAICV.text_detection.common import load_state_dict

    network = 'RepVGGEnhanceDBNet'
    input_image_size = [960, 960]

    trained_model = models.__dict__[network](**{
        'backbone_type': 'RepVGGEnhanceNetBackbone',
        'backbone_pretrained_path': '',
        'planes': [16, 16, 32, 48, 64, 80],
        'repvgg_k': 4,
        'inter_planes': 96,
        'k': 50,
        'deploy': False,
    })

    # load saved model trained parameters
    load_state_dict(trained_model_path, trained_model)
    trained_model.eval()

    # define deployed model
    deployed_model = models.__dict__[network](
        **{
            'backbone_type': 'RepVGGEnhanceNetBackbone',
            'backbone_pretrained_path': '',
            'planes': [16, 16, 32, 48, 64, 80],
            'repvgg_k': 4,
            'inter_planes': 96,
            'k': 50,
            'deploy': True,
        })

    deployed_model.eval()

    if not os.path.exists(save_deploy_model_base_path):
        os.makedirs(save_deploy_model_base_path)
    save_deployed_parameters_path = os.path.join(
        save_deploy_model_base_path,
        f'{trained_model_path.split("/")[-1][:-4]}_deployed.pth')
    deployed_model = deploy_model(trained_model, deployed_model)
    deployed_model.eval()

    if save_deployed_parameters_path:
        torch.save(deployed_model.state_dict(), save_deployed_parameters_path)

    images = torch.randn(1, 3, input_image_size[0], input_image_size[1])
    out1 = trained_model(images)
    out2 = deployed_model(images)
    print("1111", out1[0][0][0][1:10])
    print("2222", out2[0][0][0][1:10])
    print(((out1 - out2)**2).sum())  # Will be around 1e-4

    from thop import profile
    from thop import clever_format
    flops, params = profile(trained_model, inputs=(images, ), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    x = trained_model(images)
    print(
        f'1111,trained_model: flops: {flops}, params: {params},out_shape: {x.shape}'
    )

    flops, params = profile(deployed_model, inputs=(images, ), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    x = deployed_model(images)
    print(
        f'2222,deployed_model: flops: {flops}, params: {params},out_shape: {x.shape}'
    )
