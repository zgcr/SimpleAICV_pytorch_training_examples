import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn


def get_deploy_model(trained_model, deployed_model):
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
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from simpleAICV.classification import backbones
    from simpleAICV.classification.common import load_state_dict

    trained_model_path = '/root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/classification_training/imagenet/repvgg_a1/checkpoints/RepVGG_A1-acc74.082.pth'

    model_name = trained_model_path.split("/")[-1].rstrip(
        '.pth') + "_deployed.pth"
    network = 'RepVGG_A1'
    num_classes = 1000
    trained_model = backbones.__dict__[network](**{
        'deploy': False,
        'num_classes': num_classes,
    })
    load_state_dict(trained_model_path, trained_model)
    trained_model.eval()

    deployed_model = backbones.__dict__[network](**{
        'deploy': True,
        'num_classes': num_classes,
    })
    deployed_model = get_deploy_model(trained_model, deployed_model)
    deployed_model.eval()

    save_deployed_model_path = f'./checkpoints/{model_name}'
    torch.save(deployed_model.state_dict(), save_deployed_model_path)

    inputs = torch.randn(3, 3, 224, 224)
    out1 = trained_model(inputs)
    out2 = deployed_model(inputs)
    print(out1[0][1:20], out2[0][1:20])
    print(((out1 - out2)**2).sum())  # Will be around 1e-10
