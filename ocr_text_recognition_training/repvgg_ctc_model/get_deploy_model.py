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
    trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/ocr_text_recognition_training/repvgg_ctc_model/checkpoints/epoch_50.pth'
    # save deployed model parameters
    save_deploy_model_base_path = './checkpoints'

    from simpleAICV.text_recognition.models import CTCModel
    from simpleAICV.text_recognition.common import CTCTextLabelConverter, load_state_dict
    from simpleAICV.text_recognition.char_sets.final_char_table import final_char_table

    str_max_length = 80
    # please make sure your converter type is the same as 'predictor'
    converter = CTCTextLabelConverter(chars_set_list=final_char_table,
                                      str_max_length=str_max_length,
                                      garbage_char='㍿')
    # all char + '[CTCblank]' = 12111 + 1 = 12112
    num_classes = converter.num_classes

    # define trained model
    trained_model_config = {
        'backbone': {
            'name': 'RepVGGEnhanceNetBackbone',
            'param': {
                'inplanes': 1,
                'planes': [32, 64, 128, 256],
                'k': 4,
                'deploy': False,
            }
        },
        'encoder': {
            'name': 'BiLSTMEncoder',
            'param': {},
        },
        'predictor': {
            'name': 'CTCEnhancePredictor',
            'param': {
                'hidden_planes': 192,
                'num_classes': num_classes + 1,
            }
        },
    }
    trained_model = CTCModel(trained_model_config)

    load_state_dict(trained_model_path, trained_model)
    trained_model.eval()

    # define deployed model
    deployed_model_config = {
        'backbone': {
            'name': 'RepVGGEnhanceNetBackbone',
            'param': {
                'inplanes': 1,
                'planes': [32, 64, 128, 256],
                'k': 4,
                'deploy': True,
            }
        },
        'encoder': {
            'name': 'BiLSTMEncoder',
            'param': {},
        },
        'predictor': {
            'name': 'CTCEnhancePredictor',
            'param': {
                'hidden_planes': 192,
                'num_classes': num_classes + 1,
            }
        },
    }

    deployed_model = CTCModel(deployed_model_config)

    if not os.path.exists(save_deploy_model_base_path):
        os.makedirs(save_deploy_model_base_path)
    save_deployed_parameters_path = os.path.join(
        save_deploy_model_base_path,
        f'{trained_model_path.split("/")[-1][:-4]}_deployed.pth')
    deployed_model = deploy_model(trained_model, deployed_model)
    deployed_model.eval()

    if save_deployed_parameters_path:
        torch.save(deployed_model.state_dict(), save_deployed_parameters_path)

    images = torch.randn(1, 1, 32, 512)
    out1 = trained_model(images)
    out2 = deployed_model(images)
    print("1111", out1[0][1:10])
    print("2222", out2[0][1:10])
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
