'''
RepVGG: Making VGG-style ConvNets Great Again
https://arxiv.org/pdf/2101.03697.pdf
'''
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'RepVGG_A0',
    'RepVGG_A1',
    'RepVGG_A2',
    'RepVGG_B0',
    'RepVGG_B1',
    'RepVGG_B1g2',
    'RepVGG_B1g4',
    'RepVGG_B2',
    'RepVGG_B2g2',
    'RepVGG_B2g4',
    'RepVGG_B3',
    'RepVGG_B3g2',
    'RepVGG_B3g4',
]

groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in groupwise_layers}
g4_map = {l: 4 for l in groupwise_layers}

types_config = {
    'RepVGG_A0': {
        'num_blocks': [2, 4, 14, 1],
        'width_multiplier': [0.75, 0.75, 0.75, 2.5],
        'override_groups_map': None,
    },
    'RepVGG_A1': {
        'num_blocks': [2, 4, 14, 1],
        'width_multiplier': [1, 1, 1, 2.5],
        'override_groups_map': None,
    },
    'RepVGG_A2': {
        'num_blocks': [2, 4, 14, 1],
        'width_multiplier': [1.5, 1.5, 1.5, 2.75],
        'override_groups_map': None,
    },
    'RepVGG_B0': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [1, 1, 1, 2.5],
        'override_groups_map': None,
    },
    'RepVGG_B1': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [2, 2, 2, 4],
        'override_groups_map': None,
    },
    'RepVGG_B1g2': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [2, 2, 2, 4],
        'override_groups_map': g2_map,
    },
    'RepVGG_B1g4': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [2, 2, 2, 4],
        'override_groups_map': g4_map,
    },
    'RepVGG_B2': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [2.5, 2.5, 2.5, 5],
        'override_groups_map': None,
    },
    'RepVGG_B2g2': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [2.5, 2.5, 2.5, 5],
        'override_groups_map': g2_map,
    },
    'RepVGG_B2g4': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [2.5, 2.5, 2.5, 5],
        'override_groups_map': g4_map,
    },
    'RepVGG_B3': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [3, 3, 3, 5],
        'override_groups_map': None,
    },
    'RepVGG_B3g2': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [3, 3, 3, 5],
        'override_groups_map': g2_map,
    },
    'RepVGG_B3g4': {
        'num_blocks': [4, 6, 16, 1],
        'width_multiplier': [3, 3, 3, 5],
        'override_groups_map': g4_map,
    },
}


def conv_bn_layer(inplanes, planes, kernel_size, stride, padding=1, groups=1):
    layer = nn.Sequential(
        OrderedDict([
            ('conv',
             nn.Conv2d(inplanes,
                       planes,
                       kernel_size,
                       stride=stride,
                       padding=padding,
                       groups=groups,
                       bias=False)),
            ('bn', nn.BatchNorm2d(planes)),
        ]))

    return layer


class RepVGGBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 deploy=False):
        super(RepVGGBlock, self).__init__()
        self.inplanes = inplanes
        self.groups = groups
        self.deploy = deploy

        assert kernel_size == 3 and padding == 1

        if self.deploy:
            self.fuse_equivalent_conv = nn.Conv2d(inplanes,
                                                  planes,
                                                  kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=groups,
                                                  bias=True)
        else:
            self.identity = nn.BatchNorm2d(
                inplanes) if inplanes == planes and stride == 1 else None
            self.conv3x3 = conv_bn_layer(inplanes,
                                         planes,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=groups)
            self.conv1x1 = conv_bn_layer(inplanes,
                                         planes,
                                         kernel_size=1,
                                         stride=stride,
                                         padding=padding - kernel_size // 2,
                                         groups=groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            x = self.relu(self.fuse_equivalent_conv(x))

            return x

        if self.identity:
            identity_out = self.identity(x)
        else:
            identity_out = 0

        x = self.relu(self.conv3x3(x) + self.conv1x1(x) + identity_out)

        return x

    def _fuse_bn_layer(self, branch):
        '''
        fuse conv and bn layers to get equivalent conv layer kernel and bias
        '''
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            # make sure conv layer doesn't have bias
            kernel = branch.conv.weight
            running_mean, running_var = branch.bn.running_mean, branch.bn.running_var
            gamma, beta, eps = branch.bn.weight, branch.bn.bias, branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            # convert identity branch to get a equivalent 1x1 conv layer kernel and bias
            input_dim = self.inplanes // self.groups
            kernel_value = np.zeros((self.inplanes, input_dim, 3, 3),
                                    dtype=np.float32)
            for i in range(self.inplanes):
                kernel_value[i, i % input_dim, 1, 1] = 1

            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma, beta, eps = branch.weight, branch.bias, branch.eps

        # fuse conv and bn layer
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        equivalent_kernel, equivalent_bias = kernel * t, beta - running_mean * gamma / std

        return equivalent_kernel, equivalent_bias

    def get_equivalent_conv_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_layer(self.conv3x3)
        kernel1x1, bias1x1 = self._fuse_bn_layer(self.conv1x1)
        kernelidentity, biasidentity = self._fuse_bn_layer(self.identity)

        # 1x1kernel must be pad to 3x3kernel before add
        kernel, bias = kernel3x3 + F.pad(
            kernel1x1,
            [1, 1, 1, 1]) + kernelidentity, bias3x3 + bias1x1 + biasidentity
        kernel, bias = kernel.detach().cpu(), bias.detach().cpu()

        return kernel, bias


class RepVGG(nn.Module):

    def __init__(self, repvgg_type, deploy=False, num_classes=1000):
        super(RepVGG, self).__init__()
        self.superparams = types_config[repvgg_type]
        self.num_blocks = self.superparams['num_blocks']
        self.width_multiplier = self.superparams['width_multiplier']
        self.override_groups_map = self.superparams[
            'override_groups_map'] if self.superparams[
                'override_groups_map'] else dict()
        self.deploy = deploy
        self.num_classes = num_classes

        self.inplanes = min(64, int(64 * self.width_multiplier[0]))
        self.cur_layer_idx = 1
        self.stage0 = RepVGGBlock(3,
                                  self.inplanes,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  groups=1,
                                  deploy=self.deploy)

        self.stage1 = self._make_stage(int(64 * self.width_multiplier[0]),
                                       self.num_blocks[0],
                                       stride=2)
        self.stage2 = self._make_stage(int(128 * self.width_multiplier[1]),
                                       self.num_blocks[1],
                                       stride=2)
        self.stage3 = self._make_stage(int(256 * self.width_multiplier[2]),
                                       self.num_blocks[2],
                                       stride=2)
        self.stage4 = self._make_stage(int(512 * self.width_multiplier[3]),
                                       self.num_blocks[3],
                                       stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * self.width_multiplier[3]),
                            self.num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(self.inplanes,
                            planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=cur_groups,
                            deploy=self.deploy))
            self.inplanes = planes
            self.cur_layer_idx += 1

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _repvgg(arch, deploy, **kwargs):
    model = RepVGG(arch, deploy, **kwargs)

    return model


def RepVGG_A0(deploy=False, **kwargs):
    return _repvgg('RepVGG_A0', deploy, **kwargs)


def RepVGG_A1(deploy=False, **kwargs):
    return _repvgg('RepVGG_A1', deploy, **kwargs)


def RepVGG_A2(deploy=False, **kwargs):
    return _repvgg('RepVGG_A2', deploy, **kwargs)


def RepVGG_B0(deploy=False, **kwargs):
    return _repvgg('RepVGG_B0', deploy, **kwargs)


def RepVGG_B1(deploy=False, **kwargs):
    return _repvgg('RepVGG_B1', deploy, **kwargs)


def RepVGG_B1g2(deploy=False, **kwargs):
    return _repvgg('RepVGG_B1g2', deploy, **kwargs)


def RepVGG_B1g4(deploy=False, **kwargs):
    return _repvgg('RepVGG_B1g4', deploy, **kwargs)


def RepVGG_B2(deploy=False, **kwargs):
    return _repvgg('RepVGG_B2', deploy, **kwargs)


def RepVGG_B2g2(deploy=False, **kwargs):
    return _repvgg('RepVGG_B2g2', deploy, **kwargs)


def RepVGG_B2g4(deploy=False, **kwargs):
    return _repvgg('RepVGG_B2g4', deploy, **kwargs)


def RepVGG_B3(deploy=False, **kwargs):
    return _repvgg('RepVGG_B3', deploy, **kwargs)


def RepVGG_B3g2(deploy=False, **kwargs):
    return _repvgg('RepVGG_B3g2', deploy, **kwargs)


def RepVGG_B3g4(deploy=False, **kwargs):
    return _repvgg('RepVGG_B3g4', deploy, **kwargs)


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

    net = RepVGG_A0(deploy=False, num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, macs: {macs}, params: {params},out_shape: {out.shape}')

    # an example to deploy repvgg trained model
    trained_model = RepVGG_A0(deploy=False)

    # # Assuming that the model has been trained, save the model
    # torch.save(trained_model.state_dict(), 'RepVGG_A0_trained.pth')

    # # load trained parameters
    # trained_model.load_state_dict(
    #     torch.load('RepVGG_A0_trained.pth', map_location=torch.device('cpu')))

    trained_model.eval()
    # define deployed model
    deployed_model = RepVGG_A0(deploy=True)
    deployed_model = deploy_model(trained_model, deployed_model)

    # torch.save(deployed_model.state_dict(), 'RepVGG_A0_deployed.pth')

    deployed_model.eval()
    inputs = torch.randn(3, 3, 224, 224)
    out1 = trained_model(inputs)
    out2 = deployed_model(inputs)
    print(out1[0][1:20], out2[0][1:20])
    print(((out1 - out2)**2).sum())  # Will be around 1e-10