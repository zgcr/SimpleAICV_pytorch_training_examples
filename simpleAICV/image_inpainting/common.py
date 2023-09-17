import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class Opencv2PIL:

    def __init__(self):
        pass

    def __call__(self, image):
        image = Image.fromarray(np.uint8(image))

        return image


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, image):
        image = np.asarray(image).astype(np.float32)

        return image


class TorchRandomResizedCrop:

    def __init__(self,
                 resize=224,
                 scale=(0.08, 1.0),
                 ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.RandomResizedCrop = transforms.RandomResizedCrop(int(resize),
                                                              scale=scale,
                                                              ratio=ratio)

    def __call__(self, image):
        image = self.RandomResizedCrop(image)

        return image


class TorchResize:

    def __init__(self,
                 resize=224,
                 interpolation=transforms.InterpolationMode.BILINEAR):
        self.Resize = transforms.Resize(size=[int(resize),
                                              int(resize)],
                                        interpolation=interpolation)

    def __call__(self, image):
        image = self.Resize(image)

        return image


class TorchRandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.RandomHorizontalFlip = transforms.RandomHorizontalFlip(prob)

    def __call__(self, image):
        image = self.RandomHorizontalFlip(image)

        return image


class TorchColorJitter:

    def __init__(self,
                 brightness=0.05,
                 contrast=0.05,
                 saturation=0.05,
                 hue=0.05):
        self.ColorJitter = transforms.ColorJitter(brightness=brightness,
                                                  contrast=contrast,
                                                  saturation=saturation,
                                                  hue=hue)

    def __call__(self, image):
        image = self.ColorJitter(image)

        return image


class TorchRandomRotation:

    def __init__(self,
                 degrees=(0, 45),
                 interpolation=transforms.InterpolationMode.NEAREST):
        self.rotation = transforms.RandomRotation(degrees=degrees,
                                                  interpolation=interpolation)

    def __call__(self, image):
        image = self.rotation(image)

        return image


class TorchToTensor:

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        image = self.to_tensor(image)
        image = image.numpy()

        return image


class ScaleToRange:

    def __init__(self):
        pass

    def __call__(self, image):
        image = image * 2. - 1.

        return image


class ImageInpaintingCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['mask'] for s in data]

        images = np.array(images).astype(np.float32)
        masks = np.array(masks).astype(np.float32)

        images = torch.from_numpy(images).float()
        masks = torch.from_numpy(masks).float()

        return {
            'image': images,
            'mask': masks,
        }


class AverageMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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