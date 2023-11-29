import os
import cv2
import json
import numpy as np

from PIL import Image

from tqdm import tqdm

from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class Opencv2PIL:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        path, image = sample['path'], sample['image']

        image = Image.fromarray(np.uint8(image))

        return {
            'path': path,
            'image': image,
        }


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        path, image = sample['path'], sample['image']

        image = np.asarray(image).astype(np.float32)

        return {
            'path': path,
            'image': image,
        }


class TorchResize:

    def __init__(self, resize=224):
        self.Resize = transforms.Resize(int(resize))

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        path, image = sample['path'], sample['image']

        image = self.Resize(image)

        return {
            'path': path,
            'image': image,
        }


class TorchCenterCrop:

    def __init__(self, resize=224):
        self.CenterCrop = transforms.CenterCrop(int(resize))

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        path, image = sample['path'], sample['image']

        image = self.CenterCrop(image)

        return {
            'path': path,
            'image': image,
        }


class TorchMeanStdNormalize:

    def __init__(self, mean, std):
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'label' keys.
        '''
        path, image = sample['path'], sample['image']
        image = self.to_tensor(image)
        image = self.Normalize(image)
        # 3 H W ->H W 3
        image = image.permute(1, 2, 0)
        image = image.numpy()

        return {
            'path': path,
            'image': image,
        }


class ClassificationCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        paths = [s['path'] for s in data]
        images = [s['image'] for s in data]

        images = np.array(images).astype(np.float32)

        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)

        return {
            'path': paths,
            'image': images,
        }


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


class ACCV2022TestaDataset(Dataset):
    '''
    ACCV2022 Dataset:https://www.cvmart.net/race/10412/des
    '''

    def __init__(self,
                 root_dir,
                 set_name='testa',
                 transform=None,
                 broken_list_path=None):
        assert set_name in ['testa'], 'Wrong set name!'
        set_dir = os.path.join(root_dir, set_name)

        broken_list = set()
        if broken_list_path:
            with open(broken_list_path, 'r') as load_f:
                broken_list = json.load(load_f)
                broken_list = set(broken_list)
        print(f'Broken image num:{len(broken_list)}')

        self.image_path_list = []
        for per_image_name in tqdm(os.listdir(set_dir)):
            per_image_path = os.path.join(set_dir, per_image_name)
            if per_image_name in broken_list:
                continue
            self.image_path_list.append(per_image_path)

        self.transform = transform

        print(f'Dataset Size:{len(self.image_path_list)}')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        path = self.image_path_list[idx]
        image = self.load_image(idx)

        sample = {
            'path': path,
            'image': image,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)


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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import accv2022_dataset_path, accv2022_broken_list_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    accv2022testadataset = ACCV2022TestaDataset(
        root_dir=accv2022_dataset_path,
        set_name='testa',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=256),
            TorchCenterCrop(resize=224),
            PIL2Opencv(),
            # TorchMeanStdNormalize(mean=[0.485, 0.456, 0.406],
            #                       std=[0.229, 0.224, 0.225]),
        ]),
        broken_list_path=accv2022_broken_list_path)

    count = 0
    for per_sample in tqdm(accv2022testadataset):
        print(per_sample['image'].shape, type(per_sample['image']),
              per_sample['path'])

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # color = [random.randint(0, 255) for _ in range(3)]
        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image_name = per_sample['path'].split('/')[-1]
        # text = f'image_name:{image_name}'
        # cv2.putText(image,
        #             text, (30, 30),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1.5,
        #             color=color,
        #             thickness=1)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    train_loader = DataLoader(accv2022testadataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        paths, images = data['path'], data['image']
        print(images.shape)
        print(images.dtype)
        if count < 5:
            count += 1
        else:
            break
