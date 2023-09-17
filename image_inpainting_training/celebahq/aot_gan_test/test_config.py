import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import CelebAHQ_path, NVIDIA_Irregular_Mask_Dataset_test_mask_path

from simpleAICV.image_inpainting.datasets.imageinpaintingdataset import ImageInpaintingDataset
from simpleAICV.image_inpainting.common import Opencv2PIL, TorchResize, TorchRandomResizedCrop, TorchRandomHorizontalFlip, TorchColorJitter, TorchRandomRotation, TorchToTensor, ScaleToRange, ImageInpaintingCollater, load_state_dict
from simpleAICV.image_inpainting.models import aot_gan
from simpleAICV.image_inpainting.metrics.inception import InceptionV3

import torch
import torchvision.transforms as transforms


class config:
    input_image_size = 512

    generator_model = aot_gan.__dict__['AOTGANGeneratorModel'](
        **{
            'planes': [64, 128, 256],
            'rates': [1, 2, 4, 8],
            'block_num': 8,
        })
    # load total pretrained model or not
    trained_generator_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/aot_gan_train_from_scratch_on_celebahq/aot_gan_total_loss0.226_generator_model.pth'
    load_state_dict(trained_generator_model_path, generator_model)

    test_dataset_name_list = [
        '0.01-0.1',
        '0.1-0.2',
        '0.2-0.3',
        '0.3-0.4',
        '0.4-0.5',
        '0.5-0.6',
    ]
    test_dataset_list = []
    for per_sub_test_dataset_name in test_dataset_name_list:
        per_sub_test_dataset = ImageInpaintingDataset(
            image_root_dir=CelebAHQ_path,
            mask_root_dir=NVIDIA_Irregular_Mask_Dataset_test_mask_path,
            image_set_name='val',
            mask_set_name=per_sub_test_dataset_name,
            image_transform=transforms.Compose([
                Opencv2PIL(),
                TorchResize(resize=input_image_size),
                TorchToTensor(),
                ScaleToRange(),
            ]),
            mask_transform=transforms.Compose([
                Opencv2PIL(),
                TorchResize(
                    resize=input_image_size,
                    interpolation=transforms.InterpolationMode.NEAREST),
                TorchToTensor(),
            ]),
            mask_choice='inorder')
        test_dataset_list.append(per_sub_test_dataset)
    test_collater = ImageInpaintingCollater()

    # inception v3 weight https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
    fid_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/inception_v3_pytorch_weights/pt_inception-2015-12-05-6726825d.pth'
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = InceptionV3(output_blocks=[block_idx1],
                            saved_model_path=fid_model_path)

    fid_model_batch_size = 256
    fid_model_num_workers = 4

    seed = 0
    # batch_size is total size
    batch_size = 64
    # num_workers is total workers
    num_workers = 4

    save_image_dir = ''
