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
from simpleAICV.image_inpainting import losses

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
    trained_generator_model_path = ''
    load_state_dict(trained_generator_model_path, generator_model)

    discriminator_model = aot_gan.__dict__['AOTGANDiscriminatorModel'](**{
        'planes': [64, 128, 256, 512],
    })
    # load total pretrained model or not
    trained_discriminator_model_path = ''
    load_state_dict(trained_discriminator_model_path, discriminator_model)

    train_dataset = ImageInpaintingDataset(
        image_root_dir=CelebAHQ_path,
        mask_root_dir=NVIDIA_Irregular_Mask_Dataset_test_mask_path,
        image_set_name='train',
        mask_set_name=None,
        image_transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=input_image_size),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchColorJitter(0.05, 0.05, 0.05, 0.05),
            TorchToTensor(),
            ScaleToRange(),
        ]),
        mask_transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=input_image_size,
                        interpolation=transforms.InterpolationMode.NEAREST),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchRandomRotation(
                degrees=(0, 45),
                interpolation=transforms.InterpolationMode.NEAREST),
            TorchToTensor(),
        ]),
        mask_choice='random')
    train_collater = ImageInpaintingCollater()

    reconstruction_loss_list = [
        'L1Loss',
        'StyleLoss',
        'PerceptualLoss',
    ]
    reconstruction_loss_ratio = {
        'L1Loss': 1.0,
        'StyleLoss': 250.0,
        'PerceptualLoss': 0.1,
    }
    reconstruction_criterion = {}
    for loss_name in reconstruction_loss_list:
        reconstruction_criterion[loss_name] = losses.__dict__[loss_name]()

    adversarial_loss_list = [
        'SmganLoss',
    ]
    adversarial_loss_ratio = {
        'SmganLoss': 0.01,
    }
    adversarial_criterion = {}
    for loss_name in adversarial_loss_list:
        adversarial_criterion[loss_name] = losses.__dict__[loss_name]()

    generator_loss_list = [
        'L1Loss',
        'StyleLoss',
        'PerceptualLoss',
        'generator_loss',
    ]

    generator_optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 0,
            'no_weight_decay_layer_name_list': [],
            'beta1': 0.5,
            'beta2': 0.999,
        },
    )

    generator_scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [],
        },
    )

    discriminator_optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 0,
            'no_weight_decay_layer_name_list': [],
            'beta1': 0.5,
            'beta2': 0.999,
        },
    )

    discriminator_scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [],
        },
    )

    seed = 0
    # batch_size is total size
    batch_size = 12
    # num_workers is total workers
    num_workers = 24

    epochs = 100
    print_interval = 100
    save_epochs = []
    for i in range(epochs):
        if i % 10 == 0:
            save_epochs.append(i)

    sync_bn = False
    use_amp = False
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }