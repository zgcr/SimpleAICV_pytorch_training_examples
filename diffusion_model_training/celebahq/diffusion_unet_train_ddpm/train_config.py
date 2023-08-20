import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import CelebAHQ_path

from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
from simpleAICV.diffusion_model.diffusion_methods import DDPMTrainer
from simpleAICV.diffusion_model.losses import MSELoss
from simpleAICV.diffusion_model.datasets.celebahqdataset import CelebAHQDataset
from simpleAICV.diffusion_model.common import Resize, RandomHorizontalFlip, Normalize, ClassificationCollater
from simpleAICV.classification.common import load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    num_classes = None
    input_image_size = 64
    time_step = 1000

    # if use_gradient_checkpoint, use_compile set to False
    use_gradient_checkpoint = True

    model = DiffusionUNet(inplanes=3,
                          planes=128,
                          planes_multi=[1, 2, 2, 2],
                          time_embedding_ratio=4,
                          block_nums=2,
                          dropout_prob=0.1,
                          num_groups=32,
                          use_attention_planes_multi_idx=[1],
                          num_classes=num_classes,
                          use_gradient_checkpoint=use_gradient_checkpoint)

    trainer = DDPMTrainer(beta_schedule_mode='linear',
                          linear_beta_1=1e-4,
                          linear_beta_t=0.02,
                          cosine_s=0.008,
                          t=1000)

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    train_criterion = MSELoss()

    train_dataset = CelebAHQDataset(root_dir=CelebAHQ_path,
                                    set_name='train',
                                    transform=transforms.Compose([
                                        Resize(resize=input_image_size),
                                        RandomHorizontalFlip(prob=0.5),
                                        Normalize(),
                                    ]))
    train_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 512
    # num_workers is total workers
    num_workers = 30
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 4e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-4,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
            'min_lr': 1e-6,
        },
    )

    epochs = 1000
    print_interval = 10

    sync_bn = False
    use_amp = True
    # if use_compile,use_gradient_checkpoint set to False
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }

    use_ema_model = False
    ema_model_decay = 0.9999
