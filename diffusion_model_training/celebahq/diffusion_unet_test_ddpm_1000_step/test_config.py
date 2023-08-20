import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import CelebAHQ_path

from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
from simpleAICV.diffusion_model.diffusion_methods import DDPMSampler
from simpleAICV.diffusion_model.datasets.celebahqdataset import CelebAHQDataset
from simpleAICV.diffusion_model.common import Resize, RandomHorizontalFlip, Normalize, ClassificationCollater
from simpleAICV.classification.common import load_state_dict
from simpleAICV.diffusion_model.metrics.inception import InceptionV3

import torch
import torchvision.transforms as transforms


class config:
    num_classes = None
    input_image_size = 64
    time_step = 1000

    model = DiffusionUNet(inplanes=3,
                          planes=128,
                          planes_multi=[1, 2, 2, 2],
                          time_embedding_ratio=4,
                          block_nums=2,
                          dropout_prob=0.1,
                          num_groups=32,
                          use_attention_planes_multi_idx=[1],
                          num_classes=num_classes,
                          use_gradient_checkpoint=False)

    sampler = DDPMSampler(beta_schedule_mode='linear',
                          linear_beta_1=1e-4,
                          linear_beta_t=0.02,
                          cosine_s=0.008,
                          t=1000,
                          mean_type='epsilon',
                          var_type='fixedsmall',
                          clip_denoised=True)

    # load pretrained model or not
    trained_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/diffusion_model_training/celebahq/diffusion_unet_train_ddpm/checkpoints/loss0.013.pth'
    load_state_dict(trained_model_path, model)

    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    test_dataset = CelebAHQDataset(root_dir=CelebAHQ_path,
                                   set_name='train',
                                   transform=transforms.Compose([
                                       Resize(resize=input_image_size),
                                       Normalize(),
                                   ]))
    test_collater = ClassificationCollater()

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 8

    use_condition_label = False
    use_input_images = False

    # inception v3 weight https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
    fid_model_path = '/root/code/SimpleAICV_pytorch_training_examples_on_ImageNet_COCO_ADE20K/pretrained_models/inception_v3_pytorch_weights/pt_inception-2015-12-05-6726825d.pth'
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    fid_model = InceptionV3(output_blocks=[block_idx1, block_idx2],
                            saved_model_path=fid_model_path)

    save_image_dir = ''

    fid_model_batch_size = 256
    fid_model_num_workers = 8

    is_data_split_num = 10
