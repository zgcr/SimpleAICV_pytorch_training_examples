'''
Denoising Diffusion Implicit Models
https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddim_mnist.ipynb
https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddim.py
'''
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

from simpleAICV.diffusion_model.diffusion_methods.module import extract, compute_beta_schedule


class DDIMSampler(nn.Module):

    def __init__(self,
                 beta_schedule_mode='linear',
                 linear_beta_1=1e-4,
                 linear_beta_t=0.02,
                 cosine_s=0.008,
                 ddpm_t=1000,
                 ddim_t=50,
                 ddim_eta=0.0,
                 ddim_discr_method='uniform',
                 clip_denoised=True):
        super(DDIMSampler, self).__init__()
        assert beta_schedule_mode in [
            'linear',
            'cosine',
            'quad',
            'sqrt_linear',
            'const',
            'jsd',
            'sigmoid',
        ]
        assert ddim_discr_method in ['uniform', 'quad']

        # ddpm steps
        self.ddpm_t = ddpm_t

        self.beta_schedule_mode = beta_schedule_mode
        self.linear_beta_1 = linear_beta_1
        self.linear_beta_t = linear_beta_t
        self.cosine_s = cosine_s

        self.ddim_t = ddim_t
        self.ddim_eta = ddim_eta
        self.ddim_discr_method = ddim_discr_method

        self.clip_denoised = clip_denoised

        self.update_schedule(self.beta_schedule_mode, self.ddpm_t, self.ddim_t,
                             self.ddim_eta, self.ddim_discr_method)

    def update_schedule(self, beta_schedule_mode, ddpm_t, ddim_t, ddim_eta,
                        ddim_discr_method):
        assert beta_schedule_mode in [
            'linear',
            'cosine',
            'quad',
            'sqrt_linear',
            'const',
            'jsd',
            'sigmoid',
        ]

        self.beta_schedule_mode = beta_schedule_mode
        self.ddpm_t = ddpm_t
        self.ddim_t = ddim_t
        self.ddim_eta = ddim_eta
        self.ddim_discr_method = ddim_discr_method

        self.betas = compute_beta_schedule(self.beta_schedule_mode,
                                           self.ddpm_t, self.linear_beta_1,
                                           self.linear_beta_t, self.cosine_s)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1), self.alphas_cumprod[:-1]], dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 -
                                                        self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 -
                                                      self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 /
                                                      self.alphas_cumprod - 1)

        assert ddim_discr_method in ['uniform', 'quad']

        if ddim_discr_method == 'uniform':
            c = self.ddpm_t // self.ddim_t
            self.ddim_timesteps = np.asarray(list(range(0, self.ddpm_t, c)))
        elif ddim_discr_method == 'quad':
            self.ddim_timesteps = ((np.linspace(0, np.sqrt(self.ddpm_t * 0.8),
                                                self.ddim_t))**2).astype(int)

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        self.ddim_timesteps = self.ddim_timesteps + 1

        self.ddim_alphas = self.alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_prev = torch.tensor(
            np.asarray([self.alphas_cumprod[0]] +
                       self.alphas_cumprod[self.ddim_timesteps[:-1]].tolist()),
            requires_grad=False)
        self.ddim_sigmas = self.ddim_eta * torch.sqrt(
            (1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) *
            (1 - self.ddim_alphas / self.ddim_alphas_prev))
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas)

    # forward diffusion (using the nice property): q(x_t | x_0)
    def add_noise(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t,
                                        x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy

    @torch.no_grad()
    def forward(self,
                model,
                shape,
                class_label=None,
                input_images=None,
                input_masks=None,
                return_intermediates=False,
                update_beta_schedule_mode=None,
                update_ddpm_t=None,
                update_ddim_t=None,
                update_ddim_eta=None,
                update_ddim_discr_method=None):
        if update_beta_schedule_mode is not None or update_ddpm_t is not None or update_ddim_t is not None or update_ddim_eta is not None or update_ddim_discr_method is not None:
            if update_beta_schedule_mode is None:
                update_beta_schedule_mode = self.beta_schedule_mode
            if update_ddpm_t is None:
                update_ddpm_t = self.ddpm_t
            if update_ddim_t is None:
                update_ddim_t = self.ddim_t
            if update_ddim_eta is None:
                update_ddim_eta = self.ddim_eta
            if update_ddim_discr_method is None:
                update_ddim_discr_method = self.ddim_discr_method

            self.update_schedule(beta_schedule_mode=update_beta_schedule_mode,
                                 ddpm_t=update_ddpm_t,
                                 ddim_t=update_ddim_t,
                                 ddim_eta=update_ddim_eta,
                                 ddim_discr_method=update_ddim_discr_method)

        device = next(model.parameters()).device
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]

        if input_images is None:
            # start from pure noise (for each example in the batch)
            sample_images = torch.randn((b, c, h, w)).to(device)
        else:
            sample_images = input_images

        all_step_images = []
        all_steps = list(reversed(self.ddim_timesteps))
        for idx, time_step in enumerate(
                tqdm(all_steps,
                     desc='ddim sampler time step',
                     total=len(all_steps))):
            ddim_step_index_before_flip = len(all_steps) - idx - 1

            time = (torch.ones((b, )) * time_step).long().to(device)

            if input_masks is not None and input_images is not None:
                # input_masks:1. is mask region, 0. is keep region
                noise = torch.randn_like(sample_images).to(device)
                x_noisy = self.add_noise(sample_images, time, noise)
                sample_images = x_noisy * input_masks + (
                    1. - input_masks) * sample_images

            # predict noise using model
            pred_noise = model(sample_images, time, class_label=class_label)

            a_t = (torch.ones((b, 1, 1, 1)) *
                   self.ddim_alphas[ddim_step_index_before_flip]).to(device)
            a_prev_t = (
                torch.ones((b, 1, 1, 1)) *
                self.ddim_alphas_prev[ddim_step_index_before_flip]).to(device)
            sigma_t = (
                torch.ones((b, 1, 1, 1)) *
                self.ddim_sigmas[ddim_step_index_before_flip]).to(device)
            sqrt_one_minus_at = (
                torch.ones((b, 1, 1, 1)) *
                self.ddim_sqrt_one_minus_alphas[ddim_step_index_before_flip]
            ).to(device)

            # current prediction for x_0
            pred_x0 = (sample_images -
                       sqrt_one_minus_at * pred_noise) / torch.sqrt(a_t)

            if self.clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

            # direction pointing to x_t
            pred_dir_xt = torch.sqrt(1. - a_prev_t - sigma_t**2) * pred_noise
            noise = sigma_t * torch.randn_like(sample_images)

            x_prev = torch.sqrt(a_prev_t) * pred_x0 + pred_dir_xt + noise

            sample_images = x_prev

            all_step_images.append(sample_images.cpu().numpy())

        last_step_images = all_step_images[-1]

        if return_intermediates:
            return all_step_images, last_step_images
        else:
            return all_step_images


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

    from tools.path import CIFAR10_path

    import torchvision.transforms as transforms

    from simpleAICV.classification.datasets.cifar10dataset import CIFAR10Dataset
    from simpleAICV.diffusion_model.common import Opencv2PIL, TorchResize, TorchRandomHorizontalFlip, TorchMeanStdNormalize, DiffusionWithLabelCollater

    from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[0, 1, 2, 3],
                        num_classes=None,
                        use_gradient_checkpoint=False)
    net.eval()
    ddim_sampler = DDIMSampler(beta_schedule_mode='linear',
                               linear_beta_1=1e-4,
                               linear_beta_t=0.02,
                               cosine_s=0.008,
                               ddpm_t=1000,
                               ddim_t=50,
                               ddim_eta=0.0,
                               ddim_discr_method='uniform',
                               clip_denoised=True)
    all_step_images, last_step_images = ddim_sampler(net, [8, 3, 32, 32],
                                                     class_label=None,
                                                     return_intermediates=True)
    print('5555', len(all_step_images), last_step_images.shape)

    from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[0, 1, 2, 3],
                        num_classes=100,
                        use_gradient_checkpoint=False)
    net.eval()
    ddim_sampler = DDIMSampler(beta_schedule_mode='linear',
                               linear_beta_1=1e-4,
                               linear_beta_t=0.02,
                               cosine_s=0.008,
                               ddpm_t=1000,
                               ddim_t=50,
                               ddim_eta=0.0,
                               ddim_discr_method='uniform',
                               clip_denoised=True)

    labels = np.array([0., 1., 2., 3., 4., 5., 6., 7.]).astype(np.float32)
    labels = torch.from_numpy(labels).long()
    all_step_images, last_step_images = ddim_sampler(net, [8, 3, 32, 32],
                                                     class_label=labels,
                                                     return_intermediates=True)
    print('6666', len(all_step_images), last_step_images.shape)

    cifar100testdataset = CIFAR10Dataset(
        root_dir=CIFAR10_path,
        set_name='test',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=32),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

    from torch.utils.data import DataLoader
    collater = DiffusionWithLabelCollater()
    test_loader = DataLoader(cifar100testdataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=collater)

    for data in tqdm(test_loader):
        images, labels = data['image'], data['label']
        all_step_images, last_step_images = ddim_sampler(
            net, [8, 3, 32, 32],
            class_label=labels,
            input_images=images,
            input_masks=None,
            return_intermediates=True)
        print('7171', len(all_step_images), last_step_images.shape)

        device = images.device
        b, c, h, w = images.shape[0], images.shape[1], images.shape[
            2], images.shape[3]
        masks = torch.zeros((b, c, h, w)).to(device)
        h_start, w_start = np.random.randint(1, h - 1), np.random.randint(
            1, w - 1)
        remain_h, remain_w = h - h_start, w - w_start
        h_mask_length, w_mask_length = np.random.randint(
            1, remain_h), np.random.randint(1, remain_w)
        masks[:, :, h_start:h_start + h_mask_length,
              w_start:w_start + w_mask_length] = 1.
        print('7272', w_start, h_start, w_mask_length, h_mask_length,
              b * c * w_mask_length * h_mask_length, masks.sum())
        all_step_images, last_step_images = ddim_sampler(
            net, [8, 3, 32, 32],
            class_label=labels,
            input_images=images,
            input_masks=masks,
            return_intermediates=True)
        print('7373', len(all_step_images), last_step_images.shape)

        all_step_images, last_step_images = ddim_sampler(
            net, [8, 3, 32, 32],
            class_label=labels,
            input_images=images,
            input_masks=masks,
            update_ddim_t=20,
            return_intermediates=True)
        print('7474', len(all_step_images), last_step_images.shape)

        break
