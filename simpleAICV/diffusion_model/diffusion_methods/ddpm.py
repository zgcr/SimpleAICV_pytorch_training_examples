'''
Denoising Diffusion Probabilistic Model
https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddpm_mnist.ipynb
https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddpm.py
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


class DDPMTrainer(nn.Module):

    def __init__(self,
                 beta_schedule_mode='linear',
                 linear_beta_1=1e-4,
                 linear_beta_t=0.02,
                 cosine_s=0.008,
                 t=1000):
        super(DDPMTrainer, self).__init__()
        assert beta_schedule_mode in [
            'linear',
            'cosine',
            'quad',
            'sqrt_linear',
            'const',
            'jsd',
            'sigmoid',
        ]

        self.t = t

        self.beta_schedule_mode = beta_schedule_mode
        self.linear_beta_1 = linear_beta_1
        self.linear_beta_t = linear_beta_t
        self.cosine_s = cosine_s

        self.betas = compute_beta_schedule(self.beta_schedule_mode, self.t,
                                           self.linear_beta_1,
                                           self.linear_beta_t, self.cosine_s)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. -
                                                        self.alphas_cumprod)

    # forward diffusion (using the nice property): q(x_t | x_0)
    def add_noise(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t,
                                        x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy

    def forward(self, model, x_start, class_label=None):
        device = x_start.device
        t = torch.randint(0, self.t, size=(x_start.shape[0], )).to(device)
        noise = torch.randn_like(x_start).to(device)

        x_noisy = self.add_noise(x_start, t, noise)
        pred_noise = model(x_noisy, t, class_label)

        return pred_noise, noise


class DDPMSampler(nn.Module):

    def __init__(self,
                 beta_schedule_mode='linear',
                 linear_beta_1=1e-4,
                 linear_beta_t=0.02,
                 cosine_s=0.008,
                 t=1000,
                 mean_type='epsilon',
                 var_type='fixedsmall',
                 clip_denoised=True):
        super(DDPMSampler, self).__init__()
        assert beta_schedule_mode in [
            'linear',
            'cosine',
            'quad',
            'sqrt_linear',
            'const',
            'jsd',
            'sigmoid',
        ]
        assert mean_type in [
            'xprev'
            'xstart',
            'epsilon',
        ]
        assert var_type in [
            'fixedlarge',
            'fixedsmall',
        ]

        self.t = t

        self.beta_schedule_mode = beta_schedule_mode
        self.linear_beta_1 = linear_beta_1
        self.linear_beta_t = linear_beta_t
        self.cosine_s = cosine_s

        self.mean_type = mean_type
        self.var_type = var_type

        self.clip_denoised = clip_denoised

        self.update_schedule(self.beta_schedule_mode, self.t)

    def update_schedule(self, beta_schedule_mode, t):
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
        self.t = t

        self.betas = compute_beta_schedule(self.beta_schedule_mode, self.t,
                                           self.linear_beta_1,
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

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (
            1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-8))

        self.posterior_mean_coef1 = self.betas * torch.sqrt(
            self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            1.0 - self.alphas_cumprod_prev) * torch.sqrt(
                self.alphas) / (1.0 - self.alphas_cumprod)

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        # (xprev - coef2*x_t) / coef1
        return (extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1,
                        t, x_t.shape) * x_t)

    # forward diffusion (using the nice property): q(x_t | x_0)
    def add_noise(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t,
                                        x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, class_label=None):
        # predict noise using model
        pred = model(x_t, t, class_label=class_label)

        # the model predicts x_{t-1}
        if self.mean_type == 'xprev':
            x_prev = pred
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        # the model predicts x_0
        elif self.mean_type == 'xstart':
            x_0 = pred
            model_mean, _, _ = self.q_mean_variance(x_0, x_t, t)
        # the model predicts epsilon
        elif self.mean_type == 'epsilon':
            eps = pred
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _, _ = self.q_mean_variance(x_0, x_t, t)

        if self.clip_denoised:
            x_0 = torch.clamp(x_0, min=-1., max=1.)

        if self.var_type == 'fixedlarge':
            model_log_variance = torch.log(
                torch.cat([self.posterior_variance[1:2], self.betas[1:]]))
        elif self.var_type == 'fixedsmall':
            model_log_variance = self.posterior_log_variance_clipped
        model_log_variance = extract(model_log_variance, t, x_t.shape)

        return model_mean, model_log_variance

    def sample_per_time_step(self, model, x_t, t, class_label=None):
        # predict mean and variance
        model_mean, model_log_variance = self.p_mean_variance(
            model, x_t, t, class_label=class_label)

        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((1 - (t == 0).float()).view(
            -1, *([1] * (len(x_t.shape) - 1))))

        # compute x_{t-1}
        pred_img = model_mean + torch.exp(
            0.5 * model_log_variance) * noise * nonzero_mask

        return pred_img

    @torch.no_grad()
    def forward(self,
                model,
                shape,
                class_label=None,
                input_images=None,
                input_masks=None,
                return_intermediates=False,
                update_beta_schedule_mode=None,
                update_t=None):
        if update_beta_schedule_mode is not None or update_t is not None:
            if update_beta_schedule_mode is None:
                update_beta_schedule_mode = self.beta_schedule_mode
            if update_t is None:
                update_t = self.t
            self.update_schedule(beta_schedule_mode=update_beta_schedule_mode,
                                 t=update_t)

        device = next(model.parameters()).device
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]

        if input_images is None:
            # start from pure noise (for each example in the batch)
            sample_images = torch.randn((b, c, h, w)).to(device)
        else:
            sample_images = input_images

        all_step_images = []
        all_steps = list(reversed(range(0, self.t)))
        for time_step in tqdm(all_steps,
                              desc='ddpm sampler time step',
                              total=self.t):
            time = (torch.ones((b, )) * time_step).long().to(device)

            if input_masks is not None and input_images is not None:
                # input_masks:1. is mask region, 0. is keep region
                noise = torch.randn_like(sample_images).to(device)
                x_noisy = self.add_noise(sample_images, time, noise)
                sample_images = x_noisy * input_masks + (
                    1. - input_masks) * sample_images

            sample_images = self.sample_per_time_step(model,
                                                      sample_images,
                                                      time,
                                                      class_label=class_label)
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

    from tools.path import CIFAR100_path

    import torchvision.transforms as transforms

    import cv2
    import matplotlib.pyplot as plt

    from simpleAICV.diffusion_model.common import Opencv2PIL, TorchResize, TorchRandomHorizontalFlip, TorchMeanStdNormalize, ClassificationCollater
    from simpleAICV.diffusion_model.datasets.cifar100dataset import CIFAR100Dataset

    cifar100traindataset = CIFAR100Dataset(
        root_dir=CIFAR100_path,
        set_name='train',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=32),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

    ddpm_trainer = DDPMTrainer(beta_schedule_mode='linear',
                               linear_beta_1=1e-4,
                               linear_beta_t=0.02,
                               cosine_s=0.008,
                               t=1000)

    count = 0
    for per_sample in tqdm(cifar100traindataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # image = (per_sample['image'] * 0.5 + 0.5) * 255.
        # color = [random.randint(0, 255) for _ in range(3)]
        # image = np.ascontiguousarray(image, dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # label = per_sample['label']
        # text = f'label:{int(label)}'
        # cv2.putText(image,
        #             text, (30, 30),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1.5,
        #             color=color,
        #             thickness=1)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        # x_start = torch.from_numpy(per_sample['image']).permute(2, 0,
        #                                                         1).unsqueeze(0)
        # plt.figure(figsize=(16, 8))
        # for idx, t in enumerate([0, 50, 100, 200, 500, 999]):
        #     noise = torch.randn_like(x_start)
        #     x_noisy = ddpm_trainer.add_noise(x_start, torch.tensor([t]), noise)
        #     noisy_image = (x_noisy.squeeze().permute(1, 2, 0) + 1) * 127.5
        #     noisy_image = noisy_image.numpy().astype(np.uint8)
        #     plt.subplot(1, 6, 1 + idx)
        #     plt.imshow(noisy_image)
        #     # plt.axis('off')
        #     plt.title(f't={t}')
        #     output_image_name = f'idx_{count}_add_noise.png'
        #     plt.savefig(os.path.join(temp_dir, output_image_name))

        if count < 5:
            count += 1
        else:
            break

    from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=None,
                        use_gradient_checkpoint=False)
    ddpm_trainer = DDPMTrainer(beta_schedule_mode='linear',
                               linear_beta_1=1e-4,
                               linear_beta_t=0.02,
                               cosine_s=0.008,
                               t=1000)

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    train_loader = DataLoader(cifar100traindataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, labels = data['image'], data['label']
        print('1111', images.shape, labels.shape)
        print('1111', images.dtype, labels.dtype)

        pred_noise, noise = ddpm_trainer(net, images, class_label=None)
        print('2222', pred_noise.shape, noise.shape)

        if count < 5:
            count += 1
        else:
            break

    from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=100,
                        use_gradient_checkpoint=False)
    ddpm_trainer = DDPMTrainer(beta_schedule_mode='linear',
                               linear_beta_1=1e-4,
                               linear_beta_t=0.02,
                               cosine_s=0.008,
                               t=1000)

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    train_loader = DataLoader(cifar100traindataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, labels = data['image'], data['label']
        print('3333', images.shape, labels.shape)
        print('3333', images.dtype, labels.dtype)

        pred_noise, noise = ddpm_trainer(net, images, class_label=labels)
        print('4444', pred_noise.shape, noise.shape)

        if count < 5:
            count += 1
        else:
            break

    from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=None,
                        use_gradient_checkpoint=False)
    net.eval()
    ddpm_sampler = DDPMSampler(beta_schedule_mode='linear',
                               linear_beta_1=1e-4,
                               linear_beta_t=0.02,
                               cosine_s=0.008,
                               t=100,
                               mean_type='epsilon',
                               var_type='fixedsmall',
                               clip_denoised=True)
    all_step_images, last_step_images = ddpm_sampler(net, [4, 3, 32, 32],
                                                     class_label=None,
                                                     return_intermediates=True)
    print('5555', len(all_step_images), last_step_images.shape)

    cifar100testdataset = CIFAR100Dataset(
        root_dir=CIFAR100_path,
        set_name='test',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=32),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    test_loader = DataLoader(cifar100testdataset,
                             batch_size=4,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=collater)

    for data in tqdm(test_loader):
        images, labels = data['image'], data['label']
        all_step_images, last_step_images = ddpm_sampler(
            net, [4, 3, 32, 32],
            class_label=None,
            input_images=images,
            input_masks=None,
            return_intermediates=True)
        print('6666', len(all_step_images), last_step_images.shape)

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
        print('6767', w_start, h_start, w_mask_length, h_mask_length,
              b * c * w_mask_length * h_mask_length, masks.sum())
        all_step_images, last_step_images = ddpm_sampler(
            net, [4, 3, 32, 32],
            class_label=None,
            input_images=images,
            input_masks=masks,
            return_intermediates=True)
        print('6868', len(all_step_images), last_step_images.shape)

        all_step_images, last_step_images = ddpm_sampler(
            net, [4, 3, 32, 32],
            class_label=None,
            input_images=images,
            input_masks=masks,
            update_t=50,
            return_intermediates=True)
        print('6969', len(all_step_images), last_step_images.shape)

        break

    from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
    net = DiffusionUNet(inplanes=3,
                        planes=128,
                        planes_multi=[1, 2, 2, 2],
                        time_embedding_ratio=4,
                        block_nums=2,
                        dropout_prob=0.,
                        num_groups=32,
                        use_attention_planes_multi_idx=[1],
                        num_classes=100,
                        use_gradient_checkpoint=False)
    net.eval()
    ddpm_sampler = DDPMSampler(beta_schedule_mode='linear',
                               linear_beta_1=1e-4,
                               linear_beta_t=0.02,
                               cosine_s=0.008,
                               t=100,
                               mean_type='epsilon',
                               var_type='fixedsmall',
                               clip_denoised=True)

    labels = np.array([0., 1., 2., 3.]).astype(np.float32)
    labels = torch.from_numpy(labels).long()
    all_step_images, last_step_images = ddpm_sampler(net, [4, 3, 32, 32],
                                                     class_label=labels,
                                                     return_intermediates=True)
    print('7777', len(all_step_images), last_step_images.shape)

    cifar100testdataset = CIFAR100Dataset(
        root_dir=CIFAR100_path,
        set_name='test',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=32),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchMeanStdNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]))

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    test_loader = DataLoader(cifar100testdataset,
                             batch_size=4,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=collater)

    for data in tqdm(test_loader):
        images, labels = data['image'], data['label']
        all_step_images, last_step_images = ddpm_sampler(
            net, [4, 3, 32, 32],
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
        all_step_images, last_step_images = ddpm_sampler(
            net, [4, 3, 32, 32],
            class_label=labels,
            input_images=images,
            input_masks=masks,
            return_intermediates=True)
        print('7373', len(all_step_images), last_step_images.shape)

        all_step_images, last_step_images = ddpm_sampler(
            net, [4, 3, 32, 32],
            class_label=labels,
            input_images=images,
            input_masks=masks,
            update_t=50,
            return_intermediates=True)
        print('7474', len(all_step_images), last_step_images.shape)

        break