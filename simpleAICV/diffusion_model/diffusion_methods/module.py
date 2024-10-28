import math

import torch


def extract(v, t, x_shape):
    '''
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    '''
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def compute_beta_schedule(beta_schedule_mode, t, linear_beta_1, linear_beta_t,
                          cosine_s):
    assert beta_schedule_mode in [
        'linear',
        'cosine',
        'quad',
        'sqrt_linear',
        'const',
        'jsd',
        'sigmoid',
    ]

    if beta_schedule_mode == 'linear':
        betas = torch.linspace(linear_beta_1,
                               linear_beta_t,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)
    elif beta_schedule_mode == 'cosine':
        x = torch.arange(t + 1, requires_grad=False, dtype=torch.float64)
        alphas_cumprod = torch.cos(
            ((x / t) + cosine_s) / (1 + cosine_s) * math.pi * 0.5)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule_mode == 'quad':
        betas = (torch.linspace(linear_beta_1**0.5,
                                linear_beta_t**0.5,
                                t,
                                requires_grad=False,
                                dtype=torch.float64)**2)
    elif beta_schedule_mode == 'sqrt_linear':
        betas = torch.linspace(linear_beta_1,
                               linear_beta_t,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)**0.5
    elif beta_schedule_mode == 'const':
        betas = linear_beta_t * torch.ones(
            t, requires_grad=False, dtype=torch.float64)
    elif beta_schedule_mode == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / torch.linspace(
            t, 1, t, requires_grad=False, dtype=torch.float64)
    elif beta_schedule_mode == 'sigmoid':
        betas = torch.linspace(-6,
                               6,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)
        betas = torch.sigmoid(betas) * (linear_beta_t -
                                        linear_beta_1) + linear_beta_1

    return betas
