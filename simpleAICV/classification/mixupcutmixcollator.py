"""
origin code: https://github.com/rwightman/pytorch-image-models/blob/ef72ad417709b5ba6404d85d3adafd830d507b2a/timm/data/mixup.py
mixup: Beyond Empirical Risk Minimization 
papar: https://arxiv.org/abs/1710.09412
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
paper: https://arxiv.org/abs/1905.04899
code:  https://github.com/clovaai/CutMix-PyTorch
"""
import numpy as np

import torch


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)

    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.
    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.
    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]),
                              int(img_h * minmax[1]),
                              size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]),
                              int(img_w * minmax[1]),
                              size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w

    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape,
                        lam,
                        ratio_minmax=None,
                        correct_lambda=True,
                        count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lambda or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])

    return (yl, yu, xl, xu), lam


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)

    return torch.full((x.size()[0], num_classes), off_value,
                      device=device).scatter_(1, x, on_value)


def mixup_label(labels, num_classes, lam=1., smoothing=0.0):
    device = labels.device
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value

    y1 = one_hot(labels,
                 num_classes,
                 on_value=on_value,
                 off_value=off_value,
                 device=device)
    y2 = one_hot(labels.flip(0),
                 num_classes,
                 on_value=on_value,
                 off_value=off_value,
                 device=device)

    return y1 * lam + y2 * (1. - lam)


class MixupCutmixClassificationCollater:
    """ 
    Mixup/Cutmix that applies different params to each element or whole batch
    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        mix_prob (float): probability of applying mixup or cutmix per batch or element
        mixup_switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lambda (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self,
                 use_mixup=True,
                 mixup_alpha=0.8,
                 cutmix_alpha=1.0,
                 cutmix_minmax=None,
                 mix_prob=1.0,
                 mixup_switch_prob=0.5,
                 mode='batch',
                 correct_lambda=True,
                 label_smoothing=0.1,
                 num_classes=1000):
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = mix_prob
        self.mixup_switch_prob = mixup_switch_prob
        assert mode in ['elem', 'half', 'pair', 'batch']
        self.mode = mode
        # correct lambda based on clipped area for cutmix
        self.correct_lambda = correct_lambda
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def __call__(self, data):
        images = [s['image'] for s in data]
        labels = [s['label'] for s in data]

        images = np.array(images).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        # B H W 3 ->B 3 H W
        images = np.transpose(images, (0, 3, 1, 2))

        b, c, h, w = images.shape
        assert b % 2 == 0, 'Batch size should be even when using this'

        half = True if self.mode == 'half' else False
        if half:
            b //= 2

        labels = torch.from_numpy(labels).long()
        device = labels.device
        mixed_images = torch.zeros((b, c, h, w),
                                   dtype=torch.float32).to(device)

        if self.mode in ['elem', 'half']:
            lamada = self._mix_elem_collate(mixed_images, images, half=half)
        elif self.mode in ['pair']:
            lamada = self._mix_pair_collate(
                mixed_images,
                images,
            )
        elif self.mode in ['batch']:
            lamada = self._mix_batch_collate(
                mixed_images,
                images,
            )

        labels = mixup_label(labels, self.num_classes, lamada,
                             self.label_smoothing)
        labels = labels[:b]

        return {
            'image': mixed_images,
            'label': labels,
        }

    def _params_per_elem(self, batch_size):
        batch_lambda = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)

        if self.use_mixup:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(
                    batch_size) < self.mixup_switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha,
                                   self.cutmix_alpha,
                                   size=batch_size),
                    np.random.beta(self.mixup_alpha,
                                   self.mixup_alpha,
                                   size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha,
                                         self.mixup_alpha,
                                         size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha,
                                         self.cutmix_alpha,
                                         size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            batch_lambda = np.where(
                np.random.rand(batch_size) < self.mix_prob,
                lam_mix.astype(np.float32), batch_lambda)

        return batch_lambda, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.use_mixup and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.mixup_switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)

        return lam, use_cutmix

    def _mix_elem_collate(self, mixed_images, images, half=False):
        b = images.shape[0]
        element_nums = b // 2 if half else b
        assert len(mixed_images) == element_nums

        batch_lambda, use_cutmix = self._params_per_elem(element_nums)

        for i in range(element_nums):
            j = b - i - 1
            lam = batch_lambda[i]
            mixed = images[i]
            if lam != 1.:
                if use_cutmix[i]:
                    if not half:
                        mixed = mixed.copy()
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        mixed_images.shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lambda=self.correct_lambda)
                    mixed[:, yl:yh, xl:xh] = images[j][:, yl:yh, xl:xh]
                    batch_lambda[i] = lam
                else:
                    mixed = mixed.astype(np.float32) * lam + images[j].astype(
                        np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            mixed_images[i] += torch.from_numpy(mixed.astype(np.uint8))
        if half:
            batch_lambda = np.concatenate(
                (batch_lambda, np.ones(element_nums)))

        return torch.tensor(batch_lambda).unsqueeze(1)

    def _mix_pair_collate(self, mixed_images, images):
        b = images.shape[0]

        batch_lambda, use_cutmix = self._params_per_elem(b // 2)

        for i in range(b // 2):
            j = b - i - 1
            lam = batch_lambda[i]
            mixed_i = images[i]
            mixed_j = images[j]
            assert 0 <= lam <= 1.0
            if lam < 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        mixed_images.shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lambda=self.correct_lambda)
                    patch_i = mixed_i[:, yl:yh, xl:xh].copy()
                    mixed_i[:, yl:yh, xl:xh] = mixed_j[:, yl:yh, xl:xh]
                    mixed_j[:, yl:yh, xl:xh] = patch_i
                    batch_lambda[i] = lam
                else:
                    mixed_temp = mixed_i.astype(
                        np.float32) * lam + mixed_j.astype(
                            np.float32) * (1 - lam)
                    mixed_j = mixed_j.astype(
                        np.float32) * lam + mixed_i.astype(
                            np.float32) * (1 - lam)
                    mixed_i = mixed_temp
                    np.rint(mixed_j, out=mixed_j)
                    np.rint(mixed_i, out=mixed_i)
            mixed_images[i] += torch.from_numpy(mixed_i.astype(np.uint8))
            mixed_images[j] += torch.from_numpy(mixed_j.astype(np.uint8))
        batch_lambda = np.concatenate((batch_lambda, batch_lambda[::-1]))

        return torch.tensor(batch_lambda).unsqueeze(1)

    def _mix_batch_collate(self, mixed_images, images):
        b = images.shape[0]
        lam, use_cutmix = self._params_per_batch()
        if use_cutmix:
            (yl, yh, xl,
             xh), lam = cutmix_bbox_and_lam(mixed_images.shape,
                                            lam,
                                            ratio_minmax=self.cutmix_minmax,
                                            correct_lambda=self.correct_lambda)
        for i in range(b):
            j = b - i - 1
            mixed = images[i]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy(
                    )  # don't want to modify the original while iterating
                    mixed[:, yl:yh, xl:xh] = images[j][:, yl:yh, xl:xh]
                else:
                    mixed = mixed.astype(np.float32) * lam + images[j].astype(
                        np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            mixed_images[i] += torch.from_numpy(mixed.astype(np.uint8))

        return lam