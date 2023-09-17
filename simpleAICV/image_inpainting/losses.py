import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

__all__ = [
    'L1Loss',
    'StyleLoss',
    'PerceptualLoss',
    'SmganLoss',
]


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        vgg19_model = torchvision.models.vgg19(pretrained=True)

        features = vgg19_model.features

        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])

        prefix = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        posfix = [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        names = list(zip(prefix, posfix))

        self.relus = []
        for pre, pos in names:
            self.relus.append('relu{}_{}'.format(pre, pos))
            self.__setattr__('relu{}_{}'.format(pre, pos),
                             torch.nn.Sequential())

        nums = [[0, 1], [2, 3], [4, 5, 6], [7, 8], [9, 10, 11], [12, 13],
                [14, 15], [16, 17], [18, 19, 20], [21, 22], [23, 24], [25, 26],
                [27, 28, 29], [30, 31], [32, 33], [34, 35]]

        for i, layer in enumerate(self.relus):
            for num in nums[i]:
                self.__getattr__(layer).add_module(str(num), features[num])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        device = x.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        # resize and normalize input for pretrained vgg19
        x = (x + 1.0) / 2.0
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))

        features = []
        for layer in self.relus:
            x = self.__getattr__(layer)(x)
            features.append(x)

        out = {key: value for (key, value) in list(zip(self.relus, features))}

        return out


class L1Loss(nn.Module):

    def __init__(self, ):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, preds, images):
        loss = self.loss(preds, images)

        return loss


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = VGG19()
        self.loss = nn.L1Loss()

    def compute_gram(self, x):

        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)

        return G

    def forward(self, preds, images):
        device = preds.device
        self.vgg = self.vgg.eval()
        self.vgg = self.vgg.to(device)

        preds_feature, images_feature = self.vgg(preds), self.vgg(images)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.loss(
                self.compute_gram(preds_feature[f'relu{pre}_{pos}']),
                self.compute_gram(images_feature[f'relu{pre}_{pos}']))

        return style_loss


class PerceptualLoss(nn.Module):

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = weights

    def forward(self, preds, images):
        device = preds.device
        self.vgg = self.vgg.eval()
        self.vgg = self.vgg.to(device)

        preds_feature, images_feature = self.vgg(preds), self.vgg(images)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                preds_feature[f'relu{prefix[i]}_1'],
                images_feature[f'relu{prefix[i]}_1'])

        return content_loss


class GaussianBlur(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.
    Arguments:
      kernel_size (Tuple[int, int]): the size of the kernel.
      sigma (Tuple[float, float]): the standard deviation of the kernel.
    Returns:
      Tensor: the blurred tensor.
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples::
      >>> input = torch.rand(2, 4, 5, 5)
      >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
      >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._padding = self.compute_zero_padding(kernel_size)
        self.kernel = self.get_gaussian_kernel2d(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size):
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def gaussian(self, window_size, sigma):

        def gauss_fcn(x):
            return -(x - window_size // 2)**2 / float(2 * sigma**2)

        gauss = torch.stack([
            torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def get_gaussian_kernel(self, kernel_size, sigma):
        r"""Function that returns Gaussian filter coefficients.
        Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        Returns:
        Tensor: 1D tensor with gaussian filter coefficients.
        Shape:
        - Output: :math:`(\text{kernel_size})`

        Examples::
        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])
        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
        """
        if not isinstance(kernel_size,
                          int) or kernel_size % 2 == 0 or kernel_size <= 0:
            raise TypeError(
                "kernel_size must be an odd positive integer. Got {}".format(
                    kernel_size))
        window_1d = self.gaussian(kernel_size, sigma)

        return window_1d

    def get_gaussian_kernel2d(self, kernel_size, sigma):
        r"""Function that returns Gaussian filter matrix coefficients.
        Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
            Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
            direction.
        Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

        Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

        Examples::
        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
        """
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError(
                "kernel_size must be a tuple of length two. Got {}".format(
                    kernel_size))
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError(
                "sigma must be a tuple of length two. Got {}".format(sigma))
        ksize_x, ksize_y = kernel_size
        sigma_x, sigma_y = sigma
        kernel_x = self.get_gaussian_kernel(ksize_x, sigma_x)
        kernel_y = self.get_gaussian_kernel(ksize_y, sigma_y)
        kernel_2d = torch.matmul(kernel_x.unsqueeze(-1),
                                 kernel_y.unsqueeze(-1).t())
        return kernel_2d

    def forward(self, x):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError(
                "Input x type is not a torch.Tensor. Got {}".format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError(
                "Invalid input shape, we expect BxCxHxW. Got: {}".format(
                    x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # TODO: explore solution when using jit.trace since it raises a warning
        # because the shape is converted to a tensor instead to a int.
        # convolve tensor with gaussian kernel
        return F.conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


class SmganLoss(nn.Module):

    def __init__(self, ksize=71):
        super(SmganLoss, self).__init__()
        self.gaussian_blur = GaussianBlur((ksize, ksize), (10, 10))
        self.loss = nn.MSELoss()

    def __call__(self, generator_fake, discriminator_fake, discriminator_real,
                 masks):
        device = masks.device

        _, _, h, w = generator_fake.shape
        _, _, mask_h, mask_w = masks.shape

        # Handle inconsistent size between outputs and masks
        if h != mask_h or w != mask_w:
            generator_fake = F.interpolate(generator_fake,
                                           size=(mask_h, mask_w),
                                           mode='bilinear',
                                           align_corners=True)
            discriminator_fake = F.interpolate(discriminator_fake,
                                               size=(mask_h, mask_w),
                                               mode='bilinear',
                                               align_corners=True)
            discriminator_real = F.interpolate(discriminator_real,
                                               size=(mask_h, mask_w),
                                               mode='bilinear',
                                               align_corners=True)

        discriminator_fake_label = self.gaussian_blur(masks).detach().to(
            device)
        discriminator_real_label = torch.zeros_like(discriminator_real)
        generator_fake_label = torch.ones_like(generator_fake)

        discriminator_loss = self.loss(
            discriminator_fake, discriminator_fake_label) + self.loss(
                discriminator_real, discriminator_real_label)
        discriminator_loss = discriminator_loss.mean()

        generator_loss = self.loss(
            generator_fake, generator_fake_label) * masks / torch.mean(masks)
        generator_loss = generator_loss.mean()

        return discriminator_loss, generator_loss
