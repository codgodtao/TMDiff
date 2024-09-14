import os

import math
import torchvision
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from typing import Tuple, List

from tqdm import tqdm

eps = torch.finfo(torch.float32).eps

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


def data_normalize(img_dict, bit_depth):
    """ Normalize the data to [0, 1)

    Args:
        img_dict (dict[str, torch.Tensor]): images in torch.Tensor
        bit_depth (int): original data range in n-bit
    Returns:
        dict[str, torch.Tensor]: images after normalization
    """
    max_value = 2 ** bit_depth - .5
    ret = dict()
    for img_name in img_dict:
        if img_name == 'image_id':
            ret[img_name] = img_dict[img_name]
            continue
        imgs = img_dict[img_name]
        ret[img_name] = imgs / max_value
    return ret


def data_denormalize(img, bit_depth):
    """ Denormalize the data to [0, n-bit)

    Args:
        img (torch.Tensor | np.ndarray): images in torch.Tensor
        bit_depth (int): original data range in n-bit
    Returns:
        dict[str, torch.Tensor]: image after denormalize
    """
    max_value = 2 ** bit_depth - .5
    ret = img * max_value
    return ret


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img

import cv2

# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()


def get_edge(data):
    # NxHxW*C  or N*H*W
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


def res2img(img_, img_lr_up):
    img_ = img_ + img_lr_up  # res+lms = GT，模型sample的结果理论上是[-1,1]，然后加上lms [0,1] 应该变为[0,1]
    return img_


def img2res(x, img_lr_up):
    x = x - img_lr_up  # [1,0] - [1,0] = [-1,1]  GT-LMS [-1,1],模型拟合的目标结果
    return x


# def

def transform_augment(img_list):#图像转为正态分布[-1,1]
    ret_img = [img * 2.0 - 1.0 for img in img_list]
    return ret_img


def reverse_transform_augment(img_list):#图像转为[0,1]
    ret_img = [(img + 1.0) / 2.0 for img in img_list]
    return ret_img



def get_data_generator(loader, enable_tqdm, desc):
    while True:
        for data in tqdm(loader, disable=not enable_tqdm, desc=desc):
            yield data


def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [(k - 1) // 2 for k in kernel_size]
    return [computed[1], computed[1], computed[0], computed[0]]


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input.device).to(input.dtype)
    tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel
    return F.conv2d(input_pad, tmp_kernel, padding=0, stride=1, groups=c)


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def get_box_kernel2d(kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel


class BoxBlur(nn.Module):
    r"""Blurs an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 border_type: str = 'reflect',
                 normalized: bool = True) -> None:
        super(BoxBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.border_type: str = border_type
        self.kernel: torch.Tensor = get_box_kernel2d(kernel_size)
        self.normalized: bool = normalized
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               '(kernel_size=' + str(self.kernel_size) + ', ' + \
               'normalized=' + str(self.normalized) + ', ' + \
               'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor):  # type: ignore
        return filter2D(input, self.kernel, self.border_type)


# functiona api
def box_blur(input: torch.Tensor,
             kernel_size: Tuple[int, int],
             border_type: str = 'reflect',
             normalized: bool = True) -> torch.Tensor:
    r"""Blurs an image using the box filter.

    See :class:`~kornia.filters.BoxBlur` for details.
    """
    return BoxBlur(kernel_size, border_type, normalized)(input)


class PSNRLoss(nn.Module):
    r"""Creates a criterion that calculates the PSNR between 2 images. Given an m x n image,
    .. math::
    \text{MSE}(I,T) = \frac{1}{m\,n}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    Arguments:
        max_val (float): Maximum value of input

    Shape:
        - input: :math:`(*)`
        - approximation: :math:`(*)` same shape as input
        - output: :math:`()` a scalar

    Examples:
        >>> kornia.losses.psnr(torch.ones(1), 1.2*torch.ones(1), 2)
        tensor(20.0000) # 10 * log(4/((1.2-1)**2)) / log(10)

    reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """

    def __init__(self, max_val: float) -> None:
        super(PSNRLoss, self).__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        return psnr_loss(input, target, self.max_val)


def ergas(img_fake, img_real, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""

    N, C, H, W = img_real.shape
    means_real = img_real.reshape(N, C, -1).mean(dim=-1)
    mses = ((img_fake - img_real) ** 2).reshape(N, C, -1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS

    return 100 / scale * torch.sqrt((mses / (means_real ** 2 + eps)).mean())


def get_metrics_reduced(img1, img2):
    # input: img1 {the pan-sharpened image}, img2 {the ground-truth image}
    # return: (larger better) psnr, ssim, scc, (smaller better) sam, ergas
    m1 = psnr_loss(img1, img2, 1.)
    m2 = ssim(img1, img2, 5, 'mean', 1.)
    m3 = cc(img1, img2)
    m4 = sam(img1, img2)
    m5 = ergas(img1, img2)
    return [m1.item(), m2.item(), m3.item(), m4.item(), m5.item()]


def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
            eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Function that computes PSNR

    See :class:`~kornia.losses.PSNR` for details.
    """
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")
    mse_val = mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input.device).to(input.dtype)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)


def _compute_zero_padding(kernel_size: int) -> int:
    """Computes zero padding."""
    return (kernel_size - 1) // 2


def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

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
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

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
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d


class SSIM(nn.Module):
    r"""Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    Examples::

        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = kornia.losses.SSIM(5, reduction='none')
        >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(
            self,
            window_size: int,
            reduction: str = "none",
            max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.window = self.window.requires_grad_(False)  # need to disable gradients

        self.padding: int = _compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    def forward(  # type: ignore
            self,
            img1: torch.Tensor,
            img2: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                            .format(type(img1)))

        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                            .format(type(img2)))

        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                             .format(img1.shape))

        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                             .format(img2.shape))

        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}"
                             .format(img1.shape, img2.shape))

        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}"
                             .format(img1.device, img2.device))

        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}"
                             .format(img1.dtype, img2.dtype))

        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)

        # compute local mean per channel
        mu1: torch.Tensor = filter2D(img1, tmp_kernel)
        mu2: torch.Tensor = filter2D(img2, tmp_kernel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
        sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
        sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2

        ssim_map = ((2. * mu1_mu2 + self.C1) * (2. * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(ssim_map, min=0, max=1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        return loss


def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int,
        reduction: str = "mean",
        max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`~kornia.losses.SSIM` for details.
    """
    if len(img1.shape) == 3:
        N = 1
        C, H, W = img1.shape
    else:
        N, C, H, W = img1.shape
    img1 = img1.reshape(N, C, H, W)
    img2 = img2.reshape(N, C, H, W)
    return SSIM(window_size, reduction, max_val)(img1, img2)


def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


class Laplacian(nn.Module):
    r"""Creates an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (int): the size of the kernel.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Returns:
        Tensor: the tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> laplace = kornia.filters.Laplacian(5)
        >>> output = laplace(input)  # 2x4x5x5
    """

    def __init__(self,
                 kernel_size: int, border_type: str = 'reflect',
                 normalized: bool = True) -> None:
        super(Laplacian, self).__init__()
        self.kernel_size: int = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized
        self.kernel: torch.Tensor = torch.unsqueeze(
            get_laplacian_kernel2d(kernel_size), dim=0)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               '(kernel_size=' + str(self.kernel_size) + ', ' + \
               'normalized=' + str(self.normalized) + ', ' + \
               'border_type=' + self.border_type + ')'

    def forward(self, input: torch.Tensor):  # type: ignore
        return filter2D(input, self.kernel, self.border_type)


def laplacian(
        input: torch.Tensor,
        kernel_size: int,
        border_type: str = 'reflect',
        normalized: bool = True) -> torch.Tensor:
    r"""Function that returns a tensor using a Laplacian filter.

    See :class:`~kornia.filters.Laplacian` for details.
    """
    return Laplacian(kernel_size, border_type, normalized)(input)


def sam(img1, img2):
    """SAM for (N, C, H, W) image; torch.float32 [0.,1.]."""
    inner_product = (img1 * img2).sum(dim=1)
    img1_spectral_norm = torch.sqrt((img1 ** 2).sum(dim=1))
    img2_spectral_norm = torch.sqrt((img2 ** 2).sum(dim=1))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + eps)).clamp(min=0, max=1)
    cos_theta = cos_theta.reshape(cos_theta.shape[0], -1)
    return torch.mean(torch.acos(cos_theta), dim=-1).mean()


def lap_loss(img, gt):
    img = laplacian(img, 3)
    gt = laplacian(gt, 3)
    return nn.L1Loss()(img, gt)


def rmse(img, gt):
    """RMSE for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img.shape
    img = torch.reshape(img, [N, C, -1])
    gt = torch.reshape(gt, [N, C, -1])
    mse = (img - gt).pow(2).sum(dim=-1)
    rmse = mse / (gt.pow(2).sum(dim=-1) + eps)
    rmse = rmse.mean(dim=-1).sqrt()
    return rmse.mean()


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
