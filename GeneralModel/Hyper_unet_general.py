# using clip text encoding add with time steps
import functools
import string
import sys
import numpy as np
from einops import rearrange

sys.path.append('..')
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D

from core.clip import FrozenCLIPEmbedder
from config.sample_config import get_config
from GeneralModel.unet_util import *

# 频域Res block,频域attention，频域上下采样
import torch
import torch.nn as nn
import math


def modulated_convTranspose3d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width, in_depth]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width, kernel_depth]
        s,  # Style tensor: [batch_size, in_channels]
        padding=None,  # Padding: int or [padH, padW, padD]
        output_padding=None,  # Padding: int or [padH, padW, padD]
        bias=None,
        stride=None,  # stride: int or [padH, padW, padD]
        dilation=1
):
    """
    https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/training/networks_stylegan3.py
    """
    #     with misc.suppress_tracer_warnings(): # this value will be treated as a constant
    batch_size = int(x.shape[0])
    in_channels, out_channels, kh, kw, kd = w.shape

    w = w.unsqueeze(0)  # [1 I O  k k K]
    w = (w * s.unsqueeze(2).unsqueeze(3))  # [batch,I,1, 1,1,1] -> [batch,I,O, K,K,K]
    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[
                          2:])  # [batch_size, in_channels, in_height, in_width, in_depth]->[1, I*B, in_height, in_width, in_depth]
    w = w.reshape(-1, out_channels, kh, kw, kd)  # [I*B,O, in_height, in_width, in_depth]

    x = torch.nn.functional.conv_transpose3d(input=x, weight=w.to(x.dtype), bias=bias, stride=stride, padding=padding,
                                             dilation=dilation, output_padding=output_padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


def modulated_conv3d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width, in_depth]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width, kernel_depth]
        s,  # Style tensor: [batch_size, in_channels]
        padding=None,  # Padding: int or [padH, padW, padD]
        bias=None,
        stride=None,  # Padding: int or [padH, padW, padD]
        dilation=1
):
    """
    https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/training/networks_stylegan3.py
    """
    #     with misc.suppress_tracer_warnings(): # this value will be treated as a constant
    batch_size = int(x.shape[0])
    out_channels, in_channels, kd, kh, kw = w.shape

    # Modulate weights.
    w = w.unsqueeze(0)  # [1 O I  k k K]
    w = (w * s.unsqueeze(1).unsqueeze(5))  # [batch,O,I, K,K,K]
    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])  # [1, I*B, in_depth, in_height, in_width ]
    w = w.reshape(-1, in_channels, kd, kh, kw)  # [O*B,I, in_depth, in_height, in_width]

    x = torch.nn.functional.conv3d(input=x, weight=w.to(x.dtype), bias=bias, stride=stride, padding=padding,
                                   dilation=dilation, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


def gamma_embedding(gammas, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def Reverse(lst):
    return [ele for ele in reversed(lst)]


def to3D(rgb_image):
    # (b,c,h,w)-->(b,1,c,h,w)
    return rgb_image.unsqueeze(1)


def to2D(tensor_5d):
    # (b,1,c,h,w)-->(b,c,h,w)
    return torch.squeeze(tensor_5d, dim=1)


class ChannelWiseAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelWiseAttention, self).__init__()
        # 使用全局平均池化来获取通道统计信息
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(num_channels * 2, num_channels * 2 // reduction_ratio)
        self.fc2 = nn.Linear(num_channels * 2 // reduction_ratio, num_channels)
        self.relu = Swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入尺寸: [B, C, N, H, W]
        # 首先使用全局平均池化获取每个通道的全局信息
        global_max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
        global_avg_pool = self.global_avg_pool(x)

        # 将全局最大池化和全局平均池化的结果拼接起来(B,C,1,1,1)
        concat_pool = torch.cat((global_max_pool, global_avg_pool), dim=1).reshape(x.size(0), x.size(1) * 2)

        # 通过两个全连接层进行非线性变换
        hidden = self.relu(self.fc1(concat_pool))
        # 输出层使用sigmoid激活函数来获取通道注意力权重
        attention_weights = self.sigmoid(self.fc2(hidden))

        # 将注意力权重应用到原始特征上
        return x * attention_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)


class AdaptionModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim):
        super(AdaptionModulateBEST, self).__init__()
        self.conv20 = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                stride=1, padding=0)  # 通道扩张
        self.conv21 = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=(3, 3, 3),
                                stride=1, padding=1)
        self.act = Swish()
        self.dense2 = Dense(embed_dim, channel_out)

    def forward(self, h, embed, context):
        h = self.conv20(h)
        h = self.act(h)
        h = modulated_conv3d(x=h, w=self.conv21.weight, s=self.dense2(context),
                             stride=self.conv21.stride, padding=self.conv21.padding)
        return h


class ResblockDownOneModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, flag=False):
        "(b,32,h,w,c)->(b,64,h/2,w/2,c)"
        super(ResblockDownOneModulateBEST, self).__init__()
        self.conv20 = ResBlockModulateBEST(channel_in, channel_out, embed_dim, flag)
        self.down = WaveletUPorDown(act=Swish(),
                                    flag=flag,
                                    down=True,
                                    dropout=0.2,
                                    init_scale=0.,
                                    skip_rescale=False,
                                    temb_dim=embed_dim,
                                    zemb_dim=embed_dim,
                                    in_ch=channel_out)
        # self.conv21 = ResBlockModulateBEST(channel_out, channel_out, embed_dim)

    def forward(self, x, embed, prompt):
        h = self.conv20(x, embed, prompt)
        h, skipH = self.down(h, embed, prompt)  # down_samping  并且通道数量扩张
        # h = self.conv21(h, embed, prompt)
        return h, skipH


class ResblockUpOneModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim):
        "(b,32,h,w,c)->(b,64,h/2,w/2,c)"
        super(ResblockUpOneModulateBEST, self).__init__()
        self.up1 = WaveletUPorDown(act=Swish(),
                                   up=True,
                                   dropout=0.2,
                                   init_scale=0.,
                                   skip_rescale=False,
                                   temb_dim=embed_dim,
                                   zemb_dim=embed_dim,
                                   in_ch=channel_out,
                                   hi_in_ch=channel_in)
        self.conv20 = ResBlockModulateBEST(channel_in * 3, channel_out, embed_dim)

    def forward(self, x, embed, skipH, prompt):
        h = self.conv20(x, embed, prompt)
        h = self.up1(h, embed, prompt, skipH)
        return h


class ResBlockModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, flag=False):
        "(b,32,h,w,c)->(b,32,h,w,c)"
        super(ResBlockModulateBEST, self).__init__()
        self.conv20 = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                stride=1, padding=1)
        self.conv21 = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                stride=1, padding=1)
        self.dense1 = Dense(embed_dim, channel_in)
        self.dense2 = Dense(embed_dim, channel_out)
        self.dropout = nn.Dropout(0.2)
        self.res_conv = nn.Conv3d(channel_in, channel_out, 1) if channel_in != channel_out else nn.Identity()
        self.act = Swish()
        self.flag = flag
        # self.norm1 = AdaptiveGroupNorm(min(channel_in // 4, 32), channel_in, embed_dim)
        # self.norm2 = AdaptiveGroupNorm(min(channel_out // 4, 32), channel_out, embed_dim)

    def forward(self, x, embed, prompt):  # norm silu dropout conv为一个block单元
        if not self.flag:
            h = x + self.dense1(embed).unsqueeze(-1)
        else:
            h = x
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv20(h)
        h = self.act(h)
        h = self.dropout(h)
        h = modulated_conv3d(x=h, w=self.conv21.weight, s=self.dense2(prompt),
                             stride=self.conv21.stride, padding=self.conv21.padding)
        return h + self.res_conv(x)


class FinalBlockModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim):
        "(b,32,h,w,c)->(b,32,h,w,c)"
        super(FinalBlockModulateBEST, self).__init__()
        self.conv20 = ResBlockModulateBEST(channel_in * 3, channel_in, embed_dim)
        self.conv21 = ResBlockModulateBEST(channel_in, channel_in, embed_dim)
        self.conv22 = ResBlockModulateBEST(channel_in, channel_in, embed_dim)
        self.conv23 = ResBlockModulateBEST(channel_in, channel_in, embed_dim)
        self.conv24 = nn.Conv3d(channel_in, channel_out, kernel_size=1,
                                stride=1, padding=0)
        self.dense2 = Dense(embed_dim, channel_in)
        self.act = Swish()

    def forward(self, x, embed, prompt):
        h = self.conv20(x, embed, prompt)
        h = self.conv21(h, embed, prompt)
        h = self.conv22(h, embed, prompt)
        h = self.conv23(h, embed, prompt)
        h = self.act(h)
        h = modulated_conv3d(x=h, w=self.conv24.weight, s=self.dense2(prompt),
                             stride=self.conv24.stride, padding=self.conv24.padding)
        return h


from torch.nn.init import _calculate_fan_in_and_fan_out


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


class AdaptiveGroupNorm(nn.Module):

    def __init__(self, num_groups, in_channel, style_dim):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        # B C N H W input-> B C (N H W)
        style = self.style(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class WaveletUPorDown(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,flag = False,
                 dropout=0.2, skip_rescale=False, zemb_dim=None, init_scale=0., hi_in_ch=None):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch

        self.up = up
        self.down = down

        self.Conv_0 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if in_ch != out_ch or up or down:
            self.Conv_2 = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch
        if self.up:
            self.convH_0 = nn.Sequential(
                nn.Conv3d(hi_in_ch * 3, out_ch * 3, kernel_size=3, stride=1, padding=1,
                          groups=3))

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")
        self.dense1 = Dense(zemb_dim, in_ch)
        self.dense2 = Dense(zemb_dim, in_ch)
        self.flag = flag

    def forward(self, x, temb=None, zemb=None, skipH=None):
        B, C, N, H, W = x.shape  # 32 64 8 8 8
        h = self.act(x)
        h = self.Conv_0(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)  # 右侧分支，1*1卷积处理x

        hH = None
        h = rearrange(h, 'b c n h w -> b (c n) h w')  # 32 64*8 8 8
        x = rearrange(x, 'b c n h w -> b (c n) h w')
        if self.up:
            D = h.size(1)
            skipH = self.convH_0(torch.cat(skipH, dim=1) / 2.) * 2.  # B 3C N H W
            skipH = rearrange(skipH, 'b c n h w -> b (c n) h w')  # B 3C*N H W
            h = self.iwt(2. * h, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])
            x = self.iwt(2. * x, skipH[:, :D],
                         skipH[:, D: 2 * D], skipH[:, 2 * D:])  # B C*N H W

        elif self.down:
            h, hLH, hHL, hHH = self.dwt(h)  # b (c n) h/2 w/2
            x, xLH, xHL, xHH = self.dwt(x)  # 转换回去
            hLH = rearrange(hLH, 'b (c n) h w -> b c n h w', c=C)
            hHL = rearrange(hHL, 'b (c n) h w -> b c n h w', c=C)
            hHH = rearrange(hHH, 'b (c n) h w -> b c n h w', c=C)
            hH, _ = (hLH, hHL, hHH), (xLH, xHL, xHH)

            h, x = h / 2., x / 2.

        h = rearrange(h, 'b (c n) h w -> b c n h w', c=C)
        x = rearrange(x, 'b (c n) h w -> b c n h w', c=C)
        if not self.flag:
            h += self.Dense_0(temb)[:, :, None, None, None]  # 512的time转为out_ch
        h = self.act(h)
        h = self.Dropout_0(h)
        h = modulated_conv3d(x=h, w=self.Conv_1.weight, s=self.dense1(zemb),
                             stride=self.Conv_1.stride, padding=self.Conv_1.padding)

        if not self.skip_rescale:
            out = x + h
        else:  # default为True
            out = (x + h) / np.sqrt(2.)

        if not self.down:
            return out
        return out, hH


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=True, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                        eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x, ):
        B, C, N, H, W = x.shape
        x = rearrange(x, 'b c n h w -> b (c n) h w')
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        # B x H x W x H x W 的张量 通道级别的attention计算
        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)  # 对最后的HW attention矩阵softmax
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        x = rearrange(x, 'b (c n) h w -> b c n h w', c=C)
        h = rearrange(h, 'b (c n) h w -> b c n h w', c=C)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


import torch
import torch.nn as nn
import torch.nn.functional as F


class WavBEST(nn.Module):
    def __init__(self, channels=None, embed_dim=128, inter_dim=32):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128]
        self.inter_dim = inter_dim  # time_embeding dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim), Swish(), nn.Linear(embed_dim, embed_dim))
        self.embed2 = nn.Sequential(nn.Linear(768, embed_dim * 4), Swish(), nn.Linear(embed_dim * 4, embed_dim * 4),
                                    Swish(),
                                    nn.Linear(embed_dim * 4, embed_dim))
        self.conv1 = AdaptionModulateBEST(1, channels[0], embed_dim)
        self.conv2 = AdaptionModulateBEST(1, channels[0], embed_dim)

        self.down1 = ResblockDownOneModulateBEST(channels[0], channels[1], embed_dim)  # ->(b,32,h/4,w/4,c)

        self.down2 = ResblockDownOneModulateBEST(channels[1], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.down3 = ResblockDownOneModulateBEST(channels[2], channels[3], embed_dim)  # (b,128,h/8,w/8,c)

        self.down1_1 = ResblockDownOneModulateBEST(channels[0], channels[1], embed_dim, flag=True)  # ->(b,64,h/4,w/4,c)

        self.down2_1 = ResblockDownOneModulateBEST(channels[1], channels[2], embed_dim, flag=True)  # (b,128,h/4,w/4,c)

        self.down3_1 = ResblockDownOneModulateBEST(channels[2], channels[3], embed_dim, flag=True)  # (b,256,h/8,w/8,c)

        self.middle1 = ResBlockModulateBEST(channels[3], channels[3], embed_dim)

        self.up1 = ResblockUpOneModulateBEST(channels[3], channels[2], embed_dim)

        self.up2 = ResblockUpOneModulateBEST(channels[2], channels[1], embed_dim)

        self.up3 = ResblockUpOneModulateBEST(channels[1], channels[0], embed_dim)

        self.final = FinalBlockModulateBEST(channels[0], 1, embed_dim)

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")

        config = get_config()
        self.clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device="cuda")
        self.encode_prompt()
        self.act = Swish()

    def encode_prompt(self):
        self.gf2_embeding = self.clip_text_model.encode(self.get_prompt("GF2"))
        self.wv3_embeding = self.clip_text_model.encode(self.get_prompt("WV3"))
        self.qb_embeding = self.clip_text_model.encode(self.get_prompt("QB"))
        self.wv2_embeding = self.clip_text_model.encode(self.get_prompt("WV2"))
        self.wv4_embeding = self.clip_text_model.encode(self.get_prompt("WV4"))
        print(self.gf2_embeding.shape, self.wv3_embeding.shape, self.qb_embeding.shape, self.wv2_embeding.shape)

    def get_prompt(self, prompt):
        if prompt == "QB":
            return "The QuickBird satellite captures high-resolution images with notable physical properties. Its panchromatic sensor acquires images at a 0.61-meter resolution, providing crisp and detailed grayscale visuals. The multispectral sensor captures images at a 2.44-meter resolution in four spectral bands: blue (450-520 nm), green (520-600 nm), red (630-690 nm), and near-infrared (760-900 nm). These physical properties enable accurate Earth observation, supporting applications in environmental monitoring, land use planning, urban mapping, and disaster management."
        elif prompt == "WV3":
            return "The WorldView-3 satellite captures high-resolution images with exceptional physical properties. Its panchromatic sensor acquires images at a 31 cm resolution, delivering sharp and detailed grayscale visuals. The multispectral sensor captures images at a 1.24 m resolution in eight spectral bands: coastal (400-450 nm), blue (450-510 nm), green (510-580 nm), yellow (585-625 nm), red (630-690 nm), red edge (705-745 nm), near-infrared 1 (770-895 nm), and near-infrared 2 (860-1,040 nm). Additionally, WorldView-3 features a shortwave infrared (SWIR) sensor with 3.7 m resolution in eight bands (1,195-1,385 nm, 1,560-1,660 nm, 2,045-2,110 nm, etc.). These physical properties enable advanced Earth observation, supporting applications in environmental monitoring, land use planning, urban mapping, and disaster response."
        elif prompt == "GF2":
            return "The GaoFen-2 satellite captures high-resolution images with notable physical properties. Its panchromatic sensor acquires images at a 1.0-meter resolution, delivering clear and detailed grayscale visuals. The multispectral sensor captures images at a 4.0-meter resolution in four spectral bands: blue (450-520 nm), green (520-590 nm), red (630-690 nm), and near-infrared (770-890 nm). These physical properties enable accurate Earth observation, supporting applications in urban planning, environmental monitoring, disaster management, and land use analysis."
        elif prompt == "WV2":
            return "The GaoFen-2 satellite captures high-resolution images with notable physical properties. Its panchromatic sensor acquires images at a 0.5-meter resolution, delivering clear and detailed grayscale visuals. The multispectral sensor captures images at a 2.0-meter resolution in four spectral bands: blue (450-520 nm), green (520-590 nm), red (630-690 nm), and near-infrared (770-890 nm). These physical properties enable accurate Earth observation, supporting applications in urban planning, environmental monitoring, disaster management, and land use analysis."
        elif prompt == "WV4":
            return "The WorldView-4 satellite captures high-resolution images with remarkable physical properties. Its panchromatic sensor acquires images at a 31 cm resolution, providing sharp, detailed grayscale visuals. The multispectral sensor captures images at a 1.24 m resolution in four spectral bands: blue (450-510 nm), green (510-580 nm), red (630-690 nm), and near-infrared (770-895 nm). These physical properties enable precise Earth observation, facilitating applications in environmental monitoring, land use planning, and disaster response."
        return None

    def get_embeding(self, prompt):
        if prompt == "QB":
            return self.qb_embeding
        elif prompt == "WV3":
            return self.wv3_embeding
        elif prompt == "GF2":
            return self.gf2_embeding
        elif prompt == "WV2":
            return self.wv2_embeding
        elif prompt == "WV4":
            return self.wv4_embeding
        return None

    def forward(self, x_t, t_input, PAN=None, MS=None, prompt=None):
        t = t_input.view(-1, )
        prompt = self.get_embeding(prompt).repeat((x_t.shape[0], 1)).to("cuda")
        prompt = self.act(self.embed2(prompt))
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))
        b, c, h, w = MS.shape
        pan_concat = PAN.repeat(1, c, 1, 1)  # Bsx8x64x64
        cond = torch.sub(pan_concat, MS)
        cond = to3D(cond)
        x_t = to3D(x_t)

        h0_0 = self.conv1(cond, embed, prompt)
        skipHs1 = []
        h1_1, skipH = self.down1_1(h0_0, embed, prompt)  # 32
        skipHs1.append(skipH)
        h2_1, skipH = self.down2_1(h1_1, embed, prompt)  # 16
        skipHs1.append(skipH)
        h3_1, skipH = self.down3_1(h2_1, embed, prompt)  # 8
        skipHs1.append(skipH)

        skipHs = []
        h0 = self.conv2(x_t, embed, prompt)
        h1, skipH = self.down1(h0, embed, prompt)  # 32
        skipHs.append(skipH)
        h2, skipH = self.down2(h1, embed, prompt)  # 16
        skipHs.append(skipH)
        h3, skipH = self.down3(h2, embed, prompt)  # 8
        skipHs.append(skipH)

        h = self.middle1(h3, embed, prompt)

        h = self.up1(torch.cat([h, h3_1, h3], dim=1), embed, skipHs1.pop(), prompt)
        h = self.up2(torch.cat([h, h2_1, h2], dim=1), embed, skipHs1.pop(), prompt)
        h = self.up3(torch.cat([h, h1_1, h1], dim=1), embed, skipHs1.pop(), prompt)
        h = self.final(torch.cat([h, h0_0, h0], dim=1), embed, prompt)

        return to2D(h)  # (b,1,h,w,c)-->(b,c,h,w)


if __name__ == '__main__':

    # channels=None, embed_dim=256, inter_dim=32, spectrun_num=4
    model = WavBEST(channels=[32, 64, 128, 256]).cuda()
    optim_params = []
    for name, param in model.named_parameters():
        if "clip_text" in name:
            continue
        print(name, param.shape)
        optim_params.append(param)
    print(sum(p.numel() for p in optim_params) / (1024.0 * 1024.0))

    # summary(model, (1, 256, 256))
    # MS = torch.randn((32, 4, 64, 64)).cuda()
    # PAN = torch.randn((32, 1, 64, 64)).cuda()
    # x = torch.randn((32, 4, 64, 64)).cuda()
    # import time
    #
    # time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=32)).cuda()
    # t1 = time.time()
    # output = model(x, time_in, PAN, MS, "QB")
    # t2 = time.time()
    # print(t2 - t1)
    # print(output.shape)

    MS = torch.randn((1, 4, 512, 512)).cuda()
    PAN = torch.randn((1, 1, 512, 512)).cuda()
    x = torch.randn((1, 4, 512, 512)).cuda()
    import time

    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=1)).cuda()
    t1 = time.time()
    output = model(x, time_in, PAN, MS, "QB")
    t2 = time.time()
    print(t2 - t1)
    print(output.shape)
