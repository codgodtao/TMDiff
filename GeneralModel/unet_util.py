import torch
import torch.nn as nn
import math
import torch.nn.functional as F




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


class AdaptionModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim):
        super(AdaptionModulateBEST, self).__init__()
        self.conv20 = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                stride=1, padding=0)  # 通道扩张
        self.conv21 = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=(3, 3, 3),
                                stride=1, padding=(1, 0, 0))
        self.act = Swish()
        self.dense2 = Dense(embed_dim, channel_out)

    def forward(self, h, embed, context):
        h = self.conv20(h)
        h = self.act(h)
        h = modulated_conv3d(x=h, w=self.conv21.weight, s=self.dense2(context),
                             stride=self.conv21.stride, padding=self.conv21.padding)
        return h


class ResblockDownOneModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim):
        "(b,32,h,w,c)->(b,64,h/2,w/2,c)"
        super(ResblockDownOneModulateBEST, self).__init__()
        self.down = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=(3, 3, 3),
                              stride=(1, 2, 2), padding=(1, 0, 0))  # down_sampling
        self.conv20 = ResBlockModulateBEST(channel_in, channel_out, embed_dim)
        self.act = Swish()

    def forward(self, x, embed, context):
        h = self.conv20(x, embed, context)
        h = self.act(h)
        h = self.down(h)  # down_samping  并且通道数量扩张
        return h


class ResblockUpOneModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, kernel_size=(3, 3, 3),
                 stride=(1, 2, 2), out_padding=(0, 1, 1), padding=(1, 0, 0)):
        "(b,32,h,w,c)->(b,64,h/2,w/2,c)"
        super(ResblockUpOneModulateBEST, self).__init__()
        self.up1 = nn.ConvTranspose3d(in_channels=channel_out, out_channels=channel_out,
                                      kernel_size=kernel_size, stride=stride,
                                      output_padding=out_padding, padding=padding)
        self.act = Swish()
        self.conv20 = ResBlockModulateBEST(channel_in * 3, channel_out, embed_dim)

    def forward(self, x, embed, skip, cond, context):
        h = torch.cat([x, skip, cond], dim=1)
        h = self.conv20(h, embed, context)
        h = self.act(h)
        h = self.up1(h)
        return h


class ResBlockModulateBEST(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim):
        "(b,32,h,w,c)->(b,32,h,w,c)"
        super(ResBlockModulateBEST, self).__init__()
        self.conv20 = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=(3, 3, 3),
                                stride=1, padding=(1, 1, 1))
        self.conv21 = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=(3, 3, 3),
                                stride=1, padding=(1, 1, 1))
        self.dense1 = Dense(embed_dim, channel_in)
        self.dense2 = Dense(embed_dim, channel_out)
        self.dropout = nn.Dropout(0.2)
        self.res_conv = nn.Conv3d(channel_in, channel_out, 1) if channel_in != channel_out else nn.Identity()
        self.act = Swish()

    def forward(self, x, embed, context):  # norm silu dropout conv为一个block单元
        h = x + self.dense1(embed).unsqueeze(-1)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv20(h)
        h = self.act(h)
        h = self.dropout(h)
        h = modulated_conv3d(x=h, w=self.conv21.weight, s=self.dense2(context),
                             stride=self.conv21.stride, padding=self.conv21.padding)
        return h + self.res_conv(x)


class FinalBlockModulateBEST(nn.Module):
    def __init__(self, channel_out, embed_dim):
        "(b,32,h,w,c)->(b,32,h,w,c)"
        super(FinalBlockModulateBEST, self).__init__()
        self.conv20 = ResBlockModulateBEST(channel_out, channel_out, embed_dim)
        self.conv21 = ResBlockModulateBEST(channel_out, channel_out, embed_dim)
        self.dense2 = Dense(embed_dim, channel_out)
        self.act = Swish()

    def forward(self, x, embed, context):
        h = self.conv20(x, embed, context)
        h = self.act(h)
        h = self.conv21(h, embed, context)
        return h
