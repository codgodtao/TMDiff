import torch
import torch.nn as nn

import math
import torch.nn.functional as F


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


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def Reverse(lst):
    return [ele for ele in reversed(lst)]


class ResblockUpOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1):
        super(ResblockUpOne, self).__init__()

        self.conv20 = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2,
                                         output_padding=1
                                         )
        self.dense1 = Dense(embed_dim, channel_in)

        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h


class ResblockDownOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, num_group1):
        super(ResblockDownOne, self).__init__()

        self.conv20 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2)
        self.dense1 = Dense(embed_dim, channel_in)  # 转换的形状为(batch_size,inter_dim)
        self.groupnorm1 = nn.GroupNorm(num_group1, num_channels=channel_in)
        # self.groupnorm1 = nn.InstanceNorm2d(channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed)  # 输入的x每个元素都需要加上embed信息
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h


class Multi_branch_Unet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.
    输入为XT,MS 输出为PAN"""

    def __init__(self, channels=None, embed_dim=256, inter_dim=32, spectral_num=4):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim
        self.spectral_num = spectral_num
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))

        self.conv1 = nn.Conv2d(spectral_num * 2, 32, kernel_size=3, stride=1)

        self.down1 = ResblockDownOne(channels[0], channels[1], embed_dim, 4)

        self.down2 = ResblockDownOne(channels[1], channels[2], embed_dim, 32)  # 三个分支合并，通道数量*3

        self.down3 = ResblockDownOne(channels[2], channels[3], embed_dim, 32)

        self.up1 = ResblockUpOne(channels[3], channels[2], embed_dim, 32)

        self.dense2 = Dense(embed_dim, channels[2])
        self.groupnorm2 = nn.GroupNorm(32, num_channels=channels[2])
        self.up2 = nn.ConvTranspose2d(in_channels=channels[2] + channels[2], out_channels=channels[1], kernel_size=3,
                                      stride=2, output_padding=1)

        self.dense3 = Dense(embed_dim, channels[1])
        self.groupnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.up3 = nn.ConvTranspose2d(in_channels=channels[1] + channels[1], out_channels=channels[0],
                                      kernel_size=3, stride=2, output_padding=1)

        self.dense4 = Dense(embed_dim, channels[0])
        self.groupnorm4 = nn.GroupNorm(4, num_channels=channels[0])
        self.final1 = nn.ConvTranspose2d(in_channels=channels[0] + channels[0],
                                         out_channels=channels[0], kernel_size=3, stride=1)

        self.dense5 = Dense(embed_dim, channels[0])
        self.groupnorm5 = nn.GroupNorm(4, num_channels=channels[0])
        self.final2 = nn.Conv2d(channels[0] + spectral_num * 2, self.spectral_num, kernel_size=3, stride=1, padding=1)

        self.act = SiLU()

    def forward(self, x_t, t_input, MS, PAN):  # 修改模型输入内容，修改为MS,PAN,NOISE三个部分，不能切片输入
        # Obtain the Gaussian random feature embedding for t
        PAN = PAN.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        cond = torch.sub(PAN, MS)  # Bsx8x64x64
        t = t_input.view(-1, )
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理

        h1 = self.conv1(torch.concat([x_t, cond], dim=1))  # 32

        h2 = self.down1(h1, embed)  # 64
        h3 = self.down2(h2, embed)  # 128
        h4 = self.down3(h3, embed)  # 256

        # 加入一个中间层结构,从而拥有两个256*256形状的feature map作为对称结构的编码器
        h = self.up1(h4, embed)  # 128

        h += self.dense2(embed)
        h = self.groupnorm2(h)
        h = self.act(h)
        h = self.up2(torch.cat([h, h3], dim=1))  # 64

        h += self.dense3(embed)
        h = self.groupnorm3(h)
        h = self.act(h)
        h = self.up3(torch.cat([h, h2], dim=1))  # 32

        h += self.dense4(embed)
        h = self.groupnorm4(h)
        h = self.act(h)
        h = self.final1(torch.cat([h, h1], dim=1))  # 32

        h += self.dense5(embed)
        h = self.groupnorm5(h)
        h = self.act(h)
        h = self.final2(torch.cat([h, cond, x_t], dim=1))

        return h


import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchsummary import summary

if __name__ == '__main__':
    import numpy as np

    # channels=None, embed_dim=256, inter_dim=32, spectrun_num=4
    model = Multi_branch_Unet(channels=[32, 64, 128, 256], spectral_num=4)
    # summary(model, (1, 256, 256))
    MS = torch.randn((1, 4, 256, 256))
    PAN = torch.randn((1, 1, 256, 256))
    x = torch.randn((1, 4, 256, 256))
    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=1))
    output = model(x, time_in, MS, PAN)
    print(output.shape)
    # f1 = FlopCountAnalysis(model, x)
    # print(f1.total())
    # print(parameter_count_table(model))
