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
    def __init__(self, channel_in, channel_out, embed_dim):
        super(ResblockUpOne, self).__init__()

        self.conv20 = nn.ConvTranspose3d(in_channels=channel_in, out_channels=channel_out, kernel_size=(3, 3, 1),
                                         stride=(2, 2, 1),
                                         output_padding=(1, 1, 0))
        self.dense1 = Dense(embed_dim, channel_in)

        self.groupnorm1 = nn.BatchNorm3d(channel_in)
        self.act = SiLU()

    def forward(self, x, embed):  # x= hp of ms; y = hp of pan
        h = x + self.dense1(embed).unsqueeze(-1)
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h


class ResblockDownOne(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim):
        "(b,32,h,w,c)->(b,64,h/2,w/2,c)"
        super(ResblockDownOne, self).__init__()
        self.conv20 = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=(3, 3, 3),
                                stride=(2, 2, 1), padding=(0, 0, 1))
        self.dense1 = Dense(embed_dim, channel_in)  # 转换的形状为(batch_size,inter_dim)
        self.groupnorm1 = nn.BatchNorm3d(channel_in)
        # self.groupnorm1 = nn.InstanceNorm2d(channel_in)
        self.act = SiLU()

    def forward(self, x, embed):
        h = x + self.dense1(embed).unsqueeze(-1)  # B channel_in 1 1    B Channel_in H W   B Channel_in H W C
        h = self.act(self.groupnorm1(h))
        h = self.conv20(h)
        return h


def to3D(rgb_image):
    # (b,c,h,w)-->(b,1,h,w,c)
    tensor_5d = rgb_image.permute(0, 2, 3, 1).unsqueeze(1)
    return tensor_5d


def to2D(tensor_5d):
    # (b,1,h,w,c)-->(b,c,h,w)
    rgb_image = torch.squeeze(tensor_5d, dim=1)
    rgb_image = rgb_image.permute(0, 3, 1, 2)
    # print(rgb_image.shape)
    return rgb_image


class cube_block(nn.Module):
    def __init__(self, depth):
        super().__init__()
        midchannel = 16  # input:  B 1 H W C*2+1 =>1*1*1 kernel改变通道数量 B mid_channel H W C*2+1;
        self.cube_block = nn.ModuleList(
            [nn.Conv3d(1, midchannel, kernel_size=(1, 1, 1), padding=0), nn.GELU()])
        for i in range(depth - 2):
            self.cube_block.append(
                nn.Conv3d(in_channels=midchannel, out_channels=midchannel, kernel_size=(3, 3, 3), stride=(1, 1, 2),
                          padding=(1, 1, 0)),
            ).append(nn.GELU())
        self.cube_block.append(  # B 1 H W C;
            nn.Conv3d(in_channels=midchannel, out_channels=midchannel, kernel_size=(3, 3, 3), padding=(0, 0, 1)))

    def forward(self, x):
        fea = x
        for module in self.cube_block:
            fea = module(fea)
        return fea


class General_Unet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.
    输入为XT,MS 输出为PAN"""

    def __init__(self, channels=None, embed_dim=256, inter_dim=32):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128]
        self.inter_dim = inter_dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))

        self.conv1 = cube_block(3)  # (b,1,h,w,c*2+1)->(b,16,h,w,c)

        self.down1 = ResblockDownOne(channels[0], channels[1], embed_dim)  # ->(b,32,h/4,w/4,c)

        self.down2 = ResblockDownOne(channels[1], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.down3 = ResblockDownOne(channels[2], channels[3], embed_dim)  # (b,128,h/8,w/8,c)

        self.up1 = ResblockUpOne(channels[3], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.dense2 = Dense(embed_dim, channels[2])
        self.groupnorm2 = nn.BatchNorm3d(channels[2])
        self.up2 = nn.ConvTranspose3d(in_channels=channels[2] + channels[2], out_channels=channels[1],
                                      kernel_size=(3, 3, 1),
                                      stride=(2, 2, 1),
                                      output_padding=(1, 1, 0))  # ->(b,32,h/2,w/2,c)

        self.dense3 = Dense(embed_dim, channels[1])
        self.groupnorm3 = nn.BatchNorm3d(channels[1])
        self.up3 = nn.ConvTranspose3d(in_channels=channels[1] + channels[1], out_channels=channels[0],
                                      kernel_size=(3, 3, 1),
                                      stride=(2, 2, 1),
                                      output_padding=(1, 1, 0))  # (b,16,h,w,c)

        self.dense4 = Dense(embed_dim, channels[0])
        self.groupnorm4 = nn.BatchNorm3d(channels[0])
        self.final1 = nn.ConvTranspose3d(in_channels=channels[0] + channels[0],
                                         out_channels=channels[0], kernel_size=(3, 3, 1),
                                         stride=(1, 1, 1))  # (b,16,h,w,c)

        self.dense5 = Dense(embed_dim, channels[0])
        self.groupnorm5 = nn.BatchNorm3d(channels[0])
        self.final2 = nn.Conv3d(in_channels=channels[0],
                                out_channels=1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # (b,1,h,w,c)
        # 定义多个通道的输出层，处理不同类型的情况 h w可以是256 1024 512 64各种情况，C也是可以变化的，b也是可以变化的
        # 类似位置编码，不同分辨率下可以插值微调？(1,1,1,C)的形状？
        self.prompt_wv4 = nn.Parameter(torch.randn((channels[-1], 1, 1, 1)))
        self.prompt_wv2 = nn.Parameter(torch.randn((channels[-1], 1, 1, 1)))
        self.prompt_wv3 = nn.Parameter(torch.randn((channels[-1], 1, 1, 1)))
        self.prompt_gf2 = nn.Parameter(torch.randn((channels[-1], 1, 1, 1)))
        self.prompt_qb = nn.Parameter(torch.randn((channels[-1], 1, 1, 1)))
        self.act = SiLU()

    def get_prompt(self, prompt):
        if prompt == "QB":
            return self.prompt_qb
        elif prompt == "WV2":
            return self.prompt_wv2
        elif prompt == "WV4":
            return self.prompt_wv4
        elif prompt == "GF2":
            return self.prompt_gf2
        return self.prompt_wv3

    def forward(self, x_t, t_input, MS=None, PAN=None, prompt=None):
        # 输入B C H W，最终输出也是B C H W的unet结构
        t = t_input.view(-1, )
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))  # 对时间编码信息进行一步处理
        input = torch.concat([x_t, PAN, MS], dim=1)
        input = to3D(input)  # (b,1,h,w,c*2+1)
        h1 = self.conv1(input)  # 转为(b,16,h,w,c)

        h2 = self.down1(h1, embed)  # 32
        h3 = self.down2(h2, embed)  # 64
        h4 = self.down3(h3, embed)  # 128 30 30 8
        weight = self.get_prompt(prompt) #1 1 1 128
        h = self.up1(h4 + weight, embed)  # 64

        h += self.dense2(embed).unsqueeze(-1)
        h = self.groupnorm2(h)
        h = self.act(h)
        h = self.up2(torch.cat([h, h3], dim=1))  # 64

        h += self.dense3(embed).unsqueeze(-1)
        h = self.groupnorm3(h)
        h = self.act(h)
        h = self.up3(torch.cat([h, h2], dim=1))  # 32

        h += self.dense4(embed).unsqueeze(-1)
        h = self.groupnorm4(h)
        h = self.act(h)
        h = self.final1(torch.cat([h, h1], dim=1))  # 32

        h += self.dense5(embed).unsqueeze(-1)
        h = self.groupnorm5(h)
        h = self.act(h)
        h = self.final2(h)

        return to2D(h)  # (b,1,h,w,c)-->(b,c,h,w)


import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchsummary import summary

if __name__ == '__main__':
    import numpy as np

    # channels=None, embed_dim=256, inter_dim=32, spectrun_num=4
    model = General_Unet(channels=[16, 32, 64, 128])
    # summary(model, (1, 256, 256))
    MS = torch.randn((10, 8, 256, 256))
    PAN = torch.randn((10, 1, 256, 256))
    x = torch.randn((10, 8, 256, 256))
    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=10))
    output = model(x, time_in, MS, PAN, "QB")
    print(output.shape)
    # f1 = FlopCountAnalysis(model, x)
    # print(f1.total())
    # print(parameter_count_table(model))
