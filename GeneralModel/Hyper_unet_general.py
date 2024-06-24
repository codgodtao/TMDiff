import sys

import torch.nn.functional

sys.path.append('..')
from core.Attention import zero_module
from core.clip import FrozenCLIPEmbedder
from config.sample_config import get_config
from GeneralModel.unet_util import *


class HyperGeneralModel(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    """

    def __init__(self, channels=None, embed_dim=256, inter_dim=32):
        super().__init__()  # 768*3
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim))
        self.embed2 = nn.Sequential(nn.Linear(768, embed_dim))
        self.conv1 = cube_block(3, channels[0])  # (b,1,h,w,c*2+1)->(b,16,h,w,c)

        self.down1 = modulated_ResblockDownOne(channels[0], channels[1], embed_dim)  # ->(b,32,h/4,w/4,c)

        self.down2 = modulated_ResblockDownOne(channels[1], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.down3 = modulated_ResblockDownOne(channels[2], channels[3], embed_dim)  # (b,128,h/8,w/8,c)

        self.up1 = modulated_ResblockUpOne(channels[3], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.dense2 = Dense(embed_dim, channels[2])
        self.dense2_1 = Dense(embed_dim, channels[2] * 2)
        self.up2 = nn.ConvTranspose3d(in_channels=channels[2] + channels[2], out_channels=channels[1],
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2),
                                      output_padding=(0, 1, 1))  # ->(b,32,h/2,w/2,c)

        self.dense3 = Dense(embed_dim, channels[1])
        self.dense3_1 = Dense(embed_dim, channels[1] * 2)
        self.up3 = nn.ConvTranspose3d(in_channels=channels[1] + channels[1], out_channels=channels[0],
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2),
                                      output_padding=(0, 1, 1))  # (b,16,h,w,c)

        self.dense4 = Dense(embed_dim, channels[0])
        self.dense4_1 = Dense(embed_dim, channels[0] * 2)
        self.final1 = nn.ConvTranspose3d(in_channels=channels[0] + channels[0],
                                         out_channels=channels[0], kernel_size=(1, 3, 3),
                                         stride=(1, 2, 2),
                                         output_padding=(0, 1, 1))

        self.dense5 = Dense(embed_dim, channels[0])
        self.dense5_1 = Dense(embed_dim, channels[0])
        self.final2 = nn.Conv3d(in_channels=channels[0],
                                out_channels=1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # (b,1,h,w,c)
        config = get_config()
        self.clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device="cuda")
        self.encode_prompt()
        self.act = nn.GELU()

    def encode_prompt(self):
        self.gf2_embeding = self.clip_text_model.encode(self.get_prompt("GF2"))
        self.wv3_embeding = self.clip_text_model.encode(self.get_prompt("WV3"))
        self.qb_embeding = self.clip_text_model.encode(self.get_prompt("QB"))

    def get_prompt(self, prompt):
        if prompt == "QB":
            return "QuickBird"
        elif prompt == "WV3":  # A image from QB
            return "WorldView3"
        elif prompt == "GF2":
            return "GaoFen2"
        return None

    def get_embeding(self, prompt):
        if prompt == "QB":
            return self.qb_embeding
        elif prompt == "WV3":
            return self.wv3_embeding
        elif prompt == "GF2":
            return self.gf2_embeding
        return None

    def forward(self, x_t, t_input, MS=None, PAN=None, prompt=None):
        """"
        1. (b,32,h,w,c)统一输入的处理
        2.
        """
        t = t_input.view(-1, )
        contexts = self.get_embeding(prompt).repeat((x_t.shape[0], 1)).to("cuda")
        contexts = self.act(self.embed2(contexts))
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))
        input = torch.concat([x_t, PAN, MS], dim=1)
        input = to3D(input)  # (b,1,h,w,c*2+1)
        h1 = self.conv1(input)  # 转为(b,16,h,w,c)

        h2 = self.down1(h1, embed, contexts)  # 32  64
        h3 = self.down2(h2, embed, contexts)  # 16  128
        h4 = self.down3(h3, embed, contexts)  # 8  256
        h = self.up1(h4, embed, contexts)  # 16  128

        h += self.dense2(embed).unsqueeze(-1)
        h = self.act(h)
        h = modulated_convTranspose3d(x=torch.cat([h, h3], dim=1), w=self.up2.weight, s=self.dense2_1(contexts),
                                      stride=self.up2.stride, padding=self.up2.padding,
                                      output_padding=self.up2.output_padding)
        h += self.up2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        h += self.dense3(embed).unsqueeze(-1)
        h = self.act(h)
        h = modulated_convTranspose3d(x=torch.cat([h, h2], dim=1), w=self.up3.weight, s=self.dense3_1(contexts),
                                      stride=self.up3.stride, padding=self.up3.padding,
                                      output_padding=self.up3.output_padding)
        h += self.up3.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        h += self.dense4(embed).unsqueeze(-1)
        h = self.act(h)
        h = modulated_convTranspose3d(x=torch.cat([h, h1], dim=1), w=self.final1.weight, s=self.dense4_1(contexts),
                                      stride=self.final1.stride, padding=self.final1.padding,
                                      output_padding=self.final1.output_padding)
        h += self.final1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        h += self.dense5(embed).unsqueeze(-1)
        h = self.act(h)
        h = modulated_conv3d(x=h, w=self.final2.weight, s=self.dense5_1(contexts),
                             stride=self.final2.stride, padding=self.final2.padding)
        h += self.final2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return to2D(h)  # (b,1,h,w,c)-->(b,c,h,w)


def modulated_conv2d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
        s,  # Style tensor: [batch_size, in_channels]
        padding=None,  # Padding: int or [padH, padW]
        bias=None,
        stride=None,
        dilation=1
):
    """
    https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/training/networks_stylegan3.py
    """
    #     with misc.suppress_tracer_warnings(): # this value will be treated as a constant
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk]
    w = (w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4))  # [NOIkk]
    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)

    x = torch.nn.functional.conv2d(input=x, weight=w.to(x.dtype), bias=bias, stride=stride, padding=padding,
                                   dilation=dilation, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


if __name__ == '__main__':
    import numpy as np

    # channels=None, embed_dim=256, inter_dim=32, spectrun_num=4
    model = HyperGeneralModel(channels=[32, 64, 128, 256]).cuda()
    optim_params = []
    for name, param in model.named_parameters():
        if "clip_text" in name:
            continue
        print(name, param.shape)
        optim_params.append(param)
    # summary(model, (1, 256, 256))
    MS = torch.randn((1, 8, 64, 64)).cuda()
    PAN = torch.randn((1, 1, 64, 64)).cuda()
    x = torch.randn((1, 8, 64, 64)).cuda()
    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=1)).cuda()
    output = model(x, time_in, MS, PAN, "QB")
    print(output.shape)
