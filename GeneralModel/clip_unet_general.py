# using clip text encoding add with time steps

import sys
import torch

sys.path.append('..')
from core.Attention import zero_module
from core.clip import FrozenCLIPEmbedder
from config.sample_config import get_config
from GeneralModel.unet_util import *


# no prompt# wo ddpm 2.0  # loss
# few-shot  zero-shot finetune
class AblabtionNetwav(nn.Module):
    def __init__(self, channels=None, embed_dim=128, inter_dim=32):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim  # time_embeding dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim), Swish(), nn.Linear(embed_dim, embed_dim))
        self.embed2 = nn.Sequential(nn.Linear(768, embed_dim * 4), Swish(),
                                    nn.Linear(embed_dim * 4, embed_dim * 4),
                                    Swish(), nn.Linear(embed_dim * 4, embed_dim))
        self.conv1 = AdaptionModulate4(1, channels[0], embed_dim)  # (b,1,h,w,c*2+1)->(b,16,h,w,c)

        self.down1 = ResblockDownOneModulate4(channels[0], channels[1], embed_dim)  # ->(b,32,h/4,w/4,c)

        self.down2 = ResblockDownOneModulate4(channels[1], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.down3 = ResblockDownOneModulate4(channels[2], channels[3], embed_dim)  # (b,128,h/8,w/8,c)

        self.conv1_1 = AdaptionModulate4(1, channels[0], embed_dim)  # (b,1,h,w,c*2+1)->(b,32,h,w,c)

        self.conv1_2 = AdaptionModulate4(1, channels[1], embed_dim)  # (b,1,h,w,c*2+1)->(b,32,h,w,c)

        self.down1_1 = ResblockDownOneModulate4(channels[0], channels[1], embed_dim)  # ->(b,64,h/4,w/4,c)

        self.down2_1 = ResblockDownOneModulate4(channels[1], channels[2], embed_dim)  # (b,128,h/4,w/4,c)

        self.down3_1 = ResblockDownOneModulate4(channels[2], channels[3], embed_dim)  # (b,256,h/8,w/8,c)

        self.middle = ResBlockModulateFinal4(channels[3], channels[3], embed_dim)

        self.up1 = ResblockUpOneModulate4(channels[3], channels[2], embed_dim)

        self.up2 = ResblockUpOneModulate4(channels[2], channels[1], embed_dim)

        self.up3 = ResblockUpOneModulate4(channels[1], channels[0], embed_dim)

        self.final1 = ResblockUpOneModulate4(channels[0], channels[0], embed_dim, stride=1, out_padding=0)

        self.final2 = FinalBlockModulate4(channels[0], embed_dim)

        config = get_config()
        self.clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device="cuda")
        self.encode_prompt()
        self.act = Swish()

    def random_prompt(self):
        self.gf2_embeding = nn.Parameter(torch.randn((1, 768)))
        self.wv3_embeding = nn.Parameter(torch.randn((1, 768)))
        self.qb_embeding = nn.Parameter(torch.randn((1, 768)))
        self.wv2_embeding = nn.Parameter(torch.randn((1, 768)))
        self.wv4_embeding = nn.Parameter(torch.randn((1, 768)))
        print(self.gf2_embeding.shape, self.wv3_embeding.shape, self.qb_embeding.shape, self.wv2_embeding.shape)

    def encode_prompt(self):
        self.gf2_embeding = self.clip_text_model.encode(self.get_prompt("GF2"))
        self.wv3_embeding = self.clip_text_model.encode(self.get_prompt("WV3"))
        self.qb_embeding = self.clip_text_model.encode(self.get_prompt("QB"))
        self.wv2_embeding = self.clip_text_model.encode(self.get_prompt("WV2"))
        self.wv4_embeding = self.clip_text_model.encode(self.get_prompt("WV4"))
        print(self.gf2_embeding.shape, self.wv3_embeding.shape, self.qb_embeding.shape, self.wv2_embeding.shape)

    def get_prompt(self, prompt):
        if prompt == "QB":
            return "satellite images of Quick Bird"
        elif prompt == "WV3":
            return "satellite images of WorldView Three"
        elif prompt == "GF2":
            return "satellite images of GaoFen Two"
        elif prompt == "WV2":
            return "satellite images of WorldView Two"
        elif prompt == "WV4":
            return "satellite images of WorldView Four"
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

    def forward(self, x_t, t_input, PAN=None, MS=None, wav=None, prompt=None):
        t = t_input.view(-1, )
        contexts = self.get_embeding(prompt).repeat((x_t.shape[0], 1)).to("cuda")
        contexts = self.act(self.embed2(contexts))
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))
        b, c, h, w = MS.shape
        pan_concat = PAN.repeat(1, c, 1, 1)  # Bsx8x64x64

        cond = torch.sub(pan_concat, MS)
        cond = to3D(cond)
        x_t = to3D(x_t)  # (b,1,c,h,w)->输出：(b,64,c,h,w)   #(b,1,3,h,w)-> (b,64,c不定,h,w)
        wav = to3D(wav[:, :c])
        h_wav = self.conv1_2(wav, embed, contexts)

        h1_1 = self.conv1_1(cond, embed, contexts)  # 转为(b,16,h,w,c) cond部分不应该存在time相关的部分！
        h2_1 = self.down1_1(h1_1, embed, contexts)  # 32
        h2_1 += h_wav
        h3_1 = self.down2_1(h2_1, embed, contexts)  # 16
        h4_1 = self.down3_1(h3_1, embed, contexts)  # 8

        h1 = self.conv1(x_t, embed, contexts)  # 转为(b,16,h,w,c)
        h2 = self.down1(h1, embed, contexts)  # 32
        h2 += h_wav
        h3 = self.down2(h2, embed, contexts)  # 16
        h4 = self.down3(h3, embed, contexts)  # 8

        h = self.middle(h4, embed, contexts)

        h = self.up1(h, embed, h4, h4_1, contexts)  # [0-1] - [0-1] =[-1,1]  [0,1]+[-1,1]=[-1,2]
        h = self.up2(h, embed, h3, h3_1, contexts)
        h = self.up3(h, embed, h2, h2_1, contexts)
        h = self.final1(h, embed, h1, h1_1, contexts)
        h = self.final2(h, embed, contexts)

        return to2D(h)  # (b,1,h,w,c)-->(b,c,h,w)


# no prompt model is training
class Noprompt(nn.Module):
    def __init__(self, channels=None, embed_dim=128, inter_dim=32):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim  # time_embeding dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim), Swish(), nn.Linear(embed_dim, embed_dim))
        self.conv1 = AdaptionModulate(1, channels[0], embed_dim)  # (b,1,h,w,c*2+1)->(b,16,h,w,c)

        self.down1 = ResblockDownOneModulate(channels[0], channels[1], embed_dim)  # ->(b,32,h/4,w/4,c)

        self.down2 = ResblockDownOneModulate(channels[1], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.down3 = ResblockDownOneModulate(channels[2], channels[3], embed_dim)  # (b,128,h/8,w/8,c)

        self.conv1_1 = AdaptionModulate(1, channels[0], embed_dim)  # (b,1,h,w,c*2+1)->(b,32,h,w,c)

        self.down1_1 = ResblockDownOneModulate(channels[0], channels[1], embed_dim)  # ->(b,64,h/4,w/4,c)

        self.down2_1 = ResblockDownOneModulate(channels[1], channels[2], embed_dim)  # (b,128,h/4,w/4,c)

        self.down3_1 = ResblockDownOneModulate(channels[2], channels[3], embed_dim)  # (b,256,h/8,w/8,c)

        self.middle = ResBlockModulateFinal(channels[3], channels[3], embed_dim)

        self.up1 = ResblockUpOneModulate(channels[3], channels[2], embed_dim)

        self.up2 = ResblockUpOneModulate(channels[2], channels[1], embed_dim)

        self.up3 = ResblockUpOneModulate(channels[1], channels[0], embed_dim)

        self.final1 = ResblockUpOneModulate(channels[0], channels[0], embed_dim, stride=1, out_padding=0)

        self.final2 = FinalBlockModulate(channels[0], embed_dim)
        self.act = Swish()

    def forward(self, x_t, t_input, PAN=None, MS=None, wav=None, prompt=None):
        t = t_input.view(-1, )
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))
        b, c, h, w = MS.shape
        pan_concat = PAN.repeat(1, c, 1, 1)  # Bsx8x64x64

        cond = torch.sub(pan_concat, MS)
        cond = to3D(cond)
        x_t = to3D(x_t)  # (b,1,c,h,w)->输出：(b,64,c,h,w)   #(b,1,3,h,w)-> (b,64,c不定,h,w)

        h1_1 = self.conv1_1(cond, embed)  # 转为(b,16,h,w,c) cond部分不应该存在time相关的部分！
        h2_1 = self.down1_1(h1_1, embed)  # 32
        h3_1 = self.down2_1(h2_1, embed)  # 16
        h4_1 = self.down3_1(h3_1, embed)  # 8

        h1 = self.conv1(x_t, embed)  # 转为(b,16,h,w,c)
        h2 = self.down1(h1, embed)  # 32
        h3 = self.down2(h2, embed)  # 16
        h4 = self.down3(h3, embed)  # 8

        h = self.middle(h4, embed)

        h = self.up1(h, embed, h4, h4_1)  # [0-1] - [0-1] =[-1,1]  [0,1]+[-1,1]=[-1,2]
        h = self.up2(h, embed, h3, h3_1)
        h = self.up3(h, embed, h2, h2_1)
        h = self.final1(h, embed, h1, h1_1)
        h = self.final2(h, embed)

        return to2D(h)  # (b,1,h,w,c)-->(b,c,h,w)


class AblabtionNetwavBEST(nn.Module):
    def __init__(self, channels=None, embed_dim=128, inter_dim=32):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.inter_dim = inter_dim  # time_embeding dim
        self.embed = nn.Sequential(nn.Linear(inter_dim, embed_dim), Swish(), nn.Linear(embed_dim, embed_dim))
        self.embed2 = nn.Sequential(nn.Linear(768, embed_dim * 4), Swish(),
                                    nn.Linear(embed_dim * 4, embed_dim * 4),
                                    Swish(), nn.Linear(embed_dim * 4, embed_dim))
        self.conv1 = AdaptionModulateBEST(1, channels[0], embed_dim)  # (b,1,h,w,c*2+1)->(b,16,h,w,c)

        self.down1 = ResblockDownOneModulateBEST(channels[0], channels[1], embed_dim)  # ->(b,32,h/4,w/4,c)

        self.down2 = ResblockDownOneModulateBEST(channels[1], channels[2], embed_dim)  # (b,64,h/4,w/4,c)

        self.down3 = ResblockDownOneModulateBEST(channels[2], channels[3], embed_dim)  # (b,128,h/8,w/8,c)

        self.conv1_1 = AdaptionModulateBEST(1, channels[0], embed_dim)  # (b,1,h,w,c*2+1)->(b,32,h,w,c)

        self.conv1_2 = AdaptionModulateBEST(1, channels[1], embed_dim)  # (b,1,h,w,c*2+1)->(b,32,h,w,c)

        self.down1_1 = ResblockDownOneModulateBEST(channels[0], channels[1], embed_dim)  # ->(b,64,h/4,w/4,c)

        self.down2_1 = ResblockDownOneModulateBEST(channels[1], channels[2], embed_dim)  # (b,128,h/4,w/4,c)

        self.down3_1 = ResblockDownOneModulateBEST(channels[2], channels[3], embed_dim)  # (b,256,h/8,w/8,c)

        self.middle = ResBlockModulateFinalBEST(channels[3], channels[3], embed_dim)

        self.up1 = ResblockUpOneModulateBEST(channels[3], channels[2], embed_dim)

        self.up2 = ResblockUpOneModulateBEST(channels[2], channels[1], embed_dim)

        self.up3 = ResblockUpOneModulateBEST(channels[1], channels[0], embed_dim)

        self.final1 = ResblockUpOneModulateBEST(channels[0], channels[0], embed_dim, stride=1, out_padding=0)

        self.final2 = FinalBlockModulateBEST(channels[0], embed_dim)

        config = get_config()
        self.clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device="cuda")
        self.encode_prompt()
        self.act = Swish()

    def random_prompt(self):
        self.gf2_embeding = nn.Parameter(torch.randn((1, 768)))
        self.wv3_embeding = nn.Parameter(torch.randn((1, 768)))
        self.qb_embeding = nn.Parameter(torch.randn((1, 768)))
        self.wv2_embeding = nn.Parameter(torch.randn((1, 768)))
        self.wv4_embeding = nn.Parameter(torch.randn((1, 768)))
        print(self.gf2_embeding.shape, self.wv3_embeding.shape, self.qb_embeding.shape, self.wv2_embeding.shape)

    def encode_prompt(self):
        self.gf2_embeding = self.clip_text_model.encode(self.get_prompt("GF2"))
        self.wv3_embeding = self.clip_text_model.encode(self.get_prompt("WV3"))
        self.qb_embeding = self.clip_text_model.encode(self.get_prompt("QB"))
        self.wv2_embeding = self.clip_text_model.encode(self.get_prompt("WV2"))
        self.wv4_embeding = self.clip_text_model.encode(self.get_prompt("WV4"))
        print(self.gf2_embeding.shape, self.wv3_embeding.shape, self.qb_embeding.shape, self.wv2_embeding.shape)

    def get_prompt(self, prompt):
        if prompt == "QB":
            return "satellite images of Quick Bird"
        elif prompt == "WV3":
            return "satellite images of WorldView Three"
        elif prompt == "GF2":
            return "satellite images of GaoFen Two"
        elif prompt == "WV2":
            return "satellite images of WorldView-Two"
        elif prompt == "WV4":
            return "satellite images of WorldView-Four"
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

    def forward(self, x_t, t_input, PAN=None, MS=None, wav=None, prompt=None):
        t = t_input.view(-1, )
        contexts = self.get_embeding(prompt).repeat((x_t.shape[0], 1)).to("cuda")
        contexts = self.act(self.embed2(contexts))
        embed = self.act(self.embed(gamma_embedding(t, self.inter_dim)))
        b, c, h, w = MS.shape
        pan_concat = PAN.repeat(1, c, 1, 1)  # Bsx8x64x64

        cond = torch.sub(pan_concat, MS)
        cond = to3D(cond)
        x_t = to3D(x_t)  # (b,1,c,h,w)->输出：(b,64,c,h,w)   #(b,1,3,h,w)-> (b,64,c不定,h,w)
        wav = to3D(wav[:, :c])
        h_wav = self.conv1_2(wav, embed, contexts)

        h1_1 = self.conv1_1(cond, embed, contexts)  # 转为(b,16,h,w,c) cond部分不应该存在time相关的部分！
        h2_1 = self.down1_1(h1_1, embed, contexts)  # 32
        h2_1 += h_wav
        h3_1 = self.down2_1(h2_1, embed, contexts)  # 16
        h4_1 = self.down3_1(h3_1, embed, contexts)  # 8

        h1 = self.conv1(x_t, embed, contexts)  # 转为(b,16,h,w,c)
        h2 = self.down1(h1, embed, contexts)  # 32
        h2 += h_wav
        h3 = self.down2(h2, embed, contexts)  # 16
        h4 = self.down3(h3, embed, contexts)  # 8

        h = self.middle(h4, embed, contexts)

        h = self.up1(h, embed, h4, h4_1, contexts)  # [0-1] - [0-1] =[-1,1]  [0,1]+[-1,1]=[-1,2]
        h = self.up2(h, embed, h3, h3_1, contexts)
        h = self.up3(h, embed, h2, h2_1, contexts)
        h = self.final1(h, embed, h1, h1_1, contexts)
        h = self.final2(h, embed, contexts)

        return to2D(h)  # (b,1,h,w,c)-->(b,c,h,w)


if __name__ == '__main__':
    import numpy as np

    # channels=None, embed_dim=256, inter_dim=32, spectrun_num=4
    model = AblabtionNetwav(channels=[32, 64, 128, 256]).cuda()
    optim_params = []
    for name, param in model.named_parameters():
        if "clip_text" in name:
            continue
        print(name, param.shape)
        optim_params.append(param)
    # summary(model, (1, 256, 256))
    MS = torch.randn((32, 8, 64, 64)).cuda()
    PAN = torch.randn((32, 1, 64, 64)).cuda()
    wav = torch.randn((32, 11, 32, 32)).cuda()
    x = torch.randn((32, 8, 64, 64)).cuda()
    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=32)).cuda()
    output = model(x, time_in, PAN, MS, wav, "QB")
    print(output.shape)
